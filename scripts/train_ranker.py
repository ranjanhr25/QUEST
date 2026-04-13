#!/usr/bin/env python3
"""
Train the cross-modal transformer ranker for QUEST.

Designed to run on:
  - Kaggle 2×T4   (recommended — 2×16GB, ~3h for 3 epochs)
  - Google Colab T4  (1×16GB, ~5h for 3 epochs)
  - Local GPU (any 8GB+)

Usage:
    # Train from scratch
    python scripts/train_ranker.py --config configs/nextqa.yaml

    # Resume from latest checkpoint
    python scripts/train_ranker.py --config configs/nextqa.yaml --resume

    # Quick smoke-test (100 train, 50 val samples, 1 epoch)
    python scripts/train_ranker.py --config configs/nextqa.yaml --smoke_test

Prerequisites (run first):
    python -m src.data.frame_extractor  ...   # extract frames
    python -m src.data.preprocess       ...   # compute CLIP embeddings
    python scripts/build_index.py       ...   # build FAISS indexes
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import NExTQADataset, RankerTrainDataset
from src.ranking.fine_ranker import RankerTrainer
from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger

log = get_logger("train_ranker")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(
    cfg: object,
    smoke_test: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders for the RankerTrainDataset."""

    ann_dir  = Path(cfg.paths.annotations)
    emb_dir  = Path(cfg.paths.embeddings_dir)
    frame_idx = Path(cfg.paths.frames_root).parent / "frame_index.json"

    train_ann = ann_dir / cfg.annotation_files["train"]
    val_ann   = ann_dir / cfg.annotation_files["val"]

    for p in [train_ann, val_ann, frame_idx,
              emb_dir / "frame_embeddings.npy",
              emb_dir / "video_index.json"]:
        if not Path(p).exists():
            log.error("Required file missing", path=str(p))
            sys.exit(1)

    # Base datasets
    train_base = NExTQADataset(train_ann, frame_idx, split="train")
    val_base   = NExTQADataset(val_ann,   frame_idx, split="val")

    if smoke_test:
        train_base = train_base.stratified_subset(100)
        val_base   = val_base.stratified_subset(50)
        log.info("Smoke test: subsampled datasets", train=len(train_base), val=len(val_base))

    common_kwargs = dict(
        embeddings_path=emb_dir / "frame_embeddings.npy",
        video_index_path=emb_dir / "video_index.json",
        top_k_positive=3,
        n_candidates=cfg.retrieval.top_k_coarse,
        embed_dim=cfg.retrieval.embed_dim,
    )

    train_ds = RankerTrainDataset(
        base_dataset=train_base,
        question_embeddings_path=emb_dir / "question_embeddings_train.npy",
        question_index_path=emb_dir / "question_index_train.json",
        **common_kwargs,
    )
    val_ds = RankerTrainDataset(
        base_dataset=val_base,
        question_embeddings_path=emb_dir / "question_embeddings_val.npy",
        question_index_path=emb_dir / "question_index_val.json",
        **common_kwargs,
    )

    nw = cfg.training.num_workers if not smoke_test else 0
    bs = cfg.training.batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )

    log.info(
        "DataLoaders ready",
        train_samples=len(train_ds),
        val_samples=len(val_ds),
        train_batches=len(train_loader),
        batch_size=bs,
    )
    return train_loader, val_loader


def main() -> None:
    p = argparse.ArgumentParser(description="Train the QUEST temporal ranker.")
    p.add_argument("--config",     default="configs/nextqa.yaml", type=Path)
    p.add_argument("--resume",     action="store_true", help="Resume from latest checkpoint")
    p.add_argument("--smoke_test", action="store_true", help="Quick run to verify the pipeline")
    p.add_argument("--epochs",     default=None, type=int, help="Override training.epochs")
    p.add_argument("--batch_size", default=None, type=int, help="Override training.batch_size")
    args = p.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    if args.epochs is not None:
        cfg.training["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.training["batch_size"] = args.batch_size
    if args.smoke_test:
        cfg.training["epochs"] = 1

    set_seed(cfg.training.seed)

    log.info(
        "Training configuration",
        epochs=cfg.training.epochs,
        batch_size=cfg.training.batch_size,
        lr=cfg.training.lr,
        fp16=cfg.training.fp16,
        gpus=torch.cuda.device_count(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    train_loader, val_loader = build_loaders(cfg, smoke_test=args.smoke_test)

    trainer = RankerTrainer.from_config(cfg)
    trainer.train(train_loader, val_loader, resume=args.resume)

    # Final validation metrics
    log.info("Running final validation...")
    final_metrics = trainer.evaluate(val_loader)
    log.info("Final validation metrics", **{k: round(v, 4) for k, v in final_metrics.items()})

    log.info("train_ranker.py complete")


if __name__ == "__main__":
    main()