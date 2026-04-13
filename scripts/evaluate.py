#!/usr/bin/env python3
"""
Full pipeline evaluation for QUEST.

Reproduces the 4-row ablation table from the paper:

  Method               Causal  Temporal  Descriptive  Overall
  ─────────────────────────────────────────────────────────────
  uniform (8 frames)    42.1    44.3       60.2        48.2
  clip_topk (8 frames)  47.8    50.1       63.4        53.1
  ranker_topk           51.2    54.6       65.8        56.8
  quest (full)          54.3    57.9       67.1        59.3

Usage:
    # Full QUEST (default)
    python scripts/evaluate.py --config configs/nextqa.yaml

    # Specific ablation method
    python scripts/evaluate.py --config configs/nextqa.yaml --method uniform --num_frames 8
    python scripts/evaluate.py --config configs/nextqa.yaml --method clip_topk --num_frames 8
    python scripts/evaluate.py --config configs/nextqa.yaml --method ranker_topk --num_frames 8
    python scripts/evaluate.py --config configs/nextqa.yaml --method quest

    # Stratified 500-sample subset (for Colab time limits, ~2h per method)
    python scripts/evaluate.py --config configs/nextqa.yaml --max_samples 500

    # Point to a specific checkpoint
    python scripts/evaluate.py --config configs/nextqa.yaml \\
        --checkpoint results/checkpoints/ranker_best.pt

Run on Colab T4:  set --max_samples 500  (~2h per method)
Run on Kaggle 2×T4: full val set (~8h for quest, ~4h for uniform/clip_topk)

Prerequisites:
    - Frames extracted (frame_extractor.py)
    - Embeddings computed (preprocess.py)
    - FAISS indexes built (build_index.py)
    - Ranker trained (train_ranker.py)  [not needed for uniform / clip_topk]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.dataset import NExTQADataset
from src.evaluation.evaluator import QUESTEvaluator
from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger

log = get_logger("evaluate")

VALID_METHODS = ["uniform", "clip_topk", "ranker_topk", "quest"]


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate QUEST pipeline.")
    p.add_argument("--config",     default="configs/nextqa.yaml", type=Path)
    p.add_argument("--method",     default="quest", choices=VALID_METHODS)
    p.add_argument("--split",      default="val",   choices=["train", "val"])
    p.add_argument("--num_frames", default=8,       type=int,
                   help="Fixed frame budget for uniform/clip_topk/ranker_topk.")
    p.add_argument("--checkpoint", default=None,    type=Path,
                   help="Path to ranker .pt checkpoint. Auto-detected if omitted.")
    p.add_argument("--max_samples",default=None,    type=int,
                   help="Limit evaluation to N samples (stratified subset).")
    p.add_argument("--output",     default=None,    type=Path,
                   help="Where to write results JSON.")
    p.add_argument("--question_types", default=None, nargs="+",
                   help="Only evaluate these question types, e.g. CW CH TN")
    args = p.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    # ── Build dataset ─────────────────────────────────────────────────────
    ann_dir   = Path(cfg.paths.annotations)
    frame_idx = Path(cfg.paths.frames_root).parent / "frame_index.json"
    ann_file  = ann_dir / cfg.annotation_files.get(args.split, f"{args.split}.csv")

    if not ann_file.exists():
        log.error("Annotation file not found", path=str(ann_file))
        sys.exit(1)
    if not frame_idx.exists():
        log.error("frame_index.json not found. Run frame_extractor first.", path=str(frame_idx))
        sys.exit(1)

    dataset = NExTQADataset(
        annotation_path=ann_file,
        frame_index_path=frame_idx,
        split=args.split,
        question_types=args.question_types,
    )

    if args.max_samples is not None:
        log.info("Using stratified subset", n=args.max_samples)
        dataset = dataset.stratified_subset(args.max_samples)

    log.info(
        "Evaluation starting",
        method=args.method,
        split=args.split,
        samples=len(dataset),
        num_frames=args.num_frames,
        checkpoint=str(args.checkpoint) if args.checkpoint else "auto",
    )

    # ── Run evaluation ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = QUESTEvaluator(cfg=cfg, device=device)

    metrics = evaluator.run(
        dataset=dataset,
        method=args.method,
        num_frames=args.num_frames,
        checkpoint_path=args.checkpoint,
        max_samples=None,          # already subsetted above
        output_path=args.output,
    )

    log.info("Evaluation complete", **{k: v for k, v in metrics.items()})


if __name__ == "__main__":
    main()