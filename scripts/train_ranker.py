"""
Train the cross-modal TemporalRanker.

Run this on Kaggle (2x T4) or Colab free T4.
Expected training time: 2–3 hours for NExT-QA, 10 epochs.

Usage:
    python scripts/train_ranker.py --config configs/nextqa.yaml
    python scripts/train_ranker.py --config configs/msvd.yaml --override training.epochs=3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.io_utils import save_checkpoint, load_embeddings_memmap
from src.models.temporal_ranker import TemporalRanker, contrastive_ranking_loss

logger = get_logger(__name__, log_dir="results/logs")


def parse_args():
    parser = argparse.ArgumentParser(description="Train QUEST temporal ranker")
    parser.add_argument("--config", required=True, help="Path to dataset config yaml")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides key=val")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TemporalRanker(
        embed_dim=cfg.ranker.embed_dim,
        num_heads=cfg.ranker.num_heads,
        num_layers=cfg.ranker.num_layers,
        dropout=cfg.ranker.dropout,
    ).to(device)

    # DataParallel on Kaggle 2×T4
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    scaler = GradScaler(enabled=cfg.training.mixed_precision)

    start_epoch = 0
    if args.resume:
        from src.utils.io_utils import load_checkpoint
        ckpt = load_checkpoint(model, args.resume, optimizer, device)
        start_epoch = ckpt["epoch"] + 1

    # ── Training loop (pseudocode — replace DataLoader with your actual dataset) ──
    logger.info("Starting training…")
    best_val_acc = 0.0

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # TODO: replace with your actual DataLoader over precomputed embeddings
        # Each batch should provide:
        #   question_emb: (B, 512), frame_embs: (B, K, 512),
        #   temporal_pos: (B, K, 3), labels: (B, K)
        logger.info(f"Epoch {epoch+1}/{cfg.training.epochs} — replace this with real DataLoader")
        break   # Remove this break once DataLoader is connected

        for batch in train_loader:
            question_emb = batch["question_emb"].to(device)
            frame_embs = batch["frame_embs"].to(device)
            temporal_pos = batch["temporal_pos"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast(enabled=cfg.training.mixed_precision):
                scores, uncertainty = model(question_emb, frame_embs, temporal_pos)
                loss = contrastive_ranking_loss(scores, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1} | loss={avg_loss:.4f}")

        if (epoch + 1) % cfg.training.save_every_n_epochs == 0:
            save_checkpoint(
                model, optimizer, epoch,
                metrics={"train_loss": avg_loss},
                save_path=f"{cfg.paths.checkpoint_dir}/epoch_{epoch+1}.pt",
            )

    logger.info("Training complete")


if __name__ == "__main__":
    main()
