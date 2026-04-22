"""
Ranker training and evaluation for QUEST Stage 2.

This module contains RankerTrainer — the only trained component in QUEST.
Everything else (CLIP, FAISS, LLaVA) is frozen.

Training objective:
  loss = rank_weight  * listnet_loss(predicted_scores, clip_pseudo_labels)
       + uncert_weight * bce_loss(predicted_uncertainty, uncertainty_gt)

ListNet loss treats frame relevance as a probability distribution:
  - Predicted distribution: softmax(ranker_scores)
  - Target distribution:    softmax(clip_sim_scores / temperature)
  - Loss = KL divergence from target to predicted (cross-entropy form)

This is a listwise ranking loss that has a smooth gradient even when
the top-1 frame is correct, allowing the model to fine-tune the full
ranking rather than just a binary "correct/incorrect" signal.

Uncertainty label during pre-training is always 0.0 (confident) since
we don't yet know which queries will be hard. After the first evaluation
pass you can set uncertainty_gt based on retrieval errors — but for Day 2
the zero-label baseline is sufficient.

Architecture: see src/models/temporal_ranker.py (TemporalRanker, ~50M params).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.models.temporal_ranker import TemporalRanker
from src.utils.io_utils import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
)
from src.utils.logger import get_logger

log = get_logger("fine_ranker")


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def listnet_loss(
    scores: torch.Tensor,       # (B, K) raw ranker scores
    targets: torch.Tensor,      # (B, K) soft relevance labels in [0, 1]
    pad_mask: Optional[torch.Tensor] = None,  # (B, K) True = padded
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    ListNet ranking loss (top-1 probability version).

    Converts both scores and targets to probability distributions over
    frames via softmax, then computes cross-entropy between them.

    Args:
        scores:      Raw ranker output scores, shape (B, K).
        targets:     Soft relevance labels (e.g. normalised CLIP similarities), (B, K).
        pad_mask:    Boolean mask where True = padded position to ignore.
        temperature: Sharpens the target distribution. Lower → harder targets.

    Returns:
        Scalar loss value.
    """
    if pad_mask is not None:
        scores = scores.masked_fill(pad_mask, float("-inf"))
        targets = targets.masked_fill(pad_mask, float("-inf"))

    # Convert to probability distributions
    log_pred = F.log_softmax(scores, dim=-1)            # (B, K)
    target_probs = F.softmax(targets / temperature, dim=-1)  # (B, K)

    # Mask out padded positions from target_probs to avoid NaN
    if pad_mask is not None:
        target_probs = target_probs.masked_fill(pad_mask, 0.0)

    # Cross-entropy: -sum(target * log_pred)
    loss = -(target_probs * log_pred)

    if pad_mask is not None:
        loss = loss.masked_fill(pad_mask, 0.0)

    return loss.sum(dim=-1).mean()


def uncertainty_bce_loss(
    uncertainty: torch.Tensor,  # (B,) predicted uncertainty in [0, 1]
    uncertainty_gt: torch.Tensor,  # (B,) ground truth label {0.0, 1.0}
) -> torch.Tensor:
    """Binary cross-entropy on the uncertainty head."""
    return F.binary_cross_entropy(uncertainty, uncertainty_gt.float())


# ---------------------------------------------------------------------------
# Metric helpers (used during evaluation)
# ---------------------------------------------------------------------------

def recall_at_k_from_ranker(
    predicted_scores: torch.Tensor,  # (K,)
    relevance_labels: torch.Tensor,  # (K,)
    k: int,
) -> float:
    """
    Recall@K: fraction of top-3 CLIP frames that appear in the ranker's top-K.
    Used as a proxy metric during training to check the ranker is learning.
    """
    top_k_pred = set(predicted_scores.topk(k).indices.cpu().tolist())
    # Ground truth: top-3 CLIP frames
    n_gt = min(3, len(relevance_labels))
    top_gt = set(relevance_labels.topk(n_gt).indices.cpu().tolist())
    if not top_gt:
        return 1.0
    return len(top_k_pred & top_gt) / len(top_gt)


# ---------------------------------------------------------------------------
# RankerTrainer
# ---------------------------------------------------------------------------

class RankerTrainer:
    """
    Orchestrates training and evaluation of the TemporalRanker.

    Supports:
      - Multi-GPU training via DataParallel (auto-detected).
      - Mixed-precision training via torch.cuda.amp.
      - Checkpoint save/load with atomic writes.
      - Resume from latest checkpoint.
      - Cosine LR schedule with linear warmup.

    Args:
        model:    TemporalRanker instance.
        cfg:      DotDict config (from load_config).
        device:   torch.device to train on.
    """

    def __init__(
        self,
        model: TemporalRanker,
        cfg: Any,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.fp16 = cfg.training.fp16 and device.type == "cuda"

        # ── Model setup ──────────────────────────────────────────────────
        self.model = model.to(device)
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            log.info("Using DataParallel", n_gpus=n_gpus)
            self.model = nn.DataParallel(self.model)

        # ── Optimiser ────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        # Scheduler and scaler initialised in train()
        self.scheduler: Optional[Any] = None
        self.scaler = GradScaler() if self.fp16 else None

        # ── Loss weights ─────────────────────────────────────────────────
        self.rank_weight = cfg.training.loss_rank_weight
        self.uncert_weight = cfg.training.loss_uncertainty_weight

        # ── Paths ────────────────────────────────────────────────────────
        self.ckpt_dir = Path(cfg.paths.checkpoints)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")

    @classmethod
    def from_config(cls, cfg: Any, device: Optional[torch.device] = None) -> "RankerTrainer":
        """Build RankerTrainer from a loaded config."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TemporalRanker.from_config(cfg)
        log.info(
            "TemporalRanker created",
            params=f"{model.count_parameters() / 1e6:.1f}M",
            device=str(device),
        )
        return cls(model=model, cfg=cfg, device=device)

    # ── Public API ────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume: bool = False,
    ) -> None:
        """
        Full training loop.

        Args:
            train_loader: DataLoader for RankerTrainDataset (train split).
            val_loader:   DataLoader for RankerTrainDataset (val split).
            resume:       If True, look for the latest checkpoint in ckpt_dir
                          and resume from it.
        """
        cfg = self.cfg.training
        n_epochs = cfg.epochs
        total_steps = n_epochs * len(train_loader)

        # Cosine schedule with linear warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.lr,
            total_steps=total_steps,
            pct_start=cfg.warmup_steps / total_steps,
            anneal_strategy="cos",
        )

        start_epoch = 0
        if resume:
            latest = find_latest_checkpoint(self.ckpt_dir)
            if latest:
                meta = load_checkpoint(
                    latest, self._raw_model, self.optimizer, self.scheduler, self.device
                )
                self.global_step = meta["step"]
                start_epoch = meta["epoch"]
                self.best_val_loss = meta["metrics"].get("val_loss", float("inf"))
                log.info("Resumed training", from_ckpt=str(latest), epoch=start_epoch)
            else:
                log.warning("No checkpoint found for resume — starting fresh")

        log.info(
            "Training started",
            epochs=n_epochs,
            total_steps=total_steps,
            fp16=self.fp16,
            warmup_steps=cfg.warmup_steps,
        )

        for epoch in range(start_epoch, n_epochs):
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)

            log.info(
                f"Epoch {epoch + 1}/{n_epochs}",
                train_loss=round(train_metrics["loss"], 4),
                val_loss=round(val_metrics["loss"], 4),
                val_recall=round(val_metrics.get("recall@8", 0.0), 4),
                lr=round(self.optimizer.param_groups[0]["lr"], 6),
            )

            # Save best checkpoint
            val_loss = val_metrics["loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save(
                    filename="ranker_best.pt",
                    epoch=epoch + 1,
                    metrics=val_metrics,
                )
                log.info("New best checkpoint saved", val_loss=round(val_loss, 4))

            # Save periodic checkpoint
            self._save(
                filename=f"ranker_epoch{epoch + 1:02d}.pt",
                epoch=epoch + 1,
                metrics=val_metrics,
            )

        log.info("Training complete", best_val_loss=round(self.best_val_loss, 4))

    def evaluate(self, val_loader: DataLoader) -> dict[str, float]:
        """
        Run a full validation pass.

        Returns a dict with keys:
            loss, rank_loss, uncert_loss, recall@8, recall@16

        Args:
            val_loader: DataLoader for the validation split.

        Returns:
            Metrics dictionary.
        """
        self.model.eval()

        total_loss = 0.0
        total_rank_loss = 0.0
        total_uncert_loss = 0.0
        total_recall_8 = 0.0
        total_recall_16 = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                outputs, loss, rank_loss, uncert_loss = self._forward_loss(batch)
                total_loss += loss.item()
                total_rank_loss += rank_loss.item()
                total_uncert_loss += uncert_loss.item()

                # Recall@K (averaged across batch)
                relevance_scores = outputs["relevance"]  # (B, K)
                rel_labels = batch["relevance"].to(self.device)
                pad_mask = batch["pad_mask"].to(self.device)

                # Mask padded positions before topk
                masked_scores = relevance_scores.masked_fill(pad_mask, float("-inf"))

                B = relevance_scores.shape[0]
                r8_sum, r16_sum = 0.0, 0.0
                for b in range(B):
                    r8_sum += recall_at_k_from_ranker(masked_scores[b], rel_labels[b], k=8)
                    r16_sum += recall_at_k_from_ranker(masked_scores[b], rel_labels[b], k=16)
                total_recall_8 += r8_sum / B
                total_recall_16 += r16_sum / B

                n_batches += 1

        n = max(n_batches, 1)
        return {
            "loss": total_loss / n,
            "rank_loss": total_rank_loss / n,
            "uncert_loss": total_uncert_loss / n,
            "recall@8": total_recall_8 / n,
            "recall@16": total_recall_16 / n,
        }

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model weights from a checkpoint path (inference mode)."""
        load_checkpoint(path, self._raw_model, device=self.device, strict=True)

    # ── Private helpers ───────────────────────────────────────────────────

    def _train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        """Run one full training epoch."""
        self.model.train()
        cfg = self.cfg.training

        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for step, batch in enumerate(loader):
            self.optimizer.zero_grad(set_to_none=True)

            if self.fp16:
                with autocast():
                    outputs, loss, rank_loss, uncert_loss = self._forward_loss(batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs, loss, rank_loss, uncert_loss = self._forward_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            if (step + 1) % 50 == 0:
                elapsed = time.time() - t0
                fps = (step + 1) * loader.batch_size / elapsed
                log.info(
                    f"Epoch {epoch + 1} step {step + 1}/{len(loader)}",
                    loss=round(loss.item(), 4),
                    rank_loss=round(rank_loss.item(), 4),
                    uncert_loss=round(uncert_loss.item(), 4),
                    samples_per_sec=round(fps, 0),
                    lr=round(self.optimizer.param_groups[0]["lr"], 6),
                )

        return {"loss": total_loss / max(n_batches, 1)}

    def _forward_loss(
        self,
        batch: dict[str, Any],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One forward pass + loss computation.

        Moves tensors to device, runs TemporalRanker forward, computes combined loss.

        Returns:
            (outputs_dict, total_loss, rank_loss, uncert_loss)
        """
        frame_embs   = batch["frame_embs"].to(self.device, non_blocking=True)   # (B, K, D)
        q_emb        = batch["q_emb"].to(self.device, non_blocking=True)        # (B, D)
        temporal_pos = batch["temporal_pos"].to(self.device, non_blocking=True) # (B, K)
        relevance    = batch["relevance"].to(self.device, non_blocking=True)    # (B, K)
        uncert_gt    = batch["uncertainty_gt"].to(self.device, non_blocking=True)  # (B,)
        pad_mask     = batch["pad_mask"].to(self.device, non_blocking=True)     # (B, K)

        # Forward pass
        outputs = self.model(
            frame_embs=frame_embs,
            q_emb=q_emb,
            temporal_pos=temporal_pos,
            pad_mask=pad_mask,
        )

        # Handle DataParallel output (may return different dict key structure)
        if isinstance(outputs, dict):
            pred_scores = outputs["relevance"]       # (B, K)
            pred_uncert = outputs["uncertainty"]     # (B,)
        else:
            raise RuntimeError(f"Unexpected model output type: {type(outputs)}")

        # Ranking loss (ListNet)
        r_loss = listnet_loss(pred_scores, relevance, pad_mask=pad_mask)

        # Uncertainty loss (BCE)
        u_loss = uncertainty_bce_loss(pred_uncert, uncert_gt)

        total = self.rank_weight * r_loss + self.uncert_weight * u_loss

        return outputs, total, r_loss, u_loss

    def _save(self, filename: str, epoch: int, metrics: dict[str, float]) -> None:
        """Save a named checkpoint."""
        save_checkpoint(
            path=self.ckpt_dir / filename,
            model=self._raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            epoch=epoch,
            metrics=metrics,
            fp16=self.fp16,
        )

    @property
    def _raw_model(self) -> TemporalRanker:
        """Unwrap DataParallel to get the underlying model."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def get_model_for_inference(self) -> TemporalRanker:
        """Return the unwrapped model in eval mode for inference."""
        m = self._raw_model
        m.eval()
        return m