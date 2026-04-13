"""
I/O helpers: checkpoint save/load, memory-mapped embedding arrays.

Memory-mapped arrays let us store millions of frame embeddings on disk
and access them without loading everything into RAM — essential when
a dataset has 50k+ videos.
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Checkpoints ───────────────────────────────────────────────────────────────

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    save_path: str,
) -> None:
    """
    Save model weights, optimizer state, epoch, and metrics to disk.

    Args:
        model:     PyTorch model to save.
        optimizer: Optimizer whose state to include.
        epoch:     Current epoch number.
        metrics:   Dict of metric name → value to record alongside checkpoint.
        save_path: Full path including filename, e.g. "results/checkpoints/epoch_3.pt".
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        save_path,
    )
    logger.info(f"Checkpoint saved → {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Load checkpoint into model (and optionally optimizer).

    Args:
        model:           Model to load weights into.
        checkpoint_path: Path to .pt checkpoint file.
        optimizer:       If provided, optimizer state is also restored.
        device:          Device to map tensors to.

    Returns:
        The full checkpoint dict (contains epoch, metrics, etc.).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {ckpt['epoch']})")
    return ckpt


# ── Memory-mapped embedding storage ───────────────────────────────────────────

def save_embeddings_memmap(
    embeddings: np.ndarray,
    save_path: str,
) -> None:
    """
    Save a float32 embedding matrix to a memory-mapped file.

    The shape is stored in a companion .meta.npy file so we can
    reload it without knowing dimensions ahead of time.

    Args:
        embeddings: Array of shape (N, D) where N = number of frames, D = embed dim.
        save_path:  Path to .npy file (companion .meta.npy is auto-created).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mm = np.memmap(save_path, dtype="float32", mode="w+", shape=embeddings.shape)
    mm[:] = embeddings[:]
    mm.flush()

    np.save(str(save_path) + ".meta.npy", np.array(embeddings.shape))
    logger.info(f"Embeddings saved ({embeddings.shape}) → {save_path}")


def load_embeddings_memmap(save_path: str, mode: str = "r") -> np.memmap:
    """
    Load a memory-mapped embedding file.

    Args:
        save_path: Path to the .npy file written by save_embeddings_memmap.
        mode:      "r" (read-only) or "r+" (read-write). Always use "r" for inference.

    Returns:
        numpy.memmap of shape (N, D) — reads from disk on access, not loaded into RAM.
    """
    shape = tuple(np.load(str(save_path) + ".meta.npy").tolist())
    return np.memmap(save_path, dtype="float32", mode=mode, shape=shape)
