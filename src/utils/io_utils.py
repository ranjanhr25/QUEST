"""
I/O utilities for QUEST.

Covers:
  - Checkpoint save / load (model + optimiser + metadata).
  - Float16 numpy memmap creation and loading.
  - Video-to-row index for the embeddings memmap.
  - Atomic file writes (write-then-rename) to avoid corrupt checkpoints.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.logger import get_logger

log = get_logger("io_utils")


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    step: int,
    epoch: int,
    metrics: dict[str, float] | None = None,
    fp16: bool = False,
) -> None:
    """
    Atomically save a training checkpoint.

    Writes to a temp file first, then renames to avoid corruption on crash.
    If fp16=True the model state dict is stored in half precision (~50% smaller).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = model.state_dict()
    if fp16:
        state = {k: v.half() for k, v in state.items()}

    payload = {
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "step": step,
        "epoch": epoch,
        "metrics": metrics or {},
    }

    # Atomic write
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.close(tmp_fd)
        torch.save(payload, tmp_path)
        shutil.move(tmp_path, path)
        log.info("Checkpoint saved", path=str(path), step=step, epoch=epoch)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load a checkpoint.  Returns the metadata dict (step, epoch, metrics).
    Model weights are cast back to fp32 automatically if they were saved in fp16.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device, weights_only=False)

    # Cast fp16 → fp32 before loading
    state = {k: v.float() if v.dtype == torch.float16 else v
             for k, v in payload["model_state"].items()}
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if missing:
        log.warning("Missing keys in checkpoint", keys=missing)
    if unexpected:
        log.warning("Unexpected keys in checkpoint", keys=unexpected)

    if optimizer is not None and payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state"):
        scheduler.load_state_dict(payload["scheduler_state"])

    log.info("Checkpoint loaded", path=str(path),
             step=payload.get("step"), epoch=payload.get("epoch"))
    return {"step": payload.get("step", 0),
            "epoch": payload.get("epoch", 0),
            "metrics": payload.get("metrics", {})}


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Return the most recent .pt file in checkpoint_dir, or None."""
    checkpoint_dir = Path(checkpoint_dir)
    pts = sorted(checkpoint_dir.glob("*.pt"))
    return pts[-1] if pts else None


# ---------------------------------------------------------------------------
# Memmap utilities
# ---------------------------------------------------------------------------

def create_memmap(
    path: str | Path,
    shape: tuple[int, ...],
    dtype: np.dtype | str = np.float16,
) -> np.ndarray:
    """Create (or overwrite) a float16 numpy memmap at *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    log.info("Memmap created", path=str(path), shape=shape, dtype=str(dtype))
    return arr


def open_memmap(
    path: str | Path,
    shape: tuple[int, ...],
    dtype: np.dtype | str = np.float16,
    mode: str = "r",
) -> np.ndarray:
    """Open an existing memmap in read (or copy-on-write) mode."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Memmap not found: {path}")
    return np.memmap(path, dtype=dtype, mode=mode, shape=shape)


# ---------------------------------------------------------------------------
# Video-to-row index
# ---------------------------------------------------------------------------

def save_video_index(index: dict[str, tuple[int, int]], path: str | Path) -> None:
    """
    Save a mapping  video_id (str) → (start_row, end_row)  as JSON.

    This lets us slice the embeddings memmap cheaply: given a video_id,
    frames are at rows [start_row : end_row].
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {k: list(v) for k, v in index.items()}
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    log.info("Video index saved", path=str(path), n_videos=len(index))


def load_video_index(path: str | Path) -> dict[str, tuple[int, int]]:
    """Load video_id → (start_row, end_row) index from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video index not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    return {k: tuple(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Frame-list index  (video_id → list[frame_path])
# ---------------------------------------------------------------------------

def save_frame_index(index: dict[str, list[str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f)
    log.info("Frame index saved", path=str(path))


def load_frame_index(path: str | Path) -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)