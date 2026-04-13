"""
Precompute CLIP embeddings for all extracted frames and questions.

Outputs (all in embeddings_dir):
  frame_embeddings.npy     float16 memmap, shape (total_frames, embed_dim)
  video_index.json         { video_id: [start_row, end_row] }
  question_embeddings.npy  float16 memmap, shape (total_questions, embed_dim)
  question_index.json      { dataset_row_idx: embedding_row }
  metadata.json            stats: total_frames, embed_dim, model, pretrained

This script is idempotent: if the output files exist it skips re-computation
unless --force is passed.

Run locally (conda):
    python -m src.data.preprocess \
        --config configs/nextqa.yaml \
        --split train
    python -m src.data.preprocess \
        --config configs/nextqa.yaml \
        --split val

Run on Kaggle/Colab: same command after setting DATA_ROOT env var or
adjusting paths in configs/nextqa.yaml.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config import load_config, ensure_dirs
from src.utils.logger import get_logger
from src.utils.io_utils import (
    create_memmap,
    save_video_index,
    save_frame_index,
    load_frame_index,
)

log = get_logger("preprocess")


# ---------------------------------------------------------------------------
# Frame embeddings
# ---------------------------------------------------------------------------

def precompute_frame_embeddings(
    frame_index: dict[str, list[str]],
    output_dir: Path,
    embed_dim: int,
    encoder: Any,
    force: bool = False,
) -> tuple[Path, Path]:
    """
    Compute and save float16 frame embeddings memmap.

    Returns (embeddings_path, video_index_path).
    """
    emb_path = output_dir / "frame_embeddings.npy"
    idx_path = output_dir / "video_index.json"

    if emb_path.exists() and idx_path.exists() and not force:
        log.info("Frame embeddings already exist — skipping", path=str(emb_path))
        return emb_path, idx_path

    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute total frame count
    video_ids = sorted(frame_index.keys())
    frame_counts = [len(frame_index[v]) for v in video_ids]
    total_frames = sum(frame_counts)

    log.info("Allocating frame embeddings memmap",
             total_frames=total_frames, embed_dim=embed_dim,
             size_mb=round(total_frames * embed_dim * 2 / 1e6, 1))

    emb_arr = create_memmap(emb_path, shape=(total_frames, embed_dim), dtype=np.float16)
    video_index: dict[str, list[int]] = {}
    row = 0

    t0 = time.time()
    for i, vid_id in enumerate(video_ids):
        paths = frame_index[vid_id]
        if not paths:
            video_index[vid_id] = [row, row]
            continue

        embs = encoder.encode_frame_paths_batched(paths, l2_normalize=True, show_progress=False)
        n = len(embs)
        emb_arr[row : row + n] = embs.astype(np.float16)
        video_index[vid_id] = [row, row + n]
        row += n

        if (i + 1) % 50 == 0 or i == len(video_ids) - 1:
            elapsed = time.time() - t0
            fps = row / elapsed
            eta = (total_frames - row) / fps if fps > 0 else 0
            log.info(
                "Frame embedding progress",
                videos=f"{i+1}/{len(video_ids)}",
                frames=row,
                fps=round(fps, 0),
                eta_min=round(eta / 60, 1),
            )

    emb_arr.flush()
    save_video_index({k: tuple(v) for k, v in video_index.items()}, idx_path)
    log.info("Frame embeddings complete", rows=row, path=str(emb_path))
    return emb_path, idx_path


# ---------------------------------------------------------------------------
# Question embeddings
# ---------------------------------------------------------------------------

def precompute_question_embeddings(
    df: Any,  # pandas DataFrame
    output_dir: Path,
    embed_dim: int,
    encoder: Any,
    split: str,
    force: bool = False,
) -> tuple[Path, Path]:
    """
    Compute and save float16 question embeddings memmap.
    Query = "question + correct_answer_text" during training for stronger supervision.
    Query = "question" during validation/test.

    Returns (q_embeddings_path, q_index_path).
    """
    q_emb_path = output_dir / f"question_embeddings_{split}.npy"
    q_idx_path = output_dir / f"question_index_{split}.json"

    if q_emb_path.exists() and q_idx_path.exists() and not force:
        log.info("Question embeddings already exist — skipping", path=str(q_emb_path))
        return q_emb_path, q_idx_path

    is_train = (split == "train")
    queries: list[str] = []
    q_index: dict[int, int] = {}

    for row_idx, row in df.iterrows():
        if is_train:
            ans_text = str(row.get(f"a{int(row['answer'])}", ""))
            query = f"{row['question']} {ans_text}"
        else:
            query = str(row["question"])
        q_index[int(row_idx)] = len(queries)
        queries.append(query)

    log.info("Encoding questions", n=len(queries), split=split)
    embs = encoder.encode_texts(queries, l2_normalize=True)
    total_q = len(embs)

    q_emb_arr = create_memmap(q_emb_path, shape=(total_q, embed_dim), dtype=np.float16)
    q_emb_arr[:] = embs.astype(np.float16)
    q_emb_arr.flush()

    with open(q_idx_path, "w") as f:
        json.dump(q_index, f)

    log.info("Question embeddings saved", path=str(q_emb_path), rows=total_q)
    return q_emb_path, q_idx_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute CLIP embeddings for QUEST.")
    p.add_argument("--config", default="configs/nextqa.yaml", type=Path)
    p.add_argument("--split", default="train", choices=["train", "val", "all"])
    p.add_argument("--force", action="store_true", help="Recompute even if outputs exist")
    p.add_argument("--workers", default=None, type=int, help="Override data.max_workers")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    from src.models.clip_encoder import CLIPEncoder
    import pandas as pd

    encoder = CLIPEncoder.from_config(cfg)
    embed_dim = encoder.embed_dim

    ann_dir = Path(cfg.paths.annotations)
    emb_dir = Path(cfg.paths.embeddings_dir)
    frame_idx_path = Path(cfg.paths.frames_root).parent / "frame_index.json"

    if not frame_idx_path.exists():
        raise FileNotFoundError(
            f"frame_index.json not found at {frame_idx_path}. "
            "Run extract_frames first."
        )

    frame_index = load_frame_index(frame_idx_path)
    log.info("Loaded frame index", videos=len(frame_index))

    splits = ["train", "val"] if args.split == "all" else [args.split]

    for split in splits:
        ann_files = {
            "train": cfg.annotation_files.get("train", "train.csv"),
            "val": cfg.annotation_files.get("val", "val.csv"),
        }
        ann_file = ann_dir / ann_files.get(split, f"{split}.csv")
        if not ann_file.exists():
            log.warning("Annotation file missing — skipping split", split=split, path=str(ann_file))
            continue

        df = pd.read_csv(ann_file)
        df["video"] = df["video"].astype(str)

        # Filter to videos with frames
        before = len(df)
        df = df[df["video"].isin(frame_index)].reset_index(drop=True)
        log.info("Loaded annotations", split=split, rows=len(df), dropped=before - len(df))

        # Frame embeddings (shared across splits — only compute once from frame_index)
        if split == splits[0]:  # compute once
            precompute_frame_embeddings(
                frame_index=frame_index,
                output_dir=emb_dir,
                embed_dim=embed_dim,
                encoder=encoder,
                force=args.force,
            )

        # Question embeddings (per split)
        precompute_question_embeddings(
            df=df,
            output_dir=emb_dir,
            embed_dim=embed_dim,
            encoder=encoder,
            split=split,
            force=args.force,
        )

    # Save metadata
    meta = {
        "embed_dim": embed_dim,
        "model_name": cfg.retrieval.model_name,
        "pretrained": cfg.retrieval.pretrained,
        "total_videos": len(frame_index),
    }
    with open(emb_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Preprocessing complete")


if __name__ == "__main__":
    main()