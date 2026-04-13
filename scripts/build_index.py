#!/usr/bin/env python3
"""
Build FAISS per-video indexes from precomputed CLIP frame embeddings.

Run AFTER preprocess.py.

Usage (local conda / Colab / Kaggle):
    python scripts/build_index.py --config configs/nextqa.yaml

Options:
    --force     Rebuild indexes even if they already exist.
    --verify    After building, run a sanity check on 10 random videos.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Make src importable regardless of where the script is called from
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.retrieval.faiss_index import FAISSIndexBuilder, FAISSRetriever
from src.utils.config import load_config, ensure_dirs
from src.utils.io_utils import load_video_index
from src.utils.logger import get_logger

log = get_logger("build_index")


def verify_retrieval(cfg: object, n_videos: int = 10) -> None:
    """
    Sanity check: retrieve top-8 frames for random videos using a random query.
    Prints summary statistics.
    """
    from src.models.clip_encoder import CLIPEncoder
    from src.retrieval.faiss_index import FAISSRetriever

    log.info("Running retrieval sanity check", n_videos=n_videos)

    encoder = CLIPEncoder.from_config(cfg)
    retriever = FAISSRetriever.from_config(cfg)

    video_ids = list(retriever.video_index.keys())
    sample_ids = random.sample(video_ids, min(n_videos, len(video_ids)))

    dummy_queries = [
        "why did the person pick up the object",
        "what happened after the baby laughed",
        "how does the man move across the room",
        "where is this video taking place",
        "what did the girl do near the end",
    ]

    for vid_id in sample_ids:
        q = random.choice(dummy_queries)
        q_emb = encoder.encode_texts([q])[0]
        local_idx, embs, scores = retriever.retrieve_embeddings(vid_id, q_emb, top_k=8)

        start, end = retriever.video_index[vid_id]
        n_frames = end - start
        log.info(
            "Sanity check",
            video_id=vid_id,
            n_frames=n_frames,
            retrieved=len(local_idx),
            top_score=round(float(scores[0]), 4) if len(scores) else 0.0,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Build FAISS indexes for QUEST.")
    p.add_argument("--config",  default="configs/nextqa.yaml", type=Path)
    p.add_argument("--force",   action="store_true", help="Rebuild existing indexes")
    p.add_argument("--verify",  action="store_true", help="Run sanity check after building")
    args = p.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    emb_dir = Path(cfg.paths.embeddings_dir)
    emb_path      = emb_dir / "frame_embeddings.npy"
    vid_idx_path  = emb_dir / "video_index.json"
    index_dir     = Path(cfg.paths.index_dir)

    # Guard: embeddings must exist
    if not emb_path.exists():
        log.error(
            "frame_embeddings.npy not found. Run preprocess.py first.",
            expected=str(emb_path),
        )
        sys.exit(1)

    if not vid_idx_path.exists():
        log.error(
            "video_index.json not found. Run preprocess.py first.",
            expected=str(vid_idx_path),
        )
        sys.exit(1)

    video_index = load_video_index(vid_idx_path)
    log.info("Building FAISS indexes", n_videos=len(video_index), output_dir=str(index_dir))

    builder = FAISSIndexBuilder(
        embeddings_path=emb_path,
        video_index_path=vid_idx_path,
        output_dir=index_dir,
        embed_dim=cfg.retrieval.embed_dim,
    )
    builder.build(force=args.force)

    if args.verify:
        verify_retrieval(cfg, n_videos=10)

    log.info("build_index.py complete")


if __name__ == "__main__":
    main()