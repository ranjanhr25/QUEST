"""
NExT-QA PyTorch Dataset for QUEST.

Handles both the official CSV annotations (doc-doc/NExT-QA repo format)
and the HuggingFace lmms-lab/NExTQA parquet format — auto-detected by file extension.

Official CSV columns:
    video, frame_count, width, height, question, answer, qid, type, a0, a1, a2, a3, a4

Each __getitem__ returns a dict with:
    video_id        (str)
    question        (str)
    options         (List[str], len=5)
    answer_idx      (int, 0–4)
    qtype           (str, e.g. "CW")
    frame_paths     (List[str])           sorted frame jpg paths for this video
    frame_count     (int)                 total number of extracted frames
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.logger import get_logger

log = get_logger("dataset")

OPTION_COLS = ["a0", "a1", "a2", "a3", "a4"]


class NExTQADataset(Dataset):
    """
    Multi-choice NExT-QA dataset.

    Args:
        annotation_path: Path to train.csv / val.csv (or a .parquet file).
        frame_index_path: Path to frame_index.json produced by frame_extractor.
        split: "train" | "val" | "test" (informational only).
        question_types: If set, keep only rows whose 'type' is in this list.
        max_frames_per_video: Cap the number of frame paths returned (uniformly subsampled).
            Use None for no cap (return all frames). The ranker will further subsample.
        transform: Optional callable applied to each frame path list (e.g. for augmentation).
    """

    def __init__(
        self,
        annotation_path: str | Path,
        frame_index_path: str | Path,
        split: str = "val",
        question_types: list[str] | None = None,
        max_frames_per_video: int | None = None,
        transform: Any | None = None,
    ) -> None:
        self.split = split
        self.max_frames = max_frames_per_video
        self.transform = transform

        # ── Load annotations ──────────────────────────────────────────────
        annotation_path = Path(annotation_path)
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        if annotation_path.suffix == ".parquet":
            self.df = pd.read_parquet(annotation_path)
            self._normalise_hf_columns()
        else:
            self.df = pd.read_csv(annotation_path)

        self.df["video"] = self.df["video"].astype(str)

        # ── Filter by question type ───────────────────────────────────────
        if question_types:
            before = len(self.df)
            self.df = self.df[self.df["type"].isin(question_types)].reset_index(drop=True)
            log.info("Filtered by question types", types=question_types,
                     before=before, after=len(self.df))

        # ── Load frame index ──────────────────────────────────────────────
        frame_index_path = Path(frame_index_path)
        if not frame_index_path.exists():
            raise FileNotFoundError(f"Frame index not found: {frame_index_path}")
        with open(frame_index_path) as f:
            self.frame_index: dict[str, list[str]] = json.load(f)

        # ── Filter to rows that have extracted frames ─────────────────────
        has_frames = self.df["video"].isin(self.frame_index)
        n_missing = (~has_frames).sum()
        if n_missing > 0:
            log.warning(
                "Rows dropped — no extracted frames found",
                count=int(n_missing),
                examples=self.df.loc[~has_frames, "video"].unique()[:5].tolist(),
            )
        self.df = self.df[has_frames].reset_index(drop=True)

        log.info(
            "Dataset ready",
            split=split,
            rows=len(self.df),
            unique_videos=self.df["video"].nunique(),
        )

    # ── HuggingFace parquet column normalisation ──────────────────────────

    def _normalise_hf_columns(self) -> None:
        """
        HF lmms-lab/NExTQA has 'answer' as int (0-4) and uses 'video' as int.
        The official CSV also has 'answer' as int. Both are compatible.
        Ensure column names match the official format.
        """
        rename = {}
        for old, new in [("qid", "qid"), ("type", "type")]:
            if old in self.df.columns and new not in self.df.columns:
                rename[old] = new
        if rename:
            self.df.rename(columns=rename, inplace=True)
        # The HF dataset doesn't have a0-a4 columns in some versions;
        # they may be named 'choices' as a list. Expand if needed.
        if "choices" in self.df.columns and "a0" not in self.df.columns:
            choices = self.df["choices"].tolist()
            for i in range(5):
                self.df[f"a{i}"] = [c[i] if len(c) > i else "" for c in choices]

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        video_id = str(row["video"])

        # Options (a0..a4)
        options = [str(row.get(f"a{i}", "")) for i in range(5)]

        # Frame paths
        frame_paths = sorted(self.frame_index.get(video_id, []))
        if self.max_frames is not None and len(frame_paths) > self.max_frames:
            # Uniform subsample — preserves temporal order
            indices = torch.linspace(0, len(frame_paths) - 1, self.max_frames).long().tolist()
            frame_paths = [frame_paths[i] for i in indices]

        if self.transform is not None:
            frame_paths = self.transform(frame_paths)

        return {
            "video_id": video_id,
            "question": str(row["question"]),
            "options": options,
            "answer_idx": int(row["answer"]),
            "qtype": str(row.get("type", "?")),
            "frame_paths": frame_paths,
            "frame_count": len(frame_paths),
        }

    # ── Convenience ───────────────────────────────────────────────────────

    def get_video_ids(self) -> list[str]:
        return self.df["video"].unique().tolist()

    def get_type_distribution(self) -> dict[str, int]:
        return self.df["type"].value_counts().to_dict()

    def stratified_subset(self, n: int, seed: int = 42) -> "NExTQADataset":
        """
        Return a stratified subset of n samples (balanced across question types).
        Useful for quick evaluation runs on Colab.
        """
        types = self.df["type"].unique()
        per_type = max(1, n // len(types))
        sampled = (
            self.df.groupby("type", group_keys=False)
            .apply(lambda x: x.sample(min(per_type, len(x)), random_state=seed))
            .reset_index(drop=True)
        )
        clone = NExTQADataset.__new__(NExTQADataset)
        clone.split = self.split
        clone.max_frames = self.max_frames
        clone.transform = self.transform
        clone.df = sampled
        clone.frame_index = self.frame_index
        return clone


# ---------------------------------------------------------------------------
# Training pair dataset (for ranker supervision)
# ---------------------------------------------------------------------------

class RankerTrainDataset(Dataset):
    """
    Generates (video_id, question, candidate_frame_paths, relevance_scores) tuples
    for training the cross-modal transformer ranker.

    Relevance scores are weak pseudo-labels based on CLIP similarity:
      - top_k_positive frames get score 1.0
      - remaining frames get score 0.0 (with optional soft weights from CLIP sim)

    Requires that CLIP embeddings + the video index have already been computed
    (by preprocess.py). Loads them lazily on first access.

    Args:
        base_dataset: NExTQADataset instance (train split).
        embeddings_path: path to the float16 memmap of frame embeddings.
        video_index_path: path to the video_index.json (video_id → (start, end) rows).
        question_embeddings_path: path to the float16 memmap of question embeddings.
        question_index_path: path to the question_index.json (row_idx → embedding_row).
        top_k_positive: number of top-CLIP frames to treat as positives.
        n_candidates: number of candidate frames to return per sample.
        embed_dim: embedding dimension.
    """

    def __init__(
        self,
        base_dataset: NExTQADataset,
        embeddings_path: str | Path,
        video_index_path: str | Path,
        question_embeddings_path: str | Path,
        question_index_path: str | Path,
        top_k_positive: int = 3,
        n_candidates: int = 64,
        embed_dim: int = 512,
    ) -> None:
        import numpy as np
        from src.utils.io_utils import load_video_index

        self.base = base_dataset
        self.top_k_positive = top_k_positive
        self.n_candidates = n_candidates
        self.embed_dim = embed_dim

        # Load video index (lazy memmap access)
        self.video_index = load_video_index(video_index_path)
        self.q_index: dict[int, int] = {}
        with open(question_index_path) as f:
            self.q_index = {int(k): v for k, v in json.load(f).items()}

        # Compute total rows
        total_frame_rows = max(end for _, end in self.video_index.values())
        total_q_rows = len(self.q_index)

        self.frame_emb = np.memmap(
            str(embeddings_path), dtype=np.float16, mode="r",
            shape=(total_frame_rows, embed_dim),
        )
        self.q_emb = np.memmap(
            str(question_embeddings_path), dtype=np.float16, mode="r",
            shape=(total_q_rows, embed_dim),
        )

        log.info("RankerTrainDataset ready", rows=len(self.base),
                 frame_emb_shape=self.frame_emb.shape)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import numpy as np

        sample = self.base[idx]
        video_id = sample["video_id"]

        # ── Frame embeddings for this video ─────────────────────────────
        start, end = self.video_index.get(video_id, (0, 0))
        n_frames = end - start
        if n_frames == 0:
            # Fallback: return zeros
            return self._zero_sample(sample)

        frame_embs = self.frame_emb[start:end].astype(np.float32)  # (N, D)

        # ── Question embedding ────────────────────────────────────────────
        q_row = self.q_index.get(idx, 0)
        q_emb = self.q_emb[q_row].astype(np.float32)  # (D,)

        # ── CLIP similarity → pseudo relevance labels ─────────────────────
        sims = frame_embs @ q_emb  # (N,)
        norms = np.linalg.norm(frame_embs, axis=1) * np.linalg.norm(q_emb) + 1e-8
        sims = sims / norms

        # Select n_candidates (uniform subsample if video is longer)
        if n_frames > self.n_candidates:
            cand_idx = np.linspace(0, n_frames - 1, self.n_candidates, dtype=int)
        else:
            cand_idx = np.arange(n_frames)
            # Pad with zeros if fewer than n_candidates
        n_cand = len(cand_idx)

        cand_embs = frame_embs[cand_idx]   # (n_cand, D)
        cand_sims = sims[cand_idx]          # (n_cand,)

        # Relevance label: soft CLIP similarity scores (ListNet target)
        # Normalise to [0, 1] range within this sample
        sim_min, sim_max = cand_sims.min(), cand_sims.max()
        if sim_max > sim_min:
            relevance = (cand_sims - sim_min) / (sim_max - sim_min)
        else:
            relevance = np.ones(n_cand, dtype=np.float32) / n_cand

        # Uncertainty label: 1 if correct answer frame is NOT in top-k
        top_k_mask = np.zeros(n_cand, dtype=np.float32)
        topk_idx = np.argsort(cand_sims)[-self.top_k_positive:]
        top_k_mask[topk_idx] = 1.0
        # We define uncertainty_gt = 0 (confident) by default during pre-training;
        # after first eval pass we can set it based on ranker correctness.
        uncertainty_gt = np.float32(0.0)

        # Temporal positions (absolute, normalised 0-1)
        temporal_pos = cand_idx.astype(np.float32) / max(n_frames - 1, 1)

        # Pad to n_candidates if needed
        pad = self.n_candidates - n_cand
        if pad > 0:
            cand_embs   = np.pad(cand_embs,   ((0, pad), (0, 0)))
            relevance   = np.pad(relevance,   (0, pad))
            top_k_mask  = np.pad(top_k_mask,  (0, pad))
            temporal_pos = np.pad(temporal_pos, (0, pad))

        pad_mask = np.zeros(self.n_candidates, dtype=bool)
        pad_mask[n_cand:] = True  # True = padded position

        return {
            "video_id": video_id,
            "question": sample["question"],
            "q_emb": torch.from_numpy(q_emb),                          # (D,)
            "frame_embs": torch.from_numpy(cand_embs),                 # (K, D)
            "temporal_pos": torch.from_numpy(temporal_pos),            # (K,)
            "relevance": torch.from_numpy(relevance.astype(np.float32)),  # (K,)
            "top_k_mask": torch.from_numpy(top_k_mask),                # (K,)
            "uncertainty_gt": torch.tensor(uncertainty_gt),            # scalar
            "pad_mask": torch.from_numpy(pad_mask),                    # (K,)
            "answer_idx": sample["answer_idx"],
            "qtype": sample["qtype"],
        }

    def _zero_sample(self, sample: dict) -> dict[str, Any]:
        K, D = self.n_candidates, self.embed_dim
        return {
            "video_id": sample["video_id"],
            "question": sample["question"],
            "q_emb": torch.zeros(D),
            "frame_embs": torch.zeros(K, D),
            "temporal_pos": torch.zeros(K),
            "relevance": torch.zeros(K),
            "top_k_mask": torch.zeros(K),
            "uncertainty_gt": torch.tensor(0.0),
            "pad_mask": torch.ones(K, dtype=torch.bool),
            "answer_idx": sample["answer_idx"],
            "qtype": sample["qtype"],
        }