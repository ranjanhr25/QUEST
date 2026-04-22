"""
Stage 1: Coarse Retrieval pipeline wrapper for QUEST.

Given a video_id and a question string (or pre-computed query embedding),
this module finds the top-K most semantically relevant frames using
CLIP + FAISS per-video indexes.

The CoarseRetriever is the entry point for Stage 1. It wraps:
  - CLIPEncoder   (question → embedding)
  - FAISSRetriever (embedding × per-video index → top-K frame ids + embeddings)

Design choices:
  - Query embeddings are cached in-memory during evaluation to avoid re-encoding
    the same question multiple times across video candidates.
  - Batch retrieval is supported for DataLoader-style usage.
  - Falls back gracefully when a video has no FAISS index (logs a warning, returns empty).

Usage:
    retriever = CoarseRetriever.from_config(cfg)

    # Single sample
    result = retriever.retrieve(video_id="1234", question="Why did she laugh?")
    frame_indices = result.local_indices   # (K,) relative to video
    frame_embs    = result.frame_embs      # (K, 512)
    scores        = result.scores          # (K,) cosine similarity
    q_emb         = result.query_emb       # (512,)

    # With pre-computed query embedding (avoids re-encoding)
    result = retriever.retrieve(video_id="1234", query_emb=my_emb)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional

from src.utils.logger import get_logger

log = get_logger("coarse_retriever")


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    Output of a single coarse-retrieval call.

    Attributes:
        video_id:      The queried video.
        local_indices: Frame indices *relative to the video* (not global memmap row).
                       Shape (K,). Use these to index frame_paths from frame_index.json.
        frame_embs:    CLIP embeddings for the retrieved frames, shape (K, embed_dim).
        scores:        Cosine similarity scores, shape (K,), descending order.
        query_emb:     The (normalised) question embedding used for retrieval, shape (embed_dim,).
        temporal_pos:  Normalised temporal positions in [0, 1], shape (K,).
                       Computed as local_indices / max(n_total_frames - 1, 1).
    """
    video_id: str
    local_indices: np.ndarray         # (K,) int
    frame_embs: np.ndarray            # (K, embed_dim) float32
    scores: np.ndarray                # (K,) float32
    query_emb: np.ndarray             # (embed_dim,) float32
    temporal_pos: np.ndarray = field(default_factory=lambda: np.array([]))  # (K,) float32

    def __post_init__(self) -> None:
        if len(self.temporal_pos) == 0 and len(self.local_indices) > 0:
            # Compute temporal positions if not provided
            n_frames = int(self.local_indices.max()) + 1
            self.temporal_pos = self.local_indices.astype(np.float32) / max(n_frames - 1, 1)

    @property
    def n_retrieved(self) -> int:
        return len(self.local_indices)

    @property
    def is_empty(self) -> bool:
        return self.n_retrieved == 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CoarseRetriever:
    """
    Stage 1 coarse retrieval: CLIP + FAISS.

    Retrieves the top-K frames most relevant to a question from a given video.

    Args:
        encoder:    CLIPEncoder instance (for question encoding).
        retriever:  FAISSRetriever instance (per-video FAISS indexes).
        top_k:      Default number of candidate frames to retrieve.
        cache_query_embs: If True, cache question → embedding to avoid re-encoding.
    """

    def __init__(
        self,
        encoder: Any,            # CLIPEncoder
        retriever: Any,          # FAISSRetriever
        top_k: int = 64,
        cache_query_embs: bool = True,
    ) -> None:
        self.encoder = encoder
        self.retriever = retriever
        self.top_k = top_k
        self._cache: dict[str, np.ndarray] = {} if cache_query_embs else None

    @classmethod
    def from_config(cls, cfg: Any, device: Optional[str] = None) -> "CoarseRetriever":
        """
        Build a CoarseRetriever from a loaded config dict.

        Instantiates CLIPEncoder and FAISSRetriever automatically.
        """
        from src.models.clip_encoder import CLIPEncoder
        from src.retrieval.faiss_index import FAISSRetriever
        import torch
        from pathlib import Path

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        encoder = CLIPEncoder(
            model_name=cfg.retrieval.model_name,
            pretrained=cfg.retrieval.pretrained,
            device=device,
            batch_size=cfg.data.batch_size_embed,
        )

        emb_dir = Path(cfg.paths.embeddings_dir)
        retriever = FAISSRetriever(
            index_dir=Path(cfg.paths.index_dir),
            video_index_path=emb_dir / "video_index.json",
            embeddings_path=emb_dir / "frame_embeddings.npy",
            embed_dim=cfg.retrieval.embed_dim,
        )

        return cls(
            encoder=encoder,
            retriever=retriever,
            top_k=cfg.retrieval.top_k_coarse,
            cache_query_embs=True,
        )

    # ── Single-sample retrieval ───────────────────────────────────────────

    def retrieve(
        self,
        video_id: str,
        question: Optional[str] = None,
        query_emb: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve top-K frames for a question from a video.

        Exactly one of `question` or `query_emb` must be provided.

        Args:
            video_id:  String video ID (must match the FAISS index files).
            question:  Question text to encode with CLIP.
            query_emb: Pre-computed L2-normalised query embedding (skips CLIP encoding).
            top_k:     Override the default top_k for this call.

        Returns:
            RetrievalResult with frame indices, embeddings, scores, and temporal positions.
        """
        if question is None and query_emb is None:
            raise ValueError("Either 'question' or 'query_emb' must be provided.")

        k = top_k or self.top_k

        # Encode question (with cache)
        if query_emb is None:
            query_emb = self._encode_question(question)

        # FAISS retrieval
        local_indices, frame_embs, scores = self.retriever.retrieve_embeddings(
            video_id=video_id,
            query_emb=query_emb,
            top_k=k,
        )

        if len(local_indices) == 0:
            log.warning("No frames retrieved", video_id=video_id)
            d = self.encoder.embed_dim
            return RetrievalResult(
                video_id=video_id,
                local_indices=np.array([], dtype=np.int64),
                frame_embs=np.zeros((0, d), dtype=np.float32),
                scores=np.array([], dtype=np.float32),
                query_emb=query_emb,
                temporal_pos=np.array([], dtype=np.float32),
            )

        # Temporal positions: local frame index / (total frames - 1)
        start, end = self.retriever.video_index.get(video_id, (0, 0))
        n_total = max(end - start, 1)
        temporal_pos = local_indices.astype(np.float32) / max(n_total - 1, 1)

        return RetrievalResult(
            video_id=video_id,
            local_indices=local_indices,
            frame_embs=frame_embs,
            scores=scores,
            query_emb=query_emb,
            temporal_pos=temporal_pos,
        )

    # ── Batch retrieval ───────────────────────────────────────────────────

    def retrieve_batch(
        self,
        video_ids: list[str],
        questions: list[str],
        top_k: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve for a batch of (video_id, question) pairs.

        Questions are batch-encoded in a single CLIP call for efficiency.
        Returns a list of RetrievalResult, one per input pair.
        """
        k = top_k or self.top_k

        # Batch-encode all uncached questions at once
        uncached_qs = [q for q in questions if q not in (self._cache or {})]
        if uncached_qs:
            embs = self.encoder.encode_texts(list(set(uncached_qs)), l2_normalize=True)
            for q, emb in zip(set(uncached_qs), embs):
                if self._cache is not None:
                    self._cache[q] = emb

        results = []
        for vid_id, question in zip(video_ids, questions):
            q_emb = self._encode_question(question)
            results.append(self.retrieve(vid_id, query_emb=q_emb, top_k=k))

        return results

    # ── Uniform sampling (baseline method, no CLIP) ───────────────────────

    def retrieve_uniform(
        self,
        video_id: str,
        question: str,
        num_frames: int = 8,
    ) -> RetrievalResult:
        """
        Baseline: uniformly sample `num_frames` frames from the video.

        Still encodes the question so the output has a valid query_emb
        (needed for consistent result format), but frame selection is
        purely time-based, ignoring the question.

        Args:
            video_id:   String video ID.
            question:   Question text (encoded but not used for selection).
            num_frames: Number of frames to sample.

        Returns:
            RetrievalResult with uniformly sampled frames.
        """
        q_emb = self._encode_question(question)

        start, end = self.retriever.video_index.get(video_id, (0, 0))
        n_total = end - start

        if n_total == 0:
            log.warning("No frames in video index", video_id=video_id)
            d = self.encoder.embed_dim
            return RetrievalResult(
                video_id=video_id,
                local_indices=np.array([], dtype=np.int64),
                frame_embs=np.zeros((0, d), dtype=np.float32),
                scores=np.zeros(0, dtype=np.float32),
                query_emb=q_emb,
                temporal_pos=np.array([], dtype=np.float32),
            )

        # Uniform sampling (temporal spread)
        k = min(num_frames, n_total)
        local_indices = np.linspace(0, n_total - 1, k, dtype=np.int64)

        # Fetch embeddings for uniformly sampled frames
        global_indices = start + local_indices
        global_indices = np.clip(global_indices, 0, end - 1)
        frame_embs = self.retriever.emb_arr[global_indices].astype(np.float32)

        # Compute similarity scores for logging (not used for selection)
        scores = frame_embs @ q_emb  # (k,) cosine sim (already normalised)
        temporal_pos = local_indices.astype(np.float32) / max(n_total - 1, 1)

        return RetrievalResult(
            video_id=video_id,
            local_indices=local_indices,
            frame_embs=frame_embs,
            scores=scores,
            query_emb=q_emb,
            temporal_pos=temporal_pos,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _encode_question(self, question: str) -> np.ndarray:
        """Encode a question string, using cache if available."""
        if self._cache is not None and question in self._cache:
            return self._cache[question]
        emb = self.encoder.encode_texts([question], l2_normalize=True)[0]
        if self._cache is not None:
            self._cache[question] = emb
        return emb

    def clear_cache(self) -> None:
        """Clear the query embedding cache (call between datasets to free memory)."""
        if self._cache is not None:
            self._cache.clear()

    @property
    def embed_dim(self) -> int:
        return self.encoder.embed_dim