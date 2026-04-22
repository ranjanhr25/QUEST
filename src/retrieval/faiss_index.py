"""
FAISS-based coarse retrieval index for QUEST.

Builds a per-video or global FAISS index over CLIP frame embeddings,
then retrieves the top-K most relevant frames for a query embedding.

We use IndexFlatIP (exact inner product) since embeddings are L2-normalised
(inner product = cosine similarity). No approximate search needed here —
per-video retrieval is over at most ~4500 frames which is cheap.

Usage:
    builder = FAISSIndexBuilder.from_config(cfg)
    builder.build()          # builds and saves per-video FAISS indexes

    retriever = FAISSRetriever.from_config(cfg)
    frame_indices, scores = retriever.retrieve(video_id, query_emb, top_k=64)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io_utils import load_video_index, open_memmap
from src.utils.logger import get_logger

log = get_logger("faiss_index")


def _check_faiss() -> Any:
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError(
            "faiss not installed. Run: conda install -c pytorch faiss-cpu  "
            "or: pip install faiss-cpu"
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class FAISSIndexBuilder:
    """
    Builds per-video FAISS IndexFlatIP indexes and saves them to disk.

    Instead of one monolithic index we keep per-video indexes so that
    retrieval is naturally scoped: given a question about video X we only
    search frames from video X.
    """

    def __init__(
        self,
        embeddings_path: Path,
        video_index_path: Path,
        output_dir: Path,
        embed_dim: int = 512,
    ) -> None:
        self.embeddings_path = embeddings_path
        self.video_index_path = video_index_path
        self.output_dir = output_dir
        self.embed_dim = embed_dim

    @classmethod
    def from_config(cls, cfg: Any) -> "FAISSIndexBuilder":
        emb_dir = Path(cfg.paths.embeddings_dir)
        return cls(
            embeddings_path=emb_dir / "frame_embeddings.npy",
            video_index_path=emb_dir / "video_index.json",
            output_dir=Path(cfg.paths.index_dir),
            embed_dim=cfg.retrieval.embed_dim,
        )

    def build(self, force: bool = False) -> None:
        faiss = _check_faiss()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        video_index = load_video_index(self.video_index_path)

        # Load embeddings (read-only memmap)
        total_rows = max(end for _, end in video_index.values())
        emb_arr = open_memmap(
            self.embeddings_path,
            shape=(total_rows, self.embed_dim),
            dtype=np.float16,
            mode="r",
        )

        skipped, built = 0, 0
        for vid_id, (start, end) in video_index.items():
            out_path = self.output_dir / f"{vid_id}.index"
            if out_path.exists() and not force:
                skipped += 1
                continue

            n_frames = end - start
            if n_frames == 0:
                continue

            vecs = emb_arr[start:end].astype(np.float32)  # faiss needs float32
            # Ensure L2-normalised (should already be from encoder)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-8)

            index = faiss.IndexFlatIP(self.embed_dim)
            index.add(vecs)
            faiss.write_index(index, str(out_path))
            built += 1

        log.info("FAISS indexes done", built=built, skipped=skipped, total=len(video_index))

        # Write a global manifest
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump({"embed_dim": self.embed_dim, "n_videos": len(video_index)}, f)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class FAISSRetriever:
    """
    Loads per-video FAISS indexes on demand and performs retrieval.

    Indexes are cached in memory after first load.
    """

    def __init__(
        self,
        index_dir: Path,
        video_index_path: Path,
        embeddings_path: Path,
        embed_dim: int = 512,
    ) -> None:
        faiss = _check_faiss()
        self.faiss = faiss
        self.index_dir = index_dir
        self.embed_dim = embed_dim
        self.video_index = load_video_index(video_index_path)

        total_rows = max(end for _, end in self.video_index.values())
        self.emb_arr = open_memmap(
            embeddings_path,
            shape=(total_rows, embed_dim),
            dtype=np.float16,
            mode="r",
        )

        self._cache: dict[str, Any] = {}  # video_id → faiss index

    @classmethod
    def from_config(cls, cfg: Any) -> "FAISSRetriever":
        emb_dir = Path(cfg.paths.embeddings_dir)
        return cls(
            index_dir=Path(cfg.paths.index_dir),
            video_index_path=emb_dir / "video_index.json",
            embeddings_path=emb_dir / "frame_embeddings.npy",
            embed_dim=cfg.retrieval.embed_dim,
        )

    def _load_index(self, video_id: str) -> Any | None:
        if video_id in self._cache:
            return self._cache[video_id]
        path = self.index_dir / f"{video_id}.index"
        if not path.exists():
            return None
        idx = self.faiss.read_index(str(path))
        self._cache[video_id] = idx
        return idx

    def retrieve(
        self,
        video_id: str,
        query_emb: np.ndarray,
        top_k: int = 64,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve top_k frame indices (relative to video) and their scores.

        Args:
            video_id: string video ID.
            query_emb: (embed_dim,) L2-normalised float32 vector.
            top_k: number of frames to return.

        Returns:
            frame_indices: (top_k,) int array — indices relative to video start.
            scores:        (top_k,) float32 similarity scores.
        """
        index = self._load_index(video_id)
        if index is None:
            log.warning("FAISS index not found", video_id=video_id)
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        q = query_emb.astype(np.float32).reshape(1, -1)
        # Normalise query just in case
        q = q / (np.linalg.norm(q) + 1e-8)

        n_frames = index.ntotal
        k = min(top_k, n_frames)
        scores, local_indices = index.search(q, k)  # (1, k)
        return local_indices[0], scores[0]

    def retrieve_embeddings(
        self,
        video_id: str,
        query_emb: np.ndarray,
        top_k: int = 64,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Like retrieve() but also returns the frame embeddings.

        Returns:
            frame_indices: (K,) relative to video.
            frame_embs:    (K, embed_dim) float32.
            scores:        (K,) float32.
        """
        local_indices, scores = self.retrieve(video_id, query_emb, top_k)
        if len(local_indices) == 0:
            return local_indices, np.zeros((0, self.embed_dim), dtype=np.float32), scores

        start, end = self.video_index.get(video_id, (0, 0))
        global_indices = start + local_indices
        # Clamp to valid range
        global_indices = np.clip(global_indices, 0, end - 1)
        embs = self.emb_arr[global_indices].astype(np.float32)
        return local_indices, embs, scores

    def get_all_frame_embs(self, video_id: str) -> np.ndarray:
        """Return ALL frame embeddings for a video (shape: N × embed_dim)."""
        start, end = self.video_index.get(video_id, (0, 0))
        if end <= start:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        return self.emb_arr[start:end].astype(np.float32)
    
class FrameIndex:
    """
    Lightweight single-video FAISS index used by demo.py and tests.
    Wraps IndexFlatIP for a single set of frame embeddings.
    """

    def __init__(self, embed_dim: int = 512, index_type: str = "flat") -> None:
        self.embed_dim = embed_dim
        self.index_type = index_type
        self._index = None

    def build(self, embeddings: np.ndarray) -> None:
        faiss = _check_faiss()
        vecs = embeddings.astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-8)
        self._index = faiss.IndexFlatIP(self.embed_dim)
        self._index.add(vecs)

    def search(
        self, query: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Index not built — call build() first")
        q = query.astype(np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) + 1e-8)
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(q, k)
        return scores[0], indices[0]

    def save(self, path: str) -> None:
        faiss = _check_faiss()
        faiss.write_index(self._index, str(path))

    def load(self, path: str) -> None:
        faiss = _check_faiss()
        self._index = faiss.read_index(str(path))