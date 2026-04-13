"""
FAISS vector index for fast frame retrieval.

FAISS (Facebook AI Similarity Search) provides fast approximate nearest
neighbour (ANN) search over large embedding databases.

INDEX TYPES (controlled via config):
  - "flat":  Exact brute-force search. Accurate but O(N) per query.
             Use for datasets with < 100k frames total.
  - "ivf":   Inverted file index. ~10× faster, slight accuracy drop.
             Use when flat is too slow (> 500k frames).

WHY CPU FAISS:
  faiss-cpu is sufficient here because we query per-video (not across
  the full dataset at once). The index for a single 10-minute video at
  1 fps has only ~600 vectors — brute-force search on that is microseconds.
"""
from __future__ import annotations

import faiss
import numpy as np
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FrameIndex:
    """
    FAISS-backed index for a single video's frame embeddings.

    Args:
        embed_dim:   Dimensionality of embeddings (512 for ViT-B/32).
        index_type:  "flat" or "ivf".

    Example:
        >>> index = FrameIndex(embed_dim=512)
        >>> index.add(frame_embeddings)        # shape (N, 512)
        >>> scores, indices = index.search(query_emb, k=64)
    """

    def __init__(self, embed_dim: int = 512, index_type: str = "flat"):
        self.embed_dim = embed_dim
        self.index_type = index_type
        self._index: faiss.Index | None = None

    def build(self, embeddings: np.ndarray) -> None:
        """
        Build the index from a matrix of frame embeddings.

        Args:
            embeddings: Float32 array of shape (N, embed_dim).
                        Must be L2-normalised (CLIPEncoder ensures this).
        """
        embeddings = embeddings.astype(np.float32)
        N = embeddings.shape[0]

        if self.index_type == "flat":
            # Inner product on L2-normalised vectors == cosine similarity
            self._index = faiss.IndexFlatIP(self.embed_dim)
        elif self.index_type == "ivf":
            nlist = min(int(N ** 0.5), 128)   # number of Voronoi cells
            quantizer = faiss.IndexFlatIP(self.embed_dim)
            self._index = faiss.IndexIVFFlat(quantizer, self.embed_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.train(embeddings)
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        self._index.add(embeddings)
        logger.debug(f"FAISS index built: {N} vectors, type={self.index_type}")

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Find top-k most similar frames to a query embedding.

        Args:
            query: Float32 array of shape (embed_dim,) or (1, embed_dim).
            k:     Number of results to return.

        Returns:
            Tuple of (scores, indices):
              - scores:  shape (k,) cosine similarity values
              - indices: shape (k,) integer frame indices into the original embedding array
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call .build(embeddings) first.")

        query = query.astype(np.float32)
        if query.ndim == 1:
            query = query[np.newaxis, :]

        scores, indices = self._index.search(query, k)
        return scores[0], indices[0]

    def save(self, path: str) -> None:
        """Persist the index to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        logger.info(f"FAISS index saved → {path}")

    def load(self, path: str) -> None:
        """Load a previously saved index from disk."""
        self._index = faiss.read_index(str(path))
        logger.info(f"FAISS index loaded ← {path}")
