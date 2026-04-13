"""Tests for the FAISS frame index."""
import numpy as np
import pytest
import tempfile
import os
from src.retrieval.faiss_index import FrameIndex


def _random_embeddings(n: int, d: int = 512) -> np.ndarray:
    embs = np.random.randn(n, d).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


def test_flat_index_returns_correct_k():
    index = FrameIndex(embed_dim=512, index_type="flat")
    embs = _random_embeddings(100)
    index.build(embs)
    query = _random_embeddings(1)[0]
    scores, indices = index.search(query, k=10)
    assert len(scores) == 10
    assert len(indices) == 10


def test_flat_index_top1_is_self():
    """A query that equals a stored embedding should retrieve itself first."""
    index = FrameIndex(embed_dim=512, index_type="flat")
    embs = _random_embeddings(50)
    index.build(embs)
    query = embs[7]  # exact match to frame 7
    _, indices = index.search(query, k=1)
    assert indices[0] == 7


def test_index_save_load(tmp_path):
    index = FrameIndex(embed_dim=512)
    embs = _random_embeddings(30)
    index.build(embs)

    save_path = str(tmp_path / "test.index")
    index.save(save_path)

    index2 = FrameIndex(embed_dim=512)
    index2.load(save_path)
    _, indices_orig = index.search(embs[0], k=5)
    _, indices_loaded = index2.search(embs[0], k=5)
    assert list(indices_orig) == list(indices_loaded)


def test_raises_before_build():
    index = FrameIndex()
    with pytest.raises(RuntimeError, match="not built"):
        index.search(np.random.randn(512).astype(np.float32), k=5)
