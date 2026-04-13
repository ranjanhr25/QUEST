"""Tests for the DPP frame selector."""
import numpy as np
import pytest
from src.models.dpp_selector import build_dpp_kernel, greedy_dpp_select, dpp_select_frames


def test_kernel_shape():
    scores = np.array([0.9, 0.8, 0.3, 0.7, 0.5])
    timestamps = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    L = build_dpp_kernel(scores, timestamps, lambda_diversity=0.5)
    assert L.shape == (5, 5), "Kernel must be square (K x K)"


def test_kernel_symmetry():
    scores = np.random.rand(8)
    timestamps = np.sort(np.random.rand(8) * 60)
    L = build_dpp_kernel(scores, timestamps)
    assert np.allclose(L, L.T, atol=1e-6), "Kernel must be symmetric"


def test_greedy_selects_correct_count():
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    timestamps = np.arange(8, dtype=float)
    L = build_dpp_kernel(scores, timestamps)
    selected = greedy_dpp_select(L, k=4)
    assert len(selected) == 4
    assert len(set(selected)) == 4, "No duplicate indices"


def test_dpp_enforces_diversity():
    """High lambda_diversity should spread frames across the timeline."""
    scores = np.ones(20)                     # all equally relevant
    timestamps = np.arange(20, dtype=float)  # frames at t=0..19

    selected_diverse = dpp_select_frames(scores, timestamps, k=4, lambda_diversity=2.0)
    selected_greedy = dpp_select_frames(scores, timestamps, k=4, lambda_diversity=0.0)

    # Diverse selection should cover a wider time range
    ts_diverse = timestamps[selected_diverse]
    ts_greedy = timestamps[selected_greedy]
    assert ts_diverse.max() - ts_diverse.min() > ts_greedy.max() - ts_greedy.min(), \
        "High lambda should produce more temporally diverse selection"


def test_handles_single_frame():
    selected = dpp_select_frames(np.array([0.9]), np.array([0.0]), k=1)
    assert selected == [0]


def test_k_larger_than_candidates():
    scores = np.random.rand(5)
    timestamps = np.arange(5, dtype=float)
    selected = dpp_select_frames(scores, timestamps, k=10)
    assert len(selected) == 5  # should return all 5, not error
