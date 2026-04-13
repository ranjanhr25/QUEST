"""
Determinantal Point Process (DPP) frame selection — the core novel contribution.

WHY DPP INSTEAD OF TOP-K:
  Standard top-K selection picks the 8 highest-scoring frames. In practice,
  these often cluster around the same 10-second moment — the model keeps
  "finding" the same event. DPP jointly maximises relevance AND diversity,
  sampling sets where frames are simultaneously high-quality and spread
  across the video timeline.

HOW DPP WORKS (intuition):
  A DPP defines a probability distribution over subsets S of items such that:
    P(S) ∝ det(L_S)
  where L is a positive semidefinite "quality-diversity" kernel matrix.
  The determinant is large when selected items are both high-quality (diagonal)
  and dissimilar to each other (off-diagonal terms). It's maximised by sets
  that are spread out in feature space.

OUR KERNEL:
  L[i,j] = relevance[i] * relevance[j] * exp(-λ * temporal_dist(i,j))
  - Diagonal entries = squared relevance scores (quality)
  - Off-diagonal entries decay with temporal distance (diversity)
  - λ controls the diversity/relevance tradeoff

COMPUTATIONAL NOTE:
  Exact DPP sampling is O(N³) in N candidates. With N=64 candidates this
  is completely fine on CPU. We use greedy MAP inference (not sampling)
  which is O(N²·k) for selecting k items — even faster.

Reference: Kulesza & Taskar, "Determinantal Point Processes for Machine Learning" (2012)
"""
from __future__ import annotations

import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_dpp_kernel(
    relevance_scores: np.ndarray,
    frame_timestamps: np.ndarray,
    lambda_diversity: float = 0.5,
) -> np.ndarray:
    """
    Build the DPP kernel matrix L for a set of candidate frames.

    Args:
        relevance_scores: Array of shape (K,) — ranker output scores (any scale).
        frame_timestamps: Array of shape (K,) — absolute timestamp (seconds) per frame.
        lambda_diversity: Controls diversity. Higher → more diverse, less relevant.
                          Ablate: {0.1, 0.5, 1.0, 2.0}. Default 0.5 is a good start.

    Returns:
        L kernel matrix of shape (K, K). Symmetric, positive semidefinite.
    """
    K = len(relevance_scores)

    # Normalise scores to [0, 1] to keep kernel well-conditioned
    scores = relevance_scores - relevance_scores.min()
    max_score = scores.max()
    if max_score > 0:
        scores = scores / max_score

    # Quality vector: q[i] = normalised relevance score
    q = scores  # (K,)

    # Temporal distance matrix: D[i,j] = |t_i - t_j| / max_duration
    t = frame_timestamps
    duration = max(t.max() - t.min(), 1.0)
    D = np.abs(t[:, None] - t[None, :]) / duration  # (K, K), values in [0, 1]

    # Similarity kernel: high similarity for temporally close frames
    S = np.exp(-lambda_diversity * D)  # (K, K)

    # Full DPP kernel: L[i,j] = q[i] * q[j] * S[i,j]
    L = np.outer(q, q) * S  # (K, K)

    return L


def greedy_dpp_select(
    L: np.ndarray,
    k: int,
) -> list[int]:
    """
    Greedy MAP inference for DPP — selects k items that approximately
    maximise det(L_S).

    At each step, we greedily pick the item that maximally increases
    the determinant of the current selected set. This is equivalent to
    picking the item most "orthogonal" to already-selected items in the
    feature space defined by L.

    Args:
        L: DPP kernel matrix of shape (K, K).
        k: Number of frames to select.

    Returns:
        List of k selected frame indices, ordered by selection priority.

    Example:
        >>> L = build_dpp_kernel(scores, timestamps, lambda_diversity=0.5)
        >>> selected = greedy_dpp_select(L, k=8)
        >>> len(selected)
        8
    """
    K = L.shape[0]
    k = min(k, K)

    selected = []
    remaining = list(range(K))

    # Cholesky-based incremental update for efficiency
    # On each iteration, compute the conditional gain of adding each candidate
    for _ in range(k):
        best_idx = None
        best_gain = -np.inf

        for idx in remaining:
            # Marginal gain of adding idx to current selection
            # Approximated as the diagonal element of the Schur complement
            if not selected:
                gain = L[idx, idx]
            else:
                sel = np.array(selected)
                L_ss = L[np.ix_(sel, sel)]
                L_si = L[sel, idx]
                L_ii = L[idx, idx]
                # Schur complement: L_ii - L_si^T L_ss^{-1} L_si
                try:
                    gain = L_ii - L_si @ np.linalg.solve(L_ss, L_si)
                except np.linalg.LinAlgError:
                    gain = L_ii

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def dpp_select_frames(
    relevance_scores: np.ndarray,
    frame_timestamps: np.ndarray,
    k: int,
    lambda_diversity: float = 0.5,
) -> list[int]:
    """
    Full DPP frame selection pipeline: build kernel → greedy MAP select.

    Args:
        relevance_scores: Shape (K,) ranker scores for candidate frames.
        frame_timestamps: Shape (K,) timestamps in seconds.
        k:                Number of frames to select.
        lambda_diversity: Diversity weight. Tune via ablation.

    Returns:
        List of k selected indices into the candidate frame list.
    """
    L = build_dpp_kernel(relevance_scores, frame_timestamps, lambda_diversity)
    return greedy_dpp_select(L, k)
