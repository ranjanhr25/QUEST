"""
Evaluation metrics for Video QA.

Metrics implemented:
  - exact_match:   standard accuracy for multiple-choice QA
  - recall_at_k:   fraction of ground-truth relevant frames retrieved in top-K
  - wups:          Wu-Palmer Similarity for open-ended QA (MSVD-QA standard)
  - type_breakdown: per-question-type accuracy (causal / temporal / descriptive)
"""
from __future__ import annotations

import numpy as np
from collections import defaultdict


def exact_match(predictions: list[str], references: list[str]) -> float:
    """
    Compute exact string match accuracy (case-insensitive, stripped).

    Args:
        predictions: List of model-predicted answer strings.
        references:  List of ground-truth answer strings.

    Returns:
        Float in [0, 1] — fraction of correct predictions.

    Example:
        >>> exact_match(["cat", "Dog"], ["cat", "dog"])
        1.0
    """
    assert len(predictions) == len(references), "Length mismatch"
    correct = sum(
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references)
    )
    return correct / len(references)


def recall_at_k(
    retrieved_indices: list[int],
    relevant_indices: list[int],
    k: int,
) -> float:
    """
    Recall@K: what fraction of ground-truth relevant frames appear in top-K retrieved?

    Used to evaluate Stage 1 (coarse retrieval) quality independently
    from Stage 2 and the VLM — helps diagnose where the pipeline breaks down.

    Args:
        retrieved_indices: Ranked list of retrieved frame indices (sorted by score).
        relevant_indices:  List of ground-truth relevant frame indices.
        k:                 Cutoff.

    Returns:
        Float in [0, 1].
    """
    top_k = set(retrieved_indices[:k])
    relevant = set(relevant_indices)
    if not relevant:
        return 1.0
    return len(top_k & relevant) / len(relevant)


def accuracy_by_type(
    predictions: list[str],
    references: list[str],
    q_types: list[str],
) -> dict[str, float]:
    """
    Compute exact match accuracy broken down by question type.

    Useful for NExT-QA which has three types: "C" (causal), "T" (temporal), "D" (descriptive).

    Args:
        predictions: Model predictions.
        references:  Ground truth answers.
        q_types:     Question type labels (one per sample).

    Returns:
        Dict mapping type label → accuracy float.

    Example:
        >>> accuracy_by_type(preds, refs, types)
        {"C": 0.54, "T": 0.57, "D": 0.67, "overall": 0.59}
    """
    type_correct: dict[str, int] = defaultdict(int)
    type_total: dict[str, int] = defaultdict(int)

    for pred, ref, qt in zip(predictions, references, q_types):
        match = pred.strip().lower() == ref.strip().lower()
        type_correct[qt] += int(match)
        type_total[qt] += 1

    results = {
        qt: type_correct[qt] / type_total[qt]
        for qt in type_total
    }
    # Overall
    total_correct = sum(type_correct.values())
    total = sum(type_total.values())
    results["overall"] = total_correct / total if total > 0 else 0.0
    return results
