"""
Adaptive Frame Budget for QUEST (Novel Contribution).

The ranker's uncertainty head outputs u ∈ [0, 1].
  u < threshold  →  high confidence  →  use top_k_low  frames (e.g. 8)
  u ≥ threshold  →  low  confidence  →  use top_k_high frames (e.g. 16)

This avoids wasting VLM context budget on easy queries and gives more
visual evidence to hard ones.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class AdaptiveBudget:
    """
    Decides how many frames to pass to the VLM based on ranker uncertainty.

    Args:
        threshold:    uncertainty threshold separating confident from uncertain.
        top_k_low:    frame budget when confidence is HIGH (uncertainty < threshold).
        top_k_high:   frame budget when confidence is LOW  (uncertainty >= threshold).
    """

    def __init__(
        self,
        threshold: float = 0.3,
        top_k_low: int = 8,
        top_k_high: int = 16,
    ) -> None:
        self.threshold = threshold
        self.top_k_low = top_k_low
        self.top_k_high = top_k_high

    @classmethod
    def from_config(cls, cfg: Any) -> "AdaptiveBudget":
        r = cfg.ranking
        return cls(
            threshold=r.uncertainty_threshold,
            top_k_low=r.top_k_fine_low,
            top_k_high=r.top_k_fine_high,
        )

    def get_budget(self, uncertainty: float | torch.Tensor | np.floating) -> int:
        """Return the frame budget for a single sample."""
        if isinstance(uncertainty, torch.Tensor):
            uncertainty = uncertainty.item()
        return self.top_k_low if uncertainty < self.threshold else self.top_k_high

    def get_budgets(
        self, uncertainties: torch.Tensor | np.ndarray
    ) -> list[int]:
        """Return frame budgets for a batch of uncertainty values."""
        if isinstance(uncertainties, torch.Tensor):
            uncertainties = uncertainties.detach().cpu().numpy()
        return [
            self.top_k_low if u < self.threshold else self.top_k_high
            for u in uncertainties
        ]

    def select_frames(
        self,
        relevance: np.ndarray,       # (K,) ranker relevance scores
        temporal_pos: np.ndarray,    # (K,) normalised positions
        uncertainty: float,
        dpp_selector: Any,           # DPPSelector instance
        method: str = "dpp",
    ) -> list[int]:
        """
        Full selection pipeline: adaptive budget + DPP selection.

        Returns:
            Sorted list of selected frame indices (relative to candidate set).
        """
        n_select = self.get_budget(uncertainty)
        return dpp_selector.select(relevance, temporal_pos, n_select, method=method)