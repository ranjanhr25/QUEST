"""
Cross-modal temporal ranker — Stage 2 of the QUEST pipeline.

Architecture:
  1. A linear projection maps CLIP question embedding → ranker embedding space.
  2. A linear projection maps each CLIP frame embedding → ranker embedding space.
  3. A small Transformer encoder cross-attends question tokens to frame embeddings,
     producing a context-aware representation for each candidate frame.
  4. A temporal MLP fuses [frame_repr, temporal_pos_embedding, cross_attn_output].
  5. Two output heads:
     - Relevance head: scalar score per frame (higher = more relevant to question)
     - Uncertainty head: scalar for the full candidate set (used by adaptive budget)

The model is small enough (~50M params) to train from scratch on Kaggle free T4s
in under 3 hours. No pretraining needed — the CLIP embeddings are already rich.

Training objective: contrastive ranking loss with in-batch negatives.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class TemporalRanker(nn.Module):
    """
    Cross-modal transformer that scores candidate frames given a question.

    Args:
        embed_dim:   Input embedding dimension (must match CLIP output, default 512).
        hidden_dim:  Internal hidden dimension of the transformer.
        num_heads:   Number of attention heads in the cross-attention layer.
        num_layers:  Number of transformer encoder layers.
        dropout:     Dropout rate applied inside the transformer.

    Inputs (forward):
        question_emb:  Tensor of shape (B, embed_dim) — one question per sample.
        frame_embs:    Tensor of shape (B, K, embed_dim) — K candidate frames.
        temporal_pos:  Tensor of shape (B, K, 3) — temporal position features
                       [absolute_pos, scene_relative_pos, scene_id] per frame.

    Outputs (forward):
        relevance_scores: Tensor of shape (B, K) — per-frame relevance scores.
        uncertainty:      Tensor of shape (B,) — uncertainty of the retrieved set.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project CLIP embeddings into ranker space
        self.question_proj = nn.Linear(embed_dim, hidden_dim)
        self.frame_proj = nn.Linear(embed_dim, hidden_dim)

        # Scene-boundary-aware temporal position embedding (3 features → hidden_dim)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Cross-modal transformer: question attends to frames
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,          # expects (B, seq, dim)
            norm_first=True,           # pre-norm is more stable for small models
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Fusion: combine frame repr + temporal pos + cross-attn output
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.relevance_head = nn.Linear(hidden_dim, 1)   # per-frame score
        self.uncertainty_head = nn.Sequential(           # per-set uncertainty
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        question_emb: torch.Tensor,    # (B, embed_dim)
        frame_embs: torch.Tensor,      # (B, K, embed_dim)
        temporal_pos: torch.Tensor,    # (B, K, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:

        B, K, _ = frame_embs.shape

        # Project into hidden space
        q = self.question_proj(question_emb)              # (B, hidden_dim)
        f = self.frame_proj(frame_embs)                   # (B, K, hidden_dim)
        t = self.temporal_mlp(temporal_pos)               # (B, K, hidden_dim)

        # Prepend question as a CLS-like token, then run transformer
        q_token = q.unsqueeze(1)                          # (B, 1, hidden_dim)
        sequence = torch.cat([q_token, f + t], dim=1)    # (B, K+1, hidden_dim)
        encoded = self.transformer(sequence)              # (B, K+1, hidden_dim)

        # Split back: first token is question ctx, rest are frame representations
        q_ctx = encoded[:, 0, :]                          # (B, hidden_dim)
        f_ctx = encoded[:, 1:, :]                         # (B, K, hidden_dim)

        # Fuse frame repr, temporal pos, and cross-attended context
        q_expanded = q_ctx.unsqueeze(1).expand(-1, K, -1)
        fused = self.fusion(torch.cat([f_ctx, t, q_expanded], dim=-1))  # (B, K, hidden_dim)

        # Score each frame
        relevance_scores = self.relevance_head(fused).squeeze(-1)        # (B, K)

        # Compute set-level uncertainty from mean pooled representation
        mean_repr = fused.mean(dim=1)                                    # (B, hidden_dim)
        uncertainty = self.uncertainty_head(mean_repr).squeeze(-1)       # (B,)

        return relevance_scores, uncertainty


def contrastive_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Pairwise margin ranking loss.

    For each (positive, negative) frame pair in the batch, the positive
    frame should score at least `margin` higher than the negative.

    Args:
        scores: Tensor of shape (B, K) — ranker output scores.
        labels: Tensor of shape (B, K) — 1 for relevant frames, 0 for irrelevant.
        margin: Minimum score gap between positive and negative frames.

    Returns:
        Scalar loss tensor.
    """
    loss = torch.tensor(0.0, device=scores.device)
    count = 0

    for b in range(scores.shape[0]):
        pos_mask = labels[b] == 1
        neg_mask = labels[b] == 0
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue
        pos_scores = scores[b][pos_mask]
        neg_scores = scores[b][neg_mask]
        # All (pos, neg) pairs
        pairs = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)  # (P, N)
        loss += torch.clamp(margin - pairs, min=0).mean()
        count += 1

    return loss / max(count, 1)
