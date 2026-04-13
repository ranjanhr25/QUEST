"""
Cross-modal Transformer Ranker for QUEST (Stage 2).

Architecture:
  - Input: K candidate frame embeddings (D-dim) + 1 question embedding (D-dim).
  - Positional encoding: scene-boundary-aware temporal embedding (novel contribution).
  - Cross-attention: question as query, frame embeddings as key/value.
  - Self-attention: frames attend to each other for temporal coherence.
  - Output heads:
      relevance_scores: (K,)   — per-frame relevance to the question.
      uncertainty:      scalar — 1 = retrieval uncertain, 0 = confident.

Total parameters: ~50M (4 layers, 8 heads, hidden_dim=512).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Scene-boundary-aware Temporal Embedding  (Novel Contribution)
# ---------------------------------------------------------------------------

class SceneBoundaryTemporalEmbedding(nn.Module):
    """
    Encodes each frame's temporal position using two signals:
      1. Absolute temporal position (frame_idx / total_frames).
      2. Relative position to nearest scene boundary
         (distance_to_nearest_boundary / scene_length).

    Scene boundaries are detected as frames where consecutive CLIP similarity
    drops below a threshold (cosine distance spike).

    Both signals are projected to hidden_dim/2 and concatenated.
    """

    def __init__(self, hidden_dim: int, max_len: int = 4500) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        half = hidden_dim // 2

        # Sinusoidal base (absolute position)
        pe = torch.zeros(max_len, half)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, half, 2).float() * (-math.log(10000.0) / half))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:half // 2])
        self.register_buffer("abs_pe", pe)  # (max_len, half)

        # Learnable projection for boundary-relative position
        self.boundary_proj = nn.Sequential(
            nn.Linear(2, half),   # input: [scene_progress, dist_to_boundary_norm]
            nn.GELU(),
            nn.Linear(half, half),
        )

        # Merge projection
        self.merge = nn.Linear(hidden_dim, hidden_dim)

    def detect_boundaries(
        self, frame_embs: torch.Tensor, threshold: float = 0.3
    ) -> list[int]:
        """
        Detect scene boundaries from CLIP frame embeddings.
        A boundary occurs where cosine similarity between consecutive frames
        drops below (1 - threshold).

        Args:
            frame_embs: (K, D) tensor, L2-normalised.
            threshold:  cosine distance threshold.

        Returns:
            List of boundary frame indices (including 0 and K-1).
        """
        if frame_embs.shape[0] <= 1:
            return [0]
        sims = (frame_embs[:-1] * frame_embs[1:]).sum(dim=-1)  # (K-1,)
        boundaries = [0]
        for i, s in enumerate(sims.tolist()):
            if s < (1.0 - threshold):
                boundaries.append(i + 1)
        boundaries.append(frame_embs.shape[0] - 1)
        return boundaries

    def forward(
        self,
        temporal_pos: torch.Tensor,    # (B, K) normalised 0-1 frame positions
        frame_embs: torch.Tensor,      # (B, K, D) for boundary detection
    ) -> torch.Tensor:                 # (B, K, hidden_dim)
        B, K, D = frame_embs.shape

        # 1. Absolute position embedding
        # Convert normalised position to integer indices (scaled to max_len)
        max_len = self.abs_pe.shape[0]
        abs_idx = (temporal_pos * (max_len - 1)).long().clamp(0, max_len - 1)  # (B, K)
        abs_emb = self.abs_pe[abs_idx]  # (B, K, half)

        # 2. Scene-boundary-relative embedding
        boundary_feats = torch.zeros(B, K, 2, device=frame_embs.device)
        for b in range(B):
            boundaries = self.detect_boundaries(frame_embs[b])  # list of ints
            n_frames = K
            for fi in range(n_frames):
                # Find which scene this frame belongs to
                scene_start = max(j for j in boundaries if j <= fi)
                scene_end_candidates = [j for j in boundaries if j > fi]
                scene_end = scene_end_candidates[0] if scene_end_candidates else n_frames - 1

                scene_len = max(scene_end - scene_start, 1)
                scene_progress = (fi - scene_start) / scene_len  # 0→1 within scene
                dist_to_end = (scene_end - fi) / scene_len       # 0→1 to scene end

                boundary_feats[b, fi, 0] = scene_progress
                boundary_feats[b, fi, 1] = dist_to_end

        boundary_emb = self.boundary_proj(boundary_feats)  # (B, K, half)

        # 3. Concatenate and merge
        combined = torch.cat([abs_emb, boundary_emb], dim=-1)  # (B, K, hidden_dim)
        return self.merge(combined)


# ---------------------------------------------------------------------------
# Transformer Ranker
# ---------------------------------------------------------------------------

class TemporalRanker(nn.Module):
    """
    Cross-modal Transformer Ranker.

    Takes K candidate frame embeddings + 1 question embedding,
    outputs per-frame relevance scores and an uncertainty scalar.

    Args:
        embed_dim:   CLIP embedding dimension (input, e.g. 512).
        hidden_dim:  Transformer hidden dimension.
        num_layers:  Number of encoder layers.
        num_heads:   Attention heads.
        ffn_dim:     FFN intermediate dimension.
        dropout:     Dropout probability.
        max_frames:  Maximum K (for temporal embedding).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_frames: int = 4500,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Project CLIP embeddings into transformer space
        self.frame_proj = nn.Linear(embed_dim, hidden_dim)
        self.query_proj = nn.Linear(embed_dim, hidden_dim)

        # Scene-boundary-aware temporal encoding
        self.temporal_emb = SceneBoundaryTemporalEmbedding(hidden_dim, max_len=max_frames)

        # Cross-attention: question attends to frame embeddings
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        # Self-attention: frames attend to each other
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        # Layer norms
        self.cross_ln = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.self_ln  = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ffn_ln   = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # FFN
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        # Output heads
        self.relevance_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        # Uncertainty head: pool over frames then predict scalar
        self.uncertainty_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        frame_embs: torch.Tensor,    # (B, K, embed_dim)
        q_emb: torch.Tensor,         # (B, embed_dim) or (B, 1, embed_dim)
        temporal_pos: torch.Tensor,  # (B, K) normalised positions in [0, 1]
        pad_mask: torch.Tensor | None = None,  # (B, K) True = padded
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            frame_embs:  (B, K, D) CLIP frame embeddings (L2-normalised).
            q_emb:       (B, D)    CLIP question embedding (L2-normalised).
            temporal_pos:(B, K)    normalised temporal positions.
            pad_mask:    (B, K)    True for padded (invalid) frames.

        Returns dict with:
            relevance:   (B, K) raw relevance scores (pre-softmax).
            uncertainty: (B,)   uncertainty in [0, 1].
        """
        B, K, _ = frame_embs.shape

        # Project inputs
        x = self.frame_proj(frame_embs)                     # (B, K, H)
        q = self.query_proj(q_emb).unsqueeze(1)             # (B, 1, H)

        # Add temporal positional embedding
        pos_emb = self.temporal_emb(temporal_pos, frame_embs)  # (B, K, H)
        x = x + pos_emb

        # Prepare padding mask for attention (True = ignore)
        attn_key_mask = pad_mask if pad_mask is not None else None

        # Transformer layers: alternating cross-attention and self-attention
        for i in range(len(self.cross_attention_layers)):
            # Cross-attention: query attends to frames
            attended_q, _ = self.cross_attention_layers[i](
                query=q, key=x, value=x,
                key_padding_mask=attn_key_mask,
            )  # (B, 1, H)
            q = self.cross_ln[i](q + attended_q)

            # Self-attention: frames attend to each other
            attended_x, _ = self.self_attention_layers[i](
                query=x, key=x, value=x,
                key_padding_mask=attn_key_mask,
            )  # (B, K, H)
            x = self.self_ln[i](x + attended_x)

            # FFN on frames
            x = self.ffn_ln[i](x + self.ffn_layers[i](x))

        # Relevance scores (B, K)
        relevance = self.relevance_head(x).squeeze(-1)  # (B, K)
        if pad_mask is not None:
            relevance = relevance.masked_fill(pad_mask, float("-inf"))

        # Uncertainty: pool over valid frames → scalar
        if pad_mask is not None:
            valid_mask = ~pad_mask  # (B, K)
            pooled = (x * valid_mask.unsqueeze(-1).float()).sum(1)
            pooled = pooled / valid_mask.float().sum(1, keepdim=True).clamp(min=1)
        else:
            pooled = x.mean(1)  # (B, H)

        uncertainty = self.uncertainty_head(pooled).squeeze(-1)  # (B,)

        return {"relevance": relevance, "uncertainty": uncertainty}

    @classmethod
    def from_config(cls, cfg: Any) -> "TemporalRanker":
        r = cfg.ranking
        return cls(
            embed_dim=cfg.retrieval.embed_dim,
            hidden_dim=r.hidden_dim,
            num_layers=r.num_layers,
            num_heads=r.num_heads,
            ffn_dim=r.ffn_dim,
            dropout=r.dropout,
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)