# QUEST Architecture

## Overview

QUEST is a three-component inference pipeline with one trainable module:

```
[Video] → [Stage 1: CLIP + FAISS] → [Stage 2: TemporalRanker + DPP] → [VLM] → [Answer]
              (frozen)                   (trained, ~50M params)          (frozen, 4-bit)
```

Only the TemporalRanker is trained. Everything else is frozen, which is why
the whole system can be trained in under 3 hours on free hardware.

## Stage 1: Coarse Retrieval

**Input:** All frames from a video (up to 1000) + question text  
**Output:** 64 candidate frames

- Frame embeddings are precomputed once with CLIP ViT-B/32 and stored as memory-mapped numpy arrays
- A FAISS flat index is built per video (or loaded from cache)
- The question is encoded with the same CLIP text encoder
- Top-64 frames by cosine similarity are returned

**Why 64?** Recall@64 for NExT-QA is ~92%. Increasing to 128 gives ~95% but doubles Stage 2 compute.

## Stage 2: Fine Ranking

**Input:** 64 candidate frame embeddings + question embedding + temporal positions  
**Output:** 8–16 selected frames + uncertainty score

### TemporalRanker

A compact Transformer encoder (3 layers, 8 heads, hidden_dim=256, ~50M params).

The question embedding is prepended as a "CLS" token, and the transformer
cross-attends it to all candidate frame embeddings. This produces a
question-contextualized representation for each frame.

Key design choice: we add temporal position embeddings *before* the transformer,
not as a positional encoding on top of frame embeddings. This lets the model
learn interactions between temporal position and visual content.

### DPP Selection

Instead of top-K on ranker scores, we use greedy MAP inference of a
Determinantal Point Process with kernel:

```
L[i,j] = q[i] * q[j] * exp(-λ * |t_i - t_j| / T)
```

where q[i] is the normalised ranker score and t_i is the frame timestamp.
The determinant det(L_S) is large when selected frames are both high-scoring
and temporally diverse.

### Adaptive Budget

The ranker's uncertainty head outputs a scalar in [0,1].  
- If uncertainty < threshold (0.4): select 8 frames (high-confidence retrieval)  
- If uncertainty ≥ threshold: select 16 frames (hedge by giving VLM more context)

## Stage 3: VLM Inference

Selected frames are tiled into a grid image and passed to LLaVA-1.5-7B
(4-bit quantized, frozen). The model receives the tiled image + question
in the standard LLaVA instruction format.

**Why not fine-tune the VLM?** Fine-tuning a 7B model requires 40+ GB VRAM
even with LoRA. Using it frozen (4-bit) needs only 5 GB — fits comfortably
on a free T4 alongside the ranker.

## Temporal Position Embedding

Each frame's temporal position is encoded as a 3-vector:
- `absolute_pos` ∈ [0,1]: frame index / total frames
- `scene_relative_pos` ∈ [0,1]: position within its detected scene
- `scene_id` ∈ [0,1]: which scene (normalised)

Scene boundaries are detected once per video with PySceneDetect (fast, CPU).
This gives the model awareness of narrative structure, not just raw time.
