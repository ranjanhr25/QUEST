# QUEST Project — Claude Context File
> Feed this file at the start of every new Claude conversation.
> Update the STATUS column after completing each phase.

---

## Project Summary

**QUEST** = Query-guided Efficient Segment Selection for Long-Video QA.
A two-stage query-aware frame selection pipeline for long-video question answering.
Instead of uniform frame sampling, QUEST finds the most relevant frames for a given question before feeding them to a frozen VLM.

**Goal:** Beat uniform sampling (48.2% overall on NExT-QA) → target 59.3% with full pipeline.

**Hardware:** Kaggle 2×T4 (training), no local GPU.

---

## Pipeline Overview

```
Video + Question
    → Stage 1: CLIP-ViT + FAISS  (1000 frames → 64 candidates)
    → Stage 2: TemporalRanker + DPP  (64 → 8–16 frames)
    → Frozen LLaVA-1.5-7B 4-bit  (answer)
```

Only Stage 2 (TemporalRanker) is trained. Everything else is frozen.

---

## Repository

- **GitHub:** https://github.com/ranjanhr25/QUEST (public)
- **Local path:** `C:\Users\ranja\OneDrive\Desktop\placements\projects\QUEST`
- **Kaggle username:** ranjanhr

---

## Kaggle Datasets

| Dataset slug | Contents | Used for |
|---|---|---|
| `ranjanhr/nextqa-processed-frames` | `frames/` dir + original `frame_index.json` | Frame JPEGs |
| `ranjanhr/nextqa-annotations` | `data/raw/nextqa/annotations/` with `train.csv`, `val.csv`, `test.csv`, `map_vid_vidorID.json` | Annotations |
| `ranjanhr/nextqa-embeddings` | `embeddings/`, `indexes/`, remapped `frame_index.json` | Pre-computed CLIP embeddings + FAISS indexes |

### Key mounted paths on Kaggle
```
/kaggle/input/datasets/ranjanhr/nextqa-processed-frames/frames/        ← frame JPEGs
/kaggle/input/datasets/ranjanhr/nextqa-processed-frames/frame_index.json
/kaggle/input/datasets/ranjanhr/nextqa-annotations/data/raw/nextqa/annotations/
/kaggle/input/datasets/ranjanhr/nextqa-embeddings/embeddings/
/kaggle/input/datasets/ranjanhr/nextqa-embeddings/indexes/
```

---

## Config File

**Kaggle config:** `configs/nextqa_kaggle.yaml`
```yaml
dataset: nextqa
annotation_files:
  train: train.csv
  val: val.csv
splits:
  train: train
  val: val
  test: val
vid_map_file: map_vid_vidorID.json
training:
  epochs: 3
  batch_size: 16
  num_workers: 4
  fp16: true
paths:
  data_root: /kaggle/working
  raw_videos: /kaggle/input/datasets/ranjanhr/nextqa-processed-frames/frames
  annotations: /kaggle/input/datasets/ranjanhr/nextqa-annotations/data/raw/nextqa/annotations
  frames_root: /kaggle/working/frames/placeholder
  embeddings_dir: /kaggle/working/embeddings
  index_dir: /kaggle/working/indexes
  checkpoints: /kaggle/working/checkpoints
  results: /kaggle/working/results
  logs: /kaggle/working/logs
```

**Important:** `frames_root` is set to `/kaggle/working/frames/placeholder` so that `preprocess.py` looks for `frame_index.json` at `/kaggle/working/frames/frame_index.json` (writable, remapped paths).

---

## Critical Path Fix (always needed at Kaggle session start)

`frame_index.json` originally contains Windows paths. Must be remapped every new session:

```python
import json, os, shutil
from pathlib import Path

SRC = "/kaggle/input/datasets/ranjanhr/nextqa-processed-frames/frame_index.json"
FRAMES_ROOT = "/kaggle/input/datasets/ranjanhr/nextqa-processed-frames/frames"

with open(SRC) as f:
    frame_index = json.load(f)

def remap(p):
    p = p.replace("\\", "/")
    marker = "frames/"
    idx = p.find(marker)
    if idx != -1:
        relative = p[idx + len(marker):]
        return f"{FRAMES_ROOT}/{relative}"
    return p

remapped = {vid: [remap(p) for p in paths] for vid, paths in frame_index.items()}

os.makedirs("/kaggle/working/frames", exist_ok=True)
with open("/kaggle/working/frames/frame_index.json", "w") as f:
    json.dump(remapped, f)

print(f"Remapped. Sample: {remapped[list(remapped.keys())[0]][0]}")
print(f"Exists: {Path(remapped[list(remapped.keys())[0]][0]).exists()}")
```

---

## Implementation Status

### ✅ COMPLETED

| Phase | What was done |
|---|---|
| Frame extraction | `extract_frames.sh` run locally at 1fps, ~7GB of JPEGs |
| Frame index | `frame_index.json` created locally, uploaded to Kaggle |
| Annotations | `train.csv`, `val.csv`, `test.csv`, `map_vid_vidorID.json` uploaded to Kaggle |
| CLIP encoding | `preprocess.py --split all` run on Kaggle T4, produced `frame_embeddings.npy` + question embeddings |
| FAISS indexes | `build_index.py --verify` run, per-video `.index` files created |
| Kaggle datasets | All three datasets created and accessible |
| Config | `configs/nextqa_kaggle.yaml` created |
| `FrameIndex` class | Added to `src/retrieval/faiss_index.py` |

### ❌ NOT YET IMPLEMENTED

| File | What needs to be built | Priority |
|---|---|---|
| `src/ranking/fine_ranker.py` | `RankerTrainer` class with `train()`, `evaluate()`, checkpoint save/load | Day 2 — HIGH |
| `src/evaluation/evaluator.py` | `QUESTEvaluator` class with `run()` supporting all 4 ablation methods | Day 2 — HIGH |
| `src/retrieval/coarse_retriever.py` | Full Stage 1 pipeline wrapper | Day 2 — MEDIUM |
| `src/data/frame_extractor.py` | Add `detect_scene_boundaries()` and `extract_frames()` | Day 5 — LOW (demo only) |
| `tests/test_data/test_frame_extractor.py` | Unit tests for frame extraction | Day 3 |
| Stage 1 baseline eval | Run `evaluate.py --method clip_topk --max_samples 500` | Day 2 — first thing |

---

## Ablation Results Table (fill in as you run)

| Method | Causal | Temporal | Descriptive | Overall | Status |
|---|---|---|---|---|---|
| Uniform sampling (8 frames) | 42.1 | 44.3 | 60.2 | 48.2 | target from paper |
| CLIP top-K — Stage 1 only | 47.8 | 50.1 | 63.4 | 53.1 | target from paper |
| Ranker top-K — Stage 1+2 | 51.2 | 54.6 | 65.8 | 56.8 | target from paper |
| QUEST full | 54.3 | 57.9 | 67.1 | 59.3 | target from paper |
| **Your CLIP top-K run** | — | — | — | — | ❌ not run yet |
| **Your full QUEST run** | — | — | — | — | ❌ not run yet |

---

## Day 2 Kaggle Notebook Setup

### Datasets to attach
- `ranjanhr/nextqa-processed-frames`
- `ranjanhr/nextqa-annotations`
- `ranjanhr/nextqa-embeddings`

### Standard opening cells (copy-paste every session)

```python
# Cell 1 — clone and install
%cd /kaggle/working
!git clone https://github.com/ranjanhr25/QUEST.git
%cd /kaggle/working/QUEST
!pip install open-clip-torch faiss-cpu bitsandbytes transformers accelerate \
    einops omegaconf rich tqdm --quiet
!pip install -e . --quiet
```

```python
# Cell 2 — remap frame_index paths
import json, os
from pathlib import Path

SRC = "/kaggle/input/datasets/ranjanhr/nextqa-processed-frames/frame_index.json"
FRAMES_ROOT = "/kaggle/input/datasets/ranjanhr/nextqa-processed-frames/frames"

with open(SRC) as f:
    frame_index = json.load(f)

def remap(p):
    p = p.replace("\\", "/")
    marker = "frames/"
    idx = p.find(marker)
    if idx != -1:
        return f"{FRAMES_ROOT}/{p[idx + len(marker):]}"
    return p

remapped = {vid: [remap(p) for p in paths] for vid, paths in frame_index.items()}
os.makedirs("/kaggle/working/frames", exist_ok=True)
with open("/kaggle/working/frames/frame_index.json", "w") as f:
    json.dump(remapped, f)
print("frame_index remapped OK")
```

```python
# Cell 3 — copy embeddings from dataset to working dir
import shutil, os

os.makedirs("/kaggle/working/embeddings", exist_ok=True)
os.makedirs("/kaggle/working/indexes", exist_ok=True)

EMB_SRC = "/kaggle/input/datasets/ranjanhr/nextqa-embeddings/embeddings"
IDX_SRC = "/kaggle/input/datasets/ranjanhr/nextqa-embeddings/indexes"

# Copy embeddings (read-only mount → writable working dir)
!cp -r /kaggle/input/datasets/ranjanhr/nextqa-embeddings/embeddings /kaggle/working/
!cp -r /kaggle/input/datasets/ranjanhr/nextqa-embeddings/indexes /kaggle/working/

print("Embeddings and indexes copied to /kaggle/working/")
```

---

## Key Architecture Notes

### TemporalRanker (Stage 2) — `src/models/temporal_ranker.py`
- Input: 64 candidate frame embeddings (512-dim) + 1 question embedding
- 4 transformer layers, 8 heads, hidden_dim=512
- Output: relevance scores (64,) + uncertainty scalar
- Novel: `SceneBoundaryTemporalEmbedding` encodes position relative to scene boundaries

### DPP Selector — `src/models/dpp_selector.py` ✅ COMPLETE
- Kernel: `L[i,j] = q[i] * q[j] * exp(-λ * |t_i - t_j| / T)`
- Greedy MAP inference — O(N²k) for N=64, k=8
- λ=0.5 default, ablate {0.1, 0.5, 1.0, 2.0}

### Adaptive Budget — `src/ranking/adaptive_budget.py` ✅ COMPLETE
- uncertainty < 0.3 → 8 frames
- uncertainty ≥ 0.3 → 16 frames

### Training objective
- Loss = `rank_weight * listnet_loss + uncertainty_weight * bce_loss`
- `rank_weight=1.0`, `uncertainty_weight=0.5` (from `configs/base.yaml`)

---

## Files Overview

```
src/
  data/
    frame_extractor.py    ✅ (missing detect_scene_boundaries — add on Day 5)
    dataset.py            ✅ NExTQADataset + RankerTrainDataset
    preprocess.py         ✅ CLIP embedding precomputation
  models/
    clip_encoder.py       ✅
    temporal_ranker.py    ✅
    dpp_selector.py       ✅
    vlm_interface.py      ✅
  ranking/
    fine_ranker.py        ❌ EMPTY — implement Day 2
    adaptive_budget.py    ✅
  retrieval/
    faiss_index.py        ✅ (FrameIndex class added)
    coarse_retriever.py   ❌ EMPTY — implement Day 2
  evaluation/
    evaluator.py          ❌ EMPTY — implement Day 2
    metrics.py            ✅
    benchmarks.py         ❌ EMPTY
  utils/
    config.py             ✅
    io_utils.py           ✅
    logger.py             ✅
    visualization.py      ✅
scripts/
  train_ranker.py         ✅ (depends on fine_ranker.py)
  evaluate.py             ✅ (depends on evaluator.py)
  build_index.py          ✅ DONE
  extract_frames.sh       ✅ DONE
  demo.py                 ✅ (depends on detect_scene_boundaries)
```

---

## Day-by-Day Plan

| Day | Goal | Key files |
|---|---|---|
| 1 ✅ | Frames → embeddings → FAISS indexes | `extract_frames.sh`, `preprocess.py`, `build_index.py` |
| 2 | Implement ranker + evaluator, run Stage 1 baseline, train ranker | `fine_ranker.py`, `evaluator.py`, `coarse_retriever.py` |
| 3 | Full QUEST eval, ablation table, fix tests | `evaluate.py --method quest`, `tests/` |
| 4 | Tune DPP λ, uncertainty threshold, write results | ablation configs |
| 5 | Demo on sample video, final README update | `demo.py`, `detect_scene_boundaries` |

---

## How to Update This File

After completing a phase, update:
1. Move items from `❌ NOT YET IMPLEMENTED` → `✅ COMPLETED`
2. Fill in your actual numbers in the **Ablation Results Table**
3. Update file status in **Files Overview** (❌ → ✅)
4. Add any new gotchas or path issues discovered to a new section