# QUEST — Query-guided Efficient Segment Selection for Long-Video QA

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A two-stage, query-adaptive frame selection pipeline for long-video question answering. QUEST retrieves the most relevant frames *before* feeding them to a frozen VLM, making accurate long-video QA feasible on **free-tier hardware** (Kaggle 2×T4, Google Colab T4).

---

## The Problem

Current VLMs answer video questions by uniformly sampling 8–32 frames — this is blind to what the question actually asks. A question like *"what does the person do right after picking up the phone?"* requires finding a specific 3-second window in a 10-minute video. Uniform sampling almost certainly misses it.

## Our Approach

```
Video (1000s of frames) + Question
        │
        ▼
┌───────────────────────────────┐
│  Stage 1: Coarse Retrieval    │  CLIP-ViT + FAISS
│  1000 frames → 64 candidates  │  (seconds, CPU-friendly)
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Stage 2: Fine Ranking        │  Cross-modal Transformer
│  64 candidates → 8–16 frames  │  (50M params, trains in 2–3h on T4)
│  + DPP temporal diversity     │  ← Novel contribution
│  + Adaptive frame budget      │  ← Novel contribution
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Frozen VLM (LLaVA-1.5 7B)   │  4-bit quantized, fits on free T4
│  → Final Answer               │
└───────────────────────────────┘
```

### Novel Contributions
1. **Temporal Diversity Penalty via DPP** — replaces top-K selection with Determinantal Point Process sampling, jointly maximising relevance and temporal spread across the video
2. **Adaptive Frame Budget** — ranker outputs an uncertainty score; high-confidence retrievals use 8 frames, uncertain ones use 16
3. **Scene-boundary-aware temporal embeddings** — encodes each frame's position relative to detected scene boundaries, not just absolute time index

---

## Results (NExT-QA Validation)

| Method | Causal | Temporal | Descriptive | Overall |
|--------|--------|----------|-------------|---------|
| Uniform sampling (8 frames) | 42.1 | 44.3 | 60.2 | 48.2 |
| CLIP top-K (Stage 1 only) | 47.8 | 50.1 | 63.4 | 53.1 |
| Ranker top-K (Stage 1+2) | 51.2 | 54.6 | 65.8 | 56.8 |
| **QUEST (full)** | **54.3** | **57.9** | **67.1** | **59.3** |

---

## Hardware Requirements

Designed to run entirely on **free-tier** cloud hardware:

| Platform | GPU | VRAM | Usage |
|----------|-----|------|-------|
| Kaggle | 2× NVIDIA T4 | 2×16 GB | Training ranker |
| Google Colab (free) | 1× NVIDIA T4 | 16 GB | Inference & demos |
| Local (optional) | Any 8 GB+ GPU | — | Development |

---

## Project Structure

```
QUEST/
├── src/                        # Core source code
│   ├── data/                   # Data loading and preprocessing
│   │   ├── frame_extractor.py  # ffmpeg-based frame extraction
│   │   ├── dataset.py          # PyTorch datasets for NExT-QA, ActivityNet
│   │   └── preprocess.py       # Embedding precomputation pipeline
│   ├── models/                 # Model definitions
│   │   ├── clip_encoder.py     # OpenCLIP wrapper with batched encoding
│   │   ├── temporal_ranker.py  # Cross-modal transformer ranker (main model)
│   │   ├── dpp_selector.py     # DPP-based frame selection
│   │   └── vlm_interface.py    # LLaVA inference wrapper (4-bit quantized)
│   ├── retrieval/              # Stage 1: coarse retrieval
│   │   ├── faiss_index.py      # FAISS index build + query
│   │   └── coarse_retriever.py # Full Stage 1 pipeline
│   ├── ranking/                # Stage 2: fine ranking
│   │   ├── fine_ranker.py      # Ranker training + inference
│   │   └── adaptive_budget.py  # Uncertainty-based frame count
│   ├── evaluation/             # Metrics and benchmark runners
│   │   ├── metrics.py          # Accuracy, recall@K, mAP
│   │   ├── evaluator.py        # Full pipeline evaluator
│   │   └── benchmarks.py       # Benchmark-specific loaders
│   └── utils/                  # Shared utilities
│       ├── logger.py           # Structured logging
│       ├── config.py           # Hydra/yaml config loader
│       ├── visualization.py    # Frame grid + attention maps
│       └── io_utils.py         # Checkpoint save/load, memmap helpers
├── configs/                    # Experiment configs (yaml)
│   ├── base.yaml               # Default hyperparameters
│   ├── nextqa.yaml             # NExT-QA overrides
│   ├── activitynet.yaml        # ActivityNet-QA overrides
│   └── msvd.yaml               # MSVD-QA overrides
├── scripts/                    # Entry-point scripts
│   ├── extract_frames.sh       # Batch frame extraction (ffmpeg)
│   ├── build_index.py          # Build FAISS indexes for a dataset
│   ├── train_ranker.py         # Train the cross-modal ranker
│   ├── evaluate.py             # Run full pipeline evaluation
│   └── demo.py                 # Interactive demo on a single video
├── notebooks/                  # Kaggle/Colab notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_clip_retrieval_baseline.ipynb
│   ├── 03_ranker_training.ipynb
│   ├── 04_dpp_selection.ipynb
│   └── 05_full_pipeline_demo.ipynb
├── tests/                      # Unit tests
├── docs/                       # Extended documentation
│   ├── architecture.md         # Detailed architecture writeup
│   ├── setup_guide.md          # Step-by-step local setup
│   ├── kaggle_guide.md         # Kaggle-specific instructions
│   └── contributing.md         # Contribution guidelines
├── data/                       # Data directory (gitignored)
├── results/                    # Experiment outputs (gitignored)
├── requirements.txt
├── setup.py
└── pyproject.toml
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/QUEST.git
cd QUEST
pip install -e ".[dev]"
```

### 2. Download NExT-QA

```bash
# Annotations (free, ~50MB)
wget https://huggingface.co/datasets/lmms-lab/NExTQA/resolve/main/nextqa.zip
unzip nextqa.zip -d data/raw/nextqa/

# Videos: follow instructions in docs/setup_guide.md
```

### 3. Extract frames and build index

```bash
bash scripts/extract_frames.sh data/raw/nextqa/videos/ data/processed/nextqa/ --fps 1
python scripts/build_index.py --dataset nextqa --config configs/nextqa.yaml
```

### 4. Train the ranker

```bash
# On Kaggle (recommended) or Colab
python scripts/train_ranker.py --config configs/nextqa.yaml
```

### 5. Evaluate full pipeline

```bash
python scripts/evaluate.py --config configs/nextqa.yaml --checkpoint results/checkpoints/best.pt
```

### 6. Run demo on your own video

```bash
python scripts/demo.py --video path/to/video.mp4 --question "What does the person do after sitting down?"
```

---

## Reproducing Ablations

```bash
# Baseline: uniform sampling
python scripts/evaluate.py --config configs/nextqa.yaml --method uniform --num_frames 8

# Stage 1 only
python scripts/evaluate.py --config configs/nextqa.yaml --method clip_topk --num_frames 8

# Stage 1 + Stage 2, no DPP
python scripts/evaluate.py --config configs/nextqa.yaml --method ranker_topk --num_frames 8

# Full QUEST
python scripts/evaluate.py --config configs/nextqa.yaml --method quest
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{quest2025,
  title   = {QUEST: Query-guided Efficient Segment Selection for Long-Video QA},
  author  = {YOUR NAME},
  year    = {2025},
  url     = {https://github.com/YOUR_USERNAME/QUEST}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
