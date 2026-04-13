# Running QUEST on Kaggle (Free T4s)

## Setup (do this once per notebook session)

```python
# Cell 1: install dependencies
!pip install open-clip-torch faiss-cpu bitsandbytes transformers accelerate \
            scenedetect[opencv] ffmpeg-python einops omegaconf rich --quiet

# Cell 2: clone your repo
!git clone https://github.com/YOUR_USERNAME/QUEST.git
%cd QUEST
!pip install -e . --quiet
```

## Enabling 2×T4 (DataParallel training)

In your Kaggle notebook settings → Accelerator → select **GPU T4 x2**.
The training script auto-detects multiple GPUs via `torch.cuda.device_count()`.

## Saving checkpoints to Kaggle output

Kaggle sessions reset. Always save checkpoints to `/kaggle/working/` and
then copy them to your output dataset so they persist across sessions:

```python
import shutil
shutil.copy("results/checkpoints/best.pt", "/kaggle/working/quest_best.pt")
```

Then in the next session, load from `/kaggle/input/your-dataset/quest_best.pt`.

## Memory tips for free T4 (16 GB VRAM)

| Component | VRAM |
|-----------|------|
| CLIP ViT-B/32 (fp16) | ~1.5 GB |
| TemporalRanker training (batch=32) | ~4 GB |
| LLaVA-1.5-7B (4-bit) | ~5 GB |
| FAISS index (CPU) | 0 GB |
| **Total (inference)** | ~7 GB |
| **Total (training, no VLM)** | ~6 GB |

You can run training and inference in separate cells — no need to keep VLM loaded during training.

## Recommended workflow

1. **Notebook 01** — Data exploration + frame extraction (CPU, no GPU needed)
2. **Notebook 02** — CLIP encoding + FAISS baseline (GPU, ~1 hr for NExT-QA)
3. **Notebook 03** — Ranker training (2×T4, ~3 hrs)
4. **Notebook 04** — Ablation evaluation (1×T4, ~1 hr)
5. **Notebook 05** — Full pipeline demo + results
