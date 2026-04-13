# Local Setup Guide

## Prerequisites

- Python 3.9+
- ffmpeg installed system-wide (`brew install ffmpeg` or `apt install ffmpeg`)
- CUDA-capable GPU with 8+ GB VRAM (for local runs)

## Step-by-step

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/QUEST.git
cd QUEST

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install
pip install -e ".[dev]"

# 4. Verify installation
python -c "import torch; import open_clip; import faiss; print('All good')"

# 5. Download NExT-QA annotations (~50 MB)
mkdir -p data/raw/nextqa/annotations
# Download from: https://github.com/doc-doc/NExT-QA/releases
# Place nextqa_val.csv in data/raw/nextqa/annotations/

# 6. Extract frames (replace VIDEO_DIR with your video directory)
bash scripts/extract_frames.sh data/raw/nextqa/videos/ data/processed/nextqa/ --fps 1

# 7. Build FAISS index
python scripts/build_index.py --config configs/nextqa.yaml

# 8. Run demo
python scripts/demo.py --video path/to/test_video.mp4 --question "What happens first?"
```

## Running tests

```bash
pytest tests/ -v
```
