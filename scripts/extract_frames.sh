#!/usr/bin/env bash
# ===========================================================================
# extract_frames.sh
#
# Shell wrapper around src/data/frame_extractor.py.
# Extracts frames from all NExT-QA videos at 1fps using ffmpeg.
#
# Usage:
#   bash scripts/extract_frames.sh
#
# Or with overrides:
#   VIDEO_DIR=/path/to/videos   \
#   OUTPUT_DIR=/path/to/frames  \
#   FPS=1                       \
#   WORKERS=8                   \
#   bash scripts/extract_frames.sh
#
# Environment variables (all optional — defaults shown):
#   VIDEO_DIR      data/raw/nextqa/videos
#   OUTPUT_DIR     data/processed/nextqa/frames
#   ANN_DIR        data/raw/nextqa/annotations
#   VID_MAP        data/raw/nextqa/annotations/map_vid_vidorID.json
#   FPS            1
#   QUALITY        90      (JPEG quality 0-100)
#   WORKERS        8       (parallel ffmpeg processes)
#   FRAME_IDX_OUT  data/processed/nextqa/frame_index.json
#
# Prerequisites:
#   - ffmpeg installed and on PATH  (conda install -c conda-forge ffmpeg)
#   - Python env activated           (conda activate <your_env>)
# ===========================================================================

set -euo pipefail

# --- Force correct conda env ---
CONDA_ENV="/d/Anaconda/envs/quest"

export PATH="$CONDA_ENV:$CONDA_ENV/Scripts:$CONDA_ENV/Library/bin:$PATH"
# ── Defaults ──────────────────────────────────────────────────────────────
VIDEO_DIR="${VIDEO_DIR:-data/raw/nextqa/videos}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/nextqa/frames}"
ANN_DIR="${ANN_DIR:-data/raw/nextqa/annotations}"
VID_MAP="${VID_MAP:-data/raw/nextqa/annotations/map_vid_vidorID.json}"
FPS="${FPS:-1}"
QUALITY="${QUALITY:-90}"
WORKERS="${WORKERS:-8}"
FRAME_IDX_OUT="${FRAME_IDX_OUT:-data/processed/nextqa/frame_index.json}"

# ── Resolve project root (one level up from this script) ──────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ── Checks ────────────────────────────────────────────────────────────────

if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "[ERROR] VIDEO_DIR does not exist: ${VIDEO_DIR}"
    echo "        Download NExT-QA videos from: https://github.com/doc-doc/NExT-QA"
    exit 1
fi

echo "============================================"
echo "  QUEST Frame Extractor"
echo "============================================"
echo "  VIDEO_DIR   : ${VIDEO_DIR}"
echo "  OUTPUT_DIR  : ${OUTPUT_DIR}"
echo "  ANN_DIR     : ${ANN_DIR}"
echo "  FPS         : ${FPS}"
echo "  QUALITY     : ${QUALITY}"
echo "  WORKERS     : ${WORKERS}"
echo "  FRAME_IDX   : ${FRAME_IDX_OUT}"
echo "============================================"

# Build command args
CMD_ARGS=(
    --video_dir     "${VIDEO_DIR}"
    --output_dir    "${OUTPUT_DIR}"
    --fps           "${FPS}"
    --quality       "${QUALITY}"
    --workers       "${WORKERS}"
    --frame_index_out "${FRAME_IDX_OUT}"
    --annotation_dir "${ANN_DIR}"
)

# Add vid_map if the file exists (handles VidOR subdirectory layout)
if [[ -f "${VID_MAP}" ]]; then
    CMD_ARGS+=(--vid_map "${VID_MAP}")
    echo "  VID_MAP     : ${VID_MAP}"
else
    echo "  VID_MAP     : not found — using flat layout"
fi

echo ""
echo "Starting extraction..."
python -m src.data.frame_extractor "${CMD_ARGS[@]}"

echo ""
echo "[DONE] Frames written to: ${OUTPUT_DIR}"
echo "[DONE] Frame index at  : ${FRAME_IDX_OUT}"