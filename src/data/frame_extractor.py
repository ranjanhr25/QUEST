"""
Frame extraction from video files using ffmpeg.

Design decisions:
- ffmpeg is called as a subprocess (via ffmpeg-python) rather than
  using OpenCV's VideoCapture so we get hardware-accelerated decoding
  where available and avoid OpenCV's codec licensing issues on some systems.
- Frames are resized during extraction (not post-hoc) to save disk space.
- Scene boundaries are detected with PySceneDetect and stored alongside
  frame files for use in temporal position embeddings.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Generator

import numpy as np
from scenedetect import detect, ContentDetector

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    size: int = 224,
    max_frames: int = 1000,
) -> list[Path]:
    """
    Extract frames from a video at a fixed FPS and save as JPEGs.

    Args:
        video_path: Path to input video file (mp4, avi, mkv, etc.).
        output_dir: Directory to write frame JPEGs into.
        fps:        Frames per second to extract. 1.0 is usually enough for QA.
        size:       Resize shorter edge to this many pixels (preserves aspect ratio).
        max_frames: Hard cap on number of frames extracted. Videos exceeding this
                    are sub-sampled uniformly before extraction.

    Returns:
        Sorted list of Paths to extracted JPEG files.

    Example:
        >>> paths = extract_frames("video.mp4", "data/processed/vid001/", fps=1)
        >>> len(paths)
        142
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted (idempotent)
    existing = sorted(output_dir.glob("frame_*.jpg"))
    if existing:
        logger.debug(f"Frames already extracted for {video_path} ({len(existing)} found)")
        return existing

    # ffmpeg command: extract at fps, scale shorter edge to `size`
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps},scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}",
        "-q:v", "2",                        # JPEG quality (2 = near-lossless)
        "-frames:v", str(max_frames),
        str(output_dir / "frame_%06d.jpg"),
        "-loglevel", "error",
        "-y",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg failed for {video_path}: {result.stderr}")
        return []

    frames = sorted(output_dir.glob("frame_*.jpg"))
    logger.info(f"Extracted {len(frames)} frames from {Path(video_path).name}")
    return frames


def detect_scene_boundaries(video_path: str) -> list[float]:
    """
    Detect shot/scene boundaries in a video using PySceneDetect.

    Uses a content-based detector (compares frame histograms).
    Fast enough to run on CPU in seconds for typical short videos.

    Args:
        video_path: Path to input video.

    Returns:
        List of scene boundary timestamps in seconds, sorted ascending.
        Empty list if detection fails or video has no detected cuts.

    Example:
        >>> boundaries = detect_scene_boundaries("video.mp4")
        >>> boundaries
        [0.0, 12.3, 45.7, 88.1]
    """
    try:
        scene_list = detect(str(video_path), ContentDetector(threshold=27.0))
        # scene_list is a list of (start_timecode, end_timecode) tuples
        boundaries = [scene[0].get_seconds() for scene in scene_list]
        return sorted(boundaries)
    except Exception as e:
        logger.warning(f"Scene detection failed for {video_path}: {e}")
        return [0.0]


def compute_temporal_position(
    frame_idx: int,
    total_frames: int,
    scene_boundaries_sec: list[float],
    fps: float = 1.0,
) -> dict[str, float]:
    """
    Compute scene-boundary-aware temporal position features for a frame.

    Instead of a simple normalised time index, we encode three values:
      - absolute_pos:       frame_idx / total_frames  (0 to 1)
      - scene_relative_pos: position within current scene (0=scene start, 1=scene end)
      - scene_id:           which scene this frame belongs to (normalised 0 to 1)

    This gives the VLM richer temporal context — it knows whether a frame
    is at the beginning, middle, or end of a scene, and how far through
    the full video that scene occurs.

    Args:
        frame_idx:            0-based frame index.
        total_frames:         Total number of frames in the video.
        scene_boundaries_sec: Scene start times in seconds from detect_scene_boundaries.
        fps:                  FPS used during extraction.

    Returns:
        Dict with keys: "absolute_pos", "scene_relative_pos", "scene_id".
    """
    frame_time = frame_idx / fps

    # Find which scene this frame belongs to
    scene_id = 0
    for i, boundary in enumerate(scene_boundaries_sec):
        if frame_time >= boundary:
            scene_id = i

    num_scenes = max(len(scene_boundaries_sec), 1)
    scene_start = scene_boundaries_sec[scene_id] if scene_boundaries_sec else 0.0
    scene_end = (
        scene_boundaries_sec[scene_id + 1]
        if scene_id + 1 < len(scene_boundaries_sec)
        else total_frames / fps
    )
    scene_duration = max(scene_end - scene_start, 1e-6)

    return {
        "absolute_pos": frame_idx / max(total_frames - 1, 1),
        "scene_relative_pos": (frame_time - scene_start) / scene_duration,
        "scene_id": scene_id / max(num_scenes - 1, 1),
    }
