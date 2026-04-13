"""
Frame extractor for QUEST.

Extracts frames from all NExT-QA videos using ffmpeg, with:
  - Parallel extraction (ProcessPoolExecutor).
  - Resume support — skips videos whose frame dirs already exist and are complete.
  - VidOR subdirectory layout support (video_id → group/video_id.mp4).
  - Graceful error handling — failed videos are logged and skipped.
  - Progress bar via tqdm.

CLI:
    python -m src.data.frame_extractor \
        --video_dir  data/raw/nextqa/videos \
        --output_dir data/processed/nextqa/frames \
        --fps 1 \
        --quality 90 \
        --workers 8 \
        --vid_map    data/raw/nextqa/annotations/map_vid_vidorID.json

The vid_map argument is optional. If provided, the extractor uses it to locate
videos stored in VidOR's subdirectory layout (e.g. 0001/3088734333.mp4).
If not provided it falls back to a flat search for {video_id}.mp4.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

log = get_logger("frame_extractor")


# ---------------------------------------------------------------------------
# Video resolution helpers
# ---------------------------------------------------------------------------

def _find_video(
    video_id: str,
    video_dir: Path,
    vid_map: dict[str, str] | None,
) -> Optional[Path]:
    """
    Locate the video file for a given video_id.

    Strategy:
    1. Use vid_map ONLY if corresponding file actually exists on disk.
    2. Flat layout: video_dir/{video_id}.mp4
    3. Recursive search (slow — only if above two fail).
    """

    # --- FIX: check mapped path WITH extension and existence ---
    if vid_map is not None and video_id in vid_map:
        base = video_dir / vid_map[video_id]
        for ext in (".mp4", ".avi", ".mkv", ".mov"):
            candidate = base.with_suffix(ext)
            if candidate.exists():
                return candidate
        return None  # mapped entry exists but file does NOT → skip

    # Flat layout
    for ext in (".mp4", ".avi", ".mkv", ".mov"):
        candidate = video_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate

    # Last resort: recursive glob
    matches = list(video_dir.rglob(f"{video_id}.mp4"))
    if matches:
        return matches[0]

    return None


def _count_expected_frames(video_path: Path, fps: float) -> int:
    """Use ffprobe to estimate expected frame count at given fps."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=30)
        info = json.loads(out)
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                duration = float(s.get("duration", 0))
                if duration > 0:
                    return max(1, int(duration * fps))
    except Exception:
        pass
    return -1  # unknown


# ---------------------------------------------------------------------------
# Single-video extraction (run in subprocess)
# ---------------------------------------------------------------------------

def _extract_one(
    video_id: str,
    video_path: str,
    output_dir: str,
    fps: float,
    quality: int,
) -> tuple[str, int, str | None]:
    """
    Extract frames for one video using ffmpeg.

    Returns (video_id, n_frames_written, error_message_or_None).
    Runs in a separate process — must not capture any non-picklable state.
    """
    out_dir = Path(output_dir) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already complete (has ≥1 jpg)
    existing = sorted(out_dir.glob("*.jpg"))
    if existing:
        return video_id, len(existing), None  # already done — skip

    pattern = str(out_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", str(int((100 - quality) / 100 * 31 + 1)),  # ffmpeg q scale 1-31
        "-start_number", "0",
        pattern,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300, capture_output=True)
    except subprocess.CalledProcessError as e:
        return video_id, 0, e.stderr.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return video_id, 0, "ffmpeg timeout after 300s"
    except FileNotFoundError:
        return video_id, 0, "ffmpeg not found in PATH"

    frames = sorted(out_dir.glob("*.jpg"))
    return video_id, len(frames), None


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract_all(
    video_ids: list[str],
    video_dir: Path,
    output_dir: Path,
    fps: float = 1.0,
    quality: int = 90,
    max_workers: int = 8,
    vid_map: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """
    Extract frames for all video_ids in parallel.

    Returns a frame_index dict:  video_id → sorted list of absolute frame paths.
    Videos that fail are logged and excluded from the index.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve video paths
    tasks: list[tuple[str, Path]] = []
    not_found: list[str] = []
    for vid_id in video_ids:
        vpath = _find_video(vid_id, video_dir, vid_map)
        if vpath is None:
            not_found.append(vid_id)
        else:
            tasks.append((vid_id, vpath))

    if not_found:
        log.warning("Videos not found — skipping", count=len(not_found), examples=not_found[:5])

    log.info("Starting frame extraction", total=len(tasks), fps=fps, workers=max_workers)

    frame_index: dict[str, list[str]] = {}
    failed: list[str] = []

    iterator = tasks
    if tqdm is not None:
        iterator = tqdm(tasks, desc="Extracting frames", unit="video", dynamic_ncols=True)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_one, vid_id, str(vpath), str(output_dir), fps, quality): vid_id
            for vid_id, vpath in tasks
        }
        done_iter = as_completed(futures)
        if tqdm is not None:
            done_iter = tqdm(done_iter, total=len(futures), desc="Extracting", unit="video")

        for future in done_iter:
            vid_id = futures[future]
            try:
                vid_id_out, n_frames, err = future.result()
            except Exception as exc:
                log.error("Unexpected extraction error", video_id=vid_id, error=str(exc))
                failed.append(vid_id)
                continue

            if err:
                log.error("Extraction failed", video_id=vid_id_out, error=err)
                failed.append(vid_id_out)
            else:
                frame_dir = output_dir / vid_id_out
                frames = sorted(str(p) for p in frame_dir.glob("*.jpg"))
                frame_index[vid_id_out] = frames

    log.info(
        "Extraction complete",
        success=len(frame_index),
        failed=len(failed),
        not_found=len(not_found),
    )
    if failed:
        log.warning("Failed video IDs", ids=failed[:10])

    return frame_index


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract frames from NExT-QA videos.")
    p.add_argument("--video_dir",  required=True, type=Path, help="Directory containing .mp4 files")
    p.add_argument("--output_dir", required=True, type=Path, help="Where to write extracted frames")
    p.add_argument("--fps",        default=1.0, type=float, help="Frames per second to extract")
    p.add_argument("--quality",    default=90,  type=int,   help="JPEG quality 0-100")
    p.add_argument("--workers",    default=8,   type=int,   help="Parallel extraction workers")
    p.add_argument("--max_videos", default=None, type=int,
               help="Limit number of videos to process (for debugging)")
    p.add_argument("--vid_map",    default=None, type=Path,
                   help="Optional map_vid_vidorID.json (VidOR layout)")
    p.add_argument("--annotation_dir", default=None, type=Path,
                   help="If set, infer video_ids from all CSVs in this dir")
    p.add_argument("--frame_index_out", default=None, type=Path,
                   help="Where to write the resulting frame_index.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load vid_map if provided
    vid_map: dict[str, str] | None = None
    if args.vid_map and args.vid_map.exists():
        with open(args.vid_map) as f:
            vid_map = json.load(f)
        log.info("Loaded VidOR map", entries=len(vid_map))

    # Collect video IDs
    if args.annotation_dir:
        import pandas as pd
        video_ids: list[str] = []
        for csv_path in sorted(args.annotation_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            if "video" in df.columns:
                video_ids.extend(str(v) for v in df["video"].unique().tolist())
        video_ids = list(set(video_ids))
        log.info("Collected video IDs from annotations", count=len(video_ids))
    elif vid_map is not None:
        video_ids = list(vid_map.keys())
        log.info("Using all video IDs from vid_map", count=len(video_ids))
    else:
        # Fall back: enumerate all mp4s in video_dir
        video_ids = [p.stem for p in args.video_dir.rglob("*.mp4")]
        log.info("Scanned video_dir for mp4s", count=len(video_ids))
        
    # Apply optional limit
    if args.max_videos is not None:
        video_ids = video_ids[:args.max_videos]
        log.info("Limiting videos", max_videos=args.max_videos, used=len(video_ids))

    frame_index = extract_all(
        video_ids=video_ids,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        quality=args.quality,
        max_workers=args.workers,
        vid_map=vid_map,
    )

    # Save frame index
    out_path = args.frame_index_out or (args.output_dir.parent / "frame_index.json")
    from src.utils.io_utils import save_frame_index
    save_frame_index(frame_index, out_path)
    log.info("Frame index written", path=str(out_path))


if __name__ == "__main__":
    main()