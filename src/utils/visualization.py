"""
Visualization helpers for debugging and the demo.

- visualize_retrieval: shows a grid of retrieved frames with relevance scores
- visualize_dpp_vs_topk: side-by-side comparison of DPP vs top-K selection
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def make_frame_grid(
    frame_paths: list[str],
    scores: list[float] | None = None,
    cols: int = 4,
    frame_size: int = 112,
    label: str | None = None,
) -> Image.Image:
    """
    Create a grid of frames with optional relevance score labels.

    Args:
        frame_paths: Paths to frame JPEG files.
        scores:      Optional relevance score per frame (shown as text overlay).
        cols:        Number of columns in the grid.
        frame_size:  Resize each frame to this square size.
        label:       Optional title to draw above the grid.

    Returns:
        PIL Image of the full grid.
    """
    n = len(frame_paths)
    rows = (n + cols - 1) // cols
    grid_w = cols * frame_size
    grid_h = rows * frame_size + (30 if label else 0)

    grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(grid)

    if label:
        draw.text((4, 4), label, fill=(220, 220, 220))

    for i, path in enumerate(frame_paths):
        row, col = divmod(i, cols)
        y_offset = 30 if label else 0
        x = col * frame_size
        y = row * frame_size + y_offset

        img = Image.open(path).convert("RGB").resize((frame_size, frame_size))
        grid.paste(img, (x, y))

        if scores is not None:
            score_text = f"{scores[i]:.2f}"
            draw.rectangle([x, y, x + 36, y + 14], fill=(0, 0, 0, 180))
            draw.text((x + 2, y + 1), score_text, fill=(255, 215, 0))

    return grid


def compare_dpp_vs_topk(
    frame_paths: list[str],
    topk_indices: list[int],
    dpp_indices: list[int],
    scores: list[float],
    save_path: str | None = None,
) -> Image.Image:
    """
    Side-by-side comparison: top-K selected frames vs DPP-selected frames.
    Useful for the demo and for figures in your technical report.
    """
    topk_paths = [frame_paths[i] for i in topk_indices]
    dpp_paths = [frame_paths[i] for i in dpp_indices]
    topk_scores = [scores[i] for i in topk_indices]
    dpp_scores = [scores[i] for i in dpp_indices]

    topk_grid = make_frame_grid(topk_paths, topk_scores, cols=4, label="Top-K (uniform)")
    dpp_grid = make_frame_grid(dpp_paths, dpp_scores, cols=4, label="DPP (diverse)")

    w = max(topk_grid.width, dpp_grid.width)
    combined = Image.new("RGB", (w, topk_grid.height + dpp_grid.height + 8), (20, 20, 20))
    combined.paste(topk_grid, (0, 0))
    combined.paste(dpp_grid, (0, topk_grid.height + 8))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        combined.save(save_path)

    return combined
