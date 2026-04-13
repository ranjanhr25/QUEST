"""
Interactive demo: run QUEST on a single local video file.

Usage:
    python scripts/demo.py --video path/to/video.mp4 --question "What does the person do first?"
    python scripts/demo.py --video path/to/video.mp4 --question "..." --checkpoint results/checkpoints/best.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.frame_extractor import extract_frames, detect_scene_boundaries
from src.models.clip_encoder import CLIPEncoder
from src.retrieval.faiss_index import FrameIndex
from src.models.dpp_selector import dpp_select_frames
from src.models.vlm_interface import LLaVAInterface
from src.utils.logger import get_logger
from PIL import Image
import numpy as np

logger = get_logger(__name__)


def run_demo(video_path: str, question: str, checkpoint: str | None = None):
    logger.info(f"Video: {video_path}")
    logger.info(f"Question: {question}")

    # 1. Extract frames
    frame_dir = Path("results/demo_frames") / Path(video_path).stem
    frame_paths = extract_frames(video_path, str(frame_dir), fps=1, size=224)
    logger.info(f"Extracted {len(frame_paths)} frames")

    # 2. Detect scene boundaries
    boundaries = detect_scene_boundaries(video_path)
    logger.info(f"Detected {len(boundaries)} scene boundaries")

    # 3. Encode frames with CLIP
    encoder = CLIPEncoder("ViT-B-32", "openai", device="cuda")
    frame_embs = encoder.encode_frames([str(p) for p in frame_paths])  # (N, 512)
    question_emb = encoder.encode_text([question])[0]                  # (512,)

    # 4. Stage 1: FAISS coarse retrieval
    index = FrameIndex(embed_dim=512, index_type="flat")
    index.build(frame_embs)
    scores, candidate_indices = index.search(question_emb, k=min(64, len(frame_paths)))
    logger.info(f"Stage 1: retrieved {len(candidate_indices)} candidates")

    # 5. Stage 2: DPP selection (using CLIP scores as relevance proxy for demo)
    candidate_scores = scores.astype(np.float32)
    candidate_timestamps = candidate_indices.astype(np.float32)  # 1fps → idx ≈ seconds

    selected_local_indices = dpp_select_frames(
        relevance_scores=candidate_scores,
        frame_timestamps=candidate_timestamps,
        k=8,
        lambda_diversity=0.5,
    )
    selected_frame_indices = candidate_indices[selected_local_indices]
    logger.info(f"Stage 2: DPP selected frames at indices: {sorted(selected_frame_indices.tolist())}")

    # 6. Load selected frames as PIL images
    selected_images = [Image.open(frame_paths[i]).convert("RGB") for i in selected_frame_indices]

    # 7. VLM answer
    vlm = LLaVAInterface(load_in_4bit=True)
    answer = vlm.answer(selected_images, question)

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"Answer:   {answer}")
    print(f"{'='*60}\n")
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    run_demo(args.video, args.question, args.checkpoint)
