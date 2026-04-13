"""
LLaVA inference wrapper — frozen VLM for the final answer generation step.

The VLM is loaded in 4-bit quantization via bitsandbytes, which reduces
VRAM from ~14 GB (fp16) to ~5 GB, making it fit on a free T4 with room
left for the ranker and FAISS index.

IMPORTANT: we never fine-tune the VLM. It's always frozen. Only the
TemporalRanker is trained. This keeps training compute on free-tier hardware.
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLaVAInterface:
    """
    Frozen LLaVA-1.5 wrapper for multi-frame video QA inference.

    Accepts a list of frame PIL Images and a question string.
    Formats them into the LLaVA conversation template and returns the answer.

    Args:
        model_id:      HuggingFace model ID. Default: "llava-hf/llava-1.5-7b-hf".
        load_in_4bit:  Use bitsandbytes 4-bit quantization. Required for free T4.
        device:        "cuda" or "cpu".

    Example:
        >>> vlm = LLaVAInterface()
        >>> answer = vlm.answer(frames=[img1, img2, img3], question="What happens first?")
        >>> print(answer)
        "The person picks up the phone."
    """

    def __init__(
        self,
        model_id: str = "llava-hf/llava-1.5-7b-hf",
        load_in_4bit: bool = True,
        device: str = "cuda",
    ):
        self.device = device
        logger.info(f"Loading LLaVA from {model_id} (4bit={load_in_4bit})…")

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",          # automatically places layers across available GPUs
            torch_dtype=torch.float16,
        )
        self.model.eval()
        logger.info("LLaVA loaded and ready")

    @torch.no_grad()
    def answer(
        self,
        frames: list[Image.Image],
        question: str,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Generate an answer given selected frames and a question.

        Frames are concatenated horizontally into a grid image before being
        passed to LLaVA — this is the simplest way to pass multiple frames
        without modifying the model architecture.

        Args:
            frames:         List of PIL Images (the selected frames from QUEST).
            question:       Question string.
            max_new_tokens: Max tokens to generate.

        Returns:
            Generated answer string (stripped).
        """
        # Tile frames into a grid for LLaVA's single-image input
        grid = self._tile_frames(frames)

        # LLaVA conversation format
        prompt = f"USER: <image>\nThe following is a sequence of frames from a video. {question}\nASSISTANT:"

        inputs = self.processor(text=prompt, images=grid, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,            # greedy for reproducibility
        )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][input_len:]
        answer = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return answer.strip()

    @staticmethod
    def _tile_frames(frames: list[Image.Image], cols: int = 4) -> Image.Image:
        """
        Tile a list of PIL Images into a grid (left to right, top to bottom).

        Args:
            frames: Frame images (all should be the same size).
            cols:   Number of columns in the grid.

        Returns:
            A single PIL Image containing all frames tiled.
        """
        if not frames:
            raise ValueError("No frames to tile")

        w, h = frames[0].size
        rows = (len(frames) + cols - 1) // cols
        grid = Image.new("RGB", (cols * w, rows * h))

        for i, frame in enumerate(frames):
            row, col = divmod(i, cols)
            grid.paste(frame, (col * w, row * h))

        return grid
