"""
CLIP encoder wrapper for QUEST.

Uses open_clip for both frame and text encoding.
Supports batched encoding, automatic GPU/CPU selection, and AMP.

Usage:
    encoder = CLIPEncoder.from_config(cfg)
    frame_embs = encoder.encode_frames(list_of_pil_images)   # (N, D) float32 numpy
    text_embs  = encoder.encode_texts(["why did ...", ...])  # (N, D) float32 numpy
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.utils.logger import get_logger

log = get_logger("clip_encoder")


class CLIPEncoder:
    """
    Thread-safe, batched CLIP encoder.

    Attributes:
        model_name: OpenCLIP model name (e.g. "ViT-B-32").
        pretrained: OpenCLIP pretrained weights tag.
        device: torch device.
        batch_size: frames per GPU batch.
        embed_dim: embedding dimensionality.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
        batch_size: int = 256,
    ) -> None:
        try:
            import open_clip
        except ImportError:
            raise ImportError("open_clip not installed. Run: pip install open-clip-torch")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size

        log.info("Loading CLIP model", model=model_name, pretrained=pretrained, device=str(self.device))
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()

        # Infer embed_dim from a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            out = self.model.encode_image(dummy)
            self.embed_dim: int = out.shape[-1]

        log.info("CLIP encoder ready", embed_dim=self.embed_dim)

    @classmethod
    def from_config(cls, cfg: Any) -> "CLIPEncoder":
        r = cfg.retrieval
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(
            model_name=r.model_name,
            pretrained=r.pretrained,
            device=device,
            batch_size=cfg.data.batch_size_embed,
        )

    # ── Frame encoding ────────────────────────────────────────────────────

    def encode_frames(
        self,
        images: list[Union[Image.Image, str, Path]],
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of PIL images (or paths to JPEGs).
        Returns float32 numpy array of shape (N, embed_dim).
        """
        if not images:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        results: list[np.ndarray] = []
        n_batches = math.ceil(len(images) / self.batch_size)

        for i in range(n_batches):
            batch_imgs = images[i * self.batch_size : (i + 1) * self.batch_size]
            tensors: list[torch.Tensor] = []
            for img in batch_imgs:
                if isinstance(img, (str, Path)):
                    img = Image.open(img).convert("RGB")
                tensors.append(self.preprocess(img))

            batch_tensor = torch.stack(tensors).to(self.device)  # (B, 3, H, W)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                embs = self.model.encode_image(batch_tensor)  # (B, D)
                if l2_normalize:
                    embs = F.normalize(embs, dim=-1)

            results.append(embs.float().cpu().numpy())

        return np.concatenate(results, axis=0).astype(np.float32)

    # ── Text encoding ─────────────────────────────────────────────────────

    def encode_texts(
        self,
        texts: list[str],
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of strings.
        Returns float32 numpy array of shape (N, embed_dim).
        """
        if not texts:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        results: list[np.ndarray] = []
        n_batches = math.ceil(len(texts) / self.batch_size)

        for i in range(n_batches):
            batch_texts = texts[i * self.batch_size : (i + 1) * self.batch_size]
            tokens = self.tokenizer(batch_texts).to(self.device)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                embs = self.model.encode_text(tokens)   # (B, D)
                if l2_normalize:
                    embs = F.normalize(embs, dim=-1)

            results.append(embs.float().cpu().numpy())

        return np.concatenate(results, axis=0).astype(np.float32)

    # ── QA-aware query encoding ───────────────────────────────────────────

    def encode_qa_query(
        self,
        question: str,
        options: list[str],
        answer_idx: Optional[int] = None,
        l2_normalize: bool = True,
    ) -> np.ndarray:
        """
        Build a richer text query from the question + (optionally) the correct answer.
        During training we can include the correct answer for stronger supervision.
        During inference we use question only (or question + all options concatenated).
        Returns shape (embed_dim,).
        """
        if answer_idx is not None:
            # Training: use question + correct answer for strongest supervision
            query = f"{question} {options[answer_idx]}"
        else:
            # Inference: question only
            query = question
        embs = self.encode_texts([query], l2_normalize=l2_normalize)
        return embs[0]

    def encode_frame_paths_batched(
        self,
        frame_paths: list[str],
        l2_normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a large list of frame paths with optional tqdm progress.
        Loads images lazily to avoid RAM spikes.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        results: list[np.ndarray] = []
        n_batches = math.ceil(len(frame_paths) / self.batch_size)
        rng = range(n_batches)
        if show_progress and tqdm is not None:
            rng = tqdm(rng, desc="Encoding frames", unit="batch")

        for i in rng:
            batch = frame_paths[i * self.batch_size : (i + 1) * self.batch_size]
            tensors = []
            for p in batch:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self.preprocess(img))
                except Exception as e:
                    log.warning("Failed to load frame", path=p, error=str(e))
                    tensors.append(torch.zeros(3, 224, 224))

            batch_tensor = torch.stack(tensors).to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                embs = self.model.encode_image(batch_tensor)
                if l2_normalize:
                    embs = F.normalize(embs, dim=-1)
            results.append(embs.float().cpu().numpy())

        return np.concatenate(results, axis=0).astype(np.float32) if results else np.zeros((0, self.embed_dim), dtype=np.float32)