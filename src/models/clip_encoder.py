"""
OpenCLIP wrapper for encoding frames and questions.

Why OpenCLIP over HuggingFace CLIP:
  - Faster batched inference with open_clip's native DataLoader support
  - Easier to swap backbones (ViT-B/32 → ViT-L/14) via config
  - Better fp16 support out of the box

All embeddings are L2-normalised before returning so cosine similarity
can be computed as a simple dot product — important for FAISS efficiency.
"""
from __future__ import annotations

import numpy as np
import torch
import open_clip
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CLIPEncoder:
    """
    Encodes images and text into a shared embedding space using OpenCLIP.

    Args:
        model_name:  OpenCLIP model name, e.g. "ViT-B-32".
        pretrained:  Pretrained weights tag, e.g. "openai" or "laion2b_s34b_b79k".
        device:      "cuda" or "cpu".
        batch_size:  Number of images per forward pass. 256 fits on T4 at fp16.

    Example:
        >>> encoder = CLIPEncoder("ViT-B-32", "openai", device="cuda")
        >>> emb = encoder.encode_text(["a cat sitting on a mat"])
        >>> emb.shape
        (1, 512)
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
        batch_size: int = 256,
    ):
        self.device = device
        self.batch_size = batch_size

        logger.info(f"Loading CLIP {model_name} ({pretrained})…")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(device).eval()

        # Use fp16 on CUDA for ~2× speedup with negligible accuracy loss
        if device == "cuda":
            self.model = self.model.half()

        logger.info("CLIP encoder ready")

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of text strings into L2-normalised embeddings.

        Args:
            texts: List of strings (questions, captions, etc.).

        Returns:
            Float32 numpy array of shape (len(texts), embed_dim).
        """
        tokens = self.tokenizer(texts).to(self.device)
        embeddings = self.model.encode_text(tokens)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.float().cpu().numpy()

    @torch.no_grad()
    def encode_frames(self, frame_paths: list[str]) -> np.ndarray:
        """
        Encode a list of frame image paths into L2-normalised embeddings.

        Internally batches the encoding to fit GPU memory.

        Args:
            frame_paths: List of paths to JPEG/PNG frame files.

        Returns:
            Float32 numpy array of shape (len(frame_paths), embed_dim).
        """
        dataset = _FrameImageDataset(frame_paths, self.preprocess)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=2, pin_memory=True)

        all_embeddings = []
        for batch in tqdm(loader, desc="Encoding frames", leave=False):
            batch = batch.to(self.device)
            if self.device == "cuda":
                batch = batch.half()
            emb = self.model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(emb.float().cpu())

        return torch.cat(all_embeddings, dim=0).numpy()


class _FrameImageDataset(Dataset):
    """Internal dataset that loads and preprocesses frame images."""

    def __init__(self, frame_paths: list[str], preprocess):
        self.frame_paths = frame_paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.frame_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.frame_paths[idx]).convert("RGB")
        return self.preprocess(img)
