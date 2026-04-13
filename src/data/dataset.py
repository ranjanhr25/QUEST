"""
PyTorch Dataset classes for NExT-QA and MSVD-QA.

Each dataset returns a dict with:
  - video_id:  string identifier
  - question:  natural language question string
  - answer:    ground truth answer string (or option index for MC)
  - frame_dir: path to the directory of extracted JPEGs for this video

We do NOT load frames here. The retrieval pipeline handles that lazily
so we don't run out of RAM when iterating over the full dataset.
"""
from __future__ import annotations

import csv
from pathlib import Path
from torch.utils.data import Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NextQADataset(Dataset):
    """
    Dataset for NExT-QA benchmark.

    NExT-QA is a multiple-choice video QA dataset focused on causal and
    temporal questions. Each sample has 5 answer options.

    Annotation CSV columns (from official release):
        video, frame_count, width, height, possible_ans, type, a0-a4, qid, question, answer

    Args:
        annotation_file: Path to nextqa_val.csv or nextqa_train.csv.
        frame_root:      Root directory containing per-video frame subdirectories.
        split:           "train" or "val".
    """

    def __init__(self, annotation_file: str, frame_root: str, split: str = "val"):
        self.frame_root = Path(frame_root)
        self.split = split
        self.samples = self._load_annotations(annotation_file)
        logger.info(f"NExT-QA {split}: {len(self.samples)} samples loaded")

    def _load_annotations(self, annotation_file: str) -> list[dict]:
        samples = []
        with open(annotation_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Build multiple-choice options list
                options = [row[f"a{i}"] for i in range(5)]
                answer_idx = int(row["answer"])
                samples.append({
                    "video_id": row["video"],
                    "question": row["question"],
                    "options": options,
                    "answer": options[answer_idx],
                    "answer_idx": answer_idx,
                    "q_type": row["type"],   # "C" causal, "T" temporal, "D" descriptive
                    "frame_dir": str(self.frame_root / row["video"]),
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


class MSVDQADataset(Dataset):
    """
    Dataset for MSVD-QA — shorter clips, good for fast iteration and debugging.

    Use this first to verify your pipeline works before running on NExT-QA.

    Args:
        annotation_file: Path to msvd_qa_train.json or msvd_qa_test.json.
        frame_root:      Root directory containing per-video frame subdirectories.
    """

    def __init__(self, annotation_file: str, frame_root: str):
        import json
        self.frame_root = Path(frame_root)
        with open(annotation_file) as f:
            raw = json.load(f)
        self.samples = [
            {
                "video_id": item["video_id"],
                "question": item["question"],
                "answer": item["answer"],
                "frame_dir": str(self.frame_root / str(item["video_id"])),
            }
            for item in raw
        ]
        logger.info(f"MSVD-QA: {len(self.samples)} samples loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
