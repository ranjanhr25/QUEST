"""
Full pipeline evaluator for QUEST.

Supports all four ablation methods needed for the results table:

  Method         Frame selection strategy
  ──────────────────────────────────────────────────────────────
  uniform        Random uniform sampling (no CLIP, no ranker)
  clip_topk      CLIP top-K (Stage 1 only, no ranker)
  ranker_topk    CLIP top-K → Ranker top-K (no DPP)
  quest          CLIP top-K → Ranker → DPP → Adaptive budget (full)
  ──────────────────────────────────────────────────────────────

All methods pass the selected frames through the same frozen LLaVA-1.5-7B
for final answer generation, so the only variable is frame selection quality.

Usage (from evaluate.py):
    evaluator = QUESTEvaluator(cfg=cfg, device=device)
    metrics = evaluator.run(
        dataset=dataset,
        method="quest",                  # or "uniform", "clip_topk", "ranker_topk"
        num_frames=8,                    # used for uniform/clip_topk/ranker_topk
        checkpoint_path="path/to/ranker_best.pt",
        output_path="results/quest_val.json",
    )

Output JSON format:
    {
      "method": "quest",
      "split": "val",
      "n_samples": 4996,
      "metrics": {"overall": 0.593, "causal": 0.543, "temporal": 0.579, "descriptive": 0.671},
      "per_sample": [
        {"qid": "123", "video_id": "456", "pred": "A", "gt": "A", "correct": true, ...}
      ]
    }
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.evaluation.metrics import accuracy_by_type
from src.utils.logger import get_logger

log = get_logger("evaluator")

# NExT-QA question type groups for the 3-way breakdown reported in the paper
_CAUSAL_TYPES     = {"CW", "CH"}
_TEMPORAL_TYPES   = {"TN", "TC"}
_DESCRIPTIVE_TYPES = {"DL", "DC", "DO"}


def _qtype_group(qtype: str) -> str:
    """Map fine-grained NExT-QA type code to Causal / Temporal / Descriptive."""
    if qtype in _CAUSAL_TYPES:
        return "causal"
    if qtype in _TEMPORAL_TYPES:
        return "temporal"
    if qtype in _DESCRIPTIVE_TYPES:
        return "descriptive"
    return "other"


class QUESTEvaluator:
    """
    Runs end-to-end evaluation of the QUEST pipeline on NExT-QA.

    Components are initialised lazily on the first call to run() so that
    importing the class is cheap (no GPU memory allocated until needed).

    Args:
        cfg:    DotDict from load_config().
        device: torch.device for ranker inference. LLaVA uses device_map="auto".
    """

    def __init__(self, cfg: Any, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device

        # Lazy-loaded components (None until first run())
        self._coarse_retriever: Optional[Any] = None  # CoarseRetriever
        self._ranker: Optional[Any] = None             # TemporalRanker
        self._vlm: Optional[Any] = None                # LLaVAInterface
        self._adaptive_budget: Optional[Any] = None    # AdaptiveBudget

    # ── Public API ────────────────────────────────────────────────────────

    def run(
        self,
        dataset: Any,                         # NExTQADataset
        method: str = "quest",
        num_frames: int = 8,
        checkpoint_path: Optional[Path] = None,
        max_samples: Optional[int] = None,
        output_path: Optional[Path] = None,
    ) -> dict[str, float]:
        """
        Evaluate `method` on `dataset`.

        Args:
            dataset:         NExTQADataset (train or val split).
            method:          One of "uniform", "clip_topk", "ranker_topk", "quest".
            num_frames:      Frame budget for uniform/clip_topk/ranker_topk.
                             (quest uses adaptive budget from the config.)
            checkpoint_path: Path to ranker .pt checkpoint.
                             Required for "ranker_topk" and "quest".
                             Auto-detected from cfg.paths.checkpoints if None.
            max_samples:     Evaluate only the first N samples (for quick runs).
            output_path:     Write per-sample results + metrics JSON here.

        Returns:
            Dict with keys: overall, causal, temporal, descriptive,
                            and optionally other question types found in the data.
        """
        valid_methods = ["uniform", "clip_topk", "ranker_topk", "quest"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

        log.info("Evaluator.run()", method=method, dataset_size=len(dataset),
                 num_frames=num_frames, device=str(self.device))

        # ── Lazy component initialisation ─────────────────────────────────
        self._init_components(method, checkpoint_path)

        n = max_samples if max_samples is not None else len(dataset)
        n = min(n, len(dataset))

        predictions: list[str] = []
        references: list[str] = []
        q_types: list[str] = []
        per_sample_results: list[dict] = []

        t0 = time.time()

        for i in tqdm(range(n), desc=f"Evaluating [{method}]", dynamic_ncols=True):
            sample = dataset[i]
            try:
                result = self._run_one(sample, method, num_frames)
            except Exception as exc:
                log.error("Sample failed", idx=i, video_id=sample["video_id"], error=str(exc))
                result = {
                    "pred_option": sample["options"][0],  # fallback: always predict A
                    "pred_idx": 0,
                }

            # Ground truth
            gt_idx = sample["answer_idx"]
            gt_option = sample["options"][gt_idx]
            pred_option = result["pred_option"]

            correct = pred_option.strip().lower() == gt_option.strip().lower()
            predictions.append(pred_option)
            references.append(gt_option)
            q_types.append(sample["qtype"])

            per_sample_results.append({
                "idx": i,
                "video_id": sample["video_id"],
                "question": sample["question"],
                "pred": pred_option,
                "gt": gt_option,
                "pred_idx": result.get("pred_idx"),
                "gt_idx": gt_idx,
                "correct": correct,
                "qtype": sample["qtype"],
                "qtype_group": _qtype_group(sample["qtype"]),
            })

            if (i + 1) % 100 == 0:
                running_acc = sum(r["correct"] for r in per_sample_results) / (i + 1)
                elapsed = time.time() - t0
                log.info(
                    f"Progress {i + 1}/{n}",
                    running_acc=round(running_acc, 4),
                    elapsed_min=round(elapsed / 60, 1),
                    eta_min=round((elapsed / (i + 1)) * (n - i - 1) / 60, 1),
                )

        # ── Metrics ────────────────────────────────────────────────────────
        metrics = self._compute_metrics(predictions, references, q_types)

        log.info(
            "Evaluation complete",
            method=method,
            n_samples=n,
            **{k: round(v, 4) for k, v in metrics.items()},
        )

        # ── Save results ───────────────────────────────────────────────────
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output = {
                "method": method,
                "n_samples": n,
                "num_frames": num_frames,
                "metrics": {k: round(v, 6) for k, v in metrics.items()},
                "per_sample": per_sample_results,
            }
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            log.info("Results saved", path=str(output_path))

        return metrics

    # ── Per-sample inference ──────────────────────────────────────────────

    def _run_one(
        self,
        sample: dict,
        method: str,
        num_frames: int,
    ) -> dict:
        """
        Run one sample through the pipeline.

        Returns dict with:
            pred_option (str):  predicted answer text
            pred_idx    (int):  predicted answer index (0-4)
        """
        video_id = sample["video_id"]
        question = sample["question"]
        options  = sample["options"]
        frame_paths = sample["frame_paths"]

        # ── Frame selection ────────────────────────────────────────────────
        if method == "uniform":
            selected_paths = self._select_uniform(frame_paths, num_frames)

        elif method == "clip_topk":
            selected_paths = self._select_clip_topk(video_id, question, frame_paths, num_frames)

        elif method == "ranker_topk":
            selected_paths = self._select_ranker_topk(
                video_id, question, frame_paths, num_frames
            )

        else:  # "quest"
            selected_paths = self._select_quest(video_id, question, frame_paths)

        if not selected_paths:
            log.warning("No frames selected — falling back to first frame", video_id=video_id)
            selected_paths = frame_paths[:1] if frame_paths else []

        # ── VLM inference ──────────────────────────────────────────────────
        vlm_answer = self._vlm_answer(selected_paths, question, options)

        # ── Parse predicted option ─────────────────────────────────────────
        pred_idx, pred_option = self._match_option(vlm_answer, options)

        return {"pred_option": pred_option, "pred_idx": pred_idx}

    # ── Frame selection methods ───────────────────────────────────────────

    def _select_uniform(self, frame_paths: list[str], num_frames: int) -> list[str]:
        """Uniform temporal sampling — no CLIP, no ranker."""
        if not frame_paths:
            return []
        k = min(num_frames, len(frame_paths))
        indices = np.linspace(0, len(frame_paths) - 1, k, dtype=int)
        return [frame_paths[i] for i in indices]

    def _select_clip_topk(
        self,
        video_id: str,
        question: str,
        frame_paths: list[str],
        num_frames: int,
    ) -> list[str]:
        """CLIP + FAISS top-K — Stage 1 only."""
        result = self._coarse_retriever.retrieve(
            video_id=video_id,
            question=question,
            top_k=num_frames,
        )
        if result.is_empty:
            return self._select_uniform(frame_paths, num_frames)

        # Map local frame indices → frame paths
        return self._indices_to_paths(result.local_indices, frame_paths)

    def _select_ranker_topk(
        self,
        video_id: str,
        question: str,
        frame_paths: list[str],
        num_frames: int,
    ) -> list[str]:
        """CLIP top-64 → Ranker → simple top-K (no DPP, no adaptive budget)."""
        # Stage 1: CLIP top-64
        retrieval = self._coarse_retriever.retrieve(
            video_id=video_id,
            question=question,
            top_k=self.cfg.retrieval.top_k_coarse,
        )
        if retrieval.is_empty:
            return self._select_uniform(frame_paths, num_frames)

        # Stage 2: Ranker scores
        with torch.no_grad():
            frame_embs_t = torch.from_numpy(retrieval.frame_embs).unsqueeze(0).to(self.device)
            q_emb_t      = torch.from_numpy(retrieval.query_emb).unsqueeze(0).to(self.device)
            temp_pos_t   = torch.from_numpy(retrieval.temporal_pos).unsqueeze(0).to(self.device)
            outputs = self._ranker(frame_embs_t, q_emb_t, temp_pos_t)
            scores = outputs["relevance"][0].cpu().numpy()   # (K,)

        # Top-K by ranker score (no DPP)
        k = min(num_frames, len(scores))
        top_indices_local = np.argsort(scores)[::-1][:k]  # descending
        selected_cand_indices = retrieval.local_indices[top_indices_local]

        return self._indices_to_paths(selected_cand_indices, frame_paths)

    def _select_quest(
        self,
        video_id: str,
        question: str,
        frame_paths: list[str],
    ) -> list[str]:
        """Full QUEST: CLIP top-64 → Ranker → DPP → Adaptive budget."""
        from src.models.dpp_selector import dpp_select_frames

        # Stage 1
        retrieval = self._coarse_retriever.retrieve(
            video_id=video_id,
            question=question,
            top_k=self.cfg.retrieval.top_k_coarse,
        )
        if retrieval.is_empty:
            return self._select_uniform(frame_paths, self.cfg.ranking.top_k_fine_low)

        # Stage 2: Ranker
        with torch.no_grad():
            frame_embs_t = torch.from_numpy(retrieval.frame_embs).unsqueeze(0).to(self.device)
            q_emb_t      = torch.from_numpy(retrieval.query_emb).unsqueeze(0).to(self.device)
            temp_pos_t   = torch.from_numpy(retrieval.temporal_pos).unsqueeze(0).to(self.device)
            outputs = self._ranker(frame_embs_t, q_emb_t, temp_pos_t)
            scores      = outputs["relevance"][0].cpu().numpy()    # (K,)
            uncertainty = outputs["uncertainty"][0].item()          # scalar

        # Adaptive budget
        n_select = self._adaptive_budget.get_budget(uncertainty)

        # DPP frame selection
        # Timestamps: use local frame indices scaled to seconds (1fps → idx == second)
        timestamps = retrieval.local_indices.astype(np.float32)

        selected_local = dpp_select_frames(
            relevance_scores=scores,
            frame_timestamps=timestamps,
            k=n_select,
            lambda_diversity=self.cfg.ranking.dpp_lambda,
        )  # indices into the 64-candidate list

        selected_cand_indices = retrieval.local_indices[selected_local]
        return self._indices_to_paths(selected_cand_indices, frame_paths)

    # ── VLM call ──────────────────────────────────────────────────────────

    def _vlm_answer(
        self,
        frame_paths: list[str],
        question: str,
        options: list[str],
    ) -> str:
        """
        Load selected frames as PIL images and query the frozen VLM.

        The prompt is formatted as a multiple-choice question with options
        A–E, which is the standard NExT-QA evaluation format.
        """
        from PIL import Image

        images = []
        for p in frame_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
            except Exception as e:
                log.warning("Failed to load frame", path=p, error=str(e))

        if not images:
            return options[0]   # fallback

        # Build multiple-choice prompt
        opts_str = "\n".join(
            f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options) if opt.strip()
        )
        mc_question = (
            f"{question}\n"
            f"Choose the best answer from:\n{opts_str}\n"
            f"Answer with only the letter (A, B, C, D, or E)."
        )

        return self._vlm.answer(images, mc_question, max_new_tokens=16)

    # ── Option matching ───────────────────────────────────────────────────

    @staticmethod
    def _match_option(vlm_answer: str, options: list[str]) -> tuple[int, str]:
        """
        Map the VLM's free-form answer back to one of the 5 options (A–E).

        Matching strategy (in order of priority):
        1. Check if the answer starts with a letter A–E.
        2. Check for exact substring match of option text.
        3. Fall back to option 0 (A).

        Returns:
            (option_index, option_text)
        """
        ans = vlm_answer.strip()

        # Strategy 1: leading letter
        if ans and ans[0].upper() in "ABCDE":
            idx = ord(ans[0].upper()) - ord("A")
            if 0 <= idx < len(options):
                return idx, options[idx]

        # Strategy 2: substring match
        ans_lower = ans.lower()
        for i, opt in enumerate(options):
            if opt.strip().lower() in ans_lower or ans_lower in opt.strip().lower():
                return i, opt

        # Strategy 3: fallback
        return 0, options[0]

    # ── Metrics ───────────────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(
        predictions: list[str],
        references: list[str],
        q_types: list[str],
    ) -> dict[str, float]:
        """Compute overall + per-type accuracy (causal / temporal / descriptive)."""
        # Fine-grained type accuracy
        fine = accuracy_by_type(predictions, references, q_types)

        # Aggregate into the 3 groups the paper reports
        def _group_acc(group_types: set[str]) -> Optional[float]:
            preds = [p for p, qt in zip(predictions, q_types) if qt in group_types]
            refs  = [r for r, qt in zip(references,  q_types) if qt in group_types]
            if not preds:
                return None
            correct = sum(p.strip().lower() == r.strip().lower() for p, r in zip(preds, refs))
            return correct / len(preds)

        metrics: dict[str, float] = {}
        metrics["overall"] = fine.get("overall", 0.0)

        c = _group_acc(_CAUSAL_TYPES)
        if c is not None:
            metrics["causal"] = c

        t = _group_acc(_TEMPORAL_TYPES)
        if t is not None:
            metrics["temporal"] = t

        d = _group_acc(_DESCRIPTIVE_TYPES)
        if d is not None:
            metrics["descriptive"] = d

        # Also include fine-grained types
        for k, v in fine.items():
            if k != "overall":
                metrics[f"type_{k}"] = v

        return metrics

    # ── Lazy initialisation ───────────────────────────────────────────────

    def _init_components(
        self,
        method: str,
        checkpoint_path: Optional[Path],
    ) -> None:
        """Initialise pipeline components lazily (only the ones needed for `method`)."""

        # CoarseRetriever is needed for everything except uniform
        if method != "uniform" and self._coarse_retriever is None:
            log.info("Initialising CoarseRetriever...")
            from src.retrieval.coarse_retriever import CoarseRetriever
            self._coarse_retriever = CoarseRetriever.from_config(self.cfg, device=str(self.device))

        # TemporalRanker needed for ranker_topk and quest
        if method in ("ranker_topk", "quest") and self._ranker is None:
            log.info("Loading TemporalRanker...")
            from src.models.temporal_ranker import TemporalRanker
            from src.utils.io_utils import find_latest_checkpoint

            ckpt = checkpoint_path
            if ckpt is None:
                ckpt = find_latest_checkpoint(self.cfg.paths.checkpoints)
                if ckpt is not None:
                    log.info("Auto-detected checkpoint", path=str(ckpt))
            if ckpt is None or not Path(ckpt).exists():
                raise FileNotFoundError(
                    f"Ranker checkpoint not found. "
                    f"Run train_ranker.py first, or pass --checkpoint. "
                    f"Looked in: {self.cfg.paths.checkpoints}"
                )

            model = TemporalRanker.from_config(self.cfg).to(self.device).eval()
            from src.utils.io_utils import load_checkpoint
            load_checkpoint(ckpt, model, device=self.device)
            self._ranker = model

        # AdaptiveBudget needed for quest
        if method == "quest" and self._adaptive_budget is None:
            from src.ranking.adaptive_budget import AdaptiveBudget
            self._adaptive_budget = AdaptiveBudget.from_config(self.cfg)
            log.info(
                "AdaptiveBudget ready",
                threshold=self._adaptive_budget.threshold,
                low=self._adaptive_budget.top_k_low,
                high=self._adaptive_budget.top_k_high,
            )

        # VLM always needed (for final answer)
        if self._vlm is None:
            log.info("Loading LLaVA VLM (this takes ~30s the first time)...")
            from src.models.vlm_interface import LLaVAInterface
            self._vlm = LLaVAInterface(
                model_id=self.cfg.vlm.model_id,
                load_in_4bit=self.cfg.vlm.load_in_4bit,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

    # ── Path helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _indices_to_paths(
        indices: np.ndarray,
        frame_paths: list[str],
    ) -> list[str]:
        """
        Map integer frame indices (relative to a video's frame list) to actual paths.

        Frame paths are sorted, so index i maps to frame_paths[i] IF frame_paths
        is sorted the same way the index was built. We sort to be safe.
        """
        sorted_paths = sorted(frame_paths)
        result = []
        for idx in sorted(indices):  # sort to maintain temporal order
            if 0 <= idx < len(sorted_paths):
                result.append(sorted_paths[idx])
            else:
                log.warning("Frame index out of range", idx=int(idx), n_paths=len(sorted_paths))
        return result