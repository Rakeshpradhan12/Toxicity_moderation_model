"""Load NSFW and CLIP models and combine their predictions."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import pipeline

from core.s3_sync import ensure_model_files
from image_moderation.light_clip import LightClip

logger = logging.getLogger(__name__)

# Model configurations
NSFW_MODEL = os.getenv("NSFW_MODEL", "weights/nsfw_image")
NSFW_S3_PREFIX = os.getenv("NSFW_S3_PREFIX", "ml-models/image-moderation/v1/nsfw_image/")

CLIP_MODEL = os.getenv("CLIP_MODEL", "weights/clip")
CLIP_S3_PREFIX = os.getenv("CLIP_S3_PREFIX", "ml-models/image-moderation/v1/clip/")

CLIP_LIGHT_DIR = os.getenv("CLIP_LIGHT_DIR", "weights/clip_light")
CLIP_LIGHT_S3_PREFIX = os.getenv("CLIP_LIGHT_S3_PREFIX", "ml-models/image-moderation-clip-light/v1/")

CLIP_ENABLED = os.getenv("CLIP_ENABLED", "true").lower() in ("1", "true", "yes")

RUDE_LABELS = [
    "a hand making a rude middle finger gesture",
    "a person showing an offensive hand gesture",
    "a person making an obscene gesture",
    "a person giving an insulting hand sign",
]

ORDINARY_LABELS = [
    "a photo of a person",
    "a photo of a hand",
    "a photo of an object",
    "a photo of a landscape",
    "a photo of food",
    "a photo of an animal",
]

CLIP_LABELS = RUDE_LABELS + ORDINARY_LABELS

NSFW_REVIEW_THRESHOLD = float(os.getenv("NSFW_REVIEW_THRESHOLD", "0.50"))
NSFW_BLOCK_THRESHOLD = float(os.getenv("NSFW_BLOCK_THRESHOLD", "1.01"))
CLIP_REVIEW_THRESHOLD = float(os.getenv("CLIP_REVIEW_THRESHOLD", "0.70"))

MAX_CONCURRENT_INFERENCE = int(os.getenv("MAX_CONCURRENT_INFERENCE", "2"))
QUEUE_TIMEOUT_SECONDS = float(os.getenv("QUEUE_TIMEOUT_SECONDS", "10"))

DECISION_RANK = {"approved": 0, "pending_review": 1, "blocked": 2}


class ModeratorBusy(RuntimeError):
    """Too many images are being checked; the caller should retry shortly."""


def _as_batch(results: Any, n: int) -> list:
    if n == 1 and results and isinstance(results[0], dict):
        return [results]
    return results


class ImageModerator:
    """Holds NSFW classifier and CLIP gesture models in memory."""

    def __init__(self) -> None:
        self._ensure_models_available()

        # Load NSFW model
        nsfw_target = NSFW_MODEL
        if not Path(nsfw_target).exists():
            logger.warning("Local NSFW model '%s' not found; falling back to Falconsai/nsfw_image_detection", nsfw_target)
            nsfw_target = "Falconsai/nsfw_image_detection"

        logger.info("Loading NSFW classifier from %s", nsfw_target)
        self.nsfw_classifier = pipeline("image-classification", model=nsfw_target)

        # Load CLIP model
        self.clip_classifier = None
        if CLIP_ENABLED:
            if CLIP_LIGHT_DIR and Path(CLIP_LIGHT_DIR).is_dir() and (Path(CLIP_LIGHT_DIR) / "meta.json").exists():
                logger.info("Loading LightClip from %s", CLIP_LIGHT_DIR)
                self.clip_classifier = LightClip(CLIP_LIGHT_DIR)
            else:
                clip_target = CLIP_MODEL
                if not Path(clip_target).exists():
                    logger.warning("Local CLIP model '%s' not found; falling back to openai/clip-vit-base-patch32", clip_target)
                    clip_target = "openai/clip-vit-base-patch32"
                logger.info("Loading full CLIP zero-shot classifier from %s", clip_target)
                self.clip_classifier = pipeline("zero-shot-image-classification", model=clip_target)

        self._slots = threading.BoundedSemaphore(MAX_CONCURRENT_INFERENCE)

    def _ensure_models_available(self) -> None:
        # Check and auto-download NSFW model
        if NSFW_MODEL.startswith("weights/") or NSFW_MODEL.startswith("models/"):
            ensure_model_files(
                local_dir=NSFW_MODEL,
                s3_prefix=NSFW_S3_PREFIX,
                required_files=("config.json",),
            )

        # Check and auto-download CLIP Light model if configured
        if CLIP_ENABLED and CLIP_LIGHT_DIR and (CLIP_LIGHT_DIR.startswith("weights/") or CLIP_LIGHT_DIR.startswith("models/")):
            ensure_model_files(
                local_dir=CLIP_LIGHT_DIR,
                s3_prefix=CLIP_LIGHT_S3_PREFIX,
                required_files=("meta.json", "prompt_embeddings.npy"),
            )

        # Check and auto-download full CLIP model if configured and CLIP_LIGHT is not present
        if CLIP_ENABLED and (not CLIP_LIGHT_DIR or not Path(CLIP_LIGHT_DIR).exists()):
            if CLIP_MODEL.startswith("weights/") or CLIP_MODEL.startswith("models/"):
                ensure_model_files(
                    local_dir=CLIP_MODEL,
                    s3_prefix=CLIP_S3_PREFIX,
                    required_files=("config.json",),
                )

    def moderate_frames(self, frames: list[Image.Image]) -> dict[str, Any]:
        """Checks every frame and returns the strictest result."""
        if not self._slots.acquire(timeout=QUEUE_TIMEOUT_SECONDS):
            raise ModeratorBusy("moderation queue is full")
        try:
            rgb = [f.convert("RGB") for f in frames]
            nsfw_batch = _as_batch(self.nsfw_classifier(rgb), len(rgb))
            if self.clip_classifier is not None:
                clip_batch = _as_batch(
                    self.clip_classifier(rgb, candidate_labels=CLIP_LABELS), len(rgb)
                )
            else:
                clip_batch = [[] for _ in rgb]
        finally:
            self._slots.release()

        results = [self._decide(n, c) for n, c in zip(nsfw_batch, clip_batch)]
        worst = max(
            results,
            key=lambda r: (
                DECISION_RANK[r["decision"]],
                r["scores"]["nsfw"],
                r["scores"]["clip_rude_gesture_total"],
            ),
        )
        worst = dict(worst)
        worst["frames_checked"] = len(results)
        return worst

    def _decide(self, nsfw_results: list, clip_results: list) -> dict[str, Any]:
        nsfw_scores = {str(i["label"]).lower(): float(i["score"]) for i in nsfw_results}
        nsfw_score = nsfw_scores.get("nsfw", 0.0)
        normal_score = nsfw_scores.get("normal", 0.0)

        clip_scores = {str(i["label"]): float(i["score"]) for i in clip_results}
        rude_scores = {label: clip_scores.get(label, 0.0) for label in RUDE_LABELS}
        best_rude_label = max(rude_scores, key=rude_scores.get) if rude_scores else ""
        best_rude_score = rude_scores[best_rude_label] if best_rude_label else 0.0
        rude_total = sum(rude_scores.values())
        ordinary_total = sum(clip_scores.get(label, 0.0) for label in ORDINARY_LABELS)
        clip_on = self.clip_classifier is not None

        if nsfw_score >= NSFW_BLOCK_THRESHOLD:
            decision, reason = "blocked", "high_nsfw_score"
        elif nsfw_score >= NSFW_REVIEW_THRESHOLD:
            decision, reason = "pending_review", "possible_nsfw"
        elif clip_on and rude_total >= CLIP_REVIEW_THRESHOLD:
            decision, reason = "pending_review", "possible_offensive_gesture"
        else:
            decision, reason = "approved", "no_high_risk_signal"

        return {
            "decision": decision,
            "reason": reason,
            "scores": {
                "nsfw": round(nsfw_score, 4),
                "normal": round(normal_score, 4),
                "clip_rude_gesture_total": round(rude_total, 4),
                "clip_rude_gesture_max": round(best_rude_score, 4),
                "clip_rude_gesture_best_label": best_rude_label if clip_on else "",
                "clip_prompts": {l: round(s, 4) for l, s in rude_scores.items()},
                "clip_ordinary_photo": round(ordinary_total, 4),
            },
        }
