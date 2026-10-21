"""Load NSFW and CLIP models and combine their predictions."""
from __future__ import annotations

import logging
import os
import threading
from typing import Any

from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

# Use local folders in production (for example weights/nsfw_image) so the
# service does not need the internet. See .env.example.
NSFW_MODEL = os.getenv("NSFW_MODEL", "Falconsai/nsfw_image_detection")
CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
CLIP_ENABLED = os.getenv("CLIP_ENABLED", "true").lower() in ("1", "true", "yes")
# Optional smaller CLIP (image half only, see build_light_clip.py). Leave empty to use the full model.
CLIP_LIGHT_DIR = os.getenv("CLIP_LIGHT_DIR", "")

RUDE_LABELS = [
    "a hand making a rude middle finger gesture",
    "a person showing an offensive hand gesture",
    "a person making an obscene gesture",
    "a person giving an insulting hand sign",
]
# Several everyday descriptions, so ordinary photos have somewhere sensible to go
# instead of being pushed towards the rude prompts.
ORDINARY_LABELS = [
    "a photo of a person",
    "a photo of a hand",
    "a photo of an object",
    "a photo of a landscape",
    "a photo of food",
    "a photo of an animal",
]
CLIP_LABELS = RUDE_LABELS + ORDINARY_LABELS

# Starting thresholds. Calibrate on labelled images (calibrate.py, eval_dataset.py).
NSFW_REVIEW_THRESHOLD = float(os.getenv("NSFW_REVIEW_THRESHOLD", "0.50"))
# Above 1 means "never block automatically": flagged images wait for a person.
NSFW_BLOCK_THRESHOLD = float(os.getenv("NSFW_BLOCK_THRESHOLD", "1.01"))
CLIP_REVIEW_THRESHOLD = float(os.getenv("CLIP_REVIEW_THRESHOLD", "0.70"))

MAX_CONCURRENT_INFERENCE = int(os.getenv("MAX_CONCURRENT_INFERENCE", "2"))
QUEUE_TIMEOUT_SECONDS = float(os.getenv("QUEUE_TIMEOUT_SECONDS", "10"))

DECISION_RANK = {"approved": 0, "pending_review": 1, "blocked": 2}


class ModeratorBusy(RuntimeError):
    """Too many images are being checked; the caller should retry shortly."""


def _as_batch(results: Any, n: int) -> list:
    # a pipeline given one image may return a flat list instead of a list of lists
    if n == 1 and results and isinstance(results[0], dict):
        return [results]
    return results


class ImageModerator:
    """Holds model pipelines in memory for reuse across requests."""

    def __init__(self) -> None:
        self.nsfw_classifier = pipeline("image-classification", model=NSFW_MODEL)
        self.clip_classifier = None
        if CLIP_ENABLED:
            if CLIP_LIGHT_DIR:
                from models.light_clip import LightClip

                self.clip_classifier = LightClip(CLIP_LIGHT_DIR)
            else:
                self.clip_classifier = pipeline("zero-shot-image-classification", model=CLIP_MODEL)
        self._slots = threading.BoundedSemaphore(MAX_CONCURRENT_INFERENCE)

    def moderate(self, image: Image.Image) -> dict[str, Any]:
        return self.moderate_frames([image])

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
        best_rude_label = max(rude_scores, key=rude_scores.get)
        best_rude_score = rude_scores[best_rude_label]
        rude_total = sum(rude_scores.values())
        ordinary_total = sum(clip_scores.get(label, 0.0) for label in ORDINARY_LABELS)
        clip_on = self.clip_classifier is not None

        # NSFW is the main signal. CLIP results only ever go to review, never to
        # automatic blocking, because they are prompt-relative similarity scores
        # and may generate false positives.
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