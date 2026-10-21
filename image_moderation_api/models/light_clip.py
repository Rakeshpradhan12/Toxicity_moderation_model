"""Light CLIP: only the image half of CLIP plus stored prompt embeddings.

The full CLIP model has an image encoder and a text encoder. For the rude-gesture
check the prompts never change, so their text embeddings are computed once
(build_light_clip.py) and the text encoder is not needed at run time. The scores
are the same as the full model's zero-shot classification.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class LightClip:
    """Drop-in replacement for the zero-shot-image-classification pipeline."""

    def __init__(self, folder: str) -> None:
        folder_path = Path(folder)
        meta = json.loads((folder_path / "meta.json").read_text())
        self.labels: list[str] = meta["labels"]
        self.logit_scale: float = float(meta["logit_scale"])
        self.text = torch.from_numpy(np.load(folder_path / "prompt_embeddings.npy")).float()
        self.processor = CLIPImageProcessor.from_pretrained(folder_path)
        # .float() makes the model run in normal 32-bit precision even if the
        # weights were stored in half precision to save disk space
        self.model = CLIPVisionModelWithProjection.from_pretrained(folder_path).float().eval()

    def __call__(self, images, candidate_labels=None):
        if candidate_labels is not None and list(candidate_labels) != self.labels:
            raise ValueError(
                "The CLIP labels changed since the light model was built. "
                "Run build_light_clip.py again."
            )
        single = not isinstance(images, list)
        batch = [images] if single else images
        inputs = self.processor(images=batch, return_tensors="pt")
        with torch.no_grad():
            emb = self.model(pixel_values=inputs["pixel_values"]).image_embeds
            emb = emb / emb.norm(dim=-1, keepdim=True)
            probs = (self.logit_scale * emb @ self.text.T).softmax(dim=-1)
        out = [
            [{"label": label, "score": float(p)} for label, p in zip(self.labels, row)]
            for row in probs
        ]
        return out[0] if single else out