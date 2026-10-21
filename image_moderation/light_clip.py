"""Light CLIP: only the image half of CLIP plus stored prompt embeddings."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class LightClip:
    """Drop-in replacement for the zero-shot-image-classification pipeline."""

    def __init__(self, folder: str | Path) -> None:
        folder_path = Path(folder)
        meta = json.loads((folder_path / "meta.json").read_text())
        self.labels: list[str] = meta["labels"]
        self.logit_scale: float = float(meta["logit_scale"])
        self.text = torch.from_numpy(np.load(folder_path / "prompt_embeddings.npy")).float()
        self.processor = CLIPImageProcessor.from_pretrained(str(folder_path))
        self.model = CLIPVisionModelWithProjection.from_pretrained(str(folder_path)).float().eval()

    def __call__(self, images, candidate_labels=None):
        if candidate_labels is not None and list(candidate_labels) != self.labels:
            raise ValueError("The candidate labels differ from the precomputed LightClip embeddings.")
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
