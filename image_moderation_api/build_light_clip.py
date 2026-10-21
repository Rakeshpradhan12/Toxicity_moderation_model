"""
Builds the light CLIP model from the full one and checks that the scores match.

Run once from the project folder (the one containing models/ and weights/),
with your environment active:

    python build_light_clip.py --check /path/to/some/images
    python build_light_clip.py --fp16 --check /path/to/some/images

--fp16 stores the weights in half precision (half the disk size). They are
loaded back as normal 32-bit numbers, so CPU speed is unchanged.
Then set CLIP_LIGHT_DIR=weights/clip_light in .env.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModelWithProjection, pipeline

sys.path.insert(0, os.getcwd())
from models.light_clip import LightClip          # noqa: E402
from models.moderation import CLIP_LABELS        # noqa: E402

# The transformers zero-shot pipeline wraps every label in this sentence by default.
# It must match, or the scores will differ from what you use today.
DEFAULT_TEMPLATE = "This is a photo of {}."


def folder_size_mb(folder: Path) -> float:
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="weights/clip")
    ap.add_argument("--out", default="weights/clip_light")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--check", default=None, help="folder of images to compare against the full model")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loading the full CLIP model ...")
    full = CLIPModel.from_pretrained(src).float().eval()
    processor = CLIPProcessor.from_pretrained(src)

    # 1. text embeddings of the prompts, computed once
    sentences = [args.template.format(label) for label in CLIP_LABELS]
    tokens = processor.tokenizer(sentences, padding=True, return_tensors="pt")
    with torch.no_grad():
        pooled = full.text_model(
            input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
        ).pooler_output
        text_emb = full.text_projection(pooled)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    np.save(out / "prompt_embeddings.npy", text_emb.numpy().astype("float32"))
    (out / "meta.json").write_text(json.dumps({
        "labels": CLIP_LABELS,
        "template": args.template,
        "logit_scale": float(full.logit_scale.exp().item()),
    }, indent=2))

    # 2. image half only
    vision = CLIPVisionModelWithProjection.from_pretrained(src).float()
    if args.fp16:
        vision = vision.half()
    vision.save_pretrained(out)
    processor.image_processor.save_pretrained(out)
    del vision

    print(f"\nfull model : {folder_size_mb(src):7.0f} MB   ({src})")
    print(f"light model: {folder_size_mb(out):7.0f} MB   ({out}){'  [half precision]' if args.fp16 else ''}")

    # 3. check that the scores match the full model
    if not args.check:
        print("\nNo --check folder given, so the scores were NOT compared. Run again with --check.")
        return
    files = [p for p in sorted(Path(args.check).glob("*"))
             if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}][:30]
    if not files:
        sys.exit(f"No images found in {args.check}")
    images = [Image.open(p).convert("RGB") for p in files]

    reference = pipeline("zero-shot-image-classification", model=str(src))
    light = LightClip(str(out))
    ref_out = reference(images, candidate_labels=CLIP_LABELS)
    new_out = light(images, candidate_labels=CLIP_LABELS)

    worst = 0.0
    for r, n in zip(ref_out, new_out):
        a = {d["label"]: d["score"] for d in r}
        b = {d["label"]: d["score"] for d in n}
        worst = max(worst, max(abs(a[k] - b[k]) for k in a))
    limit = 0.02 if args.fp16 else 0.005
    print(f"\nchecked {len(images)} images: largest score difference {worst:.5f} (limit {limit})")
    if worst <= limit:
        print("PASS: the light model gives the same scores as the full model.")
    else:
        print("FAIL: scores differ. The most likely cause is the sentence template; "
              "try --template '{}' or check your transformers version, then run again.")


if __name__ == "__main__":
    main()