"""
Checks the NSFW model on a dataset organised in class folders
(drawings, hentai, neutral, porn, sexy).

Usage (from the image-moderation folder, with the imgmod environment active):
    python eval_dataset.py nsfw_testing/dataset/nsfw_dataset_v1
    python eval_dataset.py nsfw_testing/dataset/nsfw_dataset_v1 --limit 500

Options:
    --limit N      images sampled per class (default 300; use 0 for all, which is slow)
    --model PATH   model folder or name (default models/nsfw_image)
    --review 0.6   review threshold used in the summary
    --block 0.9    block threshold used in the summary
    --seed 1       random seed for the sample

Keep the dataset out of git and off shared drives.
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps
from transformers import pipeline

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
THRESHOLDS = [0.3, 0.5, 0.6, 0.8, 0.9, 0.95, 0.98]
EXPECT_ALLOWED = ["neutral", "drawings"]   # should NOT be flagged
EXPECT_FLAGGED = ["porn", "hentai"]        # should be flagged
BORDERLINE = ["sexy"]                      # your policy decides


def list_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def score_files(clf, files, batch_size=16):
    """Returns ({path: nsfw_score}, number_skipped)."""
    scores, skipped = {}, 0
    imgs, names = [], []

    def flush():
        if not imgs:
            return
        for name, res in zip(names, clf(imgs, batch_size=batch_size)):
            scores[name] = next((r["score"] for r in res if str(r["label"]).lower() == "nsfw"), 0.0)
        imgs.clear()
        names.clear()

    for p in files:
        try:
            with Image.open(p) as im:
                img = ImageOps.exif_transpose(im).convert("RGB")
            img.thumbnail((1024, 1024))        # saves memory; the model resizes to 224 anyway
            imgs.append(img)
            names.append(p)
        except Exception:
            skipped += 1
        if len(imgs) >= batch_size:
            flush()
    flush()
    return scores, skipped


def pct(values, th):
    return 100.0 * sum(v >= th for v in values) / len(values) if values else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--model", default="models/nsfw_image")
    ap.add_argument("--review", type=float, default=0.6)
    ap.add_argument("--block", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.root)
    classes = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    if not classes:
        sys.exit(f"No class folders found in {root}")

    clf = pipeline("image-classification", model=args.model)
    rng = random.Random(args.seed)
    by_class, all_rows = {}, []

    for folder in classes:
        files = list_images(folder)
        if args.limit and len(files) > args.limit:
            files = rng.sample(files, args.limit)
        start = time.perf_counter()
        scores, skipped = score_files(clf, files)
        secs = time.perf_counter() - start
        by_class[folder.name] = scores
        print(f"{folder.name}: scored {len(scores)} images, skipped {skipped}, {secs:.0f}s")
        for p, s in scores.items():
            all_rows.append({"class": folder.name, "file": str(p), "nsfw": round(s, 4)})

    with open("dataset_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "file", "nsfw"])
        w.writeheader()
        w.writerows(all_rows)

    print("\nShare of each class scoring at or above each threshold (%):")
    print(f"{'class':>10} {'n':>5} " + " ".join(f"{t:>6.2f}" for t in THRESHOLDS))
    for name, scores in by_class.items():
        vals = list(scores.values())
        print(f"{name:>10} {len(vals):>5} " + " ".join(f"{pct(vals, t):6.1f}" for t in THRESHOLDS))

    def group(names):
        return [s for n in names for s in by_class.get(n, {}).values()]

    allowed, flagged, border = group(EXPECT_ALLOWED), group(EXPECT_FLAGGED), group(BORDERLINE)
    print(f"\nSummary at review {args.review} and block {args.block}")
    print(f"  harmless classes ({', '.join(EXPECT_ALLOWED)}), n={len(allowed)}:")
    print(f"    sent to review or blocked: {pct(allowed, args.review):.1f}%   auto-blocked: {pct(allowed, args.block):.1f}%")
    print(f"  explicit classes ({', '.join(EXPECT_FLAGGED)}), n={len(flagged)}:")
    print(f"    sent to review or blocked: {pct(flagged, args.review):.1f}%   auto-blocked: {pct(flagged, args.block):.1f}%")
    print(f"    missed (below review threshold): {100 - pct(flagged, args.review):.1f}%")
    print(f"  borderline class ({', '.join(BORDERLINE)}), n={len(border)}:")
    print(f"    sent to review or blocked: {pct(border, args.review):.1f}%   auto-blocked: {pct(border, args.block):.1f}%")

    def worst(names, highest):
        rows = [r for r in all_rows if r["class"] in names]
        return sorted(rows, key=lambda r: r["nsfw"], reverse=highest)[:8]

    print("\nHarmless-class images with the HIGHEST scores (possible false alarms or wrong labels):")
    for r in worst(EXPECT_ALLOWED, True):
        print(f"  {r['nsfw']:.3f}  {r['class']}  {r['file']}")
    print("\nExplicit-class images with the LOWEST scores (possible misses or wrong labels):")
    for r in worst(EXPECT_FLAGGED, False):
        print(f"  {r['nsfw']:.3f}  {r['class']}  {r['file']}")
    print("\nSaved dataset_scores.csv")


if __name__ == "__main__":
    main()