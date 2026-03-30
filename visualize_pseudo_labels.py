#!/usr/bin/env python3
"""
visualize_pseudo_labels.py

Generate mask-overlay visualizations for a subset of already-saved pseudo-label
images. Reads image/mask pairs from a ConeQuest-style directory and writes
overlay PNGs to a visualize/ folder.

Usage
-----
# Visualize 200 random accepted samples
python visualize_pseudo_labels.py \
    --data_dir  /path/to/pseudo_labels \
    --split     train \
    --n         200 \
    --seed      42

# Visualize ALL samples in the split
python visualize_pseudo_labels.py \
    --data_dir  /path/to/pseudo_labels \
    --split     train \
    --n         -1

# Visualize only positives (at least 0.5% foreground pixels)
python visualize_pseudo_labels.py \
    --data_dir      /path/to/pseudo_labels \
    --split         train \
    --n             200 \
    --min_fg_ratio  0.005 \
    --fg_only

# Read per-image stats from stats.json and annotate confidence on the image
python visualize_pseudo_labels.py \
    --data_dir  /path/to/pseudo_labels \
    --split     train \
    --n         200 \
    --stats_json /path/to/pseudo_labels/stats.json
"""

import argparse
import json
import logging
import random
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_overlay(img_bgr: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.25) -> np.ndarray:
    """Brightness-only overlay: foreground brightened, background dimmed."""
    img_f  = img_bgr.astype(np.float32)
    fg     = mask > 0
    result = img_f.copy()
    result[fg]  = np.clip(img_f[fg]  + alpha * 255, 0, 255)
    result[~fg] = np.clip(img_f[~fg] - alpha * 255, 0, 255)
    return result.astype(np.uint8)


def annotate(vis: np.ndarray, text: str) -> np.ndarray:
    """Draw text with a dark shadow for readability."""
    cv2.putText(vis, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 0),     1, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fg_ratio(mask: np.ndarray, fg_class_id: int = 1) -> float:
    return float((mask == fg_class_id).sum()) / mask.size


def load_stats(stats_json: str) -> dict:
    """Return {stem: {confidence, fg_ratio, ...}} from stats.json."""
    with open(stats_json) as f:
        data = json.load(f)
    samples = data.get("samples", data)   # support both {summary, samples} and flat list
    return {Path(s["file"]).stem: s for s in samples}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize pseudo-label mask overlays for manual inspection."
    )
    parser.add_argument("--data_dir",   required=True,
                        help="Root pseudo-labels directory "
                             "(contains data/{split}/images and masks/).")
    parser.add_argument("--split",      default="train",
                        help="Split subfolder (default: train).")
    parser.add_argument("--out_dir",    default=None,
                        help="Output directory for visualizations. "
                             "Defaults to <data_dir>/visualize/.")
    parser.add_argument("--n",          type=int, default=200,
                        help="Number of images to visualize. -1 = all.")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--alpha",      type=float, default=0.25,
                        help="Brightness shift for overlay (default: 0.25).")

    # Filtering
    parser.add_argument("--fg_only",    action="store_true",
                        help="Only visualize images that contain foreground pixels.")
    parser.add_argument("--min_fg_ratio", type=float, default=0.0,
                        help="Min foreground pixel ratio to include (default: 0).")
    parser.add_argument("--fg_class_id", type=int, default=1,
                        help="Foreground class ID (default: 1).")

    # Stats annotation
    parser.add_argument("--stats_json", default=None,
                        help="Path to stats.json from generate_pseudo_labels.py. "
                             "If provided, confidence and fg_ratio are shown on each image.")

    args = parser.parse_args()
    random.seed(args.seed)

    data_root = Path(args.data_dir)
    img_dir   = data_root / "data" / args.split / "images"
    mask_dir  = data_root / "data" / args.split / "masks"
    out_dir   = Path(args.out_dir) if args.out_dir else data_root / "visualize"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    # Collect matched image/mask pairs
    img_paths  = sorted([p for p in img_dir.iterdir()
                         if p.suffix.lower() in IMAGE_EXTENSIONS])
    mask_paths = {p.stem: p for p in mask_dir.iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS}

    pairs = [(p, mask_paths[p.stem]) for p in img_paths if p.stem in mask_paths]
    if not pairs:
        raise FileNotFoundError("No matching image/mask pairs found.")
    logger.info(f"Found {len(pairs)} image-mask pairs in {img_dir}")

    # Load stats if provided
    stats = load_stats(args.stats_json) if args.stats_json else {}

    # Filter by fg_ratio
    min_fg = args.min_fg_ratio if not args.fg_only else max(args.min_fg_ratio, 1e-6)
    if min_fg > 0:
        filtered = []
        for img_p, mask_p in pairs:
            mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if fg_ratio(mask, args.fg_class_id) >= min_fg:
                filtered.append((img_p, mask_p))
        logger.info(f"After fg_ratio >= {min_fg}: {len(filtered)} pairs remain")
        pairs = filtered

    # Subsample
    if args.n > 0 and len(pairs) > args.n:
        pairs = random.sample(pairs, args.n)
        logger.info(f"Randomly sampled {args.n} pairs")

    # Generate visualizations
    n_done = 0
    for img_p, mask_p in pairs:
        img  = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            logger.warning(f"Could not read {img_p.name}, skipping.")
            continue

        # Resize mask to image size if needed
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        vis = make_overlay(img, mask, alpha=args.alpha)

        # Build annotation text
        stem = img_p.stem
        r    = fg_ratio(mask, args.fg_class_id)
        if stem in stats:
            s   = stats[stem]
            txt = (f"conf={s.get('confidence', 0):.2f}  "
                   f"fg={r*100:.1f}%  "
                   f"{'POS' if r >= args.min_fg_ratio else 'NEG'}")
        else:
            txt = f"fg={r*100:.1f}%  {'POS' if r > 0 else 'NEG'}"

        annotate(vis, txt)
        cv2.imwrite(str(out_dir / f"{stem}.png"), vis)
        n_done += 1

    logger.info(f"Saved {n_done} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
