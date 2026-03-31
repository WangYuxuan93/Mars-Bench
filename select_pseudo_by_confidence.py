#!/usr/bin/env python3
"""
select_pseudo_by_confidence.py

从已生成的伪标签数据集（data/train/images|masks）中，
按置信度从高到低选取指定数量的样本，构建新的 ConeQuest 格式数据集。

Usage
-----
python select_pseudo_by_confidence.py \\
    --src_dir  /path/to/pseudo_labels \\
    --out_dir  /path/to/selected \\
    --n        5000 \\
    [--neg_pos_ratio 1.0] \\
    [--min_fg_ratio  0.005] \\
    [--pos_only] \\
    [--split    train] \\
    [--visualize] [--vis_n 200]
"""

import argparse
import json
import logging
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_stats(src_dir: Path) -> list:
    shard_files = sorted(src_dir.glob("stats_shard_*.json"))
    if shard_files:
        records = []
        for f in shard_files:
            with open(f) as fp:
                data = json.load(fp)
            records.extend(data.get("samples", []))
        logger.info(f"Loaded {len(records)} records from {len(shard_files)} shard files")
        return records
    stats_file = src_dir / "stats.json"
    if stats_file.exists():
        with open(stats_file) as fp:
            data = json.load(fp)
        records = data.get("samples", data)
        logger.info(f"Loaded {len(records)} records from stats.json")
        return records
    raise FileNotFoundError(f"No stats_shard_*.json or stats.json found in {src_dir}")


def fg_ratio_histogram(records: list, bin_size: float = 0.05) -> dict:
    positives = [r for r in records if r.get("fg_ratio", 0) > 0]
    n_bins = int(1.0 / bin_size)
    bins = {}
    for i in range(n_bins):
        lo, hi = i * bin_size, (i + 1) * bin_size
        bins[f"{lo*100:.0f}%-{hi*100:.0f}%"] = sum(
            1 for r in positives if lo <= r.get("fg_ratio", 0) < hi)
    bins[f"{n_bins*bin_size*100:.0f}%-100%"] = sum(
        1 for r in positives if r.get("fg_ratio", 0) >= n_bins * bin_size)
    return bins


def make_overlay(img_bgr, mask, alpha=0.25):
    img_f  = img_bgr.astype(np.float32)
    result = img_f.copy()
    fg = mask > 0
    result[fg]  = np.clip(img_f[fg]  + alpha * 255, 0, 255)
    result[~fg] = np.clip(img_f[~fg] - alpha * 255, 0, 255)
    return result.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir",  required=True,
                        help="generate_pseudo_labels.py 的输出根目录")
    parser.add_argument("--out_dir",  required=True)
    parser.add_argument("--n",        type=int, required=True,
                        help="选取的总样本数量")
    parser.add_argument("--split",    default="train")
    parser.add_argument("--min_fg_ratio", type=float, default=0.005)
    parser.add_argument("--pos_only",     action="store_true")
    parser.add_argument("--neg_pos_ratio", type=float, default=1.0,
                        help="负/正比例上限（默认1:1）；-1=保留所有负例；0=不要负例")
    parser.add_argument("--visualize",    action="store_true")
    parser.add_argument("--vis_n",    type=int, default=200)
    args = parser.parse_args()

    src_dir  = Path(args.src_dir)
    out_dir  = Path(args.out_dir)
    src_img_dir  = src_dir / "data" / args.split / "images"
    src_mask_dir = src_dir / "data" / args.split / "masks"

    if not src_img_dir.exists():
        raise FileNotFoundError(f"Not found: {src_img_dir}")

    out_img_dir  = out_dir / "data" / args.split / "images"
    out_mask_dir = out_dir / "data" / args.split / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    if args.visualize:
        vis_dir = out_dir / "visualize"
        vis_dir.mkdir(parents=True, exist_ok=True)

    # 加载统计，只保留 data/train/ 里实际存在的
    records = load_stats(src_dir)
    existing = {p.stem for p in src_mask_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS}
    records = [r for r in records if Path(r["file"]).stem in existing]
    logger.info(f"Records matched in data/{args.split}/: {len(records)}")

    # 分正负例，各自按置信度降序
    positives = sorted(
        [r for r in records if r.get("fg_ratio", 0) >= args.min_fg_ratio],
        key=lambda r: r.get("confidence", 0), reverse=True)
    negatives = sorted(
        [r for r in records if r.get("fg_ratio", 0) < args.min_fg_ratio],
        key=lambda r: r.get("confidence", 0), reverse=True)

    logger.info(f"Positives: {len(positives)}, Negatives: {len(negatives)}")

    if args.pos_only or args.neg_pos_ratio == 0:
        selected_pos = positives[:min(args.n, len(positives))]
        selected_neg = []
    elif args.neg_pos_ratio < 0:
        selected_neg = negatives
        selected_pos = positives[:max(0, min(args.n - len(negatives), len(positives)))]
    else:
        n_pos = min(math.ceil(args.n / (1 + args.neg_pos_ratio)), len(positives))
        n_neg = min(args.n - n_pos, int(n_pos * args.neg_pos_ratio), len(negatives))
        selected_pos = positives[:n_pos]
        selected_neg = negatives[:n_neg]

    selected = selected_pos + selected_neg
    logger.info(f"Selected: {len(selected)} (pos={len(selected_pos)}, neg={len(selected_neg)})")

    # fg_ratio 分布
    hist = fg_ratio_histogram(selected_pos)
    logger.info("Positive fg_ratio distribution (5% bins):")
    max_count = max(hist.values()) if hist else 1
    for label, count in hist.items():
        bar = "█" * (count * 30 // max(max_count, 1))
        logger.info(f"  {label:>12}  {count:5d}  {bar}")

    # 复制
    vis_count = 0
    vis_limit = args.vis_n if args.vis_n >= 0 else len(selected)

    for r in selected:
        stem = Path(r["file"]).stem
        src_img  = src_img_dir  / f"{stem}.png"
        src_mask = src_mask_dir / f"{stem}.png"

        if not src_img.exists():
            for ext in IMAGE_EXTENSIONS:
                cand = src_img_dir / f"{stem}{ext}"
                if cand.exists():
                    src_img = cand
                    break

        if not src_img.exists() or not src_mask.exists():
            logger.warning(f"Missing files for {stem}, skipping.")
            continue

        shutil.copy2(str(src_img),  str(out_img_dir  / f"{stem}.png"))
        shutil.copy2(str(src_mask), str(out_mask_dir / f"{stem}.png"))

        if args.visualize and vis_count < vis_limit:
            img  = cv2.imread(str(src_img),  cv2.IMREAD_COLOR)
            mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
            if img is not None and mask is not None:
                if img.shape[:2] != mask.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                vis = make_overlay(img, mask)
                label = (f"conf={r.get('confidence',0):.3f}  "
                         f"fg={r.get('fg_ratio',0)*100:.1f}%  "
                         f"{'POS' if r.get('fg_ratio',0) >= args.min_fg_ratio else 'NEG'}")
                cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0,0,0),       1, cv2.LINE_AA)
                cv2.imwrite(str(vis_dir / f"{stem}.png"), vis)
            vis_count += 1

    conf_all = [r.get("confidence", 0) for r in selected]
    out_stats = {
        "summary": {
            "n_selected": len(selected),
            "n_pos": len(selected_pos),
            "n_neg": len(selected_neg),
            "conf_max":  round(max(conf_all), 4) if conf_all else 0,
            "conf_min":  round(min(conf_all), 4) if conf_all else 0,
            "conf_mean": round(sum(conf_all)/len(conf_all), 4) if conf_all else 0,
            "fg_ratio_distribution_5pct": hist,
        },
        "samples": [{"file": r["file"],
                     "confidence": round(r.get("confidence",0), 6),
                     "fg_ratio":   round(r.get("fg_ratio",  0), 6)}
                    for r in selected],
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(out_stats, f, indent=2)

    logger.info(f"Output: {out_img_dir.parent}")


if __name__ == "__main__":
    main()
