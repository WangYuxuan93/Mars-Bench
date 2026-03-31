#!/usr/bin/env python3
"""
select_pseudo_by_confidence.py

从已生成的伪标签数据集中，按置信度从高到低选取指定数量的样本，
支持正负例均衡控制，构建新的 ConeQuest 格式数据集。

源目录结构（generate_pseudo_labels.py 输出，需要 --save_all_preds）：
  src_dir/
    all/{split}/images/
    all/{split}/masks/
    stats_shard_*.json  或  stats.json

输出目录结构（ConeQuest 格式）：
  out_dir/
    data/{split}/images/
    data/{split}/masks/
    visualize/           （--visualize 时生成）
    stats.json

Usage
-----
# 选置信度最高的 5000 张，正负比 1:1
python select_pseudo_by_confidence.py \\
    --src_dir  /path/to/pseudo_labels \\
    --out_dir  /path/to/selected \\
    --n        5000 \\
    --neg_pos_ratio 1.0

# 只选正例
python select_pseudo_by_confidence.py \\
    --src_dir  /path/to/pseudo_labels \\
    --out_dir  /path/to/selected \\
    --n        2000 \\
    --pos_only \\
    --visualize
"""

import argparse
import json
import logging
import random
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
    """统计正例中前景占比的分布，以 bin_size 为区间。"""
    positives = [r for r in records if r.get("fg_ratio", 0) > 0]
    bins = {}
    n_bins = int(1.0 / bin_size)
    for i in range(n_bins):
        lo = i * bin_size
        hi = lo + bin_size
        label = f"{lo*100:.0f}%-{hi*100:.0f}%"
        bins[label] = sum(1 for r in positives
                          if lo <= r.get("fg_ratio", 0) < hi)
    # 最后一个 bin 包含 100%
    bins[f"{(n_bins)*bin_size*100:.0f}%-100%"] = sum(
        1 for r in positives if r.get("fg_ratio", 0) >= n_bins * bin_size)
    return bins


def make_overlay(img_bgr: np.ndarray, mask: np.ndarray,
                 alpha: float = 0.25) -> np.ndarray:
    img_f  = img_bgr.astype(np.float32)
    fg     = mask > 0
    result = img_f.copy()
    result[fg]  = np.clip(img_f[fg]  + alpha * 255, 0, 255)
    result[~fg] = np.clip(img_f[~fg] - alpha * 255, 0, 255)
    return result.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(
        description="按置信度从高到低选取伪标签样本，支持正负例均衡"
    )
    parser.add_argument("--src_dir",  required=True)
    parser.add_argument("--out_dir",  required=True)
    parser.add_argument("--n",        type=int, required=True,
                        help="选取的总样本数量")
    parser.add_argument("--split",    default="train")
    parser.add_argument("--min_fg_ratio", type=float, default=0.005,
                        help="正例最小前景占比（默认：0.005）")
    parser.add_argument("--fg_class_id",  type=int, default=1)
    parser.add_argument("--pos_only",     action="store_true",
                        help="只选正例，忽略 --neg_pos_ratio")
    parser.add_argument("--neg_pos_ratio", type=float, default=1.0,
                        help="负例/正例比例上限（默认 1.0 即 1:1）；"
                             "-1=保留所有负例；0=不保留负例")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis_n",    type=int, default=200,
                        help="可视化数量，-1=全部")
    args = parser.parse_args()

    random.seed(args.seed)
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)

    out_img_dir  = out_dir / "data" / args.split / "images"
    out_mask_dir = out_dir / "data" / args.split / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    if args.visualize:
        vis_dir = out_dir / "visualize"
        vis_dir.mkdir(parents=True, exist_ok=True)

    src_img_dir  = src_dir / "all" / args.split / "images"
    src_mask_dir = src_dir / "all" / args.split / "masks"
    if not src_img_dir.exists() or not src_mask_dir.exists():
        raise FileNotFoundError(
            f"Source all/{args.split}/images|masks not found in {src_dir}.\n"
            "Run generate_pseudo_labels.py with --save_all_preds first."
        )

    # 加载并过滤存在文件的记录
    records = load_stats(src_dir)
    existing_stems = {p.stem for p in src_mask_dir.iterdir()
                      if p.suffix.lower() in IMAGE_EXTENSIONS}
    records = [r for r in records if Path(r["file"]).stem in existing_stems]
    logger.info(f"Records with existing masks: {len(records)}")

    # 分正负例，各自按置信度从高到低排序
    positives = sorted(
        [r for r in records if r.get("fg_ratio", 0) >= args.min_fg_ratio],
        key=lambda r: r.get("confidence", 0), reverse=True
    )
    negatives = sorted(
        [r for r in records if r.get("fg_ratio", 0) < args.min_fg_ratio],
        key=lambda r: r.get("confidence", 0), reverse=True
    )

    logger.info(f"Total positives: {len(positives)}, negatives: {len(negatives)}")

    if args.pos_only:
        # 只取正例，按置信度取前 n 个
        n_pos = min(args.n, len(positives))
        selected_pos = positives[:n_pos]
        selected_neg = []
    else:
        # 先确定正例数量：在总数 n 的约束下，正例取尽量多
        if args.neg_pos_ratio <= 0:
            # 0: 不要负例
            n_pos = min(args.n, len(positives))
            n_neg = 0
        elif args.neg_pos_ratio < 0:
            # -1: 保留所有负例
            n_neg = len(negatives)
            n_pos = min(args.n - n_neg, len(positives))
            n_pos = max(n_pos, 0)
        else:
            # 按比例分配，总数 = n
            # n_pos + n_neg = n, n_neg <= n_pos * ratio
            # n_pos = ceil(n / (1 + ratio))
            import math
            n_pos = min(math.ceil(args.n / (1 + args.neg_pos_ratio)), len(positives))
            n_neg = min(args.n - n_pos, int(n_pos * args.neg_pos_ratio), len(negatives))

        selected_pos = positives[:n_pos]
        # 负例取置信度最高的 n_neg 个
        selected_neg = negatives[:n_neg]

    selected = selected_pos + selected_neg
    total = len(selected)

    # fg_ratio 分布（仅正例）
    hist = fg_ratio_histogram(selected_pos)

    # 打印汇总
    conf_all = [r.get("confidence", 0) for r in selected]
    logger.info(
        f"\n{'='*50}\n"
        f"Selected total : {total}  (pos={len(selected_pos)}, neg={len(selected_neg)})\n"
        f"Confidence     : {max(conf_all):.4f} ~ {min(conf_all):.4f}  "
        f"(mean={sum(conf_all)/len(conf_all):.4f})\n"
        f"\nPositive fg_ratio distribution (5% bins):"
    )
    for label, count in hist.items():
        bar = "█" * (count * 40 // max(max(hist.values()), 1))
        logger.info(f"  {label:>12}  {count:5d}  {bar}")
    logger.info("=" * 50)

    # 复制文件
    vis_count = 0
    vis_limit = args.vis_n if args.vis_n >= 0 else total

    for r in selected:
        stem     = Path(r["file"]).stem
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
                is_pos = r.get("fg_ratio", 0) >= args.min_fg_ratio
                label = (f"conf={r.get('confidence',0):.3f}  "
                         f"fg={r.get('fg_ratio',0)*100:.1f}%  "
                         f"{'POS' if is_pos else 'NEG'}")
                cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 0),     1, cv2.LINE_AA)
                cv2.imwrite(str(vis_dir / f"{stem}.png"), vis)
            vis_count += 1

    # 保存 stats.json
    out_stats = {
        "summary": {
            "n_requested":    args.n,
            "n_selected":     total,
            "n_pos":          len(selected_pos),
            "n_neg":          len(selected_neg),
            "neg_pos_ratio":  args.neg_pos_ratio,
            "min_fg_ratio":   args.min_fg_ratio,
            "conf_max":       max(conf_all) if conf_all else 0,
            "conf_min":       min(conf_all) if conf_all else 0,
            "conf_mean":      round(sum(conf_all)/len(conf_all), 4) if conf_all else 0,
            "fg_ratio_distribution_5pct": hist,
        },
        "samples": [
            {"file": r["file"],
             "confidence": round(r.get("confidence", 0), 6),
             "fg_ratio":   round(r.get("fg_ratio",   0), 6),
             "is_positive": r.get("fg_ratio", 0) >= args.min_fg_ratio}
            for r in selected
        ],
    }
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(out_stats, f, indent=2)

    logger.info(
        f"Output   : {out_img_dir.parent}\n"
        f"Stats    : {stats_path}"
        + (f"\nVisualize: {vis_dir}" if args.visualize else "")
    )


if __name__ == "__main__":
    main()
