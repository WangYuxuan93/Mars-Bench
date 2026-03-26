#!/usr/bin/env python3
"""
merge_stats.py

Merge per-shard stats_shardN.json files (produced by generate_pseudo_labels.py
with --num_shards > 1) into a single stats.json.

Usage
-----
python merge_stats.py --output_dir /path/to/pseudo_labels [--num_shards 4]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_shards", type=int, default=None,
                        help="If omitted, auto-detect all stats_shardN.json files.")
    args = parser.parse_args()

    out_root = Path(args.output_dir)

    # Find shard files
    if args.num_shards:
        shard_paths = [out_root / f"stats_shard{i}.json"
                       for i in range(args.num_shards)]
    else:
        shard_paths = sorted(out_root.glob("stats_shard*.json"))

    if not shard_paths:
        raise FileNotFoundError(f"No stats_shardN.json files found in {out_root}")

    print(f"Merging {len(shard_paths)} shard files …")

    all_samples = []
    first_summary = None

    for p in shard_paths:
        with open(p) as f:
            data = json.load(f)
        all_samples.extend(data["samples"])
        if first_summary is None:
            first_summary = data["summary"]

    total     = len(all_samples)
    accepted  = sum(1 for s in all_samples if s["status"].startswith("accepted"))
    acc_pos   = sum(1 for s in all_samples if s["status"] == "accepted_pos")
    acc_neg   = sum(1 for s in all_samples if s["status"] == "accepted_neg")
    rej_neg   = sum(1 for s in all_samples if s["status"] == "rejected_neg")
    rej_conf  = sum(1 for s in all_samples if s["status"] == "rejected_conf")

    all_conf  = [s["confidence"] for s in all_samples]

    # Rebuild confidence distribution
    buckets = {}
    bucket_lines = []
    for lo in range(0, 100, 10):
        hi    = lo + 10
        key   = f"{lo:02d}-{hi:02d}%"
        count = sum(1 for c in all_conf
                    if lo / 100 <= c < hi / 100 or (hi == 100 and c == 1.0))
        pct   = count / total * 100 if total else 0
        buckets[key] = {"count": count, "pct": round(pct, 1)}
        bucket_lines.append(f"  {key}: {count:6d}  ({pct:5.1f}%)")

    summary = {
        **first_summary,
        "total":           total,
        "accepted":        accepted,
        "accepted_pos":    acc_pos,
        "accepted_neg":    acc_neg,
        "rejected_neg":    rej_neg,
        "rejected_conf":   rej_conf,
        "mean_conf":       round(float(np.mean(all_conf)),   4),
        "median_conf":     round(float(np.median(all_conf)), 4),
        "conf_distribution": buckets,
    }

    out_path = out_root / "stats.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "samples": all_samples}, f, indent=2)

    dist_str = "\n".join(reversed(bucket_lines))
    print(
        f"Merged {len(shard_paths)} shards → {total} samples\n"
        f"  accepted={accepted} (pos={acc_pos}, neg={acc_neg})\n"
        f"  rejected={rej_neg + rej_conf}\n"
        f"\nConfidence distribution (high → low):\n{dist_str}\n"
        f"\nSaved → {out_path}"
    )


if __name__ == "__main__":
    main()
