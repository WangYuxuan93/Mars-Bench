#!/usr/bin/env python3
"""
generate_pseudo_labels.py

Batch-predict pseudo-labels for unlabeled CTX images using a trained
Mask2Former checkpoint (MarsBench .ckpt or HF model directory), with:

  - Confidence-based quality filtering
  - Positive/negative sample classification + negative downsampling
  - Mask overlay visualizations for manual inspection
  - ConeQuest-compatible output layout for MarsBench training

Output layout
-------------
<output_dir>/
  data/{split}/
    images/   <stem>.png   – accepted image (RGB)
    masks/    <stem>.png   – uint8 pseudo-label (pixel values = class IDs 0/1)
  visualize/
    <stem>.png             – original image with mask overlay (always saved for accepted)
  confidence/              – optional (--save_confidence)
    <stem>.png             – uint8 where 255 = 100 % confidence
  rejected/                – optional (--save_rejected)
    {split}/images|masks/
  stats.json

Usage
-----
python generate_pseudo_labels.py \\
  --image_dir      /path/to/unlabeled_ctx \\
  --checkpoint     /path/to/best.ckpt \\
  --base_model     facebook/mask2former-swin-large-ade-semantic \\
  --num_classes    2 \\
  --output_dir     /path/to/pseudo_labels \\
  --conf_threshold 0.80 \\
  --min_fg_ratio   0.005 \\
  --neg_pos_ratio  1.0 \\
  [--save_confidence] [--save_rejected] \\
  [--per_gpu_batch_size 8] [--input_size 512] [--num_workers 4] [--device cuda]
"""

import argparse
import json
import logging
import random
import tempfile
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Visualization color palette: index → BGR color


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_dir: str, input_size: int):
        self.paths = sorted([
            p for p in Path(image_dir).iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])
        if not self.paths:
            raise FileNotFoundError(f"No images found in {image_dir}")
        logger.info(f"Found {len(self.paths)} images in {image_dir}")

        self.transform = A.Compose([
            A.Resize(height=input_size, width=input_size,
                     interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD,
                        max_pixel_value=255.0),
            A.ToTensorV2(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("uint8")
        orig_h, orig_w = img.shape[:2]
        tensor = self.transform(image=img)["image"]
        return tensor, str(path), orig_h, orig_w


def collate_fn(batch):
    tensors, paths, orig_hs, orig_ws = zip(*batch)
    return torch.stack(tensors), list(paths), list(orig_hs), list(orig_ws)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint: str, base_model: str, num_classes: int,
               device: torch.device) -> Mask2FormerForUniversalSegmentation:
    """Load Mask2Former from a MarsBench Lightning .ckpt or HF model directory.

    MarsBench saves Lightning checkpoints where the HF model is stored as
    self.model, so all state-dict keys are prefixed with 'model.'.
    We strip that prefix to recover the plain HF state-dict.
    base_model is only used to fetch the config (no weight download).
    """
    from transformers import Mask2FormerConfig

    ckpt_path = Path(checkpoint)

    if ckpt_path.is_dir():
        logger.info(f"Loading from HF model directory: {checkpoint}")
        hf_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint, num_labels=num_classes, ignore_mismatched_sizes=True,
        )
    elif ckpt_path.suffix in (".ckpt", ".pt", ".pth", ".bin"):
        logger.info(f"Loading MarsBench Lightning checkpoint: {checkpoint}")
        config = Mask2FormerConfig.from_pretrained(base_model)
        config.num_labels = num_classes
        hf_model = Mask2FormerForUniversalSegmentation(config)

        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = raw.get("state_dict", raw)

        prefix = "model."
        sd = {k[len(prefix):]: v for k, v in state_dict.items()
              if k.startswith(prefix)}
        if not sd:
            logger.warning("No 'model.*' keys found; trying raw state-dict.")
            sd = state_dict

        missing, unexpected = hf_model.load_state_dict(sd, strict=False)
        real_missing = [k for k in missing if "criterion" not in k]
        if real_missing:
            logger.warning(f"Missing keys (first 10): {real_missing[:10]}")
        if unexpected:
            logger.warning(f"Unexpected keys (first 10): {unexpected[:10]}")
        logger.info("Checkpoint loaded.")
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint}")

    hf_model.eval().to(device)
    return hf_model


# ---------------------------------------------------------------------------
# Inference with confidence
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_batch(
    model, image_processor, pixel_tensors, orig_sizes, device,
):
    processed = image_processor(
        list(pixel_tensors),
        return_tensors="pt",
        do_resize=False, do_rescale=False, do_normalize=False,
    )
    pixel_values = processed["pixel_values"].to(device=device,
                                                   dtype=next(model.parameters()).dtype)
    pixel_mask   = processed["pixel_mask"].to(device)
    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    class_q   = outputs.class_queries_logits          # [B, Q, C+1]
    masks_q   = outputs.masks_queries_logits          # [B, Q, h, w]
    masks_cls = class_q.softmax(dim=-1)[..., :-1]     # [B, Q, C] drop null
    masks_prb = masks_q.sigmoid()                     # [B, Q, h, w]

    preds, conf_maps = [], []
    for i, (oh, ow) in enumerate(orig_sizes):
        seg = torch.einsum("qc,qhw->chw", masks_cls[i], masks_prb[i])
        seg = F.interpolate(seg.unsqueeze(0), size=(oh, ow),
                            mode="bilinear", align_corners=False).squeeze(0)
        seg_sum  = seg.sum(0, keepdim=True).clamp(min=1e-6)
        seg_norm = seg / seg_sum
        conf, pred = seg_norm.max(0)
        preds.append(pred.cpu().numpy().astype(np.uint8))
        conf_maps.append(conf.cpu().numpy().astype(np.float32))
    return preds, conf_maps


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def compute_sample_stats(pred: np.ndarray, conf_map: np.ndarray,
                          fg_class_id: int) -> dict:
    total_pixels = pred.size
    fg_pixels    = int((pred == fg_class_id).sum())
    return {
        "confidence":  float(conf_map.mean()),
        "fg_pixels":   fg_pixels,
        "fg_ratio":    fg_pixels / total_pixels,
        "is_positive": fg_pixels > 0,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_overlay(image_path: str, mask: np.ndarray,
                 alpha: float = 0.25) -> np.ndarray:
    """Brightness-only mask visualization: foreground brightened, background dimmed.

    Returns a BGR uint8 image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.resize(img, (mask.shape[1], mask.shape[0]),
                     interpolation=cv2.INTER_LINEAR)
    img_f  = img.astype(np.float32)
    fg     = mask > 0

    result = img_f.copy()
    result[fg]  = np.clip(img_f[fg]  + alpha * 255, 0, 255)
    result[~fg] = np.clip(img_f[~fg] - alpha * 255, 0, 255)

    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_image_copy(src: str, dst: Path):
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(src)
    cv2.imwrite(str(dst), img)


def save_mask(mask: np.ndarray, dst: Path):
    cv2.imwrite(str(dst), mask)


def save_confidence_map(conf: np.ndarray, dst: Path):
    cv2.imwrite(str(dst), (conf * 255).clip(0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels with pos/neg balancing."
    )
    parser.add_argument("--image_dir",      default=None,
                        help="Required for inference mode.")
    parser.add_argument("--checkpoint",     default=None,
                        help="Required for inference mode.")
    parser.add_argument("--base_model",     default=None,
                        help="HF model name used only for config/architecture. "
                             "Required for inference mode.")
    parser.add_argument("--num_classes",    type=int, default=None,
                        help="Required for inference mode.")
    parser.add_argument("--output_dir",     required=True,
                        help="Root output directory. For --refilter, also the "
                             "source of stats.json and all/ predictions.")
    parser.add_argument("--refilter_output_dir", default=None,
                        help="Write re-filtered results here instead of "
                             "overwriting --output_dir. Only used with --refilter.")
    parser.add_argument("--split",          default="train",
                        help="ConeQuest split name (default: train).")

    # Quality filtering
    parser.add_argument("--conf_threshold", type=float, default=0.80,
                        help="Min mean-confidence to keep a sample (default: 0.80).")
    parser.add_argument("--min_fg_ratio",   type=float, default=0.005,
                        help="Min foreground pixel ratio for a 'positive' image "
                             "(default: 0.005 = 0.5%%).")
    parser.add_argument("--fg_class_id",    type=int,   default=1,
                        help="Class ID treated as foreground (default: 1).")

    # Pos/neg balancing
    parser.add_argument("--neg_pos_ratio",  type=float, default=1.0,
                        help="Max negatives kept per positive sample "
                             "(default: 1.0 → 1:1). Set 0 to drop all negatives, "
                             "-1 to keep all negatives that pass conf_threshold.")
    parser.add_argument("--seed",           type=int,   default=42)

    # Optional outputs
    parser.add_argument("--save_confidence", action="store_true")
    parser.add_argument("--save_rejected",   action="store_true")
    parser.add_argument("--visualize",       action="store_true",
                        help="Save mask-overlay visualizations to visualize/. "
                             "Skipped by default to save time on large datasets.")
    parser.add_argument("--save_all_preds",  action="store_true",
                        help="Save ALL predicted masks+images to all/{split}/ "
                             "so you can re-filter later with --refilter.")

    # Re-filter mode (no inference needed)
    parser.add_argument("--refilter", action="store_true",
                        help="Skip inference. Re-apply thresholds using the "
                             "existing stats.json and all/{split}/ predictions. "
                             "Requires a previous run with --save_all_preds.")

    # Inference
    parser.add_argument("--per_gpu_batch_size",  type=int, default=8,
                        help="Batch size per GPU (total = batch_size × num_gpus).")
    parser.add_argument("--input_size",  type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="DataLoader prefetch factor per worker (default: 4).")
    parser.add_argument("--fp16", action="store_true",
                        help="Run model in float16 for faster inference (~1.5-2x).")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model (~20-30%% speedup, "
                             "slow first batch).")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    device   = torch.device(args.device)
    out_root = Path(args.output_dir)

    # For refilter: write accepted results to a separate dir if specified
    write_root = Path(args.refilter_output_dir) if (
        args.refilter and args.refilter_output_dir
    ) else out_root

    # ------------------------------------------------------------------ dirs
    accepted_img_dir  = write_root / "data" / args.split / "images"
    accepted_mask_dir = write_root / "data" / args.split / "masks"
    vis_dir           = write_root / "visualize"
    for d in [accepted_img_dir, accepted_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    all_img_dir  = out_root / "all" / args.split / "images"
    all_mask_dir = out_root / "all" / args.split / "masks"
    if args.save_all_preds or args.refilter:
        all_img_dir.mkdir(parents=True, exist_ok=True)
        all_mask_dir.mkdir(parents=True, exist_ok=True)

    if args.save_rejected:
        rej_img_dir  = out_root / "rejected" / args.split / "images"
        rej_mask_dir = out_root / "rejected" / args.split / "masks"
        rej_img_dir.mkdir(parents=True, exist_ok=True)
        rej_mask_dir.mkdir(parents=True, exist_ok=True)

    if args.save_confidence:
        conf_dir = out_root / "confidence"
        conf_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================ REFILTER MODE
    if args.refilter:
        stats_path = out_root / "stats.json"   # always read from original out_root
        if not stats_path.exists():
            raise FileNotFoundError(
                f"stats.json not found at {stats_path}. "
                "Run without --refilter first (with --save_all_preds)."
            )
        if not any(all_mask_dir.iterdir()):
            raise FileNotFoundError(
                f"No predictions found in {all_mask_dir}. "
                "Run without --refilter first (with --save_all_preds)."
            )

        logger.info(f"Re-filter mode: reading stats from {stats_path}")
        with open(stats_path) as f:
            saved = json.load(f)

        # Rebuild all_samples from disk.
        # Original images are NOT copied to all/ (to save space/time),
        # so fall back to the recorded original path in stats.json,
        # or search in all_img_dir for backward compatibility.
        all_samples = []
        for rec in saved["samples"]:
            stem     = Path(rec["file"]).stem
            msk_path = all_mask_dir / f"{stem}.png"
            if not msk_path.exists():
                logger.warning(f"Missing mask in all/ for {stem}, skipping.")
                continue
            # Use original_path if recorded, otherwise try all_img_dir
            orig_path = rec.get("original_path")
            if orig_path and Path(orig_path).exists():
                img_path = orig_path
            else:
                img_path = all_img_dir / f"{stem}.png"
                if not img_path.exists():
                    logger.warning(f"Image not found for {stem}, skipping.")
                    continue
            all_samples.append({
                "path":        str(img_path),
                "stem":        stem,
                "pred":        cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE),
                "confidence":  rec["confidence"],
                "fg_ratio":    rec["fg_ratio"],
                "fg_pixels":   rec["fg_pixels"],
                "is_positive": rec["is_positive"],
            })
        logger.info(f"Loaded {len(all_samples)} samples from all/ dir.")
    # ============================================================ INFERENCE MODE
    else:
        missing = [n for n, v in [("--image_dir",  args.image_dir),
                                   ("--checkpoint", args.checkpoint),
                                   ("--base_model", args.base_model),
                                   ("--num_classes",args.num_classes)]
                   if v is None]
        if missing:
            parser.error(f"Inference mode requires: {', '.join(missing)}")

        model = load_model(args.checkpoint, args.base_model,
                           args.num_classes, device)
        if args.fp16:
            model = model.half()
            logger.info("Model cast to float16.")
        if args.compile:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile.")
        n_gpus = torch.cuda.device_count() if device.type == "cuda" else 1
        if device.type == "cuda" and n_gpus > 1:
            model = torch.nn.DataParallel(model)
            logger.info(f"Using {n_gpus} GPUs via DataParallel.")
        image_processor = Mask2FormerImageProcessor(ignore_index=255)

        total_batch = args.per_gpu_batch_size * n_gpus
        logger.info(f"batch_size per GPU={args.per_gpu_batch_size}, "
                    f"total batch={total_batch} ({n_gpus} GPU(s))")
        dataset    = UnlabeledImageDataset(args.image_dir, args.input_size)
        dataloader = DataLoader(
            dataset, batch_size=total_batch, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_fn,
            pin_memory=(args.device == "cuda"),
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            persistent_workers=args.num_workers > 0,
        )

        all_samples = []

        logger.info("Phase 1: running inference …")
        for pixel_tensors, paths, orig_hs, orig_ws in tqdm(dataloader, desc="Predict"):
            orig_sizes = list(zip(orig_hs, orig_ws))
            preds, conf_maps = predict_batch(model, image_processor,
                                             pixel_tensors, orig_sizes, device)

            for path, pred, conf_map in zip(paths, preds, conf_maps):
                stats = compute_sample_stats(pred, conf_map, args.fg_class_id)
                if args.save_confidence:
                    save_confidence_map(conf_map,
                                        conf_dir / f"{Path(path).stem}.png")
                if args.save_all_preds:
                    stem = Path(path).stem
                    # Only save mask; record original image path to avoid
                    # copying 10M images (saves time and disk space)
                    save_mask(pred, all_mask_dir / f"{stem}.png")
                all_samples.append({
                    "path":      path,
                    "stem":      Path(path).stem,
                    "pred":      pred,
                    **stats,
                })

    # ---------------------------------- Phase 2: quality filter + balancing
    logger.info("Phase 2: filtering and balancing …")

    conf_ok    = [s for s in all_samples
                  if s["confidence"] >= args.conf_threshold]
    conf_bad   = [s for s in all_samples
                  if s["confidence"] <  args.conf_threshold]

    positives  = [s for s in conf_ok if s["fg_ratio"] >= args.min_fg_ratio]
    negatives  = [s for s in conf_ok if s["fg_ratio"] <  args.min_fg_ratio]

    n_pos = len(positives)

    if args.neg_pos_ratio < 0:
        neg_keep = negatives                     # keep all
    elif args.neg_pos_ratio == 0:
        neg_keep = []                            # drop all
    else:
        n_neg_max = int(n_pos * args.neg_pos_ratio)
        neg_keep  = random.sample(negatives, min(len(negatives), n_neg_max))

    neg_keep_ids = {id(s) for s in neg_keep}
    neg_drop     = [s for s in negatives if id(s) not in neg_keep_ids]

    accepted = positives + neg_keep
    rejected = conf_bad + neg_drop

    logger.info(
        f"  All predicted   : {len(all_samples)}\n"
        f"  Conf threshold  : {args.conf_threshold}  → {len(conf_ok)} pass / "
        f"{len(conf_bad)} fail\n"
        f"  Positives (fg≥{args.min_fg_ratio:.3f}): {n_pos}\n"
        f"  Negatives total : {len(negatives)}  "
        f"→ keep {len(neg_keep)} (ratio {args.neg_pos_ratio})\n"
        f"  Final accepted  : {len(accepted)}"
    )

    # ----------------------------------------- Phase 3: save accepted
    logger.info("Phase 3: saving accepted samples …")
    for s in tqdm(accepted, desc="Save accepted"):
        stem = s["stem"]
        save_image_copy(s["path"], accepted_img_dir  / f"{stem}.png")
        save_mask(s["pred"],       accepted_mask_dir / f"{stem}.png")

        if args.visualize:
            vis = make_overlay(s["path"], s["pred"])
            label = (f"conf={s['confidence']:.2f}  "
                     f"fg={s['fg_ratio']*100:.1f}%  "
                     f"{'POS' if s['is_positive'] else 'NEG'}")
            cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0),     1, cv2.LINE_AA)
            cv2.imwrite(str(vis_dir / f"{stem}.png"), vis)

    if args.save_rejected:
        logger.info("Saving rejected samples …")
        for s in tqdm(rejected, desc="Save rejected"):
            stem = s["stem"]
            save_image_copy(s["path"], rej_img_dir  / f"{stem}.png")
            save_mask(s["pred"],       rej_mask_dir / f"{stem}.png")

    # --------------------------------------------------- stats.json
    def _sample_record(s, status):
        return {
            "file":          Path(s["path"]).name,
            "original_path": s["path"],   # kept for --refilter without image copy
            "status":        status,
            "confidence":    round(s["confidence"], 6),
            "fg_ratio":      round(s["fg_ratio"],   6),
            "fg_pixels":     s["fg_pixels"],
            "is_positive":   s["is_positive"],
        }

    records = (
        [_sample_record(s, "accepted_pos") for s in positives] +
        [_sample_record(s, "accepted_neg") for s in neg_keep]  +
        [_sample_record(s, "rejected_neg") for s in neg_drop]  +
        [_sample_record(s, "rejected_conf") for s in conf_bad]
    )

    all_conf = [s["confidence"] for s in all_samples]
    total    = len(all_conf)

    # Confidence distribution in 10% buckets
    buckets = {}
    bucket_lines = []
    for lo in range(0, 100, 10):
        hi    = lo + 10
        key   = f"{lo:02d}-{hi:02d}%"
        count = sum(1 for c in all_conf if lo / 100 <= c < hi / 100)
        # include 100% in the last bucket
        if hi == 100:
            count = sum(1 for c in all_conf if lo / 100 <= c <= 1.0)
        pct   = count / total * 100 if total else 0
        buckets[key] = {"count": count, "pct": round(pct, 1)}
        bucket_lines.append(f"    {key}: {count:5d}  ({pct:5.1f}%)")

    summary  = {
        "total":           total,
        "accepted":        len(accepted),
        "accepted_pos":    len(positives),
        "accepted_neg":    len(neg_keep),
        "rejected_neg":    len(neg_drop),
        "rejected_conf":   len(conf_bad),
        "conf_threshold":  args.conf_threshold,
        "min_fg_ratio":    args.min_fg_ratio,
        "neg_pos_ratio":   args.neg_pos_ratio,
        "mean_conf":       round(float(np.mean(all_conf)),   4),
        "median_conf":     round(float(np.median(all_conf)), 4),
        "conf_distribution": buckets,
    }

    stats_path = write_root / "stats.json"
    with open(stats_path, "w") as f:
        json.dump({"summary": summary, "samples": records}, f, indent=2)

    dist_str = "\n".join(reversed(bucket_lines))   # high → low
    vis_line = f"\n  Visualizations → {vis_dir}" if args.visualize else ""
    logger.info(
        f"\nDone.\n"
        f"  Accepted : {len(accepted)} "
        f"(pos={len(positives)}, neg={len(neg_keep)})\n"
        f"  Rejected : {len(rejected)}\n"
        f"\nConfidence distribution (high → low):\n{dist_str}\n"
        f"{vis_line}\n"
        f"  Stats          → {stats_path}\n"
        f"  Pseudo-labels  → {accepted_img_dir.parent}"
    )


if __name__ == "__main__":
    main()
