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
  [--batch_size 8] [--input_size 512] [--num_workers 4] [--device cuda]
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
_PALETTE = {
    0: None,               # background – no overlay
    1: (0, 200, 0),        # foreground – green
    2: (200, 0, 0),        # class 2 – blue
    3: (0, 0, 200),        # class 3 – red
}


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
    pixel_values = processed["pixel_values"].to(device)
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
                 alpha: float = 0.45) -> np.ndarray:
    """Draw a semi-transparent colored mask overlay on the original image.

    Returns a BGR uint8 image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.resize(img, (mask.shape[1], mask.shape[0]),
                     interpolation=cv2.INTER_LINEAR)
    overlay = img.copy()

    for cls_id, color in _PALETTE.items():
        if color is None:
            continue
        region = mask == cls_id
        if not region.any():
            continue
        overlay[region] = color

    vis = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Draw a thin contour around foreground regions for clarity
    for cls_id, color in _PALETTE.items():
        if color is None:
            continue
        region = (mask == cls_id).astype(np.uint8)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 1)

    return vis


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
    parser.add_argument("--image_dir",      required=True)
    parser.add_argument("--checkpoint",     required=True)
    parser.add_argument("--base_model",     required=True,
                        help="HF model name used only for config/architecture.")
    parser.add_argument("--num_classes",    type=int, required=True)
    parser.add_argument("--output_dir",     required=True)
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

    # Inference
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--input_size",  type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    device   = torch.device(args.device)
    out_root = Path(args.output_dir)

    # ------------------------------------------------------------------ dirs
    accepted_img_dir  = out_root / "data" / args.split / "images"
    accepted_mask_dir = out_root / "data" / args.split / "masks"
    vis_dir           = out_root / "visualize"
    for d in [accepted_img_dir, accepted_mask_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.save_rejected:
        rej_img_dir  = out_root / "rejected" / args.split / "images"
        rej_mask_dir = out_root / "rejected" / args.split / "masks"
        rej_img_dir.mkdir(parents=True, exist_ok=True)
        rej_mask_dir.mkdir(parents=True, exist_ok=True)

    if args.save_confidence:
        conf_dir = out_root / "confidence"
        conf_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ load model
    model           = load_model(args.checkpoint, args.base_model,
                                 args.num_classes, device)
    image_processor = Mask2FormerImageProcessor(ignore_index=255)

    dataset    = UnlabeledImageDataset(args.image_dir, args.input_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=(args.device == "cuda"))

    # ------------------------------------ Phase 1: predict, buffer metadata
    # We keep (path, pred, conf_map) in a flat list, then do balancing offline.
    # For very large datasets consider streaming to a temp dir instead.
    all_samples = []   # list of dicts

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
            all_samples.append({
                "path":      path,
                "stem":      Path(path).stem,
                "pred":      pred,
                # conf_map is large (float32 H×W); only keep scalar confidence
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

        # visualization – always generated for accepted samples
        vis = make_overlay(s["path"], s["pred"])
        # annotate with stats in corner
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
            "file":        Path(s["path"]).name,
            "status":      status,
            "confidence":  round(s["confidence"], 6),
            "fg_ratio":    round(s["fg_ratio"],   6),
            "fg_pixels":   s["fg_pixels"],
            "is_positive": s["is_positive"],
        }

    records = (
        [_sample_record(s, "accepted_pos") for s in positives] +
        [_sample_record(s, "accepted_neg") for s in neg_keep]  +
        [_sample_record(s, "rejected_neg") for s in neg_drop]  +
        [_sample_record(s, "rejected_conf") for s in conf_bad]
    )

    all_conf = [s["confidence"] for s in all_samples]
    summary  = {
        "total":           len(all_samples),
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
    }

    stats_path = out_root / "stats.json"
    with open(stats_path, "w") as f:
        json.dump({"summary": summary, "samples": records}, f, indent=2)

    logger.info(
        f"\nDone.\n"
        f"  Accepted : {len(accepted)} "
        f"(pos={len(positives)}, neg={len(neg_keep)})\n"
        f"  Rejected : {len(rejected)}\n"
        f"  Visualizations → {vis_dir}\n"
        f"  Stats          → {stats_path}\n"
        f"  Pseudo-labels  → {accepted_img_dir.parent}"
    )


if __name__ == "__main__":
    main()
