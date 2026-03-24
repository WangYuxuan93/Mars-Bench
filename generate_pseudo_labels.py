#!/usr/bin/env python3
"""
generate_pseudo_labels.py

Batch-predict pseudo-labels for unlabeled CTX images using a trained
Mask2Former checkpoint (MarsBench .ckpt or HF model directory), then
filter by per-image confidence and export results in a format that is
directly compatible with Transformers semantic segmentation training
(imagefolder / MarsBench ConeQuest directory layout).

Output layout
-------------
<output_dir>/
  accepted/
    images/   <stem>.png   – copy of the original image (RGB)
    masks/    <stem>.png   – uint8 pseudo-label (pixel values = class IDs)
  rejected/                – created only with --save_rejected
    images/
    masks/
  confidence/              – created only with --save_confidence
    <stem>.png             – uint8 image where 255 = 100 % confidence
  stats.json               – per-image scores + overall statistics

Usage
-----
python generate_pseudo_labels.py \\
  --image_dir   /path/to/unlabeled_ctx \\
  --checkpoint  /path/to/best.ckpt \\
  --base_model  facebook/mask2former-swin-large-ade-semantic \\
  --num_classes 2 \\
  --output_dir  /path/to/pseudo_labels \\
  --conf_threshold 0.80 \\
  [--save_confidence] \\
  [--save_rejected] \\
  [--batch_size 8] \\
  [--input_size 512] \\
  [--num_workers 4] \\
  [--device cuda]
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ImageNet RGB statistics – must match MarsBench's segmentation transform
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UnlabeledImageDataset(Dataset):
    """Loads unlabeled images from a directory, applies resize + normalize."""

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
        tensor = self.transform(image=img)["image"]   # [3, H, W] float32
        return tensor, str(path), orig_h, orig_w


def collate_fn(batch):
    tensors, paths, orig_hs, orig_ws = zip(*batch)
    return torch.stack(tensors), list(paths), list(orig_hs), list(orig_ws)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint: str, base_model: str, num_classes: int,
               device: torch.device) -> Mask2FormerForUniversalSegmentation:
    """Load Mask2Former from a MarsBench Lightning .ckpt or an HF model directory.

    MarsBench Lightning checkpoint structure
    ----------------------------------------
    The Lightning module stores the HF model as ``self.model``
    (BaseSegmentationModel.__init__ → self.model = self._initialize_model()).
    Therefore every key in the Lightning state_dict is prefixed with 'model.':

        model.model.pixel_level_module.*   (Mask2FormerModel internals)
        model.class_predictor.*
        model.criterion.*
        ...

    We strip the leading 'model.' to recover the plain HF state-dict.

    The ``base_model`` argument is only used to download the *config* so we
    can instantiate the correct architecture – the pretrained HF weights are
    NOT downloaded, saving time and bandwidth.
    """
    from transformers import Mask2FormerConfig

    ckpt_path = Path(checkpoint)

    if ckpt_path.is_dir():
        # HF model directory (model.safetensors / config.json already present)
        logger.info(f"Loading from HF model directory: {checkpoint}")
        hf_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    elif ckpt_path.suffix in (".ckpt", ".pt", ".pth", ".bin"):
        logger.info(f"Loading MarsBench Lightning checkpoint: {checkpoint}")

        # Build the architecture from config only (no weight download)
        config = Mask2FormerConfig.from_pretrained(base_model)
        config.num_labels = num_classes
        hf_model = Mask2FormerForUniversalSegmentation(config)

        # Load checkpoint
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = raw.get("state_dict", raw)  # handle both Lightning and raw

        # Strip Lightning wrapper prefix: 'model.' → ''
        prefix = "model."
        sd = {k[len(prefix):]: v
              for k, v in state_dict.items()
              if k.startswith(prefix)}

        if not sd:
            logger.warning(
                "No 'model.*' keys found in checkpoint. "
                "Trying to load the raw state-dict directly."
            )
            sd = state_dict

        missing, unexpected = hf_model.load_state_dict(sd, strict=False)
        # 'criterion.*' keys may differ between HF versions – usually fine to ignore
        real_missing = [k for k in missing if "criterion" not in k]
        if real_missing:
            logger.warning(f"Missing keys (first 10): {real_missing[:10]}")
        if unexpected:
            logger.warning(f"Unexpected keys (first 10): {unexpected[:10]}")
        logger.info("Checkpoint loaded successfully.")
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint}")

    hf_model.eval()
    hf_model.to(device)
    return hf_model


# ---------------------------------------------------------------------------
# Inference with confidence
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_batch(
    model: Mask2FormerForUniversalSegmentation,
    image_processor: Mask2FormerImageProcessor,
    pixel_tensors: torch.Tensor,   # [B, 3, H, W]  normalised
    orig_sizes: list[tuple[int, int]],
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Run Mask2Former on a batch and return per-image predictions + confidence.

    We replicate the logic of post_process_semantic_segmentation but also
    compute per-pixel confidence as:
        conf[x,y] = argmax_class_score[x,y] / sum_class_scores[x,y]

    Returns
    -------
    preds       list of [H, W] uint8 arrays (class IDs, original resolution)
    conf_maps   list of [H, W] float32 arrays in [0, 1]
    """
    # Mask2FormerImageProcessor expects un-normalised images for its internal
    # pipeline, but MarsBench bypasses its resize/rescale/normalize steps and
    # passes pre-normalised tensors directly.  We replicate that here.
    processed = image_processor(
        list(pixel_tensors),          # list of [3, H, W] float tensors
        return_tensors="pt",
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )
    pixel_values = processed["pixel_values"].to(device)
    pixel_mask   = processed["pixel_mask"].to(device)

    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    # ---- replicate post_process_semantic_segmentation with confidence ----
    # class_queries_logits : [B, Q, num_labels + 1]  (last = null/bg class)
    # masks_queries_logits : [B, Q, H/4, W/4]
    class_q = outputs.class_queries_logits          # [B, Q, C+1]
    masks_q = outputs.masks_queries_logits          # [B, Q, h, w]

    masks_classes = class_q.softmax(dim=-1)[..., :-1]  # [B, Q, C]  drop null
    masks_probs   = masks_q.sigmoid()                  # [B, Q, h, w]

    preds     = []
    conf_maps = []

    for i, (oh, ow) in enumerate(orig_sizes):
        # Aggregate: [C, h, w]
        seg = torch.einsum(
            "qc,qhw->chw", masks_classes[i], masks_probs[i]
        )
        # Upsample to original resolution
        seg = F.interpolate(
            seg.unsqueeze(0), size=(oh, ow),
            mode="bilinear", align_corners=False
        ).squeeze(0)                                   # [C, H, W]

        # Normalise scores → confidence
        seg_sum  = seg.sum(dim=0, keepdim=True).clamp(min=1e-6)
        seg_norm = seg / seg_sum                       # [C, H, W]

        conf, pred = seg_norm.max(dim=0)               # [H, W]

        preds.append(pred.cpu().numpy().astype(np.uint8))
        conf_maps.append(conf.cpu().numpy().astype(np.float32))

    return preds, conf_maps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sample_confidence(conf_map: np.ndarray) -> float:
    """Scalar confidence score for a single image = mean pixel confidence."""
    return float(conf_map.mean())


def save_image_copy(src: str, dst: Path):
    """Save a copy of the original image as PNG (RGB)."""
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(src)
    cv2.imwrite(str(dst), img)


def save_mask(mask: np.ndarray, dst: Path):
    """Save a uint8 single-channel mask (pixel values = class IDs)."""
    cv2.imwrite(str(dst), mask)


def save_confidence_map(conf: np.ndarray, dst: Path):
    """Save confidence map as uint8 PNG (0 = 0 %, 255 = 100 %)."""
    vis = (conf * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(dst), vis)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels for unlabeled CTX images."
    )
    parser.add_argument("--image_dir",      required=True,
                        help="Directory containing unlabeled images.")
    parser.add_argument("--checkpoint",     required=True,
                        help="MarsBench .ckpt file or HF model directory.")
    parser.add_argument("--base_model",     required=True,
                        help="HF hub name used to initialise the model architecture "
                             "(e.g. facebook/mask2former-swin-large-ade-semantic).")
    parser.add_argument("--num_classes",    type=int, required=True,
                        help="Number of classes (including background).")
    parser.add_argument("--output_dir",     required=True,
                        help="Root directory for all outputs.")
    parser.add_argument("--conf_threshold", type=float, default=0.80,
                        help="Per-image mean-confidence threshold. "
                             "Images below this are rejected (default: 0.80).")
    parser.add_argument("--save_confidence", action="store_true",
                        help="Save per-pixel confidence maps to output_dir/confidence/.")
    parser.add_argument("--save_rejected",   action="store_true",
                        help="Save low-confidence samples to output_dir/rejected/.")
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--input_size",     type=int, default=512,
                        help="Resize images to this square size before inference.")
    parser.add_argument("--num_workers",    type=int, default=4)
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_root = Path(args.output_dir)

    # Prepare output directories
    accepted_img_dir  = out_root / "accepted" / "images"
    accepted_mask_dir = out_root / "accepted" / "masks"
    accepted_img_dir.mkdir(parents=True, exist_ok=True)
    accepted_mask_dir.mkdir(parents=True, exist_ok=True)

    if args.save_rejected:
        rejected_img_dir  = out_root / "rejected" / "images"
        rejected_mask_dir = out_root / "rejected" / "masks"
        rejected_img_dir.mkdir(parents=True, exist_ok=True)
        rejected_mask_dir.mkdir(parents=True, exist_ok=True)

    if args.save_confidence:
        conf_dir = out_root / "confidence"
        conf_dir.mkdir(parents=True, exist_ok=True)

    # Load model + image processor
    model = load_model(args.checkpoint, args.base_model, args.num_classes, device)
    image_processor = Mask2FormerImageProcessor(
        ignore_index=255,
        reduce_labels=False,
    )

    # Build dataloader
    dataset    = UnlabeledImageDataset(args.image_dir, args.input_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(args.device == "cuda"),
    )

    # Inference loop
    stats = {
        "conf_threshold": args.conf_threshold,
        "num_classes": args.num_classes,
        "samples": [],
    }
    n_accepted = 0
    n_rejected = 0

    for pixel_tensors, paths, orig_hs, orig_ws in tqdm(dataloader, desc="Predicting"):
        orig_sizes = list(zip(orig_hs, orig_ws))
        preds, conf_maps = predict_batch(
            model, image_processor, pixel_tensors, orig_sizes, device
        )

        for path, pred, conf_map in zip(paths, preds, conf_maps):
            stem  = Path(path).stem
            score = sample_confidence(conf_map)
            accepted = score >= args.conf_threshold

            stats["samples"].append({
                "file": Path(path).name,
                "confidence": round(score, 6),
                "accepted": accepted,
            })

            if args.save_confidence:
                save_confidence_map(conf_map, conf_dir / f"{stem}.png")

            if accepted:
                save_image_copy(path, accepted_img_dir  / f"{stem}.png")
                save_mask(pred,       accepted_mask_dir / f"{stem}.png")
                n_accepted += 1
            else:
                n_rejected += 1
                if args.save_rejected:
                    save_image_copy(path, rejected_img_dir  / f"{stem}.png")
                    save_mask(pred,       rejected_mask_dir / f"{stem}.png")

    # Summary statistics
    all_scores = [s["confidence"] for s in stats["samples"]]
    stats["summary"] = {
        "total":    len(all_scores),
        "accepted": n_accepted,
        "rejected": n_rejected,
        "mean_conf":   round(float(np.mean(all_scores)),  4) if all_scores else 0,
        "median_conf": round(float(np.median(all_scores)),4) if all_scores else 0,
        "min_conf":    round(float(np.min(all_scores)),   4) if all_scores else 0,
        "max_conf":    round(float(np.max(all_scores)),   4) if all_scores else 0,
    }

    stats_path = out_root / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"\nDone.  accepted={n_accepted}  rejected={n_rejected}  "
        f"(threshold={args.conf_threshold})\n"
        f"Stats saved to {stats_path}\n"
        f"Pseudo-labels in {accepted_img_dir.parent}"
    )


if __name__ == "__main__":
    main()
