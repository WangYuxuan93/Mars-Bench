"""
ViTMAE (Semantic Segmentation) model implementation for Mars surface image segmentation.

This replaces Mask2Former with transformers' ViTMAEForSemanticSegmentation,
and loads your MAE-pretrained encoder weights (vit.*) into the segmentation model.
"""

import logging
import os
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt
import torch
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torchvision.utils import make_grid
from transformers import ViTMAEConfig, ViTMAEForSemanticSegmentation, ViTMAEForPreTraining

from .BaseSegmentationModel import BaseSegmentationModel

logger = logging.getLogger(__name__)


def _find_weight_file(ckpt_dir: str) -> str:
    candidates = ["model.safetensors", "pytorch_model.bin"]
    for name in candidates:
        p = os.path.join(ckpt_dir, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Cannot find weights under {ckpt_dir}. Expected one of: {candidates}")


def _load_state_dict(weight_path: str) -> Dict[str, torch.Tensor]:
    if weight_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(weight_path)
    return torch.load(weight_path, map_location="cpu")


def _load_mae_encoder_into_seg_model(
    seg_model: ViTMAEForSemanticSegmentation,
    mae_ckpt_dir: str,
    *,
    skip_position_embeddings: bool = True,
) -> Tuple[int, int]:
    """
    Load MAE-pretrained encoder weights into seg_model.
    - Only loads keys starting with "vit."
    - Optionally skips "position_embeddings" to avoid shape mismatch when image_size differs.

    Returns:
        (num_missing_keys, num_unexpected_keys)
    """
    weight_path = _find_weight_file(mae_ckpt_dir)
    state = _load_state_dict(weight_path)

    filtered = {}
    for k, v in state.items():
        if not k.startswith("vit."):
            continue
        if skip_position_embeddings and "position_embeddings" in k:
            continue
        filtered[k] = v

    missing, unexpected = seg_model.load_state_dict(filtered, strict=False)

    # Log a small summary
    logger.info(
        f"Loaded MAE encoder weights from {mae_ckpt_dir}. "
        f"Loaded keys={len(filtered)}, missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if len(unexpected) > 0:
        logger.debug(f"Unexpected keys (first 20): {unexpected[:20]}")
    if len(missing) > 0:
        logger.debug(f"Missing keys (first 20): {missing[:20]}")

    return len(missing), len(unexpected)


class ViTMAESemSeg(BaseSegmentationModel):
    """
    Expected batch format from your datamodule:
      - pixel_values: FloatTensor (B, C, H, W)
      - labels:      LongTensor  (B, H, W)
      - orig_mask:   list[LongTensor] or Tensor (B, H, W)  (for visualization / metric target)

    Notes:
      - ViTMAE in transformers enforces fixed input size (config.image_size). Ensure your datamodule outputs that size.
      - num_labels should equal cfg.data.num_classes (including background).
    """

    def __init__(self, cfg):
        super(ViTMAESemSeg, self).__init__(cfg)

    def _build_config(self) -> ViTMAEConfig:
        """
        Build segmentation config, starting from your MAE checkpoint config if provided,
        otherwise fall back to a base config.
        """
        # You can store this in cfg.model.pretrained_ckpt or cfg.model.mae_ckpt_dir, etc.
        mae_ckpt_dir = getattr(self.cfg.model, "mae_ckpt_dir", None)

        if mae_ckpt_dir:
            config = ViTMAEConfig.from_pretrained(mae_ckpt_dir)
        else:
            config = ViTMAEConfig()

        # --- Required / common overrides for semantic segmentation ---
        config.num_labels = int(self.cfg.data.num_classes)
        # ignore_index used by loss
        config.semantic_loss_ignore_index = int(self.cfg.training.ignore_index)

        # IMPORTANT: ViTMAE patch embeddings hard-check input H/W == config.image_size
        # Use cfg.data.image_size if you have it; otherwise assume 512 like your current pipeline.
        image_size = int(getattr(self.cfg.data, "image_size", 512))
        config.image_size = image_size

        # Must be 4 indices for this implementation
        # Base default: [3, 5, 7, 11] (works for 12-layer encoder).
        # If your encoder has fewer layers (tiny/small), adjust accordingly.
        out_indices = getattr(self.cfg.model, "out_indices", None)
        if out_indices is None:
            config.out_indices = [3, 5, 7, 11]
        else:
            if len(out_indices) != 4:
                raise ValueError("cfg.model.out_indices must have 4 integers for ViTMAEForSemanticSegmentation.")
            config.out_indices = list(out_indices)

        # UPerHead / auxiliary head knobs (make sure they exist)
        if not hasattr(config, "pool_scales"):
            config.pool_scales = (1, 2, 3, 6)

        use_aux = bool(getattr(self.cfg.model, "use_auxiliary_head", False))
        config.use_auxiliary_head = use_aux

        if not hasattr(config, "auxiliary_channels"):
            config.auxiliary_channels = config.hidden_size
        if not hasattr(config, "auxiliary_num_convs"):
            config.auxiliary_num_convs = 1
        if not hasattr(config, "auxiliary_concat_input"):
            config.auxiliary_concat_input = False
        if not hasattr(config, "auxiliary_loss_weight"):
            config.auxiliary_loss_weight = 0.4

        return config

    def _initialize_model(self):
        pretrained = bool(getattr(self.cfg.model, "pretrained", False))
        freeze_layers = bool(getattr(self.cfg.model, "freeze_layers", False))

        # Where your MAE pretraining output is saved (must contain config.json + weights)
        mae_ckpt_dir = getattr(self.cfg.model, "mae_ckpt_dir", None)

        config = self._build_config()
        model = ViTMAEForSemanticSegmentation(config)

        # Load MAE-pretrained encoder weights into seg model
        if not mae_ckpt_dir:
            raise ValueError(
                "cfg.model.pretrained=True but cfg.model.mae_ckpt_dir is not set. "
                "Please set it to your MAE pretraining output directory."
            )

        # If you kept the SAME image_size between pretrain and finetune, you can set skip_position_embeddings=False.
        # If you changed image_size, keep it True to avoid shape mismatch.
        skip_pos = bool(getattr(self.cfg.model, "skip_position_embeddings", True))
        _load_mae_encoder_into_seg_model(model, mae_ckpt_dir, skip_position_embeddings=skip_pos)

        # Handle layer freezing
        if freeze_layers and not pretrained:
            logger.warning("freeze_layers=True but pretrained=False. Setting freeze_layers=False.")
            freeze_layers = False

        if pretrained and freeze_layers:
            # Freeze only the encoder (vit.*); keep seg heads trainable
            for p in model.vit.parameters():
                p.requires_grad = False
            for p in model.decode_head.parameters():
                p.requires_grad = True
            if model.auxiliary_head is not None:
                for p in model.auxiliary_head.parameters():
                    p.requires_grad = True
            logger.info("Froze ViTMAE encoder (vit.*), kept decode_head (+ auxiliary_head) trainable.")
        else:
            for p in model.parameters():
                p.requires_grad = True
            if pretrained:
                logger.info("Using MAE-pretrained encoder weights with all layers trainable.")
            else:
                logger.info("Training ViTMAE segmentation from scratch with all layers trainable.")

        return model

    def forward(self, pixel_values, labels=None):
        # ViTMAEForSemanticSegmentation expects pixel_values + optional labels
        return self.model(pixel_values=pixel_values, labels=labels)

    def _shared_step(self, batch, batch_idx, prefix):
        pixel_values = batch["pixel_values"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = self(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        logits = outputs.logits  # (B, num_labels, H', W')
        preds = logits.argmax(dim=1)

        # Targets for metrics: prefer orig_mask if your pipeline keeps it
        if "orig_mask" in batch:
            target_masks = torch.stack(batch["orig_mask"], dim=0).to(self.device).long()
        else:
            target_masks = labels

        metrics = {f"{prefix}/loss": loss}
        metrics.update(self._calculate_metrics_for_step(prefix, preds, target_masks))
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        if (
            self.log_images_every_n_epochs is not None
            and batch_idx == 0
            and self.trainer.is_global_zero
            and self.current_epoch % self.log_images_every_n_epochs == 0
        ):
            # for visualization, use target_masks for GT
            self._log_visualizations(batch["pixel_values"], target_masks, preds, prefix)

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        images = batch["pixel_values"].to(self.device)
        outputs = self(pixel_values=images, labels=None)
        preds = outputs.logits.argmax(dim=1)
        return preds

    def _log_visualizations(self, images, masks, preds, prefix):
        """
        Create and log visualization images.
        """
        num_samples = min(self.max_samples, len(images))
        colormap = plt.cm.get_cmap("tab20", self.cfg.data.num_classes)

        all_images = []
        for i in range(num_samples):
            img = images[i].cpu()
            mask = masks[i].cpu()
            pred = preds[i].cpu()

            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)

            m = torch.tensor(self.cfg.transforms.rgb.mean).view(3, 1, 1)
            s = torch.tensor(self.cfg.transforms.rgb.std).view(3, 1, 1)
            img = img * s + m
            img = img.clamp(0, 1)

            # diff overlay
            diff_mask = torch.zeros((3, mask.shape[0], mask.shape[1]), dtype=torch.float32)
            correct = mask == pred
            diff_mask[1, correct] = 1.0
            incorrect = mask != pred
            diff_mask[0, incorrect] = 1.0
            alpha = self.overlay_alpha
            diff_overlay = img * (1 - alpha) + diff_mask * alpha

            # class color maps
            mask_vis = torch.zeros((3, mask.shape[0], mask.shape[1]), dtype=torch.float32)
            pred_vis = torch.zeros((3, pred.shape[0], pred.shape[1]), dtype=torch.float32)
            for class_idx in range(self.cfg.data.num_classes):
                color = torch.tensor(colormap(class_idx)[:3], dtype=torch.float32)
                mask_locations = mask == class_idx
                pred_locations = pred == class_idx
                for c in range(3):
                    mask_vis[c][mask_locations] = color[c]
                    pred_vis[c][pred_locations] = color[c]

            all_images.extend([img, mask_vis, pred_vis, diff_overlay])

        grid = make_grid(all_images, nrow=4)
        caption = [f"Epoch {self.current_epoch}: Original | Ground Truth | Prediction | Diff (green=correct, red=error)"]

        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                f"visualizations/{prefix}",
                [grid],
                caption=caption,
                step=self.current_epoch,
            )
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                f"visualizations/{prefix}",
                grid,
                global_step=self.current_epoch,
            )
        else:
            logger.warning(f"Logger {self.logger} does not support image logging.")