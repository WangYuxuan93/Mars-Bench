"""
Mask2Former model implementation for Mars surface image segmentation.
"""

import glob
import logging

import torch
from transformers import Mask2FormerForUniversalSegmentation
from transformers import Mask2FormerImageProcessor

from .BaseSegmentationModel import BaseSegmentationModel

logger = logging.getLogger(__name__)


class Mask2Former(BaseSegmentationModel):
    def __init__(self, cfg):
        super(Mask2Former, self).__init__(cfg)

        self.image_processor = Mask2FormerImageProcessor(
            ignore_index=self.cfg.training.ignore_index,
            reduce_labels=False,
        )

    # ------------------------------------------------------------------
    # Backbone checkpoint loading (SwinForMaskedImageModeling / MIMWithDistill)
    # ------------------------------------------------------------------
    @staticmethod
    def _load_swin_backbone(model, backbone_ckpt: str):
        """Replace Mask2Former's Swin encoder with weights from a Swin checkpoint.

        Supports:
        - HuggingFace model id       (e.g. "microsoft/swin-large-patch4-window12-384")
        - MIMWithDistill checkpoint  (keys start with "student.swin.")
        - SwinForMaskedImageModeling (keys start with "swin.")
        - Bare Swin state dict       (keys have no prefix)
        - Local directory            (first .safetensors / pytorch_model.bin used)
        """
        import os

        encoder = model.model.pixel_level_module.encoder
        backbone = encoder.model if hasattr(encoder, "model") else encoder

        # ---- HuggingFace model id (string that is not a local path) ----
        if not os.path.exists(backbone_ckpt):
            logger.info(f"Loading Swin encoder from HuggingFace: {backbone_ckpt}")
            # Use the same class as the backbone to ensure key format matches.
            # SwinBackbone uses "encoder.stages.*"; SwinModel uses "encoder.layers.*".
            hf_swin = backbone.__class__.from_pretrained(backbone_ckpt)
            result = backbone.load_state_dict(hf_swin.state_dict(), strict=False)
            logger.info(
                f"Loaded Swin encoder from HuggingFace '{backbone_ckpt}'\n"
                f"  missing : {result.missing_keys[:10]}\n"
                f"  unexpected: {result.unexpected_keys[:10]}"
            )
            return

        # ---- Local directory → find weight file ----
        if os.path.isdir(backbone_ckpt):
            candidates = (
                glob.glob(os.path.join(backbone_ckpt, "model.safetensors"))
                + glob.glob(os.path.join(backbone_ckpt, "pytorch_model.bin"))
            )
            if not candidates:
                raise FileNotFoundError(f"No weight file found in {backbone_ckpt}")
            backbone_ckpt = candidates[0]

        if backbone_ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file
            raw_sd = load_file(backbone_ckpt, device="cpu")
        else:
            raw_sd = torch.load(backbone_ckpt, map_location="cpu", weights_only=True)

        # Strip wrapper prefixes: MIMWithDistill → student.swin.xxx, MIM → swin.xxx
        if any(k.startswith("student.") for k in raw_sd):
            raw_sd = {k[len("student."):]: v for k, v in raw_sd.items()
                      if k.startswith("student.")}
        if any(k.startswith("swin.") for k in raw_sd):
            raw_sd = {k[len("swin."):]: v for k, v in raw_sd.items()
                      if k.startswith("swin.")}

        if not raw_sd:
            raise ValueError("Empty state dict after stripping prefix – check checkpoint format.")

        result = backbone.load_state_dict(raw_sd, strict=False)
        logger.info(
            f"Loaded Swin encoder from {backbone_ckpt}\n"
            f"  missing : {result.missing_keys[:10]}\n"
            f"  unexpected: {result.unexpected_keys[:10]}"
        )

    def _initialize_model(self):
        pretrained = self.cfg.model.pretrained
        freeze_layers = self.cfg.model.freeze_layers
        model_name = self.cfg.model.get("model_name", None)

        backbone_ckpt = self.cfg.model.get("backbone_checkpoint", None)

        if model_name:
            logger.info(
                f"[Decoder] Pretrained from HuggingFace: {model_name}\n"
                f"[Encoder] {'Will be replaced by: ' + backbone_ckpt if backbone_ckpt else 'Using encoder bundled with ' + model_name}"
            )
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name,
                num_labels=self.cfg.data.num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            # Random decoder: derive architecture config from backbone to ensure dim alignment
            _swin_to_m2f = {
                "microsoft/swin-large-patch4-window12-384": "facebook/mask2former-swin-large-ade-semantic",
                "microsoft/swin-base-patch4-window12-384":  "facebook/mask2former-swin-base-ade-semantic",
                "microsoft/swin-tiny-patch4-window7-224":   "facebook/mask2former-swin-tiny-ade-semantic",
                "microsoft/swin-small-patch4-window7-224":  "facebook/mask2former-swin-small-ade-semantic",
            }
            from transformers import Mask2FormerConfig
            # ref_model: explicit config source (yaml field) > lookup by backbone_ckpt > fallback
            explicit_ref = self.cfg.model.get("ref_model", None)
            ref_model = (
                explicit_ref
                or (_swin_to_m2f.get(backbone_ckpt, None) if backbone_ckpt else None)
            )
            if ref_model:
                config = Mask2FormerConfig.from_pretrained(ref_model)
                logger.info(
                    f"[Decoder] Random init (architecture config borrowed from {ref_model})\n"
                    f"[Encoder] Will be replaced by: {backbone_ckpt}"
                )
            else:
                config = Mask2FormerConfig()
                logger.info(
                    f"[Decoder] Random init (default swin-base architecture config)\n"
                    f"[Encoder] {'Will be replaced by: ' + backbone_ckpt if backbone_ckpt else 'Random init'}"
                )
            config.num_labels = self.cfg.data.num_classes
            model = Mask2FormerForUniversalSegmentation(config)
            pretrained = False

        if backbone_ckpt:
            self._load_swin_backbone(model, backbone_ckpt)

        if freeze_layers and not pretrained:
            logger.warning("freeze_layers is True but model is not pretrained – setting freeze_layers to False")
            freeze_layers = False

        if pretrained and freeze_layers:
            for param in model.model.pixel_level_module.encoder.parameters():
                param.requires_grad = False
            for param in model.model.pixel_level_module.decoder.parameters():
                param.requires_grad = True
            for param in model.model.transformer_module.parameters():
                param.requires_grad = True
            logger.info("Froze pixel-level encoder, keeping decoder and transformer trainable")
        else:
            for param in model.model.parameters():
                param.requires_grad = True
            logger.info("Using pretrained weights with all layers trainable" if pretrained else
                        "Training from scratch with all layers trainable")

        return model

    def forward(self, pixel_values, mask_labels=None, class_labels=None, pixel_mask=None):
        return self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
            pixel_mask=pixel_mask,
        )

    def _shared_step(self, batch, batch_idx, prefix):
        pixel_values = batch["pixel_values"].to(self.device)
        mask_labels = [m.to(self.device) for m in batch["mask_labels"]]
        class_labels = [c.to(self.device) for c in batch["class_labels"]]
        pixel_mask = batch["pixel_mask"].to(self.device)

        outputs = self(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
            pixel_mask=pixel_mask,
        )

        loss = outputs.loss

        target_masks = torch.stack(batch["orig_mask"], dim=0).to(self.device).long()
        h, w = target_masks.shape[-2:]
        target_sizes = [(h, w)] * target_masks.shape[0]
        pred_indices = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )
        pred_indices = torch.stack(pred_indices).to(self.device)

        # update base-class MetricCollection
        metrics = getattr(self, f"{prefix}_metrics")
        metrics.update(pred_indices, target_masks)

        self.log(f"{prefix}/loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        # store samples for visualisation (uses base-class _store_vis / _log_vis_grid)
        if (self.current_epoch % self.vis_every == 0
                or self.current_epoch == self.trainer.max_epochs - 1):
            self._store_vis(prefix, pixel_values, target_masks, pred_indices)

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        images = batch["pixel_values"].to(self.device)
        # no labels needed for inference – call self.model directly
        outputs = self.model(pixel_values=images)
        h, w = images.shape[-2:]
        target_sizes = [(h, w)] * images.shape[0]
        pred_indices = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )
        return torch.stack(pred_indices).to(self.device)
