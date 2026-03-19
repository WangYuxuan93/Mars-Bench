"""
Mask2Former model implementation for Mars surface image segmentation.
"""

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
            ignore_index=255,   # standard Mask2Former ignore value after reduce_labels
            reduce_labels=True, # shift labels: 0(bg)→255(ignored), 1(cone)→0(foreground)
        )

    def _initialize_model(self):
        pretrained = self.cfg.model.pretrained
        freeze_layers = self.cfg.model.freeze_layers

        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.cfg.model.model_name,
            num_labels=self.cfg.data.num_classes - 1,
            ignore_mismatched_sizes=True,
        )

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

        target_sizes = [(512, 512)] * len(batch["orig_mask"])
        pred_indices = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )
        pred_indices = torch.stack(pred_indices).to(self.device)
        target_masks = torch.stack(batch["orig_mask"], dim=0).to(self.device).long()

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
        target_sizes = [(512, 512)] * images.shape[0]
        pred_indices = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )
        return torch.stack(pred_indices).to(self.device)
