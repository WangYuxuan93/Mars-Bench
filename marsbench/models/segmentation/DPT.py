"""
DPT model implementation for Mars surface image segmentation.
"""

import logging
import torch

import segmentation_models_pytorch as smp

from .BaseSegmentationModel import BaseSegmentationModel

logger = logging.getLogger(__name__)


class DPT(BaseSegmentationModel):
    def __init__(self, cfg):
        super(DPT, self).__init__(cfg)

    def _initialize_model(self):
        """Initialize DPT model with configuration parameters."""
        in_channels = self._get_in_channels()
        num_classes = self.cfg.data.num_classes
        encoder_name = self.cfg.model.get("encoder_name", "tu-vit_base_patch16_224.augreg_in21k")
        pretrained = self.cfg.model.pretrained
        freeze_layers = self.cfg.model.freeze_layers

        # Set encoder weights based on pretrained flag
        encoder_weights = "imagenet" if pretrained else None

        model = smp.DPT(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )

        ckpt = self.cfg.model.get("encoder_checkpoint_path", None)
        if ckpt:
            sd = torch.load(ckpt, map_location="cpu")
            # 关键：TimmViTEncoder 里 timm 模型在 model.encoder.model
            model.encoder.model.load_state_dict(sd, strict=False)
            logger.info(f"Loaded MAE-pretrained encoder weights from: {ckpt}")
            # ---- sanity checks (a few tensors) ----
            enc = model.encoder.model
            with torch.no_grad():
                pe = enc.patch_embed.proj.weight
                logger.info(f"[ENC] patch_embed.proj.weight mean={pe.mean().item():.6f} std={pe.std().item():.6f}")

                qkv_w = enc.blocks[0].attn.qkv.weight
                logger.info(f"[ENC] blocks.0.attn.qkv.weight mean={qkv_w.mean().item():.6f} std={qkv_w.std().item():.6f}")

                n1 = enc.blocks[0].norm1.weight
                logger.info(f"[ENC] blocks.0.norm1.weight mean={n1.mean().item():.6f} std={n1.std().item():.6f}")

                mlp1 = enc.blocks[0].mlp.fc1.weight
                logger.info(f"[ENC] blocks.0.mlp.fc1.weight mean={mlp1.mean().item():.6f} std={mlp1.std().item():.6f}")

        # Handle layer freezing
        if freeze_layers and not pretrained:
            logger.warning("freeze_layers is set to True but model is not pretrained. Setting freeze_layers to False")
            freeze_layers = False

        if pretrained and freeze_layers:
            # Freeze encoder layers
            for param in model.encoder.parameters():
                param.requires_grad = False
            # Keep decoder trainable for fine-tuning
            for param in model.decoder.parameters():
                param.requires_grad = True
            for param in model.segmentation_head.parameters():
                param.requires_grad = True
            logger.info("Froze encoder layers, keeping decoder and segmentation head trainable")
        else:
            # Make all layers trainable
            for param in model.parameters():
                param.requires_grad = True
            if pretrained:
                logger.info("Using pretrained weights with all layers trainable")
            else:
                logger.info("Training from scratch with all layers trainable")

        return model
