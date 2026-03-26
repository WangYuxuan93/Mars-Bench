"""
DINOv2-based semantic segmentation model for Mars surface image segmentation.

Architecture
------------
DINOv2 ViT backbone (Dinov2Model from HuggingFace) + lightweight linear decode head.

The decode head follows the approach from the DINOv2 paper (linear probe variant):
  1. Extract patch tokens from the last hidden state            [B, N, D]
  2. Reshape to spatial grid                                    [B, D, H/p, W/p]
  3. Linear classifier (1×1 conv) → [B, num_classes, H/p, W/p]
  4. Bilinear upsample to input resolution                      [B, num_classes, H, W]

p = patch_size (14 for all DINOv2 models).

Usage
-----
model_name: dinov2_segmentation
Config:  marsbench/configs/model/segmentation/dinov2_segmentation.yaml
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model

from .BaseSegmentationModel import BaseSegmentationModel

logger = logging.getLogger(__name__)


class _LinearHead(nn.Module):
    """Simple 1×1 conv classification head used on DINOv2 patch tokens.

    input : [B, hidden_size, h, w]   (spatial patch grid)
    output: [B, num_classes, h, w]
    """

    def __init__(self, hidden_size: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(hidden_size, num_classes, kernel_size=1, bias=True)
        nn.init.normal_(self.conv.weight, std=0.01)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class _DINOv2SegModel(nn.Module):
    """DINOv2 backbone + linear decode head wrapped as a single nn.Module.

    Kept as an inner module so BaseSegmentationModel.self.model works as usual.
    """

    def __init__(self, backbone: Dinov2Model, head: _LinearHead,
                 patch_size: int, num_register_tokens: int = 0):
        super().__init__()
        self.backbone             = backbone
        self.head                 = head
        self.patch_size           = patch_size
        self.num_register_tokens  = num_register_tokens  # 0 for standard models, >0 for *-with-registers

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return full-resolution logits [B, C, H, W]."""
        B, _, H, W = pixel_values.shape

        outputs = self.backbone(pixel_values, output_hidden_states=False)
        # last_hidden_state: [B, 1 + num_register_tokens + N, D]
        # Skip CLS token (index 0) and any register tokens (indices 1..num_register_tokens)
        skip = 1 + self.num_register_tokens
        hidden = outputs.last_hidden_state[:, skip:, :]        # [B, N, D]

        # Reshape to spatial: N = (H/p)*(W/p)
        h_pat = H // self.patch_size
        w_pat = W // self.patch_size
        hidden = hidden.permute(0, 2, 1)                       # [B, D, N]
        hidden = hidden.reshape(B, -1, h_pat, w_pat)           # [B, D, h_pat, w_pat]

        logits = self.head(hidden)                             # [B, C, h_pat, w_pat]
        logits = F.interpolate(logits, size=(H, W),
                               mode="bilinear", align_corners=False)
        return logits                                          # [B, C, H, W]


class DINOv2Segmentation(BaseSegmentationModel):
    """Semantic segmentation using a frozen/finetuned DINOv2 backbone.

    Config fields (dinov2_segmentation.yaml):
      model_name    : HF hub name, e.g. "facebook/dinov2-large"
      pretrained    : bool – load DINOv2 pretrained weights
      freeze_layers : bool – freeze backbone, train head only
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------

    def _initialize_model(self) -> nn.Module:
        model_name    = self.cfg.model.model_name
        num_classes   = self.cfg.data.num_classes
        pretrained    = self.cfg.model.pretrained
        freeze_layers = self.cfg.model.freeze_layers

        # --- backbone ---
        if pretrained:
            backbone = Dinov2Model.from_pretrained(model_name)
            logger.info(f"Loaded pretrained DINOv2 backbone from {model_name}")
        else:
            from transformers import Dinov2Config
            config   = Dinov2Config.from_pretrained(model_name)
            backbone = Dinov2Model(config)
            logger.info("Initialised DINOv2 backbone from scratch (random weights).")

        hidden_size          = backbone.config.hidden_size
        patch_size           = backbone.config.patch_size          # always 14 for DINOv2
        num_register_tokens  = getattr(backbone.config, "num_register_tokens", 0)

        logger.info(f"DINOv2 hidden_size={hidden_size}, patch_size={patch_size}, "
                    f"num_register_tokens={num_register_tokens}")

        # --- decode head ---
        head = _LinearHead(hidden_size, num_classes)

        # --- freeze logic ---
        if freeze_layers and not pretrained:
            logger.warning("freeze_layers=True but pretrained=False – "
                           "setting freeze_layers=False")
            freeze_layers = False

        if pretrained and freeze_layers:
            for param in backbone.parameters():
                param.requires_grad = False
            n_frozen = sum(p.numel() for p in backbone.parameters())
            n_train  = sum(p.numel() for p in head.parameters())
            logger.info(f"Froze DINOv2 backbone ({n_frozen/1e6:.1f}M params), "
                        f"head trainable ({n_train/1e6:.1f}M params)")
        else:
            for param in backbone.parameters():
                param.requires_grad = True
            logger.info("All parameters trainable "
                        f"({'pretrained' if pretrained else 'scratch'}).")

        return _DINOv2SegModel(backbone, head, patch_size, num_register_tokens)

    # ------------------------------------------------------------------
    # Forward – delegates to _DINOv2SegModel, returns [B, C, H, W]
    # BaseSegmentationModel.forward() calls self.model(x), which works
    # since _DINOv2SegModel.forward() already returns full-res logits.
    # No override needed – kept here for clarity.
    # ------------------------------------------------------------------

    # predict_step is inherited from BaseSegmentationModel:
    #   x = batch[0]; return self(x).argmax(1)   ← already correct
