"""
Transformers DPT model implementation for Mars surface image segmentation.
Supports loading ViTMAE encoder weights as initialization.
"""

import logging
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision.utils import make_grid
from transformers import (
    DPTForSemanticSegmentation,
    DPTConfig,
    AutoImageProcessor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from .BaseSegmentationModel import BaseSegmentationModel

logger = logging.getLogger(__name__)


# =========================
# Utils
# =========================

def strip_prefix_if_present(state_dict, prefixes=("model.", "module.")):
    out = state_dict
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if all(k.startswith(p) for k in out.keys()):
                out = {k[len(p):]: v for k, v in out.items()}
                changed = True
    return out


def load_vitmae_encoder_into_dpt(model, vitmae_ckpt_path):
    """
    Load HF ViTMAE encoder weights into HF DPT model.
    Only loads encoder-related parameters.
    Skips MAE decoder and position embeddings.
    """

    logger.info(f"Loading ViTMAE weights from: {vitmae_ckpt_path}")
    ckpt = torch.load(vitmae_ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        vit_sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        vit_sd = ckpt
    else:
        raise ValueError("Unsupported checkpoint format.")

    vit_sd = strip_prefix_if_present(vit_sd)

    mapped = {}

    for k, v in vit_sd.items():

        # Skip decoder
        if k.startswith("decoder.") or ".decoder." in k:
            continue

        # Skip MAE position embedding
        if "position_embeddings" in k:
            continue

        new_k = None

        if k.startswith("vit.embeddings.patch_embeddings."):
            new_k = k.replace(
                "vit.embeddings.patch_embeddings.",
                "dpt.embeddings.patch_embeddings.",
                1,
            )

        elif k == "vit.embeddings.cls_token":
            new_k = "dpt.embeddings.cls_token"

        elif k.startswith("vit.encoder."):
            new_k = k.replace("vit.encoder.", "dpt.encoder.", 1)

        elif k.startswith("vit.layernorm."):
            new_k = k.replace("vit.layernorm.", "dpt.layernorm.", 1)

        if new_k is not None:
            mapped[new_k] = v

    missing, unexpected = model.load_state_dict(mapped, strict=False)

    logger.info(f"Loaded {len(mapped)} encoder parameters from ViTMAE")
    logger.info(f"Missing keys (sample): {missing[:20]}")
    logger.info(f"Unexpected keys (sample): {unexpected[:20]}")

    return model


def reinit_module(module):
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


# =========================
# Model
# =========================

class TransformersDPT(BaseSegmentationModel):

    def __init__(self, cfg):
        super().__init__(cfg)

        backbone_id = self.cfg.model.get(
            "hf_backbone",
            "Intel/dpt-large-ade"
        )

        self.image_processor = AutoImageProcessor.from_pretrained(backbone_id)

    def _initialize_model(self):

        num_labels = self.cfg.data.num_classes
        freeze_layers = self.cfg.model.freeze_layers
        use_hf_init = self.cfg.model.pretrained
        vitmae_ckpt = self.cfg.model.get("vitmae_ckpt", None)

        backbone_id = self.cfg.model.get(
            "hf_backbone",
            "Intel/dpt-large-ade"
        )

        # 1️⃣ 先加载HF结构+权重
        model = DPTForSemanticSegmentation.from_pretrained(
            backbone_id,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )

        # 2️⃣ 如果不想用HF head/neck初始化
        if not use_hf_init:
            logger.info("Reinitializing neck and head weights.")
            reinit_module(model.neck)
            reinit_module(model.head)
            if getattr(model, "auxiliary_head", None) is not None:
                reinit_module(model.auxiliary_head)

        # 3️⃣ 覆盖encoder为ViTMAE权重
        if vitmae_ckpt is not None:
            model = load_vitmae_encoder_into_dpt(model, vitmae_ckpt)

        # 4️⃣ 冻结策略
        if freeze_layers:
            logger.info("Freezing DPT backbone (encoder).")
            for p in model.dpt.parameters():
                p.requires_grad = False
        else:
            for p in model.parameters():
                p.requires_grad = True

        return model

    # =========================
    # Forward + Training
    # =========================

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)
    

    def _unpack_batch(self, batch):
        # Your dataloader returns list/tuple: [pixel_values, labels]
        if isinstance(batch, (list, tuple)):
            if len(batch) < 2:
                raise ValueError(f"Expected batch len>=2, got {len(batch)}")
            pixel_values, labels = batch[0], batch[1]
            return pixel_values, labels

        # For completeness (if you later switch datamodule)
        if isinstance(batch, dict):
            pixel_values = batch["pixel_values"]
            labels = batch.get("labels", None)
            if labels is None:
                # MarsBench sometimes uses orig_mask as GT
                labels = batch.get("orig_mask", None)
            return pixel_values, labels

        raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _shared_step(self, batch, batch_idx, prefix):
        pixel_values, labels = self._unpack_batch(batch)

        pixel_values = pixel_values.to(self.device)

        # labels should be (B, H, W) long
        if isinstance(labels, list):
            labels = torch.stack(labels, dim=0)
        labels = labels.to(self.device).long()

        outputs = self.model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        # logits: (B, num_labels, h, w)  -> upsample to labels size
        logits = outputs.logits
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        pred_indices = logits.argmax(dim=1)

        metrics = {f"{prefix}/loss": loss}
        metrics.update(self._calculate_metrics_for_step(prefix, pred_indices, labels))
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        # 如果你要保留可视化（按你 Mask2Former 那套），这里也能接上
        # 注意：images 这里是 pixel_values；GT 是 labels；pred 是 pred_indices

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"].to(self.device)
        outputs = self(pixel_values=pixel_values)
        logits = outputs.logits

        logits = F.interpolate(
            logits,
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )

        preds = logits.argmax(dim=1)
        return preds

    # =========================
    # Visualization
    # =========================

    def _log_visualizations(self, images, masks, preds, prefix):

        num_samples = min(self.max_samples, len(images))
        colormap = plt.cm.get_cmap(
            "tab20",
            self.cfg.data.num_classes,
        )

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

            diff_mask = torch.zeros_like(img)
            correct = mask == pred
            incorrect = mask != pred
            diff_mask[1, correct] = 1.0
            diff_mask[0, incorrect] = 1.0

            alpha = self.overlay_alpha
            diff_overlay = img * (1 - alpha) + diff_mask * alpha

            mask_vis = torch.zeros_like(img)
            pred_vis = torch.zeros_like(img)

            for class_idx in range(self.cfg.data.num_classes):
                color = torch.tensor(colormap(class_idx)[:3])
                mask_vis[:, mask == class_idx] = color.view(3, 1)
                pred_vis[:, pred == class_idx] = color.view(3, 1)

            all_images.extend([img, mask_vis, pred_vis, diff_overlay])

        grid = make_grid(all_images, nrow=4)

        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                f"visualizations/{prefix}",
                [grid],
                step=self.current_epoch,
            )
        elif isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image(
                f"visualizations/{prefix}",
                grid,
                global_step=self.current_epoch,
            )