"""
Multi-task Mask2Former: shared backbone + per-task projection layers and classification heads.

Architecture
------------
- Shared : Mask2FormerModel  (SwinBackbone + PixelDecoder + TransformerDecoder)
- Per-task: projection layer  → Conv(in,64,3,1) + BN + ReLU + Conv(64,in,1)
            class_predictor   → nn.Linear(hidden_dim, num_classes + 1)
            Mask2FormerLoss   → with task-specific num_labels
            MetricCollection  → with task-specific num_classes

Training
--------
- AlternatingTaskSampler ensures every batch contains samples from exactly one task.
- Task is selected uniformly at random per batch.
- Only the shared backbone, the active task's projection layer, and the active task's
  head receive gradients for that batch.
- Validation runs one DataLoader per task; metrics are logged independently.

Usage
-----
Replace MarsDataModule with MultiTaskDataModule in your training script:

    from marsbench.models.segmentation.multitask_mask2former import (
        MultitaskMask2Former, MultiTaskDataModule
    )
    cfg = ...  # Hydra config with cfg.model.tasks list
    model = MultitaskMask2Former(cfg)
    dm    = MultiTaskDataModule(cfg)
    trainer.fit(model, dm)
"""

from __future__ import annotations

import itertools
import logging
import os
import random
from copy import deepcopy
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Sampler,
)
from torchmetrics import Accuracy, MetricCollection, Precision, Recall
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)

from marsbench.data import DATASET_REGISTRY
from marsbench.data.segmentation.Mask2FormerWrapper import Mask2FormerWrapper
from marsbench.utils.transforms import get_transforms

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers: projection layer, metrics, loss
# ---------------------------------------------------------------------------

def _build_proj_layer(in_channels: int) -> nn.Sequential:
    """Two-layer task projection that preserves (C, H, W)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, in_channels, kernel_size=1),
    )


def _build_metrics(num_classes: int, weight_type: str = "square") -> MetricCollection:
    return MetricCollection(
        {
            "acc":  Accuracy(task="multiclass", num_classes=num_classes, average=None),
            "prec": Precision(task="multiclass", num_classes=num_classes, average=None),
            "rec":  Recall(task="multiclass", num_classes=num_classes, average=None),
            "dice": GeneralizedDiceScore(
                num_classes=num_classes, weight_type=weight_type,
                per_class=True, input_format="index",
            ),
            "iou":  MeanIoU(num_classes=num_classes, per_class=True, input_format="index"),
        },
        compute_groups=False,
    )


def _build_mask2former_loss(config: Mask2FormerConfig):
    """Construct a Mask2FormerLoss with weight_dict matching the original forward()."""
    from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

    weight_dict: Dict[str, float] = {
        "loss_cross_entropy": config.class_weight,
        "loss_mask":          config.mask_weight,
        "loss_dice":          config.dice_weight,
    }
    if config.use_auxiliary_loss:
        for i in range(config.decoder_layers - 1):
            weight_dict[f"loss_cross_entropy_{i}"] = config.class_weight
            weight_dict[f"loss_mask_{i}"]          = config.mask_weight
            weight_dict[f"loss_dice_{i}"]          = config.dice_weight
    return Mask2FormerLoss(config, weight_dict)


# ---------------------------------------------------------------------------
# Data: task-tagged dataset + alternating-task batch sampler
# ---------------------------------------------------------------------------

class TaskTaggedDataset(Dataset):
    """Wraps a Mask2FormerWrapper dataset and appends task_id to each sample."""

    def __init__(self, wrapped: Mask2FormerWrapper, task_id: str):
        self.wrapped = wrapped
        self.task_id = task_id

    def __len__(self) -> int:
        return len(self.wrapped)

    def __getitem__(self, idx: int):
        # Mask2FormerWrapper returns (image, mask, orig_image, orig_mask)
        image, mask, orig_image, orig_mask = self.wrapped[idx]
        return image, mask, orig_image, orig_mask, self.task_id


class AlternatingTaskSampler(Sampler):
    """
    Batch sampler that yields batches from exactly one task per iteration.

    Tasks are chosen uniformly at random. Within a task the samples are
    shuffled at the start of each epoch. Each task contributes
    floor(task_size / batch_size) batches; remaining samples are dropped.
    """

    def __init__(self, task_sizes: List[int], batch_size: int):
        self.task_sizes  = task_sizes
        self.batch_size  = batch_size
        # Offsets into the ConcatDataset's global index space
        self.offsets     = list(itertools.accumulate([0] + task_sizes))
        self._n_batches  = sum(s // batch_size for s in task_sizes)

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        # Per-task shuffled indices (local → global)
        task_indices = [
            (torch.randperm(size) + offset).tolist()
            for size, offset in zip(self.task_sizes, self.offsets)
        ]
        # Build a list of (task_id, start_in_task) pairs, one per batch
        batch_list: List[Tuple[int, int]] = []
        for t, size in enumerate(self.task_sizes):
            n_batches = size // self.batch_size
            for b in range(n_batches):
                batch_list.append((t, b * self.batch_size))
        random.shuffle(batch_list)

        for task_id, start in batch_list:
            yield task_indices[task_id][start : start + self.batch_size]


# ---------------------------------------------------------------------------
# Multi-task DataModule
# ---------------------------------------------------------------------------

class MultiTaskDataModule(pl.LightningDataModule):
    """
    Loads one segmentation dataset per task.

    train  : ConcatDataset of all tasks, served via AlternatingTaskSampler
             (each batch is pure single-task).
    val    : list of per-task DataLoaders (one per task).
    test   : list of per-task DataLoaders (one per task).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg        = cfg
        self.task_cfgs  = list(cfg.model.tasks)
        self.batch_size = cfg.training.batch_size
        self.task_names = [t.name for t in self.task_cfgs]

        sys_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) \
            if "SLURM_JOB_ID" in os.environ else (os.cpu_count() or 2) // 2
        req_workers = int(cfg.training.get("num_workers", -1))
        self.num_workers = min(sys_workers, req_workers) if req_workers >= 0 else sys_workers

        self.image_processor = Mask2FormerImageProcessor(
            ignore_index=cfg.training.ignore_index,
            reduce_labels=False,
        )

    # ------------------------------------------------------------------ setup

    def setup(self, stage=None):
        train_tagged, val_tagged, test_tagged = [], [], []

        for task_cfg in self.task_cfgs:
            # Build a per-task cfg by merging task data config over global config.
            # Convert to plain dict first to drop Hydra's struct flag (which forbids new keys
            # like load_from_hf / repo_id that aren't in the global data schema).
            base_dict = OmegaConf.to_container(self.cfg, resolve=False, throw_on_missing=False)
            task_data_dict = OmegaConf.to_container(task_cfg.data, resolve=True)
            base_dict["data"] = {**base_dict.get("data", {}), **task_data_dict}
            per_cfg = OmegaConf.create(base_dict)

            transforms = get_transforms(per_cfg)
            use_hf = task_cfg.data.get("load_from_hf", False)

            if use_hf:
                # HuggingFace dataset path
                from marsbench.data.segmentation.HFSegmentation import HFSegmentation
                repo_id = task_cfg.data.get("repo_id", "")
                if not repo_id:
                    raise ValueError(f"Task '{task_cfg.name}': load_from_hf=true but repo_id is empty")

                def _make(split, t=task_cfg, tc=per_cfg, tf=transforms, rid=repo_id):
                    base_ds = HFSegmentation(
                        cfg=tc,
                        repo_id=rid,
                        transform=tf[0] if split == "train" else tf[1],
                        split=split,
                    )
                    return TaskTaggedDataset(Mask2FormerWrapper(base_ds), t.name)
            else:
                # Local dataset path
                dataset_cls = DATASET_REGISTRY["segmentation"][task_cfg.data.name]

                def _make(split, t=task_cfg, tc=per_cfg, tf=transforms, dc=dataset_cls):
                    annot_csv = t.data.get("annot_csv", None)
                    kwargs = dict(
                        cfg=tc,
                        data_dir=t.data.data_dir,
                        transform=tf[0] if split == "train" else tf[1],
                        split=split,
                    )
                    import inspect
                    if "annot_csv" in inspect.signature(dc.__init__).parameters:
                        kwargs["annot_csv"] = annot_csv
                    base_ds = dc(**kwargs)
                    return TaskTaggedDataset(Mask2FormerWrapper(base_ds), t.name)

            train_tagged.append(_make("train"))
            val_tagged.append(_make("val"))
            test_tagged.append(_make("test"))

        self.train_concat    = ConcatDataset(train_tagged)
        self._train_sizes    = [len(d) for d in train_tagged]
        self.val_datasets    = val_tagged
        self.test_datasets   = test_tagged

    # -------------------------------------------------------------- collate

    def _collate(self, batch):
        images, seg_maps, orig_images, orig_masks, task_ids = zip(*batch)
        processed = self.image_processor(
            list(images),
            segmentation_maps=list(seg_maps),
            return_tensors="pt",
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
        )
        processed["orig_image"] = list(orig_images)
        processed["orig_mask"]  = list(orig_masks)
        processed["task_ids"]   = list(task_ids)
        return processed

    # ----------------------------------------------------------- dataloaders

    def train_dataloader(self) -> DataLoader:
        sampler = AlternatingTaskSampler(self._train_sizes, self.batch_size)
        return DataLoader(
            self.train_concat,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self._collate,
            )
            for ds in self.val_datasets
        ]

    def test_dataloader(self) -> List[DataLoader]:
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self._collate,
            )
            for ds in self.test_datasets
        ]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultitaskMask2Former(pl.LightningModule):
    """
    Multi-task Mask2Former with shared backbone and per-task projection + head.

    Config layout (cfg.model)
    -------------------------
    model_name       : str    HuggingFace model id for the pretrained Mask2Former
    in_channels      : int    Input channels (default 3)
    freeze_backbone  : bool   Freeze Swin encoder (default false)
    backbone_checkpoint: str  Optional MIM checkpoint path (same format as Mask2Former)
    tasks:                    List of task configs, each with:
      - name         : str    Unique task identifier (used as dict key / log prefix)
        num_classes  : int    Number of semantic classes for this task
        data         : dict   Data config (name, data_dir, image_type, num_classes, …)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg        = cfg
        self.task_cfgs  = list(cfg.model.tasks)
        self.task_names = [t.name for t in self.task_cfgs]
        self.ignore_index = cfg.training.get("ignore_index", -100)

        # ------------------------------------------------------------------
        # 1. Load pretrained Mask2Former; extract the shared Mask2FormerModel
        # ------------------------------------------------------------------
        logger.info(f"Loading pretrained Mask2Former: {cfg.model.model_name}")
        base_m2f = Mask2FormerForUniversalSegmentation.from_pretrained(
            cfg.model.model_name,
            num_labels=self.task_cfgs[0].num_classes,   # arbitrary; head is replaced
            ignore_mismatched_sizes=True,
        )
        # shared_model: Mask2FormerModel (backbone + pixel dec + transformer dec)
        self.shared_model: nn.Module = base_m2f.model
        base_config: Mask2FormerConfig = base_m2f.config
        hidden_dim: int = base_config.hidden_dim

        # Optional MIM backbone weights (reuse Mask2Former's static helper logic)
        backbone_ckpt = cfg.model.get("backbone_checkpoint", None)
        if backbone_ckpt:
            self._load_swin_backbone(base_m2f, backbone_ckpt)

        # Optional backbone freeze
        if cfg.model.get("freeze_backbone", False):
            for p in self.shared_model.pixel_level_module.encoder.parameters():
                p.requires_grad = False
            logger.info("Swin encoder frozen.")

        # ------------------------------------------------------------------
        # 2. Per-task components
        # ------------------------------------------------------------------
        in_ch = cfg.model.get("in_channels", 3)

        self.proj_layers: nn.ModuleDict = nn.ModuleDict({
            t.name: _build_proj_layer(in_ch) for t in self.task_cfgs
        })

        self.task_heads: nn.ModuleDict = nn.ModuleDict({
            t.name: nn.Linear(hidden_dim, t.num_classes + 1) for t in self.task_cfgs
        })

        # Per-task Mask2FormerLoss (Hungarian matching, no learnable params)
        task_criteria: Dict[str, nn.Module] = {}
        for t in self.task_cfgs:
            task_cfg_copy = deepcopy(base_config)
            task_cfg_copy.num_labels = t.num_classes
            task_criteria[t.name] = _build_mask2former_loss(task_cfg_copy)
        self.task_criteria: nn.ModuleDict = nn.ModuleDict(task_criteria)

        # ------------------------------------------------------------------
        # 3. Image processor (shared, stateless)
        # ------------------------------------------------------------------
        self.image_processor = Mask2FormerImageProcessor(
            ignore_index=self.ignore_index,
            reduce_labels=False,
        )

        # ------------------------------------------------------------------
        # 4. Per-task metrics  (train / val / test × num_tasks)
        # ------------------------------------------------------------------
        weight_type = cfg.training.criterion.get("weight_type", "square")
        self.train_metrics: nn.ModuleDict = nn.ModuleDict({
            t.name: _build_metrics(t.num_classes, weight_type).clone(prefix=f"train_{t.name}/")
            for t in self.task_cfgs
        })
        self.val_metrics: nn.ModuleDict = nn.ModuleDict({
            t.name: _build_metrics(t.num_classes, weight_type).clone(prefix=f"val_{t.name}/")
            for t in self.task_cfgs
        })
        self.test_metrics: nn.ModuleDict = nn.ModuleDict({
            t.name: _build_metrics(t.num_classes, weight_type).clone(prefix=f"test_{t.name}/")
            for t in self.task_cfgs
        })

        self.test_results: Dict[str, float] = {}
        self.save_hyperparameters(cfg)

    # ------------------------------------------------------------------
    # Backbone loading (mirrors Mask2Former._load_swin_backbone)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_swin_backbone(model: Mask2FormerForUniversalSegmentation, ckpt_path: str):
        import glob as _glob

        if os.path.isdir(ckpt_path):
            candidates = (
                _glob.glob(os.path.join(ckpt_path, "model.safetensors"))
                + _glob.glob(os.path.join(ckpt_path, "pytorch_model.bin"))
            )
            if not candidates:
                raise FileNotFoundError(f"No weight file in {ckpt_path}")
            ckpt_path = candidates[0]

        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            raw_sd = load_file(ckpt_path, device="cpu")
        else:
            raw_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        if any(k.startswith("student.") for k in raw_sd):
            raw_sd = {k[len("student."):]: v for k, v in raw_sd.items()
                      if k.startswith("student.")}

        swin_sd = {k[len("swin."):]: v for k, v in raw_sd.items() if k.startswith("swin.")}
        if not swin_sd:
            raise ValueError("No 'swin.*' keys found — not a Swin MIM checkpoint?")

        encoder = model.model.pixel_level_module.encoder
        backbone = encoder.model if hasattr(encoder, "model") else encoder
        result = backbone.load_state_dict(swin_sd, strict=False)
        logger.info(
            f"Loaded Swin backbone from {ckpt_path}\n"
            f"  missing:    {result.missing_keys[:10]}\n"
            f"  unexpected: {result.unexpected_keys[:10]}"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _forward_task(
        self,
        pixel_values: torch.Tensor,
        task_id: str,
        pixel_mask: Optional[torch.Tensor] = None,
        mask_labels: Optional[List[torch.Tensor]] = None,
        class_labels: Optional[List[torch.Tensor]] = None,
    ) -> SimpleNamespace:
        """
        1. Task-specific projection
        2. Shared Mask2FormerModel forward
        3. Task-specific class prediction
        4. Task-specific loss (if labels provided)
        """
        # -- projection --
        projected = self.proj_layers[task_id](pixel_values)

        # -- shared backbone --
        model_out = self.shared_model(pixel_values=projected, pixel_mask=pixel_mask)
        # transformer_decoder_last_hidden_state: [B, num_queries, hidden_dim]
        seq_out = model_out.transformer_decoder_last_hidden_state

        # -- task head --
        class_queries_logits = self.task_heads[task_id](seq_out)  # [B, Q, C+1]
        # In this version of transformers, masks_queries_logits is a tuple of per-layer
        # predictions (one tensor per decoder layer). Take the last (final) layer only.
        _mql = model_out.masks_queries_logits
        masks_queries_logits = _mql[-1] if isinstance(_mql, tuple) else _mql  # [B, Q, H/4, W/4]

        # -- loss --
        loss = None
        if mask_labels is not None and class_labels is not None:
            loss_dict = self.task_criteria[task_id](
                masks_queries_logits=masks_queries_logits,
                class_queries_logits=class_queries_logits,
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=None,   # skip aux loss for simplicity
            )
            # Weight and sum exactly as Mask2FormerForUniversalSegmentation does
            weight_dict = self.task_criteria[task_id].weight_dict
            loss = sum(
                loss_dict[k] * w
                for k, w in weight_dict.items()
                if k in loss_dict
            )

        # Return a namespace that post_process_semantic_segmentation understands
        return SimpleNamespace(
            loss=loss,
            class_queries_logits=class_queries_logits,
            masks_queries_logits=masks_queries_logits,
        )

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _shared_step(
        self,
        batch: dict,
        phase: str,
        task_id: str,
        metric_coll: MetricCollection,
    ) -> torch.Tensor:
        pixel_values  = batch["pixel_values"].to(self.device)
        pixel_mask    = batch["pixel_mask"].to(self.device)
        mask_labels   = [m.to(self.device) for m in batch["mask_labels"]]
        class_labels  = [c.to(self.device) for c in batch["class_labels"]]

        outputs = self._forward_task(
            pixel_values, task_id, pixel_mask, mask_labels, class_labels
        )
        loss = outputs.loss

        # Post-process → dense predictions for metrics
        target_sizes = [(512, 512)] * len(batch["orig_mask"])
        preds = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )
        preds   = torch.stack(preds).to(self.device)
        targets = torch.stack(batch["orig_mask"], dim=0).to(self.device).long()

        metric_coll.update(preds, targets)

        self.log(
            f"{phase}_{task_id}/loss", loss,
            on_step=(phase == "train"), on_epoch=True,
            sync_dist=True, prog_bar=True,
            add_dataloader_idx=False,
        )
        return loss

    # ------------------------------------------------------------------
    # Lightning step hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        task_id = batch["task_ids"][0]   # AlternatingTaskSampler guarantees same task per batch
        return self._shared_step(batch, "train", task_id, self.train_metrics[task_id])

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        task_id = self.task_names[dataloader_idx]
        self._shared_step(batch, "val", task_id, self.val_metrics[task_id])

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        task_id = self.task_names[dataloader_idx]
        self._shared_step(batch, "test", task_id, self.test_metrics[task_id])

    # ------------------------------------------------------------------
    # Epoch-end metric emission
    # ------------------------------------------------------------------

    def _emit_metrics(self, metric_dict: nn.ModuleDict):
        """Emit all per-task metrics; barrier before compute() to avoid NCCL mismatch."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        for task_name, coll in metric_dict.items():
            out = coll.compute()
            for full_key, tensor in out.items():
                if torch.isnan(tensor).any():
                    tensor = torch.nan_to_num(tensor, nan=-1.0)
                if tensor.ndim == 1:
                    valid = tensor[tensor.ge(0)]
                    val = valid.mean() if valid.numel() > 0 else torch.tensor(-1.0, device=tensor.device)
                else:
                    val = tensor
                self.log(full_key, val, on_step=False, on_epoch=True,
                         sync_dist=False, add_dataloader_idx=False)
            coll.reset()

    def on_train_epoch_end(self):
        self._emit_metrics(self.train_metrics)

    def on_validation_epoch_end(self):
        self._emit_metrics(self.val_metrics)

    def on_test_epoch_end(self):
        self._emit_metrics(self.test_metrics)
        for key, val in self.trainer.callback_metrics.items():
            if any(key.startswith(f"test_{t}/") or key.startswith(f"test_{t}_") for t in self.task_names):
                self.test_results[key] = round(float(val), 4)

    # ------------------------------------------------------------------
    # Optimizer & scheduler (mirrors Mask2Former / BaseSegmentationModel)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        opt_cfg = self.cfg.training.optimizer
        opt_name = opt_cfg.name.lower()
        kw = dict(lr=opt_cfg.lr, weight_decay=opt_cfg.get("weight_decay", 0.0))
        if opt_name == "adam":
            from torch.optim import Adam
            opt = Adam(self.parameters(), **kw)
        elif opt_name == "adamw":
            from torch.optim import AdamW
            opt = AdamW(self.parameters(), **kw)
        elif opt_name == "sgd":
            from torch.optim import SGD
            kw["momentum"] = opt_cfg.get("momentum", 0.9)
            opt = SGD(self.parameters(), **kw)
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.name}")

        sched_cfg = self.cfg.training.get("scheduler", {})
        if not sched_cfg.get("enabled", False):
            return opt

        name = sched_cfg.name.lower()
        if name == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=sched_cfg.get("t_max", self.cfg.training.trainer.max_epochs),
                eta_min=sched_cfg.get("eta_min", 0),
            )
        elif name == "step":
            sched = torch.optim.lr_scheduler.StepLR(
                opt,
                step_size=sched_cfg.get("step_size", 10),
                gamma=sched_cfg.get("gamma", 0.1),
            )
        elif name == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                patience=sched_cfg.get("patience", 5),
                factor=sched_cfg.get("factor", 0.1),
                mode=sched_cfg.get("mode", "min"),
            )
            return {"optimizer": opt, "lr_scheduler": sched,
                    "monitor": self.cfg.training.monitor_on}
        else:
            raise ValueError(f"Unknown scheduler: {sched_cfg.name}")

        return {"optimizer": opt, "lr_scheduler": sched}
