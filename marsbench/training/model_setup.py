"""
Model initialization utilities for MarsBench.
"""

import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from marsbench.models import import_model_class

log = logging.getLogger(__name__)


def setup_model(cfg: DictConfig) -> pl.LightningModule:
    """Set up model based on configuration.
    Args:
        cfg: Configuration object
    Returns:
        Initialized model
    """
    # PyTorch 2.6+ changed torch.load default to weights_only=True, which breaks
    # Lightning checkpoints that embed OmegaConf objects. Patch torch.load to
    # force weights_only=False (safe here since we only load our own checkpoints).
    import functools
    import torch
    _orig_torch_load = torch.load

    @functools.wraps(_orig_torch_load)
    def _patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    # Import the model class based on configuration
    model_class = import_model_class(cfg)
    # Load weights from checkpoint if specified
    if cfg.get("checkpoint_path"):
        log.info(f"Loading model from checkpoint: {cfg.checkpoint_path}")
        checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_path)
        try:
            # Use the model class to load the checkpoint
            model = model_class.load_from_checkpoint(checkpoint_path, cfg=cfg)
            # Update model name to reflect checkpoint usage
            # Keep the original model name so collate_fn detection still works
            ckpt_base_name = os.path.basename(cfg.checkpoint_path).split(".")[0]
            log.info(f"Loaded checkpoint: {ckpt_base_name}")
        except Exception as e:
            log.error(f"Failed to load from checkpoint: {e}")
            raise
    else:
        # Create a new model instance
        log.info(f"Creating new model: {cfg.model.name}")
        model = model_class(cfg)
    return model
