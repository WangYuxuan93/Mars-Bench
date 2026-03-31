"""
Model initialization utilities for MarsBench.
"""

import logging
import os

import hydra
import pytorch_lightning as pl
import torch
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
    # PyTorch 2.6+ requires explicitly allowlisting non-tensor globals in checkpoints.
    # Lightning checkpoints embed OmegaConf objects, so we allowlist all common ones.
    try:
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode, IntegerNode, FloatNode, BooleanNode, StringNode, EnumNode
        torch.serialization.add_safe_globals([
            DictConfig, ListConfig,
            ContainerMetadata, Metadata,
            AnyNode, IntegerNode, FloatNode, BooleanNode, StringNode, EnumNode,
        ])
    except Exception:
        pass
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
            ckpt_base_name = os.path.basename(cfg.checkpoint_path).split(".")[0]
            cfg.model.name = f"ckpt_{ckpt_base_name}"
        except Exception as e:
            log.error(f"Failed to load from checkpoint: {e}")
            raise
    else:
        # Create a new model instance
        log.info(f"Creating new model: {cfg.model.name}")
        model = model_class(cfg)
    return model
