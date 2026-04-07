"""
Training entry point for MultitaskMask2Former.

Uses MultiTaskDataModule instead of the standard MarsDataModule.

Example
-------
python train_multitask.py \\
    model=segmentation/multitask_mask2former \\
    training=train_multitask_mask2former \\
    dataset_path=/data/MarsBench \\
    output_path=./outputs \\
    task=segmentation \\
    data_name=multitask
"""

import logging

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from marsbench.models.segmentation.multitask_mask2former import (
    MultitaskMask2Former,
    MultiTaskDataModule,
)
from marsbench.training import setup_callbacks
from marsbench.training.model_setup import setup_model
from marsbench.training.results import save_benchmark_results
from marsbench.utils.logger import setup_loggers
from marsbench.utils.seed import seed_everything

torch.set_float32_matmul_precision("medium")

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="marsbench/configs", config_name="config")
def main(cfg: DictConfig):
    if "seed" in cfg:
        seed_everything(cfg.seed)

    # Model: supports checkpoint_path via setup_model
    model = setup_model(cfg)

    # DataModule: multi-task aware
    data_module = MultiTaskDataModule(cfg)

    callbacks = setup_callbacks(cfg)
    loggers   = setup_loggers(cfg)

    trainer_config = {k: v for k, v in cfg.training.trainer.items() if k != "logger"}
    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=hydra.utils.get_original_cwd(),
        **trainer_config,
    )

    mode = cfg.get("mode", "train")
    if mode == "train":
        log.info("Starting multi-task training")
        trainer.fit(model, data_module)
        if cfg.get("test_after_training", False):
            log.info("Running test after training")
            trainer.test(model, data_module)
            save_benchmark_results(cfg, model.test_results)
    elif mode == "test":
        log.info("Starting multi-task testing")
        trainer.test(model, data_module)
        save_benchmark_results(cfg, model.test_results)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
