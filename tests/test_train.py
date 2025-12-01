# pylint: disable=protected-access
"""
Default test to make sure train runs
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.train import train_loop
from sdfstudio.configs.base_config import Config
from sdfstudio.configs.method_configs import method_configs
from sdfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig

try:
    import tinycudann

    TCNN_EXISTS = True
except ImportError:
    TCNN_EXISTS = False

BLACKLIST = ["base", "semantic-nerfw", "instant-ngp", "nerfacto", "phototourism"]


def set_reduced_config(config: Config):
    """Reducing the config settings to speedup test"""
    config.machine.num_gpus = torch.cuda.device_count()
    config.trainer.max_num_iterations = 2
    # reduce dataset factors; set dataset to test
    config.pipeline.datamanager.dataparser = BlenderDataParserConfig(data=Path("tests/data/lego_test"))
    config.pipeline.datamanager.train_num_images_to_sample_from = 1
    config.pipeline.datamanager.train_num_rays_per_batch = 4

    # use tensorboard logging instead of wandb
    config.vis = "tensorboard"
    config.logging.relative_log_dir = Path("/tmp/")

    # reduce model factors
    if hasattr(config.pipeline.model, "num_coarse_samples"):
        config.pipeline.model.num_coarse_samples = 4
    if hasattr(config.pipeline.model, "num_importance_samples"):
        config.pipeline.model.num_importance_samples = 4
    # remove viewer
    config.viewer.enable = False

    # model specific config settings
    if config.method_name == "instant-ngp" and not TCNN_EXISTS:
        config.pipeline.model.field_implementation = "torch"

    return config


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@torch.amp.autocast('cuda', enabled=False)
def test_train():
    """test run train script works properly"""
    all_config_names = method_configs.keys()
    for config_name in all_config_names:
        if config_name in BLACKLIST:
            print("skipping", config_name)
            continue
        print(f"testing run for: {config_name}")
        config = method_configs[config_name]
        config = set_reduced_config(config)

        world_size = config.machine.num_gpus * config.machine.num_machines
        train_loop(local_rank=0, world_size=world_size, config=config)


if __name__ == "__main__":
    test_train()
