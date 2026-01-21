# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluation utils
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
import yaml
from rich.console import Console
from typing_extensions import Literal

from sdfstudio.pipelines.base_pipeline import Pipeline

if TYPE_CHECKING:
    from sdfstudio.configs.base_config import Config, TrainerConfig

CONSOLE = Console(width=120)


def eval_load_checkpoint(config: TrainerConfig, pipeline: Pipeline) -> Path:
    ## TODO: ideally eventually want to get this to be the same as whatever is used to load train checkpoint too
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    """
    assert config.load_dir is not None
    if config.load_step is None:
        CONSOLE.print("Loading latest checkpoint from load_dir")
        # NOTE: this is specific to the checkpoint name format
        if not os.path.exists(config.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(
                f"No checkpoint directory found at {config.load_dir}, ",
                justify="center",
            )
            CONSOLE.print(
                "Please make sure the checkpoint exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = config.load_step
    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"
    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    # Prefer safe loading, but fall back for older checkpoints that require full unpickling.
    try:
        loaded_state = torch.load(load_path, map_location="cpu", weights_only=True)
    except (RuntimeError, pickle.UnpicklingError):
        CONSOLE.print(
            "[yellow]Secure checkpoint loading failed; falling back to weights_only=False. "
            "Only do this with checkpoints from trusted sources.[/yellow]"
        )
        loaded_state = torch.load(load_path, map_location="cpu", weights_only=False)
    pipeline.load_pipeline(loaded_state["pipeline"])
    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    return load_path


CONFIG_DIR_PLACEHOLDER = "__CONFIG_DIR__"


def _resolve_config_relative_path(path: Path, config_dir: Path) -> Path:
    """Resolve a path that may contain the CONFIG_DIR_PLACEHOLDER."""
    path_str = str(path)
    if path_str.startswith(CONFIG_DIR_PLACEHOLDER):
        relative_part = path_str[len(CONFIG_DIR_PLACEHOLDER) :].lstrip("/\\")
        return (config_dir / relative_part).resolve()
    return path


def eval_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    data_path: Optional[Path] = None,
) -> tuple[Config, Pipeline, Path]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test datset into memory
            'inference': does not load any dataset into memory
        data_path: Override the data path stored in config. Useful when loading
            configs created in Docker or on different machines.


    Returns:
        Loaded config, pipeline module, and corresponding checkpoint.
    """
    from sdfstudio.configs.base_config import Config

    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, Config)

    config_dir = config_path.parent.resolve()

    # Handle data path: explicit override > placeholder resolution > original
    if data_path is not None:
        config.pipeline.datamanager.dataparser.data = data_path
    else:
        stored_path = config.pipeline.datamanager.dataparser.data
        if stored_path is not None:
            config.pipeline.datamanager.dataparser.data = _resolve_config_relative_path(stored_path, config_dir)

    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    # load checkpoints from wherever they were saved
    # Derive checkpoint dir from config file location (not stored output_dir) for portability
    # Config file lives at {base_dir}/config.yml, checkpoints at {base_dir}/{relative_model_dir}
    config.trainer.load_dir = config_dir / config.trainer.relative_model_dir
    config.pipeline.datamanager.eval_image_indices = None

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    # load checkpointed information
    checkpoint_path = eval_load_checkpoint(config.trainer, pipeline)

    return config, pipeline, checkpoint_path
