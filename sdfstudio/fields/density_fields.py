from __future__ import annotations

import warnings
from typing import Optional

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
Proposal network field.
"""

import numpy as np
import torch
from torch import Tensor as TensorType
from torch.nn.parameter import Parameter

from sdfstudio.cameras.rays import RaySamples
from sdfstudio.data.scene_box import SceneBox
from sdfstudio.field_components.activations import trunc_exp
from sdfstudio.field_components.encodings import HashEncoding
from sdfstudio.field_components.mlp import MLP
from sdfstudio.field_components.spatial_distortions import SpatialDistortion
from sdfstudio.fields.base_field import Field

try:
    import tinycudann as tcnn  # type: ignore
except (ImportError, ModuleNotFoundError, OSError) as e:
    tcnn = None  # type: ignore
    warnings.warn(
        f"tinycudann not available ({type(e).__name__}: {e}); proposal networks will use PyTorch fallback.",
        RuntimeWarning,
        stacklevel=2,
    )


class HashMLPDensityField(Field):
    """A lightweight density field module.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        spatial_distortion: spatial distortion module
        use_linear: whether to skip the MLP and use a single linear layer instead
    """

    def __init__(
        self,
        aabb,
        num_layers: int = 2,
        hidden_dim: int = 64,
        spatial_distortion: Optional[SpatialDistortion] = None,
        use_linear=False,
        num_levels=8,
        max_res=1024,
        base_res=16,
        log2_hashmap_size=18,
        features_per_level=2,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.use_linear = use_linear
        if device is not None and device.startswith("cuda") and tcnn is None:
            warnings.warn(
                "CUDA device requested but tinycudann is unavailable; using PyTorch HashMLPDensityField fallback.",
                RuntimeWarning,
                stacklevel=2,
            )
        self._use_tcnn = tcnn is not None and device is not None and device.startswith("cuda")
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

        config = {
            "encoding": {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        }

        if self._use_tcnn:
            if not self.use_linear:
                self.mlp_base = tcnn.NetworkWithInputEncoding(  # type: ignore[union-attr]
                    n_input_dims=3,
                    n_output_dims=1,
                    encoding_config=config["encoding"],
                    network_config=config["network"],
                )
            else:
                self.encoding = tcnn.Encoding(n_input_dims=3, encoding_config=config["encoding"])  # type: ignore[union-attr]
                self.linear = torch.nn.Linear(self.encoding.n_output_dims, 1)
        else:
            self.encoding = HashEncoding(
                num_levels=num_levels,
                min_res=base_res,
                max_res=max_res,
                log2_hashmap_size=log2_hashmap_size,
                features_per_level=features_per_level,
                implementation="torch",
            )
            if not self.use_linear:
                self.mlp_base = MLP(
                    in_dim=self.encoding.get_out_dim(),
                    num_layers=num_layers,
                    layer_width=hidden_dim,
                    out_dim=1,
                    activation=torch.nn.ReLU(),
                )
            else:
                self.linear = torch.nn.Linear(self.encoding.get_out_dim(), 1)

    def get_density(self, ray_samples: RaySamples):
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(ray_samples.frustums.get_positions())
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        positions_flat = positions.view(-1, 3)
        if not self.use_linear:
            if self._use_tcnn:
                density_before_activation = (
                    self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1).to(positions)  # type: ignore[attr-defined]
                )
            else:
                x = self.encoding(positions_flat).to(positions)  # type: ignore[operator]
                density_before_activation = self.mlp_base(x).view(*ray_samples.frustums.shape, -1)  # type: ignore[attr-defined]
        else:
            x = self.encoding(positions_flat).to(positions)  # type: ignore[operator]
            density_before_activation = self.linear(x).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation)
        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        return {}
