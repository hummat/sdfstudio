# Copyright 2022-2025 The Nerfstudio Team. All rights reserved.
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

"""Minimal NeuS2-style variant: reuse Neuralangelo schedules but compute curvature with
analytic second-order (double backward through tcnn) instead of finite differences."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sdfstudio.models.neuralangelo import NeuralangeloModel, NeuralangeloModelConfig
from sdfstudio.models.neus import NeuSModel


@dataclass
class NeuS2ModelConfig(NeuralangeloModelConfig):
    """NeuS2-style Model Config.

    Keeps the Neuralangelo progressive hash / curvature schedules but switches curvature
    to analytic second-order (double backward) and expects the SDF field to run without
    numerical gradients.
    """

    _target: type = field(default_factory=lambda: NeuS2Model)

    # Defaults for a NeuS2-style run:
    # - analytic gradients only (no numerical-gradients schedule)
    # - same eikonal/fg-mask defaults as NeuS
    # - no explicit curvature loss unless enabled via CLI.
    enable_numerical_gradients_schedule: bool = False
    curvature_loss_multi: float = 0.0


class NeuS2Model(NeuralangeloModel):
    """NeuS2-style model: same sampler/schedules as Neuralangelo, analytic curvature."""

    config: NeuS2ModelConfig

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> dict:
        # Bypass Neuralangelo's finite-difference curvature; start from plain NeuS losses.
        loss_dict = NeuSModel.get_loss_dict(self, outputs, batch, metrics_dict)

        if self.training and self.config.curvature_loss_multi > 0.0:
            ray_samples = outputs["ray_samples"]
            positions = ray_samples.frustums.get_start_positions().view(-1, 3)

            if self.field.spatial_distortion is not None:
                positions = self.field.spatial_distortion(positions)

            positions.requires_grad_(True)

            with torch.enable_grad():
                sdf = self.field.forward_geonetwork(positions)[:, :1]
                grad = torch.autograd.grad(
                    outputs=sdf,
                    inputs=positions,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                # Trace of Hessian (Laplacian) via double backward.
                lap = torch.autograd.grad(
                    outputs=grad,
                    inputs=positions,
                    grad_outputs=torch.ones_like(grad),
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )[0]

            curvature = lap.sum(dim=-1, keepdim=True)
            loss_dict["curvature_loss"] = (
                curvature.abs().mean() * self.config.curvature_loss_multi * self.curvature_loss_multi_factor
            )

        return loss_dict
