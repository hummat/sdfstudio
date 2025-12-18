from __future__ import annotations

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
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as TensorType
from torch import nn
from torch.nn.parameter import Parameter

from sdfstudio.cameras.rays import RaySamples
from sdfstudio.field_components.embedding import Embedding
from sdfstudio.field_components.encodings import (
    NeRFEncoding,
    PeriodicVolumeEncoding,
    TensorVMEncoding,
)
from sdfstudio.field_components.field_heads import FieldHeadNames
from sdfstudio.field_components.spatial_distortions import SpatialDistortion
from sdfstudio.fields.base_field import Field, FieldConfig

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, sdf: TensorType, beta: Optional[TensorType] = None) -> TensorType:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SigmoidDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Sigmoid density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, sdf: TensorType, beta: Optional[TensorType] = None) -> TensorType:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta

        # negtive sdf will have large density
        return alpha * torch.sigmoid(-sdf * alpha)

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SingleVarianceNetwork(nn.Module):
    """Variance network in NeuS

    Args:
        nn (_type_): init value in NeuS variance network
    """

    def __init__(self, init_val):
        super().__init__()
        self.register_parameter("variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, x):
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)

    def get_variance(self):
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)


@dataclass
class SDFFieldConfig(FieldConfig):
    """Nerfacto Model Config"""

    _target: type = field(default_factory=lambda: SDFField)
    num_layers: int = 8
    """Number of layers for geometric network"""
    hidden_dim: int = 256
    """Number of hidden dimension of geometric network"""
    geo_feat_dim: int = 256
    """Dimension of geometric feature"""
    num_layers_color: int = 4
    """Number of layers for color network"""
    hidden_dim_color: int = 256
    """Number of hidden dimension of color network"""
    appearance_embedding_dim: int = 32
    """Dimension of the per-image appearance (color) embedding."""
    use_appearance_embedding: bool = False
    """Whether to feed a learned per-image appearance embedding into the color MLP.

    When enabled, each training image gets a small latent code that can absorb exposure / white-balance /
    lighting variation in the color network instead of pushing it into geometry or BRDF parameters."""
    bias: float = 0.8
    """Controls the radius/offset of the geometric SDF initialization.

    With ``geometric_init=True``, the last SDF MLP layer is initialized to a sphere whose radius is
    roughly ``bias`` (up to sign conventions via ``inside_outside``). Smaller values focus the initial
    field on a tight object-centric region; larger values cover a bigger volume and spread gradients
    more widely at the start."""
    geometric_init: bool = True
    """Whether to use geometric initialization"""
    inside_outside: bool = True
    """whether to revert signed distance value, set to True for indoor scene"""
    weight_norm: bool = True
    """Whether to use weight norm for linear laer"""
    use_grid_feature: bool = False
    """Whether to use multi-resolution feature grids"""
    divide_factor: float = 2.0
    """Normalization factor for multi-resolution grids"""
    beta_init: float = 0.1
    """Initial softness scale for SDF→density/alpha around the surface.

    Used to initialize both the VolSDF Laplace density ``beta`` and the NeuS variance network
    (which exponentiates this value internally to produce the global sharpness ``s_val``). Smaller
    values make the initial transition sharper; larger values make it thicker and more forgiving."""
    encoding_type: Literal["hash", "periodic", "tensorf_vm"] = "hash"
    """feature grid encoding type"""
    position_encoding_max_degree: int = 6
    """Positional encoding max degree for spatial inputs."""
    use_diffuse_color: bool = False
    """Enable a Ref-NeRF-style diffuse/specular split.

    Adds a view-independent diffuse head fed from geo features + appearance embedding, while the main
    color MLP focuses on view-dependent (specular-like) effects. This makes it easier to keep albedo
    separate from highlights and reduces pressure to encode specular behavior into geometry."""
    use_specular_tint: bool = False
    """Enable Ref-NeRF-style specular tint.

    Learns an RGB tint for the specular component so the network can represent colored specular (metals)
    instead of assuming neutral/white highlights."""
    use_reflections: bool = False
    """Use reflection directions for view encoding as in Ref-NeRF.

    When enabled, the color MLP sees features derived from both the view direction and its reflection
    about the surface normal, improving specular highlight placement and reflection-like view dependence."""
    use_n_dot_v: bool = False
    """Provide n·v (cosine of view incidence) to the color MLP.

    This gives the network an explicit angle-of-incidence cue, making limb darkening, foreshortening,
    and Fresnel-like intensity ramps easier to learn."""
    enable_pred_roughness: bool = False
    """Predict a PBR-style roughness in [0, 1] and use it to mix view and reflection encodings.

    With `use_reflections=True`, roughness=0 (specular) relies purely on reflection directions and
    roughness=1 (diffuse) uses only view directions, giving an interpretable roughness map and a
    simple roughness-dependent bias on specular sharpness (no full microfacet BRDF)."""
    rgb_padding: float = 0.001
    """Padding added to the RGB outputs"""
    off_axis: bool = False
    """whether to use off axis encoding from mipnerf360"""
    use_numerical_gradients: bool = False
    """whether to use numercial gradients"""
    num_levels: int = 16
    """number of levels for multi-resolution hash grids"""
    max_res: int = 2048
    """max resolution for multi-resolution hash grids"""
    base_res: int = 16
    """base resolution for multi-resolution hash grids"""
    log2_hashmap_size: int = 19
    """log2 hash map size for multi-resolution hash grids"""
    hash_features_per_level: int = 2
    """number of features per level for multi-resolution hash grids"""
    hash_smoothstep: bool = True
    """whether to use smoothstep for multi-resolution hash grids"""
    use_position_encoding: bool = True
    """whether to use positional encoding as input for geometric network"""


class SDFField(Field):
    """_summary_

    Args:
        Field (_type_): _description_
    """

    config: SDFFieldConfig

    def __init__(
        self,
        config: SDFFieldConfig,
        aabb,
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.config = config

        # TODO do we need aabb here?
        self.aabb = Parameter(aabb, requires_grad=False)

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images

        self.embedding_appearance = Embedding(self.num_images, self.config.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_grid_feature = self.config.use_grid_feature
        self.divide_factor = self.config.divide_factor

        self.num_levels = self.config.num_levels
        self.max_res = self.config.max_res
        self.base_res = self.config.base_res
        self.log2_hashmap_size = self.config.log2_hashmap_size
        self.features_per_level = self.config.hash_features_per_level
        use_hash = True
        smoothstep = self.config.hash_smoothstep
        self.growth_factor = np.exp((np.log(self.max_res) - np.log(self.base_res)) / (self.num_levels - 1))

        if self.config.encoding_type == "hash":
            # feature encoding
            self.encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid" if use_hash else "DenseGrid",
                    "n_levels": self.num_levels,
                    "n_features_per_level": self.features_per_level,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": self.base_res,
                    "per_level_scale": self.growth_factor,
                    "interpolation": "Smoothstep" if smoothstep else "Linear",
                },
            )
            self.hash_encoding_mask = torch.ones(
                self.num_levels * self.features_per_level,
                dtype=torch.float32,
            )

        elif self.config.encoding_type == "periodic":
            print("using periodic encoding")
            self.encoding = PeriodicVolumeEncoding(
                num_levels=self.num_levels,
                min_res=self.base_res,
                max_res=self.max_res,
                log2_hashmap_size=18,  # 64 ** 3 = 2^18
                features_per_level=self.features_per_level,
                smoothstep=smoothstep,
            )
        elif self.config.encoding_type == "tensorf_vm":
            print("using tensor vm")
            self.encoding = TensorVMEncoding(128, 24, smoothstep=smoothstep)

        # we concat inputs position ourselves
        self.position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=self.config.position_encoding_max_degree,
            min_freq_exp=0.0,
            max_freq_exp=self.config.position_encoding_max_degree - 1,
            include_input=False,
            off_axis=self.config.off_axis,
        )

        self.direction_encoding = NeRFEncoding(
            in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
        )

        # TODO move it to field components
        # MLP with geometric initialization
        dims = [self.config.hidden_dim for _ in range(self.config.num_layers)]
        in_dim = 3 + self.position_encoding.get_out_dim() + self.encoding.n_output_dims
        dims = [in_dim] + dims + [1 + self.config.geo_feat_dim]
        self.num_layers = len(dims)
        # TODO check how to merge skip_in to config
        self.skip_in = [4]

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if self.config.geometric_init:
                if l == self.num_layers - 2:
                    if not self.config.inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -self.config.bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, self.config.bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.config.weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)
            setattr(self, "glin" + str(l), lin)

        # laplace function for transform sdf to density from VolSDF
        self.laplace_density = LaplaceDensity(init_val=self.config.beta_init)
        # self.laplace_density = SigmoidDensity(init_val=self.config.beta_init)

        # TODO use different name for beta_init for config
        # deviation_network to compute alpha from sdf from NeuS
        self.deviation_network = SingleVarianceNetwork(init_val=self.config.beta_init)

        # diffuse, specular tint, and roughness layers
        if self.config.use_diffuse_color:
            self.diffuse_color_pred = nn.Linear(self.config.geo_feat_dim, 3)
        if self.config.use_specular_tint:
            self.specular_tint_pred = nn.Linear(self.config.geo_feat_dim, 3)
        if self.config.enable_pred_roughness:
            self.roughness_pred = nn.Linear(self.config.geo_feat_dim, 1)

        # view dependent color network
        dims = [self.config.hidden_dim_color for _ in range(self.config.num_layers_color)]
        if self.config.use_diffuse_color:
            in_dim = (
                self.direction_encoding.get_out_dim()
                + self.config.geo_feat_dim
                + self.embedding_appearance.get_out_dim()
            )
        else:
            # point, view_direction, normal, feature, embedding
            in_dim = (
                3
                + self.direction_encoding.get_out_dim()
                + 3
                + self.config.geo_feat_dim
                + self.embedding_appearance.get_out_dim()
            )
        if self.config.use_n_dot_v:
            in_dim += 1

        dims = [in_dim] + dims + [3]
        self.num_layers_color = len(dims)

        for l in range(0, self.num_layers_color - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            torch.nn.init.kaiming_uniform_(lin.weight.data)
            torch.nn.init.zeros_(lin.bias.data)

            if self.config.weight_norm:
                lin = nn.utils.parametrizations.weight_norm(lin)
            setattr(self, "clin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self._cos_anneal_ratio = 1.0
        self.numerical_gradients_delta = 0.0001

    def set_cos_anneal_ratio(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def update_mask(self, level: int):
        self.hash_encoding_mask[:] = 1.0
        self.hash_encoding_mask[level * self.features_per_level :] = 0

    def forward_geonetwork(self, inputs):
        """forward the geonetwork"""
        if self.use_grid_feature:
            # TODO normalize inputs depending on the whether we model the background or not
            positions = (inputs + 2.0) / 4.0
            # positions = (inputs + 1.0) / 2.0
            feature = self.encoding(positions)
            # mask feature
            feature = feature * self.hash_encoding_mask.to(feature.device)
        else:
            feature = torch.zeros_like(inputs[:, :1].repeat(1, self.encoding.n_output_dims))

        pe = self.position_encoding(inputs)
        if not self.config.use_position_encoding:
            pe = torch.zeros_like(pe)

        inputs = torch.cat((inputs, pe, feature), dim=-1)

        x = inputs

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "glin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x

    def get_sdf(self, ray_samples: RaySamples):
        """predict the sdf value for ray samples"""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        h = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, _ = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
        return sdf

    def set_numerical_gradients_delta(self, delta: float) -> None:
        """Set the delta value for numerical gradient."""
        self.numerical_gradients_delta = delta

    def gradient(self, x, skip_spatial_distortion=False, return_sdf=False):
        """compute the gradient of the ray"""
        if self.spatial_distortion is not None and not skip_spatial_distortion:
            x = self.spatial_distortion(x)

        # compute gradient in contracted space
        if self.config.use_numerical_gradients:
            # https://github.com/bennyguo/instant-nsr-pl/blob/main/models/geometry.py#L173
            delta = self.numerical_gradients_delta
            points = torch.stack(
                [
                    x + torch.as_tensor([delta, 0.0, 0.0]).to(x),
                    x + torch.as_tensor([-delta, 0.0, 0.0]).to(x),
                    x + torch.as_tensor([0.0, delta, 0.0]).to(x),
                    x + torch.as_tensor([0.0, -delta, 0.0]).to(x),
                    x + torch.as_tensor([0.0, 0.0, delta]).to(x),
                    x + torch.as_tensor([0.0, 0.0, -delta]).to(x),
                ],
                dim=0,
            )

            points_sdf = self.forward_geonetwork(points.view(-1, 3))[..., 0].view(6, *x.shape[:-1])
            gradients = torch.stack(
                [
                    0.5 * (points_sdf[0] - points_sdf[1]) / delta,
                    0.5 * (points_sdf[2] - points_sdf[3]) / delta,
                    0.5 * (points_sdf[4] - points_sdf[5]) / delta,
                ],
                dim=-1,
            )
        else:
            x.requires_grad_(True)

            y = self.forward_geonetwork(x)[:, :1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
        if not return_sdf:
            return gradients
        else:
            return gradients, points_sdf

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_start_positions()
        positions_flat = positions.view(-1, 3)
        h = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, geo_feature = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
        density = self.laplace_density(sdf)
        return density, geo_feature

    @torch.amp.autocast("cuda", enabled=False)
    def get_alpha(self, ray_samples: RaySamples, sdf=None, gradients=None):
        """compute alpha from sdf as in NeuS"""
        if sdf is None or gradients is None:
            inputs = ray_samples.frustums.get_start_positions()
            inputs.requires_grad_(True)
            with torch.enable_grad():
                h = self.forward_geonetwork(inputs)
                sdf, _ = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

        inv_s = self.deviation_network.get_variance()  # Single parameter

        true_cos = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)

        # anneal as NeuS
        cos_anneal_ratio = self._cos_anneal_ratio

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)

        # HF-NeuS
        # # sigma
        # cdf = torch.sigmoid(sdf * inv_s)
        # e = inv_s * (1 - cdf) * (-iter_cos) * ray_samples.deltas
        # alpha = (1 - torch.exp(-e)).clip(0.0, 1.0)

        return alpha

    def get_occupancy(self, sdf):
        """compute occupancy as in UniSurf"""
        occupancy = self.sigmoid(-10.0 * sdf)
        return occupancy

    def get_colors(self, points, directions, gradients, geo_features, camera_indices):
        """Compute view-dependent colors with Ref-NeRF-inspired BRDF signals.

        - Diffuse/specular split (`use_diffuse_color`) separates view-independent albedo from
          view-dependent components.
        - Specular tint (`use_specular_tint`) lets the model learn colored specular (metals).
        - Reflection encoding (`use_reflections`) provides reflection-direction features for
          more accurate highlight and reflection placement.
        - Roughness (`enable_pred_roughness`) in [0, 1] mixes view and reflection encodings
          (0 = fully specular / reflection-dir, 1 = fully diffuse / view-dir).
        - n·v (`use_n_dot_v`) explicitly encodes angle of incidence for Fresnel/foreshortening-
          like behavior.

        All effects are realized through a learned MLP; this is not a full analytic microfacet
        BRDF and there is no explicit environment lighting.

        Args:
            points: Sample positions (unused for shading, reserved for spatial variation).
            directions: View directions, shape (..., 3).
            gradients: SDF gradients used to compute normals, shape (..., 3).
            geo_features: Geometry features from the SDF network.
            camera_indices: Per-ray camera indices for appearance embedding.

        Returns:
            If `use_diffuse_color` is False: RGB colors, shape (..., 3).
            If `use_diffuse_color` is True: tuple (rgb, diffuse, specular, tint),
                each with shape (..., 3).
        """

        # diffuse color, specular tint, and roughness
        if self.config.use_diffuse_color:
            raw_rgb_diffuse = self.diffuse_color_pred(geo_features.view(-1, self.config.geo_feat_dim))
        if self.config.use_specular_tint:
            tint = self.sigmoid(self.specular_tint_pred(geo_features.view(-1, self.config.geo_feat_dim)))
        if self.config.enable_pred_roughness:
            roughness = self.sigmoid(self.roughness_pred(geo_features.view(-1, self.config.geo_feat_dim)))

        normals = F.normalize(gradients, p=2, dim=-1)

        # encode view and (optionally) reflection directions
        d_view = self.direction_encoding(directions)
        if self.config.use_reflections:
            # https://github.com/google-research/multinerf/blob/5d4c82831a9b94a87efada2eee6a993d530c4226/internal/ref_utils.py#L22
            refdirs = 2.0 * torch.sum(normals * -directions, axis=-1, keepdims=True) * normals + directions
            d_ref = self.direction_encoding(refdirs)

            if self.config.enable_pred_roughness:
                # roughness in [0, 1]: 0=specular (reflection dir), 1=diffuse (view dir)
                # directly usable for PBR roughness map export
                d = d_view * roughness + d_ref * (1.0 - roughness)
            else:
                # if roughness is not predicted, fall back to a simple average
                d = 0.5 * (d_view + d_ref)
        else:
            d = d_view

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
            # set it to zero if don't use it
            if not self.config.use_appearance_embedding:
                embedded_appearance = torch.zeros_like(embedded_appearance)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                )
        if self.config.use_diffuse_color:
            h = [
                d,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
            ]
        else:
            h = [
                points,
                d,
                gradients,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
            ]

        if self.config.use_n_dot_v:
            n_dot_v = torch.sum(normals * directions, dim=-1, keepdims=True)
            h.append(n_dot_v)

        h = torch.cat(h, dim=-1)

        for l in range(0, self.num_layers_color - 1):
            lin = getattr(self, "clin" + str(l))

            h = lin(h)

            if l < self.num_layers_color - 2:
                h = self.relu(h)

        rgb = self.sigmoid(h)

        # Adapted from https://github.com/google-research/multinerf/blob/main/internal/image.py#L48
        def linear_to_srgb(linear: TensorType, eps: Optional[float] = None) -> TensorType:
            if eps is None:
                eps = torch.finfo(linear.dtype).eps
            srgb0 = (323 / 25) * linear
            srgb1 = (211 * torch.clamp_min(linear, eps).pow(5 / 12) - 11) / 200
            return torch.where(linear <= 0.0031308, srgb0, srgb1)

        if self.config.use_diffuse_color:
            # Initialize linear diffuse color around 0.25, so that the combined
            # linear color is initialized around 0.5.
            diffuse_linear = self.sigmoid(raw_rgb_diffuse - math.log(3.0))
            if self.config.use_specular_tint:
                specular_linear = tint * rgb
            else:
                specular_linear = 0.5 * rgb

            # Combine specular and diffuse components and tone map to sRGB.
            rgb = linear_to_srgb(specular_linear + diffuse_linear).clamp(0, 1)

        # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
        rgb = rgb * (1 + 2 * self.config.rgb_padding) - self.config.rgb_padding
        if self.config.use_diffuse_color:
            components: list[TensorType] = [
                rgb,
                linear_to_srgb(2 * diffuse_linear).clamp(0, 1),
                linear_to_srgb(specular_linear).clamp(0, 1),
            ]
            if self.config.use_specular_tint:
                components.append(linear_to_srgb(tint).clamp(0, 1))
            if self.config.enable_pred_roughness:
                components.append(roughness)
            return tuple(components)
        return rgb

    def get_outputs(self, ray_samples: RaySamples, return_alphas: bool = False, return_occupancy: bool = False):
        """compute output of ray samples"""
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        outputs = {}

        camera_indices = ray_samples.camera_indices.squeeze()

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        if self.spatial_distortion is not None:
            inputs = self.spatial_distortion(inputs)
        points_norm = inputs.norm(dim=-1)
        # compute gradient in constracted space
        inputs.requires_grad_(True)
        with torch.enable_grad():
            h = self.forward_geonetwork(inputs)
            sdf, geo_feature = torch.split(h, [1, self.config.geo_feat_dim], dim=-1)

        if self.config.use_numerical_gradients:
            gradients, sampled_sdf = self.gradient(
                inputs,
                skip_spatial_distortion=True,
                return_sdf=True,
            )
            sampled_sdf = (
                sampled_sdf.view(-1, *ray_samples.frustums.directions.shape[:-1]).permute(1, 2, 0).contiguous()
            )
        else:
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=inputs,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            sampled_sdf = None

        rgb = self.get_colors(inputs, directions_flat, gradients, geo_feature, camera_indices)
        if isinstance(rgb, tuple):
            components = list(rgb)
            rgb = components[0]
            diffuse = components[1]
            specular = components[2]
            outputs["diffuse"] = diffuse.view(*ray_samples.frustums.directions.shape[:-1], -1)
            outputs["specular"] = specular.view(*ray_samples.frustums.directions.shape[:-1], -1)

            idx = 3
            if self.config.use_specular_tint:
                tint = components[idx]
                outputs["tint"] = tint.view(*ray_samples.frustums.directions.shape[:-1], -1)
                idx += 1
            if self.config.enable_pred_roughness:
                roughness = components[idx]
                outputs["roughness"] = roughness.view(*ray_samples.frustums.directions.shape[:-1], -1)

        density = self.laplace_density(sdf)

        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        density = density.view(*ray_samples.frustums.directions.shape[:-1], -1)
        gradients = gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
        normals = F.normalize(gradients, p=2, dim=-1)
        points_norm = points_norm.view(*ray_samples.frustums.directions.shape[:-1], -1)

        outputs.update(
            {
                FieldHeadNames.RGB: rgb,
                FieldHeadNames.DENSITY: density,
                FieldHeadNames.SDF: sdf,
                FieldHeadNames.NORMAL: normals,
                FieldHeadNames.GRADIENT: gradients,
                "points_norm": points_norm,
                "sampled_sdf": sampled_sdf,
            }
        )

        if return_alphas:
            # TODO use mid point sdf for NeuS
            alphas = self.get_alpha(ray_samples, sdf, gradients)
            outputs.update({FieldHeadNames.ALPHA: alphas})

        if return_occupancy:
            occupancy = self.get_occupancy(sdf)
            outputs.update({FieldHeadNames.OCCUPANCY: occupancy})

        return outputs

    def forward(self, ray_samples: RaySamples, return_alphas=False, return_occupancy=False):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        field_outputs = self.get_outputs(ray_samples, return_alphas=return_alphas, return_occupancy=return_occupancy)
        return field_outputs
