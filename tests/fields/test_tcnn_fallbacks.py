from __future__ import annotations

import torch

from sdfstudio.field_components.encodings import HashEncoding
from sdfstudio.field_components.mlp import MLP
from sdfstudio.fields.density_fields import HashMLPDensityField
from sdfstudio.fields.sdf_field import SDFField, SDFFieldConfig


def test_hashmlp_density_field_uses_torch_on_cpu_even_if_tcnn_installed() -> None:
    aabb = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    field = HashMLPDensityField(aabb=aabb, spatial_distortion=None, device="cpu")

    assert field._use_tcnn is False
    assert isinstance(field.encoding, HashEncoding)
    assert isinstance(field.mlp_base, MLP)


def test_sdf_field_hash_encoding_uses_torch_on_cpu_even_if_tcnn_installed() -> None:
    aabb = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    config = SDFFieldConfig()
    config.use_grid_feature = True
    config.encoding_type = "hash"

    field = SDFField(config=config, aabb=aabb, num_images=1, spatial_distortion=None, device="cpu")
    assert isinstance(field.encoding, HashEncoding)
