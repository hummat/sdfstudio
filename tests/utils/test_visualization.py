"""
Test colormaps
"""
import torch

from sdfstudio.utils import colormaps


def test_apply_colormap():
    """Test adding a colormap to data"""
    data = torch.rand((10, 20, 1))
    colored_data = colormaps.apply_colormap(data)

    assert colored_data.shape == (10, 20, 3)
    assert torch.min(colored_data) >= 0
    assert torch.max(colored_data) <= 1


def test_apply_depth_colormap():
    """Test adding a colormap to depth data"""
    data = torch.rand((10, 20, 1))
    accum = torch.rand((10, 20, 1))
    accum = accum / torch.max(accum)
    colored_data = colormaps.apply_depth_colormap(depth=data, accumulation=accum)

    assert colored_data.shape == (10, 20, 3)
    assert torch.min(colored_data) >= 0
    assert torch.max(colored_data) <= 1


def test_apply_boolean_colormap():
    """Test adding a colormap to boolean data"""
    data = torch.rand((10, 20, 1))
    data = data > 0.5
    colored_data = colormaps.apply_boolean_colormap(data)

    assert colored_data.shape == (10, 20, 3)
    assert torch.min(colored_data) >= 0
    assert torch.max(colored_data) <= 1
