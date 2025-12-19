"""
Test losses
"""

import torch

from sdfstudio.model_components.losses import orientation_loss


def test_orientation_loss_uses_point_to_camera_viewdirs():
    """Orientation loss should be zero when normals face the camera.

    In this codebase `viewdirs` are ray directions (camera→point), and `orientation_loss()`
    internally uses `v = -viewdirs` so that the dot product uses the point→camera view vector.
    """

    weights = torch.ones((1, 1, 1))
    normals = torch.tensor([[[0.0, 0.0, 1.0]]])  # facing +Z

    viewdirs_camera_to_point = torch.tensor([[0.0, 0.0, -1.0]])
    loss_good = orientation_loss(weights, normals, viewdirs_camera_to_point)
    assert torch.allclose(loss_good, torch.zeros_like(loss_good))

    viewdirs_point_to_camera = torch.tensor([[0.0, 0.0, 1.0]])
    loss_bad = orientation_loss(weights, normals, viewdirs_point_to_camera)
    assert torch.all(loss_bad > 0.0)
