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
Code for camera paths.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

import sdfstudio.utils.poses as pose_utils
from sdfstudio.cameras import camera_utils
from sdfstudio.cameras.camera_utils import get_interpolated_poses_many
from sdfstudio.cameras.cameras import Cameras
from sdfstudio.viewer.server.utils import three_js_perspective_camera_focal_length


def get_interpolated_camera_path(cameras: Cameras, steps: int) -> Cameras:
    """Generate a camera path between two cameras.

    Args:
        cameras: Cameras object containing intrinsics of all cameras.
        steps: The number of steps to interpolate between the two cameras.

    Returns:
        A new set of cameras along a path.
    """
    Ks = cameras.get_intrinsics_matrices().cpu().numpy()
    poses = cameras.camera_to_worlds().cpu().numpy()
    poses, Ks = get_interpolated_poses_many(poses, Ks, steps_per_transition=steps)

    cameras = Cameras(fx=Ks[:, 0, 0], fy=Ks[:, 1, 1], cx=Ks[0, 0, 2], cy=Ks[0, 1, 2], camera_to_worlds=poses)
    return cameras


def get_spiral_path(
    camera: Cameras,
    steps: int = 30,
    radius: Optional[float] = None,
    radiuses: Optional[Tuple[float]] = None,
    rots: int = 2,
    zrate: float = 0.5,
) -> Cameras:
    """
    Returns a list of camera in a sprial trajectory.

    Args:
        camera: The camera to start the spiral from.
        steps: The number of cameras in the generated path.
        radius: The radius of the spiral for all xyz directions.
        radiuses: The list of radii for the spiral in xyz directions.
        rots: The number of rotations to apply to the camera.
        zrate: How much to change the z position of the camera.

    Returns:
        A spiral camera path.
    """

    assert radius is not None or radiuses is not None, "Either radius or radiuses must be specified."
    assert camera.ndim == 1, "We assume only one batch dim here"
    if radius is not None and radiuses is None:
        rad = torch.tensor([radius] * 3, device=camera.device)
    elif radiuses is not None and radius is None:
        rad = torch.tensor(radiuses, device=camera.device)
    else:
        raise ValueError("Only one of radius or radiuses must be specified.")

    camera = camera.flatten()

    up = camera.camera_to_worlds[0, :3, 2]  # scene is z up
    focal = torch.min(camera.fx[0], camera.fy[0])
    target = torch.tensor([0, 0, -focal], device=camera.device)  # camera looking in -z direction

    c2w = camera.camera_to_worlds[0]
    c2wh_global = pose_utils.to4x4(c2w)

    local_c2whs = []
    for theta in torch.linspace(0.0, 2.0 * torch.pi * rots, steps + 1)[:-1]:
        center = (
            torch.tensor([torch.cos(theta), -torch.sin(theta), -torch.sin(theta * zrate)], device=camera.device) * rad
        )
        lookat = center - target
        c2w = camera_utils.viewmatrix(lookat, up, center)
        c2wh = pose_utils.to4x4(c2w)
        local_c2whs.append(c2wh)

    new_c2ws = []
    for local_c2wh in local_c2whs:
        c2wh = torch.matmul(c2wh_global, local_c2wh)
        new_c2ws.append(c2wh[:3, :4])
    new_c2ws = torch.stack(new_c2ws, dim=0)

    return Cameras(
        fx=camera.fx[0],
        fy=camera.fy[0],
        cx=camera.cx[0],
        cy=camera.cy[0],
        camera_to_worlds=new_c2ws,
    )


def get_path_from_json(camera_path: Dict[str, Any]) -> Cameras:
    """Takes a camera path dictionary and returns a trajectory as a Camera instance.

    Args:
        camera_path: A dictionary of the camera path information coming from the viewer.

    Returns:
        A Cameras instance with the camera path.
    """

    image_height = camera_path["render_height"]
    image_width = camera_path["render_width"]

    c2ws = []
    fxs = []
    fys = []
    for camera in camera_path["camera_path"]:
        # pose
        c2w = torch.tensor(camera["camera_to_world"]).view(4, 4)[:3]
        c2ws.append(c2w)
        # field of view
        fov = camera["fov"]
        focal_length = three_js_perspective_camera_focal_length(fov, image_height)
        fxs.append(focal_length)
        fys.append(focal_length)

    camera_to_worlds = torch.stack(c2ws, dim=0)
    fx = torch.tensor(fxs)
    fy = torch.tensor(fys)
    return Cameras(
        fx=fx,
        fy=fy,
        cx=image_width / 2,
        cy=image_height / 2,
        camera_to_worlds=camera_to_worlds,
    )


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


# https://github.com/google-research/multinerf/blob/47fad9688748b3cc962990c19898aff78b45968e/internal/camera_utils.py#L144
def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


# https://github.com/google-research/multinerf/blob/47fad9688748b3cc962990c19898aff78b45968e/internal/camera_utils.py#L230
def generate_ellipse_path(
    cameras: Cameras, n_frames: int = 120, const_speed: bool = True, z_variation: float = 0.0, z_phase: float = 0.0
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""

    poses = np.stack(cameras.camera_to_worlds.cpu().numpy())
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                z_variation * (z_low[2] + (z_high - z_low)[2] * (np.cos(theta + 2 * np.pi * z_phase) * 0.5 + 0.5)),
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    if const_speed:
        # Resample theta angles so that the velocity is closer to constant.
        # lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        # theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
        # positions = get_positions(theta)
        raise NotImplementedError

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    render_c2ws = np.stack([viewmatrix(p - center, up, p) for p in positions])
    render_c2ws = torch.from_numpy(render_c2ws)

    # use intrinsic of first camera
    camera_path = Cameras(
        fx=cameras[0].fx,
        fy=cameras[0].fy,
        cx=cameras[0].cx,
        cy=cameras[0].cy,
        height=cameras[0].height,
        width=cameras[0].width,
        camera_to_worlds=render_c2ws[:, :3, :4],
        camera_type=cameras[0].camera_type,
    )
    return camera_path
