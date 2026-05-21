from __future__ import annotations

import json

import pytest
import torch
from PIL import Image

from sdfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from sdfstudio.data.scene_box import SceneBox
from sdfstudio.model_components.scene_colliders import NearFarCollider
from sdfstudio.models.neus import NeuSModel, NeuSModelConfig


def _write_nerfstudio_dataset(data_dir, distances: tuple[float, ...]) -> None:
    frames = []
    for index, distance in enumerate(distances):
        image_path = data_dir / f"{index:04d}.png"
        Image.new("RGB", (2, 2), color=(255, 255, 255)).save(image_path)

        transform = torch.eye(4)
        transform[2, 3] = distance
        frames.append({"file_path": image_path.name, "transform_matrix": transform.tolist()})

    transforms = {
        "fl_x": 1.0,
        "fl_y": 1.0,
        "cx": 1.0,
        "cy": 1.0,
        "h": 2,
        "w": 2,
        "frames": frames,
    }
    (data_dir / "transforms.json").write_text(json.dumps(transforms), encoding="utf-8")


def test_nerfstudio_dataparser_records_camera_distance_bounds(tmp_path):
    _write_nerfstudio_dataset(tmp_path, (2.0, 4.0))
    config = NerfstudioDataParserConfig(
        data=tmp_path,
        orientation_method="none",
        center_method="none",
        auto_scale_poses="",
        scale_factor=1.5,
        train_split_fraction=1.0,
    )

    outputs = config.setup().get_dataparser_outputs(split="train")

    assert outputs.metadata["camera_distance_bounds"] == pytest.approx({"near_plane": 3.0, "far_plane": 6.0})


def test_surface_model_uses_auto_near_far_metadata():
    scene_box = SceneBox(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
    config = NeuSModelConfig(
        auto_near_far_plane=True,
        auto_near_plane_margin=0.8,
        auto_far_plane_margin=1.25,
        auto_near_plane_min=0.1,
    )

    model = NeuSModel(
        config=config,
        scene_box=scene_box,
        num_train_data=1,
        metadata={"camera_distance_bounds": {"near_plane": 0.2, "far_plane": 2.0}},
        device="cpu",
    )

    assert isinstance(model.collider, NearFarCollider)
    assert model.collider.near_plane == pytest.approx(0.16)
    assert model.collider.far_plane == pytest.approx(2.5)


def test_explicit_near_far_override_wins_over_auto_metadata():
    scene_box = SceneBox(aabb=torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
    config = NeuSModelConfig(
        auto_near_far_plane=True,
        overwrite_near_far_plane=True,
        near_plane=7.0,
        far_plane=8.0,
    )

    model = NeuSModel(
        config=config,
        scene_box=scene_box,
        num_train_data=1,
        metadata={"camera_distance_bounds": {"near_plane": 0.2, "far_plane": 2.0}},
        device="cpu",
    )

    assert isinstance(model.collider, NearFarCollider)
    assert model.collider.near_plane == pytest.approx(7.0)
    assert model.collider.far_plane == pytest.approx(8.0)
