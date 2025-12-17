"""
Script to texture an existing mesh file.

Supports multiple methods:
- legacy: Original implementation (v1)
- cpu: v2 with CPU rasterization
- gpu: v2 with GPU rasterization (nvdiffrast)
- open3d: v2 multiview render-and-reproject (Open3D)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro
from rich.console import Console

from sdfstudio.exporter import texture_utils, texture_utils_v2
from sdfstudio.exporter.exporter_utils import get_mesh_from_filename
from sdfstudio.utils.eval_utils import eval_setup

CONSOLE = Console(width=120)


@dataclass
class TextureMesh:
    """
    Export a textured mesh with color computed from the NeRF.
    """

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""
    input_mesh_filename: Path
    """Mesh filename to texture."""
    num_pixels_per_side: int = 2048
    """Pixels per side of the texture image."""
    target_num_faces: int | float | None = 50000
    """Target number of faces for the mesh to texture. If < 1, it is a fraction of the original mesh faces."""
    method: Literal["legacy", "cpu", "gpu", "open3d"] = "gpu"
    """Texturing method.

    Notes:
      - "cpu"/"gpu" use xatlas for UV unwrap + per-texel NeRF queries.
      - "open3d" uses Open3D's UV atlas + multiview projection when available.
        It uses a different UV V convention than xatlas, so the exporter must not
        unconditionally flip V for OBJ output.
    """
    num_directions: int = 6
    """Number of ray directions per texel for averaging (v2 cpu/gpu only)."""
    pad_px: int = 32
    """Number of pixels to dilate charts outward (v2 cpu/gpu/open3d)."""
    num_views: int = 30
    """Number of synthetic views for --method open3d."""
    render_pixels_per_side: int = 768
    """Square render resolution for --method open3d."""
    fov_degrees: float = 60.0
    """Vertical field of view for --method open3d."""
    elev_min_degrees: float = -30.0
    """Minimum elevation angle for --method open3d."""
    elev_max_degrees: float = 60.0
    """Maximum elevation angle for --method open3d."""
    radius_mult: float = 2.0
    """Multiplier for mesh bounding sphere radius for --method open3d."""

    def main(self) -> None:
        """Export textured mesh"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        # Load the mesh
        mesh = get_mesh_from_filename(str(self.input_mesh_filename), target_num_faces=self.target_num_faces)

        # Load the pipeline
        _, pipeline, _ = eval_setup(self.load_config, test_mode="inference")

        if self.method == "legacy":
            CONSOLE.print("[yellow]Using legacy (v1) texture export with xatlas")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                output_dir=self.output_dir,
                unwrap_method="xatlas",
                num_pixels_per_side=self.num_pixels_per_side,
            )
        elif self.method in ("cpu", "gpu"):
            CONSOLE.print(f"[green]Using v2 texture export ({self.method})")
            texture_utils_v2.export_textured_mesh_v2(
                mesh,
                pipeline,
                output_dir=self.output_dir,
                texture_size=self.num_pixels_per_side,
                num_directions=self.num_directions,
                use_gpu_rasterization=(self.method == "gpu"),
                pad_px=self.pad_px,
            )
        elif self.method == "open3d":
            CONSOLE.print("[green]Using v2 multiview texture export (open3d)")
            texture_utils_v2.export_textured_mesh_multiview(
                mesh,
                pipeline,
                output_dir=self.output_dir,
                texture_size=self.num_pixels_per_side,
                num_views=self.num_views,
                render_pixels_per_side=self.render_pixels_per_side,
                fov_degrees=self.fov_degrees,
                elevation_range=(self.elev_min_degrees, self.elev_max_degrees),
                radius_mult=self.radius_mult,
                pad_px=self.pad_px,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[TextureMesh]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(TextureMesh)  # noqa
