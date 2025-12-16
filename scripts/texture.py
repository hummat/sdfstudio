"""
Script to texture an existing mesh file.

Supports two implementations:
- v1 (legacy): Original implementation with CPU rasterization
- v2: Improved implementation with GPU rasterization, fast I/O, multi-direction averaging
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
    px_per_uv_triangle: int = 4
    """Number of pixels per UV square (v1 only)."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh (v1 only)."""
    num_pixels_per_side: int = 2048
    """Pixels per side of the texture image."""
    target_num_faces: int | float | None = 50000
    """Target number of faces for the mesh to texture. If < 1, it is a fraction of the original mesh faces."""
    implementation: Literal["v1", "v2"] = "v2"
    """Which implementation to use: v1 (legacy) or v2 (improved)."""
    num_directions: int = 6
    """Number of ray directions per texel for averaging (v2 only)."""
    use_gpu_rasterization: bool = True
    """Use GPU rasterization if available (v2 only, requires nvdiffrast)."""

    def main(self) -> None:
        """Export textured mesh"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        # Load the mesh
        mesh = get_mesh_from_filename(str(self.input_mesh_filename), target_num_faces=self.target_num_faces)

        # Load the pipeline
        _, pipeline, _ = eval_setup(self.load_config, test_mode="inference")

        if self.implementation == "v1":
            CONSOLE.print("[yellow]Using v1 (legacy) texture export")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                px_per_uv_triangle=self.px_per_uv_triangle,
                output_dir=self.output_dir,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )
        else:
            CONSOLE.print("[green]Using v2 (improved) texture export")
            texture_utils_v2.export_textured_mesh_v2(
                mesh,
                pipeline,
                output_dir=self.output_dir,
                texture_size=self.num_pixels_per_side,
                num_directions=self.num_directions,
                use_gpu_rasterization=self.use_gpu_rasterization,
            )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[TextureMesh]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(TextureMesh)  # noqa
