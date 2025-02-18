"""
Script to texture an existing mesh file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch
import torchvision

import tyro
from rich.console import Console
from typing_extensions import Literal

from sdfstudio.exporter import texture_utils
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
    """Number of pixels per UV square."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int | float] = 50000
    """Target number of faces for the mesh to texture. If < 1, it is a fraction of the original mesh faces."""

    def main(self) -> None:
        """Export textured mesh"""
        # pylint: disable=too-many-statements

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        # load the Mesh
        mesh = get_mesh_from_filename(str(self.input_mesh_filename), target_num_faces=self.target_num_faces)

        # load the Pipeline
        _, pipeline, _ = eval_setup(self.load_config, test_mode="inference")

        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        texture_utils.export_textured_mesh(
            mesh,
            pipeline,
            px_per_uv_triangle=self.px_per_uv_triangle,
            output_dir=self.output_dir,
            unwrap_method=self.unwrap_method,
            num_pixels_per_side=self.num_pixels_per_side,
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[TextureMesh]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(TextureMesh)  # noqa
