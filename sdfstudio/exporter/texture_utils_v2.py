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
Texture utils v2 - Improved texture export with GPU rasterization and multi-view support.

Path A: GPU rasterization (nvdiffrast) + fast mesh I/O (trimesh) + multi-direction averaging
Path B: Render-and-reproject using synthetic NeRF views
"""

# pylint: disable=no-member

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch
import trimesh  # type: ignore
import xatlas  # type: ignore
from rich.console import Console
from torch import Tensor

from sdfstudio.cameras import camera_utils
from sdfstudio.cameras.cameras import Cameras
from sdfstudio.cameras.rays import RayBundle
from sdfstudio.exporter.exporter_utils import Mesh
from sdfstudio.pipelines.base_pipeline import Pipeline
from sdfstudio.utils import poses as pose_utils
from sdfstudio.utils.rich_utils import get_progress

# Optional nvdiffrast import
try:
    import nvdiffrast.torch as dr  # type: ignore

    NVDIFFRAST_AVAILABLE = True
except ImportError:
    NVDIFFRAST_AVAILABLE = False
    dr = None  # type: ignore

# Check for open3d availability (Path B)
OPEN3D_AVAILABLE = importlib.util.find_spec("open3d") is not None

CONSOLE = Console(width=120)


def _default_reproject_chunk_size(eval_num_rays_per_chunk: int) -> int:
    """Heuristic chunk size for multiview reprojection.

    Tied to `ModelConfig.eval_num_rays_per_chunk` so users can trade VRAM for speed
    with a single knob.
    """
    # 64x provides good throughput while keeping peak memory bounded.
    return int(max(50_000, min(1_000_000, 64 * eval_num_rays_per_chunk)))


# =============================================================================
# Path A: GPU Rasterization + Fast I/O + Multi-direction Averaging
# =============================================================================


def rasterize_uv_gpu(
    uv_coords: Tensor,
    faces: Tensor,
    texture_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Rasterize UV coordinates to get per-pixel triangle IDs and barycentric coords.

    Uses nvdiffrast for GPU-accelerated rasterization.

    Args:
        uv_coords: UV coordinates per vertex (num_verts, 2) in [0, 1]
        faces: Face indices (num_faces, 3)
        texture_size: Output texture resolution (square)
        device: Torch device

    Returns:
        triangle_ids: (H, W) tensor of triangle indices (-1 for empty pixels)
        bary_coords: (H, W, 3) tensor of barycentric coordinates
        rast_out: (1, H, W, 4) raw raster output from nvdiffrast
    """
    if not NVDIFFRAST_AVAILABLE:
        raise RuntimeError("nvdiffrast not available, use rasterize_uv_cpu instead")

    # Convert UVs to clip space [-1, 1] for nvdiffrast
    # UV (0,0) is bottom-left, clip space (-1,-1) is bottom-left
    clip_coords = uv_coords * 2.0 - 1.0

    # Add z=0 and w=1 for homogeneous coordinates
    vertices_clip = (
        torch.cat(
            [
                clip_coords,
                torch.zeros_like(clip_coords[..., :1]),
                torch.ones_like(clip_coords[..., :1]),
            ],
            dim=-1,
        )
        .unsqueeze(0)
        .contiguous()
    )  # (1, V, 4)

    faces_int = faces.int().contiguous()  # nvdiffrast needs int32

    # Create rasterization context
    assert dr is not None, "nvdiffrast is required for GPU rasterization"
    glctx = dr.RasterizeCudaContext() if device.type == "cuda" else dr.RasterizeGLContext()

    # Rasterize
    rast_out, _ = dr.rasterize(glctx, vertices_clip, faces_int, resolution=[texture_size, texture_size])

    # rast_out is (1, H, W, 4): [u, v, z, triangle_id]
    # triangle_id is 1-indexed (0 means no triangle)
    triangle_ids = rast_out[0, :, :, 3].long() - 1  # Convert to 0-indexed, -1 for empty

    # Barycentric coordinates from u, v
    # nvdiffrast outputs (u, v) where w = 1 - u - v
    u = rast_out[0, :, :, 0]
    v = rast_out[0, :, :, 1]
    w = 1.0 - u - v
    bary_coords = torch.stack([w, u, v], dim=-1)  # (H, W, 3)

    return triangle_ids, bary_coords, rast_out


def rasterize_uv_cpu(
    uv_coords: Tensor,
    faces: Tensor,
    texture_size: int,
) -> tuple[Tensor, Tensor, None]:
    """CPU fallback for UV rasterization using vectorized operations.

    Args:
        uv_coords: UV coordinates per vertex (num_verts, 2) in [0, 1]
        faces: Face indices (num_faces, 3)
        texture_size: Output texture resolution (square)

    Returns:
        triangle_ids: (H, W) tensor of triangle indices (-1 for empty pixels)
        bary_coords: (H, W, 3) tensor of barycentric coordinates
        rast_out: Always None (API parity with rasterize_uv_gpu)
    """
    device = uv_coords.device
    H = W = texture_size
    num_faces = faces.shape[0]

    # Create pixel grid - pixel centers
    px_size = 1.0 / texture_size
    u_coords = torch.linspace(px_size / 2, 1 - px_size / 2, W, device=device)
    v_coords = torch.linspace(px_size / 2, 1 - px_size / 2, H, device=device)
    grid_u, grid_v = torch.meshgrid(u_coords, v_coords, indexing="xy")
    pixel_uvs = torch.stack([grid_u, grid_v], dim=-1)  # (H, W, 2)

    # Get triangle UVs
    tri_uvs = uv_coords[faces]  # (F, 3, 2)

    # Initialize outputs
    triangle_ids = torch.full((H, W), -1, dtype=torch.long, device=device)
    bary_coords = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
    min_dist = torch.full((H, W), float("inf"), device=device)

    # Flatten pixel UVs for broadcasting: (1, H*W, 2)
    p = pixel_uvs.reshape(1, -1, 2)
    pixel_arange = torch.arange(H * W, device=device)

    # Process faces in chunks to avoid OOM
    # Adaptive chunk size based on texture resolution
    chunk_size = max(1, 100000 // (H * W // 1000 + 1))

    progress = get_progress("Rasterizing UVs (CPU)")
    with progress:
        for start in progress.track(range(0, num_faces, chunk_size)):
            end = min(start + chunk_size, num_faces)

            # Get UVs for this chunk: (chunk, 3, 2)
            v0 = tri_uvs[start:end, 0:1, :]  # (chunk, 1, 2)
            v1 = tri_uvs[start:end, 1:2, :]
            v2 = tri_uvs[start:end, 2:3, :]

            # Compute barycentric coordinates using the standard method:
            # Given point P and triangle (V0, V1, V2), solve:
            #   P = u*V0 + v*V1 + w*V2, where u + v + w = 1
            # Using vectors from V0:
            #   P - V0 = v*(V1-V0) + w*(V2-V0)
            v0v1 = v1 - v0  # (chunk, 1, 2)
            v0v2 = v2 - v0
            v0p = p - v0  # (chunk, H*W, 2) via broadcasting

            # Solve 2x2 system using dot products (Cramer's rule)
            d00 = (v0v1 * v0v1).sum(-1)  # (chunk, 1)
            d01 = (v0v1 * v0v2).sum(-1)
            d11 = (v0v2 * v0v2).sum(-1)
            d20 = (v0p * v0v1).sum(-1)  # (chunk, H*W)
            d21 = (v0p * v0v2).sum(-1)

            denom = d00 * d11 - d01 * d01
            # Reject degenerate UV triangles.
            #
            # If we "fix" denom by setting it to 1 for near-zero-area triangles,
            # barycentrics collapse to (1,0,0) and every pixel appears "inside"
            # that triangle. That inflates coverage to ~100% and makes the CPU
            # path query the NeRF for atlas background pixels.
            degenerate = torch.abs(denom) < 1e-12
            denom_safe = torch.where(degenerate, torch.ones_like(denom), denom)

            # bary_v is weight for V1, bary_w is weight for V2
            bary_v = (d11 * d20 - d01 * d21) / denom_safe  # (chunk, H*W)
            bary_w = (d00 * d21 - d01 * d20) / denom_safe
            bary_u = 1.0 - bary_v - bary_w  # weight for V0

            # Conservative inside test to avoid cracks between adjacent triangles.
            #
            # GPU rasterizers typically cover shared edges consistently; a strict
            # float barycentric test on CPU often misses boundary pixels, creating
            # "gaps" that later show up as seams. Use an epsilon tied to texel size
            # and allow slight overshoot above 1 due to numeric error.
            eps = max(1e-5, 2.0 / float(texture_size))
            inside = (
                (bary_u >= -eps)
                & (bary_v >= -eps)
                & (bary_w >= -eps)
                & (bary_u <= 1.0 + eps)
                & (bary_v <= 1.0 + eps)
                & (bary_w <= 1.0 + eps)
            )  # (chunk, H*W)
            inside = inside & (~degenerate)  # broadcast (chunk, 1) over pixels

            # Distance metric for tie-breaking: deviation from center (1/3, 1/3, 1/3)
            dist = torch.abs(bary_u - 1 / 3) + torch.abs(bary_v - 1 / 3) + torch.abs(bary_w - 1 / 3)
            dist = torch.where(inside, dist, torch.full_like(dist, float("inf")))

            # Find best triangle per pixel within this chunk
            best_dist_chunk, best_idx_chunk = dist.min(dim=0)  # (H*W,)

            # Reshape for 2D operations
            best_dist_chunk_2d = best_dist_chunk.reshape(H, W)
            best_idx_chunk_2d = best_idx_chunk.reshape(H, W)

            # Update where this chunk has better (closer to center) triangles
            update_mask = best_dist_chunk_2d < min_dist

            # Update minimum distance
            min_dist = torch.where(update_mask, best_dist_chunk_2d, min_dist)

            # Update triangle IDs (add chunk offset to get global face index)
            new_tri_ids = start + best_idx_chunk_2d
            triangle_ids = torch.where(update_mask, new_tri_ids, triangle_ids)

            # Gather and update barycentric coordinates
            # best_idx_chunk indexes into chunk dimension [0, chunk_size_actual)
            # pixel_arange indexes into pixel dimension [0, H*W)
            new_bary = torch.stack(
                [
                    bary_u[best_idx_chunk, pixel_arange],
                    bary_v[best_idx_chunk, pixel_arange],
                    bary_w[best_idx_chunk, pixel_arange],
                ],
                dim=-1,
            ).reshape(H, W, 3)
            bary_coords = torch.where(update_mask.unsqueeze(-1), new_bary, bary_coords)

    # Clamp/renormalize barycentrics for numeric stability on boundary pixels we
    # accepted via the conservative test above.
    valid = triangle_ids >= 0
    if bool(valid.any()):
        bc = bary_coords[valid].clamp_min(0.0)
        bc = bc / bc.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        bary_coords[valid] = bc

    return triangle_ids, bary_coords, None


def pad_textures_nearest(
    textures: dict[str, Tensor],
    valid_mask: Tensor,
    pad_px: int = 32,
) -> dict[str, Tensor]:
    """Pad texture charts by propagating colors outward from valid pixels.

    This avoids black seams from rasterization gaps and prevents mipmap/bilinear
    bleeding from undefined atlas regions, without querying the NeRF outside the
    rasterized triangles.

    Note:
      - This is seam dilation, not atlas-background filling. Large empty regions
        are expected for typical unwraps, and should remain uncolored.
    """
    if pad_px <= 0:
        return textures

    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.bool()

    # Nothing to do if already full coverage
    if bool(valid_mask.all()):
        return textures

    def shift_hw(x: Tensor, dy: int, dx: int) -> Tensor:
        h, w = x.shape[:2]
        out = torch.zeros_like(x)

        src_y0 = max(0, -dy)
        src_y1 = h - max(0, dy)
        src_x0 = max(0, -dx)
        src_x1 = w - max(0, dx)

        dst_y0 = max(0, dy)
        dst_y1 = h - max(0, -dy)
        dst_x0 = max(0, dx)
        dst_x1 = w - max(0, -dx)

        out[dst_y0:dst_y1, dst_x0:dst_x1] = x[src_y0:src_y1, src_x0:src_x1]
        return out

    neighbors = (
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )

    padded: dict[str, Tensor] = {}
    for name, tex in textures.items():
        padded[name] = tex

    valid = valid_mask.clone()
    for _ in range(pad_px):
        if bool(valid.all()):
            break

        filled_any = False
        inv = ~valid

        for dy, dx in neighbors:
            neigh_valid = shift_hw(valid, dy, dx)
            fill = inv & neigh_valid
            if not bool(fill.any()):
                continue

            for name, tex in padded.items():
                if tex.ndim != 3:
                    continue
                neigh_tex = shift_hw(tex, dy, dx)
                padded[name] = torch.where(fill.unsqueeze(-1), neigh_tex, tex)

            valid = valid | fill
            filled_any = True
            inv = ~valid

        if not filled_any:
            break

    return padded


def generate_hemisphere_directions(normal: Tensor, num_dirs: int = 6) -> Tensor:
    """Generate directions on a hemisphere oriented along the normal.

    Args:
        normal: Surface normal (*, 3)
        num_dirs: Number of directions to generate

    Returns:
        directions: (*, num_dirs, 3) unit vectors on hemisphere
    """
    device = normal.device
    shape = normal.shape[:-1]

    # Create orthonormal basis from normal
    # Find a vector not parallel to normal
    up = torch.zeros(*shape, 3, device=device)
    up[..., 2] = 1.0

    # If normal is parallel to up, use different vector
    parallel_mask = torch.abs(normal[..., 2]) > 0.99
    up[parallel_mask] = torch.tensor([1.0, 0.0, 0.0], device=device)

    # Gram-Schmidt to get tangent and bitangent
    tangent = torch.cross(up, normal, dim=-1)
    tangent = tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-8)
    bitangent = torch.cross(normal, tangent, dim=-1)

    # Generate directions: normal + ring around it at 45 degrees
    directions = []

    # Central direction (along normal)
    directions.append(normal)

    # Ring of directions at ~45 degrees from normal
    cos_45 = math.cos(math.pi / 4)
    sin_45 = math.sin(math.pi / 4)

    for i in range(num_dirs - 1):
        angle = 2 * math.pi * i / (num_dirs - 1)
        # Direction = cos(45)*normal + sin(45)*(cos(angle)*tangent + sin(angle)*bitangent)
        ring_dir = cos_45 * normal + sin_45 * (math.cos(angle) * tangent + math.sin(angle) * bitangent)
        # Normalize to ensure unit vectors
        ring_dir = ring_dir / (ring_dir.norm(dim=-1, keepdim=True) + 1e-8)
        directions.append(ring_dir)

    return torch.stack(directions, dim=-2)  # (*, num_dirs, 3)


def unwrap_mesh_with_xatlas_v2(
    vertices: Tensor,
    faces: Tensor,
    vertex_normals: Tensor,
    texture_size: int = 1024,
    use_gpu: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Unwrap mesh using xatlas and rasterize to get per-pixel 3D positions/normals.

    Args:
        vertices: Mesh vertices (V, 3)
        faces: Mesh faces (F, 3)
        vertex_normals: Vertex normals (V, 3)
        texture_size: Texture resolution
        use_gpu: Whether to use GPU rasterization (requires nvdiffrast)

    Returns:
        texture_uvs: Per-face UV coordinates (F, 3, 2)
        faces_uv_order: Per-face vertex indices aligned to UV corner order (F, 3)
        origins: Per-pixel 3D positions (H, W, 3)
        normals: Per-pixel normals (H, W, 3)
        valid_mask: Per-pixel validity mask (H, W)
    """
    device = vertices.device

    # Run xatlas UV unwrapping
    CONSOLE.print("Running xatlas UV unwrapping...")
    vertices_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()
    normals_np = vertex_normals.cpu().numpy()

    vmapping, indices, uvs = xatlas.parametrize(vertices_np, faces_np, normals_np)

    # uvs: (num_new_verts, 2), indices: (F, 3) indexing into uvs
    uv_coords = torch.from_numpy(uvs.astype(np.float32)).to(device)
    uv_faces = torch.from_numpy(indices.astype(np.int64)).to(device)
    texture_uvs = uv_coords[uv_faces]  # (F, 3, 2)

    # Map UV vertices back to the original mesh vertex indices.
    #
    # xatlas creates seam-split UV vertices and can reorder triangle corners. Since we
    # rasterize in UV space, barycentric weights are with respect to that UV-corner
    # order. Using vmapping keeps UV rasterization -> 3D interpolation aligned, which
    # avoids corner swaps / wrong surface points.
    vmapping_t = torch.from_numpy(vmapping.astype(np.int64)).to(device)  # (num_new_verts,)
    faces_uv_order = vmapping_t[uv_faces]  # (F, 3) original vertex indices per UV corner

    if not torch.equal(faces_uv_order, faces):
        CONSOLE.print("[yellow]xatlas remapped face corner order; using vmapping for geometry alignment")

    CONSOLE.print(f"[green]UV unwrapping complete: {len(uvs)} UV vertices")

    # Rasterize UV space
    CONSOLE.print(f"Rasterizing UV space ({texture_size}x{texture_size})...")

    if use_gpu and NVDIFFRAST_AVAILABLE and device.type == "cuda":
        triangle_ids, bary_coords, rast_out = rasterize_uv_gpu(uv_coords, uv_faces, texture_size, device)
    else:
        if use_gpu and not NVDIFFRAST_AVAILABLE:
            CONSOLE.print("[yellow]nvdiffrast not available, falling back to CPU rasterization")
        triangle_ids, bary_coords, rast_out = rasterize_uv_cpu(uv_coords, uv_faces, texture_size)

    # Valid mask: pixels that belong to a triangle (atlas background stays invalid)
    valid_mask = triangle_ids >= 0

    if use_gpu and NVDIFFRAST_AVAILABLE and device.type == "cuda" and rast_out is not None:
        # Interpolate on the UV vertex domain directly using nvdiffrast.
        #
        # Why:
        # - Interpreting rasterizer barycentrics manually is easy to get subtly wrong
        #   once seam splits / corner reindexing enter the picture.
        # - dr.interpolate(...) performs interpolation in exactly the same space as
        #   rasterization, which fixed the global "grain/noise" pattern seen in the
        #   v2 GPU texture output.
        assert dr is not None, "nvdiffrast is required for GPU interpolation"
        pos_uv = vertices[vmapping_t].contiguous()  # (V_uv, 3)
        nrm_uv = vertex_normals[vmapping_t].contiguous()  # (V_uv, 3)
        uv_faces_int = uv_faces.int().contiguous()

        interp_pos, _ = dr.interpolate(pos_uv.unsqueeze(0), rast_out, uv_faces_int)  # type: ignore
        interp_nrm, _ = dr.interpolate(nrm_uv.unsqueeze(0), rast_out, uv_faces_int)  # type: ignore

        origins = interp_pos[0]
        normals = torch.nn.functional.normalize(interp_nrm[0], dim=-1)
    else:
        # Clamp triangle IDs for gathering (invalid will be masked out anyway)
        triangle_ids_clamped = triangle_ids.clamp(min=0)

        # Get vertex indices for each pixel's triangle (aligned to UV corner order)
        pixel_faces = faces_uv_order[triangle_ids_clamped]  # (H, W, 3)

        # Gather vertex positions and normals
        pixel_verts = vertices[pixel_faces]  # (H, W, 3, 3)
        pixel_norms = vertex_normals[pixel_faces]  # (H, W, 3, 3)

        # Interpolate using barycentric coordinates
        bary = bary_coords.unsqueeze(-1)  # (H, W, 3, 1)
        origins = (pixel_verts * bary).sum(dim=2)  # (H, W, 3)
        normals = (pixel_norms * bary).sum(dim=2)  # (H, W, 3)
        normals = torch.nn.functional.normalize(normals, dim=-1)

    CONSOLE.print("[green]Rasterization complete")

    return texture_uvs, faces_uv_order, origins, normals, valid_mask


def query_nerf_textures(
    pipeline: Pipeline,
    origins: Tensor,
    normals: Tensor,
    valid_mask: Tensor,
    num_directions: int = 6,
    ray_length: float = 0.1,
) -> dict[str, Tensor]:
    """Query NeRF to extract all texture maps (RGB, normals, PBR).

    RGB is averaged across multiple directions (view-dependent).
    Normals and material properties (diffuse, specular, roughness, tint)
    are extracted from a single query along the surface normal (intrinsic).

    Args:
        pipeline: NeRF pipeline
        origins: Surface positions (H, W, 3)
        normals: Surface normals (H, W, 3)
        valid_mask: Valid pixel mask (H, W)
        num_directions: Number of ray directions per texel for RGB averaging
        ray_length: Length of rays to cast

    Returns:
        Dict with texture maps. Always contains "rgb". May also contain
        "normal", "diffuse", "specular", "roughness", "tint" if available.
        All tensors are (H, W, 3) or (H, W, 1) depending on channels.
    """
    device = pipeline.device
    H, W = origins.shape[:2]

    # Work only on valid texels to avoid allocating (H*W*num_dirs) buffers.
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.bool()
    flat_surface_origins = origins[valid_mask].reshape(-1, 3)  # (N, 3)
    flat_surface_normals = normals[valid_mask].reshape(-1, 3)  # (N, 3)

    N = flat_surface_origins.shape[0]
    if N == 0:
        return {"rgb": torch.zeros(H, W, 3, device=device)}

    CONSOLE.print(f"Querying NeRF: {N} valid texels x {num_directions} directions")

    # Generate multiple directions per texel (pointing into surface / away from camera).
    # We pass -normals so hemisphere points into the surface.
    directions = generate_hemisphere_directions(-flat_surface_normals, num_directions)  # (N, num_dirs, 3)
    directions = torch.nn.functional.normalize(directions, dim=-1)

    # Storage for all outputs
    all_rgb = []
    other_outputs: dict[str, Tensor] = {}

    with torch.no_grad():
        for d in range(num_directions):
            CONSOLE.print(f"[cyan]Rendering direction {d + 1}/{num_directions}")
            dir_dirs = directions[:, d, :]  # (N, 3)
            dir_origins = flat_surface_origins - 0.5 * ray_length * dir_dirs  # (N, 3)

            # Reshape to (1, N) "image" - the model will chunk internally
            ray_bundle = RayBundle(
                origins=dir_origins.unsqueeze(0),  # (1, N, 3)
                directions=dir_dirs.unsqueeze(0),  # (1, N, 3)
                pixel_area=torch.ones(1, N, 1, device=device),
                camera_indices=torch.zeros(1, N, 1, device=device, dtype=torch.long),
                directions_norm=torch.ones(1, N, 1, device=device),
                nears=torch.zeros(1, N, 1, device=device),
                fars=torch.full((1, N, 1), ray_length, device=device),
            )

            # Inner progress bar shows ray chunking (many samples = good ETA)
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle, progress=True)

            # First direction: extract all available outputs for normals/PBR
            if d == 0:
                for key in ["normal", "diffuse", "specular", "roughness", "tint"]:
                    if key in outputs:
                        # Output shape is (1, N, C), flatten to (N, C)
                        other_outputs[key] = outputs[key].reshape(N, -1)

            # All directions: collect RGB for averaging
            all_rgb.append(outputs["rgb"].reshape(N, 3))

    # Average RGB across directions
    stacked_rgb = torch.stack(all_rgb, dim=1)  # (N, num_dirs, 3)
    avg_rgb = stacked_rgb.mean(dim=1)  # (N, 3)

    # Reconstruct full textures
    result = {}

    # RGB (averaged)
    rgb = torch.zeros(H, W, 3, device=device)
    rgb[valid_mask] = avg_rgb.to(device)
    result["rgb"] = rgb

    # Other outputs (single query, not averaged)
    for key, value in other_outputs.items():
        channels = value.shape[-1]
        tex = torch.zeros(H, W, channels, device=device)
        tex[valid_mask] = value.to(device)
        result[key] = tex

    return result


def write_textured_mesh_fast(
    vertices: Tensor,
    faces: Tensor,
    vertex_normals: Tensor,
    texture_uvs: Tensor,
    texture_image: np.ndarray,
    output_dir: Path,
    mesh_name: str = "mesh",
    *,
    flip_v: bool = True,
) -> None:
    """Write textured mesh (OBJ+MTL + GLB).

    Args:
        vertices: Mesh vertices (V, 3)
        faces: Mesh faces (F, 3)
        vertex_normals: Vertex normals (V, 3)
        texture_uvs: Per-face UV coordinates (F, 3, 2)
        texture_image: Texture image (H, W, 3) as numpy array [0, 1]
        output_dir: Output directory
        mesh_name: Base name for output files
        flip_v: Whether to flip V when exporting OBJ/GLB.

            This is intentionally explicit because different UV sources used by
            this file have different V conventions relative to how viewers sample
            textures:
            - xatlas unwrap + image-space textures: typically needs V flip for OBJ.
            - Open3D uvatlas + project_images_to_albedo: already correct for OBJ;
              flipping V breaks mapping (islands land on wrong parts).
    """
    import PIL.Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy
    verts_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()
    normals_np = vertex_normals.cpu().numpy()
    uvs_np = texture_uvs.cpu().numpy()  # (F, 3, 2)

    # Trimesh wants per-face UVs as (F*3, 2) with corresponding face_indices
    # Or we can use TextureVisuals with uv per vertex after de-duplicating

    # For OBJ export with per-face UVs, we need to expand vertices
    # Each face gets its own copy of vertices with unique UVs
    num_faces = faces_np.shape[0]

    # Expand: each face becomes 3 unique vertices
    expanded_verts = verts_np[faces_np].reshape(-1, 3)  # (F*3, 3)
    expanded_normals = normals_np[faces_np].reshape(-1, 3)  # (F*3, 3)
    expanded_uvs = uvs_np.reshape(-1, 2)  # (F*3, 2)
    expanded_faces = np.arange(num_faces * 3).reshape(-1, 3)  # (F, 3)

    if flip_v:
        # OBJ importers usually interpret V as increasing upward; when the texture
        # was generated in image space, we flip to keep sampling consistent.
        expanded_uvs[:, 1] = 1.0 - expanded_uvs[:, 1]

    # Export OBJ + MTL ourselves.
    #
    # Trimesh's OBJ exporter tends to emit an extra `material_0.png` next to the OBJ,
    # duplicating `texture.png` and confusing downstream tooling. We want a single
    # canonical filename: `texture.png`.
    obj_path = output_dir / f"{mesh_name}.obj"
    mtl_path = output_dir / f"{mesh_name}.mtl"

    # MTL assumes texture.png already exists in output_dir.
    mtl_path.write_text(
        "\n".join(
            [
                "newmtl material_0",
                "Ka 0 0 0",
                "Kd 1 1 1",
                "Ks 0 0 0",
                "d 1.0",
                "illum 1",
                "map_Kd texture.png",
                "",
            ]
        )
    )

    lines: list[str] = []
    lines.append(f"mtllib {mtl_path.name}")
    lines.append(f"o {mesh_name}")
    for v in expanded_verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for vt in expanded_uvs:
        lines.append(f"vt {vt[0]:.6f} {vt[1]:.6f}")
    for vn in expanded_normals:
        lines.append(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}")
    lines.append("usemtl material_0")
    for i in range(num_faces):
        a = 3 * i + 1
        b = 3 * i + 2
        c = 3 * i + 3
        lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
    obj_path.write_text("\n".join(lines) + "\n")

    # Also export GLB for convenience
    texture_uint8 = (np.clip(texture_image, 0, 1) * 255).astype(np.uint8)
    texture_pil = PIL.Image.fromarray(texture_uint8)
    material = trimesh.visual.material.SimpleMaterial(  # type: ignore
        image=texture_pil,
        diffuse=[255, 255, 255, 255],
    )
    visuals = trimesh.visual.TextureVisuals(  # type: ignore
        uv=expanded_uvs,
        material=material,
    )
    mesh_tm = trimesh.Trimesh(
        vertices=expanded_verts,
        faces=expanded_faces,
        vertex_normals=expanded_normals,
        visual=visuals,
        process=False,  # Don't merge vertices
    )
    glb_path = output_dir / f"{mesh_name}.glb"
    mesh_tm.export(glb_path, file_type="glb")


def export_textured_mesh_v2(
    mesh: Mesh,
    pipeline: Pipeline,
    output_dir: Path,
    texture_size: int = 2048,
    num_directions: int = 6,
    use_gpu_rasterization: bool = True,
    pad_px: int = 32,
) -> None:
    """Export textured mesh using improved pipeline.

    Args:
        mesh: Input mesh
        pipeline: Trained NeRF/SDF pipeline
        output_dir: Output directory
        texture_size: Texture resolution
        num_directions: Number of ray directions per texel for averaging
        use_gpu_rasterization: Whether to use GPU rasterization
        pad_px: Number of pixels to dilate charts outward to avoid seam bleeding
    """
    device = pipeline.device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove legacy duplicate artifacts if present (from older exporters / trimesh OBJ export).
    for legacy_name in ("material_0.png", "material_0.mtl"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    vertices = mesh.vertices.to(device)
    faces = mesh.faces.to(device)
    vertex_normals = mesh.normals.to(device)

    CONSOLE.print(f"[bold]Texturing mesh: {len(vertices)} vertices, {len(faces)} faces")

    # Step 1: UV unwrap and rasterize
    texture_uvs, faces_uv_order, origins, normals, valid_mask = unwrap_mesh_with_xatlas_v2(
        vertices,
        faces,
        vertex_normals,
        texture_size=texture_size,
        use_gpu=use_gpu_rasterization,
    )
    # Helpful debug signal: many unwraps have low coverage because the atlas is
    # sparse by design. This is not the same thing as rasterization "gaps".
    coverage = float(valid_mask.float().mean().item() * 100.0)
    CONSOLE.print(f"UV atlas coverage: {coverage:.1f}%")

    # Step 2: Compute ray length from mesh scale
    face_verts = vertices[faces_uv_order]
    edge_lengths = torch.norm(face_verts[:, 1] - face_verts[:, 0], dim=-1)
    ray_length = 2.0 * edge_lengths.mean().item()
    CONSOLE.print(f"Ray length: {ray_length:.4f}")

    # Step 3: Query NeRF with multi-direction averaging (RGB) and single query (normals/PBR)
    textures = query_nerf_textures(
        pipeline,
        origins,
        normals,
        valid_mask,
        num_directions=num_directions,
        ray_length=ray_length,
    )

    # Step 3.5: Seam dilation to avoid seam bleeding / raster gaps without querying outside triangles.
    textures = pad_textures_nearest(textures, valid_mask, pad_px=pad_px)

    # Step 4: Save all texture maps
    import mediapy as media  # type: ignore

    saved_textures = []
    for name, tex in textures.items():
        img = tex.cpu().numpy()

        # Normals are in [-1, 1], convert to [0, 1] for image saving
        if name == "normal":
            img = img * 0.5 + 0.5

        # Use "texture.png" for RGB to maintain compatibility with mesh.mtl
        filename = "texture.png" if name == "rgb" else f"{name}.png"
        filepath = output_dir / filename
        media.write_image(str(filepath), img)
        saved_textures.append(filepath)

    # Write mesh (uses texture.png for the main material)
    texture_image = textures["rgb"].cpu().numpy()
    write_textured_mesh_fast(
        vertices,
        faces_uv_order,
        vertex_normals,
        texture_uvs,
        texture_image,
        output_dir,
        mesh_name="mesh",
        flip_v=True,
    )

    # Log results
    CONSOLE.print("[bold green]Texture export complete!")
    CONSOLE.print(f"  Mesh: {output_dir / 'mesh.obj'} (+ mesh.mtl)")
    CONSOLE.print(f"  Binary: {output_dir / 'mesh.glb'}")
    CONSOLE.print("  Textures:")
    for tex_path in saved_textures:
        CONSOLE.print(f"    - {tex_path}")


# =============================================================================
# Multiview render-and-project/reproject
# =============================================================================


def generate_camera_poses_on_sphere(
    center: Tensor,
    radius: float,
    num_views: int = 30,
    elevation_range: tuple[float, float] = (-30, 60),
    image_size: tuple[int, int] = (1024, 1024),
    fov_degrees: float = 60.0,
) -> tuple[Tensor, Tensor]:
    """Generate camera poses on a sphere looking at center.

    Args:
        center: Center point to look at (3,)
        radius: Distance from center
        num_views: Number of camera views
        elevation_range: Min/max elevation angles in degrees
        image_size: Render resolution (H, W) used to construct intrinsics
        fov_degrees: Vertical field of view in degrees

    Returns:
        intrinsics: Camera intrinsics (num_views, 3, 3)
        extrinsics: Camera-to-world poses (num_views, 4, 4)
    """
    device = center.device
    center_f = center.to(dtype=torch.float32)

    elev_min_deg, elev_max_deg = elevation_range
    elev_min = math.radians(elev_min_deg)
    elev_max = math.radians(elev_max_deg)

    # Deterministic sampling: golden-angle spiral in azimuth + linear elevation sweep.
    # Coordinates are in a z-up world.
    i = torch.arange(num_views, device=device, dtype=torch.float32)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    azimuth = i * golden_angle
    elevation = torch.linspace(elev_min, elev_max, num_views, device=device, dtype=torch.float32)

    cos_e = torch.cos(elevation)
    sin_e = torch.sin(elevation)
    x = cos_e * torch.cos(azimuth)
    y = cos_e * torch.sin(azimuth)
    z = sin_e

    positions = center_f[None, :] + radius * torch.stack([x, y, z], dim=-1)  # (N, 3)

    # Camera-to-world poses (3x4)
    up_default = torch.tensor([0.0, 0.0, 1.0], device=device)
    up_fallback = torch.tensor([0.0, 1.0, 0.0], device=device)
    c2ws = []
    for pos in positions:
        # camera_utils.viewmatrix expects "lookat" to be the camera's +Z axis (back direction);
        # for cameras looking at the center, that's (pos - center).
        lookat = pos - center_f
        lookat_n = lookat / (lookat.norm() + 1e-8)
        up = up_fallback if torch.abs(torch.dot(lookat_n, up_default)) > 0.99 else up_default
        c2w = camera_utils.viewmatrix(lookat, up, pos)  # (3, 4)
        c2ws.append(c2w)
    c2ws_t = torch.stack(c2ws, dim=0)  # (N, 3, 4)
    extrinsics = pose_utils.to4x4(c2ws_t)  # (N, 4, 4)

    # Intrinsics (same for all views)
    H, W = image_size
    # Use vertical fov to derive focal length in pixels.
    fy = 0.5 * H / math.tan(math.radians(fov_degrees) / 2.0)
    fx = fy
    cx = W / 2.0
    cy = H / 2.0

    intrinsics = torch.zeros((num_views, 3, 3), device=device, dtype=torch.float32)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy
    intrinsics[:, 2, 2] = 1.0

    return intrinsics, extrinsics


def render_views_from_nerf(
    pipeline: Pipeline,
    intrinsics: Tensor,
    extrinsics: Tensor,
    image_size: tuple[int, int] = (1024, 1024),
) -> tuple[list[Tensor], list[Tensor]]:
    """Render synthetic views from NeRF at given camera poses.

    Args:
        pipeline: Trained NeRF pipeline
        intrinsics: Camera intrinsics (N, 3, 3)
        extrinsics: Camera-to-world poses (N, 4, 4)
        image_size: Output image resolution (H, W)

    Returns:
        rgbs: List of rendered RGB images (N,) each (H, W, 3)
        depths: List of rendered depth images (N,) each (H, W, 1)
    """
    device = pipeline.device
    H, W = image_size

    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics of shape (N, 3, 3), got {tuple(intrinsics.shape)}")
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"Expected extrinsics of shape (N, 4, 4), got {tuple(extrinsics.shape)}")
    if intrinsics.shape[0] != extrinsics.shape[0]:
        raise ValueError("intrinsics and extrinsics must have the same number of views")

    N = intrinsics.shape[0]
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    cameras = Cameras(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=W,
        height=H,
        camera_to_worlds=extrinsics[:, :3, :4],
    ).to(device)

    rgbs: list[Tensor] = []
    depths: list[Tensor] = []
    progress = get_progress("Rendering synthetic views")
    with progress:
        for cam_idx in progress.track(range(N)):
            camera_ray_bundle = cameras.generate_rays(camera_indices=cam_idx).to(device)
            with torch.no_grad():
                # Avoid nested rich progress bars:
                # rich only allows one active "Live" display at a time, so the inner
                # model ray-chunk progress must be disabled when we already show an
                # outer view-render progress bar.
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle, progress=False)
            if "rgb" not in outputs:
                raise KeyError(f"Model outputs did not contain 'rgb' (available: {list(outputs.keys())})")
            rgbs.append(outputs["rgb"])
            if "depth" in outputs:
                depths.append(outputs["depth"])
            elif "expected_depth" in outputs:
                depths.append(outputs["expected_depth"])
            else:
                raise KeyError(
                    f"Model outputs did not contain 'depth' (available: {list(outputs.keys())}); "
                    "Multiview reprojection needs depth for visibility"
                )

    return rgbs, depths


def project_views_to_texture_open3d(
    mesh: Mesh,
    images: list[Tensor],
    intrinsics: Tensor,
    extrinsics: Tensor,
    texture_size: int = 2048,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Project rendered views onto mesh texture using Open3D.

    Args:
        mesh: Input mesh
        images: Rendered RGB images
        intrinsics: Camera intrinsics (N, 3, 3)
        extrinsics: Camera-to-world poses (N, 4, 4)
        texture_size: Output texture resolution

    Returns:
        texture: Projected texture (H, W, 3)
        texture_uvs: Per-face UV coordinates (F, 3, 2) in [0, 1]
        vertices: Vertex positions for the UV-mapped mesh (V, 3)
        faces: Triangle indices for the UV-mapped mesh (F, 3)
        vertex_normals: Vertex normals for the UV-mapped mesh (V, 3)
    """
    if not OPEN3D_AVAILABLE:
        raise RuntimeError("Open3D not available for multiview projection")

    import open3d as o3d  # type: ignore
    import open3d.core as o3c  # type: ignore

    if intrinsics.shape[0] != extrinsics.shape[0] or len(images) != intrinsics.shape[0]:
        raise ValueError("images, intrinsics, and extrinsics must have the same number of views")

    # Convert mesh to Open3D
    verts_np = mesh.vertices.detach().cpu().numpy().astype(np.float32)
    faces_np = mesh.faces.detach().cpu().numpy().astype(np.int32)

    mesh_legacy = o3d.geometry.TriangleMesh()  # type: ignore[attr-defined]
    mesh_legacy.vertices = o3d.utility.Vector3dVector(verts_np)  # type: ignore[attr-defined]
    mesh_legacy.triangles = o3d.utility.Vector3iVector(faces_np)  # type: ignore[attr-defined]
    # Keep normals sane if Open3D decides to duplicate / reorder vertices during UV atlas generation.
    mesh_legacy.compute_vertex_normals()  # type: ignore[attr-defined]

    if not hasattr(o3d.t.geometry.TriangleMesh, "from_legacy"):
        raise RuntimeError("Open3D Tensor TriangleMesh.from_legacy() not available")
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)  # type: ignore[attr-defined]

    # Compute UV atlas.
    #
    # Important: compute_uvatlas() mutates the mesh and returns stats (tuple).
    # Assigning its return would replace the mesh with a tuple and make subsequent
    # projection calls fail in misleading ways.
    if not hasattr(mesh_t, "compute_uvatlas"):
        raise RuntimeError("Open3D TriangleMesh.compute_uvatlas() not available")
    try:
        # NOTE: compute_uvatlas() mutates the mesh and returns statistics
        # (max_stretch, num_charts, num_partitions), not a mesh.
        mesh_t.compute_uvatlas(size=texture_size)  # type: ignore[call-arg]
    except TypeError:
        mesh_t.compute_uvatlas(texture_size)  # type: ignore[misc]

    # Convert rendered images to Open3D tensor images (uint8 RGB)
    o3d_images = []
    for img in images:
        img_np = (img.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
        o3d_images.append(o3d.t.geometry.Image(o3c.Tensor(img_np)))  # type: ignore[attr-defined]

    # Open3D expects extrinsics as world-to-camera and uses an OpenCV-style camera
    # convention (x right, y down, z forward).
    #
    # SDFStudio's Cameras uses x right, y up, z backward (forward is -z). The axis
    # flip (diag([1, -1, -1])) converts conventions so Open3D projects colors to the
    # correct surface locations.
    w2c_sdf = torch.linalg.inv(extrinsics).detach().cpu().numpy().astype(np.float32)
    axis_flip = np.eye(4, dtype=np.float32)
    axis_flip[1, 1] = -1.0
    axis_flip[2, 2] = -1.0
    w2c = axis_flip[None, :, :] @ w2c_sdf
    K = intrinsics.detach().cpu().numpy().astype(np.float32)
    intrinsic_matrices = [o3c.Tensor(K[i]) for i in range(K.shape[0])]
    extrinsic_matrices = [o3c.Tensor(w2c[i]) for i in range(w2c.shape[0])]

    tex_out = None
    try:
        # Open3D expects lists of per-view matrices (not stacked tensors) and the
        # kwarg name is `tex_size` (not `texture_size`).
        tex_out = mesh_t.project_images_to_albedo(
            o3d_images,
            intrinsic_matrices,
            extrinsic_matrices,
            tex_size=texture_size,
            update_material=False,
        )
    except AttributeError as e:
        version = getattr(o3d, "__version__", "unknown")
        raise RuntimeError(
            f"Open3D {version} does not expose TriangleMesh.project_images_to_albedo() on this build "
            f"(mesh type: {type(mesh_t)})."
        ) from e
    except TypeError as e:
        version = getattr(o3d, "__version__", "unknown")
        raise RuntimeError(f"Open3D {version}: project_images_to_albedo() call failed (type mismatch/signature)") from e

    # Extract texture tensor
    if isinstance(tex_out, o3d.t.geometry.Image):  # type: ignore[attr-defined]
        tex_tensor = tex_out.as_tensor()
    elif isinstance(tex_out, o3c.Tensor):
        tex_tensor = tex_out
    else:
        # Some versions may return a dict or a mesh; try to extract a known field.
        tex_tensor = getattr(tex_out, "as_tensor", None)
        if callable(tex_tensor):
            tex_tensor = tex_out.as_tensor()
        elif isinstance(tex_out, dict):
            for k in ("albedo", "texture", "image"):
                if k in tex_out:
                    v = tex_out[k]
                    tex_tensor = v.as_tensor() if hasattr(v, "as_tensor") else v
                    break
        if tex_tensor is None:
            raise RuntimeError(f"Unexpected project_images_to_albedo() return type: {type(tex_out)}")

    tex_np = tex_tensor.numpy()
    if tex_np.dtype != np.float32:
        tex_np = tex_np.astype(np.float32)
    if tex_np.max() > 1.0:
        tex_np = tex_np / 255.0

    # IMPORTANT: compute_uvatlas() may reorder triangles and/or vertices.
    # We export the UV-mapped geometry from Open3D and use its triangle_uvs as the
    # single source of truth. Mixing faces from one mesh with UVs from another is a
    # common failure mode that looks like "texture islands on the wrong parts".
    legacy_with_uv = mesh_t.to_legacy()  # type: ignore[attr-defined]
    verts_out_np = np.asarray(legacy_with_uv.vertices, dtype=np.float32)  # type: ignore[attr-defined]
    faces_out_np = np.asarray(legacy_with_uv.triangles, dtype=np.int64)  # type: ignore[attr-defined]

    # Extract per-face UVs from the UV-mapped mesh (authoritative).
    texture_uvs_np = np.asarray(legacy_with_uv.triangle_uvs, dtype=np.float32)  # type: ignore[attr-defined]
    if texture_uvs_np.size == 0:
        raise RuntimeError("Open3D UV atlas did not populate triangle_uvs")

    # Normalize/reshape to (F, 3, 2)
    if texture_uvs_np.ndim == 2 and texture_uvs_np.shape[1] == 2:
        if texture_uvs_np.shape[0] == faces_out_np.shape[0] * 3:
            texture_uvs_np = texture_uvs_np.reshape(faces_out_np.shape[0], 3, 2)
        else:
            raise RuntimeError(f"Unexpected UV shape: {texture_uvs_np.shape}")
    elif texture_uvs_np.ndim == 3 and texture_uvs_np.shape[1:] == (3, 2):
        pass
    else:
        raise RuntimeError(f"Unexpected UV shape: {texture_uvs_np.shape}")

    normals_out_np = None
    try:
        normals_out_np = np.asarray(legacy_with_uv.vertex_normals, dtype=np.float32)  # type: ignore[attr-defined]
        if normals_out_np.shape != verts_out_np.shape:
            normals_out_np = None
    except Exception:  # noqa: BLE001
        normals_out_np = None
    if normals_out_np is None:
        # Fallback: if vertex normals are missing, compute in Open3D and re-read.
        legacy_with_uv.compute_vertex_normals()  # type: ignore[attr-defined]
        normals_out_np = np.asarray(legacy_with_uv.vertex_normals, dtype=np.float32)  # type: ignore[attr-defined]

    texture = torch.from_numpy(tex_np)
    texture_uvs = torch.from_numpy(texture_uvs_np.astype(np.float32))
    vertices_out = torch.from_numpy(verts_out_np)
    faces_out = torch.from_numpy(faces_out_np)
    vertex_normals_out = torch.from_numpy(normals_out_np)

    return texture, texture_uvs, vertices_out, faces_out, vertex_normals_out


def project_views_to_texture_reproject(
    origins: Tensor,
    normals: Tensor,
    valid_mask: Tensor,
    rgbs: list[Tensor],
    depths: list[Tensor],
    intrinsics: Tensor,
    extrinsics: Tensor,
    *,
    depth_tolerance: float,
    chunk_size: int,
) -> Tensor:
    """Project rendered view images back onto the UV texture by reprojection.

    Uses per-texel 3D positions (origins) and normals (from UV rasterization) and
    blends colors from multiple synthetic views. Occlusions are handled by comparing
    the projected texel depth against the rendered depth map.
    """
    device = origins.device
    H_tex, W_tex = origins.shape[:2]

    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.bool()

    texel_coords = valid_mask.nonzero(as_tuple=False)  # (M, 2)
    if texel_coords.numel() == 0:
        return torch.zeros((H_tex, W_tex, 3), device=device)

    P_all = origins[valid_mask].reshape(-1, 3)  # (M, 3)
    N_all = normals[valid_mask].reshape(-1, 3)  # (M, 3)

    M = P_all.shape[0]
    accum = torch.zeros((M, 3), device=device, dtype=torch.float32)
    wsum = torch.zeros((M, 1), device=device, dtype=torch.float32)

    num_views = len(rgbs)
    if num_views != len(depths) or num_views != intrinsics.shape[0] or num_views != extrinsics.shape[0]:
        raise ValueError("rgbs/depths/intrinsics/extrinsics must have the same number of views")

    for view_idx in range(num_views):
        K = intrinsics[view_idx].to(device=device, dtype=torch.float32)
        c2w = extrinsics[view_idx, :3, :4].to(device=device, dtype=torch.float32)
        R = c2w[:, :3]  # (3, 3)
        t = c2w[:, 3]  # (3,)

        rgb_img = rgbs[view_idx].to(device=device, dtype=torch.float32)
        depth_img = depths[view_idx].to(device=device, dtype=torch.float32)

        if rgb_img.ndim != 3 or rgb_img.shape[-1] != 3:
            raise ValueError(f"Expected rgb image (H, W, 3), got {tuple(rgb_img.shape)}")
        if depth_img.ndim == 3 and depth_img.shape[-1] == 1:
            pass
        elif depth_img.ndim == 2:
            depth_img = depth_img.unsqueeze(-1)
        else:
            raise ValueError(f"Expected depth image (H, W, 1) or (H, W), got {tuple(depth_img.shape)}")

        H_img, W_img = rgb_img.shape[:2]
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Prepare for grid_sample
        rgb_nchw = rgb_img.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        depth_nchw = depth_img.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)

        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            P = P_all[start:end]  # (C, 3)
            N = N_all[start:end]  # (C, 3)

            Pc = (P - t) @ R  # (C, 3) world -> camera
            z = -Pc[:, 2]  # camera looks along -Z

            in_front = z > 1e-6
            denom = z.clamp_min(1e-6)
            x_norm = Pc[:, 0] / denom
            y_norm = Pc[:, 1] / denom

            u = fx * x_norm + cx
            v = cy - fy * y_norm

            in_bounds = (u >= 0.0) & (u <= (W_img - 1)) & (v >= 0.0) & (v <= (H_img - 1))

            # Facing term: prefer views where the surface normal points towards the camera.
            view_dir = (t[None, :] - P)
            view_dir = view_dir / (view_dir.norm(dim=-1, keepdim=True) + 1e-8)
            cos = (N * view_dir).sum(dim=-1).clamp(min=0.0)  # (C,)
            facing = cos > 0.0

            # Sample rendered rgb/depth
            u_norm_gs = (u / (W_img - 1)) * 2.0 - 1.0
            v_norm_gs = (v / (H_img - 1)) * 2.0 - 1.0
            grid = torch.stack([u_norm_gs, v_norm_gs], dim=-1).view(1, -1, 1, 2)  # (1, C, 1, 2)

            rgb_s = torch.nn.functional.grid_sample(rgb_nchw, grid, align_corners=True, mode="bilinear")
            rgb_s = rgb_s[0, :, :, 0].permute(1, 0)  # (C, 3)

            depth_s = torch.nn.functional.grid_sample(depth_nchw, grid, align_corners=True, mode="bilinear")
            depth_s = depth_s[0, 0, :, 0]  # (C,)

            depth_ok = torch.abs(depth_s - z) <= depth_tolerance

            m = in_front & in_bounds & facing & depth_ok
            w = (cos * cos).unsqueeze(-1) * m.float().unsqueeze(-1)  # (C, 1)

            accum[start:end] = accum[start:end] + w * rgb_s
            wsum[start:end] = wsum[start:end] + w

    rgb_out = torch.zeros((H_tex, W_tex, 3), device=device, dtype=torch.float32)
    rgb_valid = accum / wsum.clamp_min(1e-6)
    rgb_out[valid_mask] = rgb_valid
    return rgb_out


def export_textured_mesh_multiview(
    mesh: Mesh,
    pipeline: Pipeline,
    output_dir: Path,
    texture_size: int = 2048,
    num_views: int = 30,
    render_pixels_per_side: int = 768,
    fov_degrees: float = 60.0,
    elevation_range: tuple[float, float] = (-30.0, 60.0),
    radius_mult: float = 2.0,
    pad_px: int = 32,
) -> None:
    """Export textured mesh using multiview render-and-project/reproject.

    This approach:
    1. Generates camera poses around the object
    2. Renders synthetic views from the NeRF
    3. Projects/blends views onto UV-mapped mesh texture

    Args:
        mesh: Input mesh
        pipeline: Trained NeRF/SDF pipeline
        output_dir: Output directory
        texture_size: Texture resolution
        num_views: Number of synthetic views to render
        render_pixels_per_side: Render resolution (square) for NeRF views
        fov_degrees: Vertical field of view for synthetic cameras
        elevation_range: Min/max elevation angles in degrees
        radius_mult: Multiplier for the mesh bounding sphere radius
        pad_px: Number of pixels to dilate charts outward to avoid seam bleeding (fallback path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove legacy duplicate artifacts if present (from older exporters / trimesh OBJ export).
    for legacy_name in ("material_0.png", "material_0.mtl"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()

    device = pipeline.device
    vertices = mesh.vertices.to(device)
    faces = mesh.faces.to(device)
    vertex_normals = mesh.normals.to(device)

    # Step 1: bounding sphere for camera placement
    center = vertices.mean(dim=0)
    radius = (vertices - center).norm(dim=-1).max().item() * radius_mult
    CONSOLE.print(f"[bold]Cameras: {num_views} views, radius={radius:.4f}, fov={fov_degrees:.1f}°")

    # Step 2: camera poses
    intrinsics, extrinsics = generate_camera_poses_on_sphere(
        center=center,
        radius=radius,
        num_views=num_views,
        elevation_range=elevation_range,
        image_size=(render_pixels_per_side, render_pixels_per_side),
        fov_degrees=fov_degrees,
    )

    # Step 3: render synthetic views
    rgbs, depths = render_views_from_nerf(
        pipeline, intrinsics, extrinsics, image_size=(render_pixels_per_side, render_pixels_per_side)
    )

    # Step 4: project to UV texture via Open3D if available, otherwise fall back to reprojection.
    #
    # We keep an explicit flip flag because Open3D UVs and xatlas UVs differ in V
    # convention relative to how typical OBJ viewers sample textures.
    texture = None
    texture_uvs = None
    faces_for_export = None
    flip_v_for_export = True

    if OPEN3D_AVAILABLE:
        try:
            texture, texture_uvs, vertices_for_export, faces_for_export, normals_for_export = project_views_to_texture_open3d(
                mesh, rgbs, intrinsics, extrinsics, texture_size=texture_size
            )
            # Open3D UVs are already in the same convention as typical OBJ importers.
            flip_v_for_export = False
        except RuntimeError as e:
            if "project_images_to_albedo" not in str(e):
                raise
            CONSOLE.print("[yellow]Open3D projection API not available; falling back to xatlas + reprojection")

    if texture is None or texture_uvs is None or faces_for_export is None:
        if not OPEN3D_AVAILABLE:
            CONSOLE.print("[yellow]Open3D not available; using xatlas + reprojection instead")

        # UV unwrap & per-texel surface points for reprojection
        texture_uvs, faces_uv_order, origins, surf_normals, valid_mask = unwrap_mesh_with_xatlas_v2(
            vertices,
            faces,
            vertex_normals,
            texture_size=texture_size,
            use_gpu=True,
        )
        # Helpful debug signal: many unwraps have low coverage because the atlas is
        # sparse by design. This is not the same thing as rasterization "gaps".
        coverage = float(valid_mask.float().mean().item() * 100.0)
        CONSOLE.print(f"UV atlas coverage: {coverage:.1f}%")
        texture = project_views_to_texture_reproject(
            origins,
            surf_normals,
            valid_mask,
            rgbs,
            depths,
            intrinsics,
            extrinsics,
            depth_tolerance=0.01 * radius,
            chunk_size=_default_reproject_chunk_size(pipeline.model.config.eval_num_rays_per_chunk),
        )
        texture = pad_textures_nearest({"rgb": texture}, valid_mask, pad_px=pad_px)["rgb"]
        faces_for_export = faces_uv_order
        vertices_for_export = vertices
        normals_for_export = vertex_normals
        flip_v_for_export = True

    # Step 5: write results (texture + mesh)
    import mediapy as media  # type: ignore

    texture_np = texture.detach().cpu().numpy().clip(0.0, 1.0)
    media.write_image(str(output_dir / "texture.png"), texture_np)

    write_textured_mesh_fast(
        vertices_for_export,
        faces_for_export,
        normals_for_export,
        texture_uvs.to(device),
        texture_np,
        output_dir,
        mesh_name="mesh",
        flip_v=flip_v_for_export,
    )

    CONSOLE.print("[bold green]Multiview texture export complete!")
    CONSOLE.print(f"  Mesh: {output_dir / 'mesh.obj'} (+ mesh.mtl)")
    CONSOLE.print(f"  Binary: {output_dir / 'mesh.glb'}")
    CONSOLE.print(f"  Texture: {output_dir / 'texture.png'}")
