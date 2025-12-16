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
Path B: Render-and-reproject using synthetic NeRF views (stubs)
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

from sdfstudio.cameras.rays import RayBundle
from sdfstudio.exporter.exporter_utils import Mesh
from sdfstudio.pipelines.base_pipeline import Pipeline
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


# =============================================================================
# Path A: GPU Rasterization + Fast I/O + Multi-direction Averaging
# =============================================================================


def rasterize_uv_gpu(
    uv_coords: Tensor,
    faces: Tensor,
    texture_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
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

    return triangle_ids, bary_coords


def rasterize_uv_cpu(
    uv_coords: Tensor,
    faces: Tensor,
    texture_size: int,
) -> tuple[Tensor, Tensor]:
    """CPU fallback for UV rasterization using vectorized operations.

    Args:
        uv_coords: UV coordinates per vertex (num_verts, 2) in [0, 1]
        faces: Face indices (num_faces, 3)
        texture_size: Output texture resolution (square)

    Returns:
        triangle_ids: (H, W) tensor of triangle indices (-1 for empty pixels)
        bary_coords: (H, W, 3) tensor of barycentric coordinates
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
            denom = torch.where(torch.abs(denom) < 1e-10, torch.ones_like(denom), denom)

            # bary_v is weight for V1, bary_w is weight for V2
            bary_v = (d11 * d20 - d01 * d21) / denom  # (chunk, H*W)
            bary_w = (d00 * d21 - d01 * d20) / denom
            bary_u = 1.0 - bary_v - bary_w  # weight for V0

            # Point is inside triangle if all barycentric coords in [0, 1]
            eps = 1e-5
            inside = (bary_u >= -eps) & (bary_v >= -eps) & (bary_w >= -eps)  # (chunk, H*W)

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

    return triangle_ids, bary_coords


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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Unwrap mesh using xatlas and rasterize to get per-pixel 3D positions/normals.

    Args:
        vertices: Mesh vertices (V, 3)
        faces: Mesh faces (F, 3)
        vertex_normals: Vertex normals (V, 3)
        texture_size: Texture resolution
        use_gpu: Whether to use GPU rasterization (requires nvdiffrast)

    Returns:
        texture_uvs: Per-face UV coordinates (F, 3, 2)
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

    CONSOLE.print(f"[green]UV unwrapping complete: {len(uvs)} UV vertices")

    # Rasterize UV space
    CONSOLE.print(f"Rasterizing UV space ({texture_size}x{texture_size})...")

    if use_gpu and NVDIFFRAST_AVAILABLE and device.type == "cuda":
        triangle_ids, bary_coords = rasterize_uv_gpu(uv_coords, uv_faces, texture_size, device)
    else:
        if use_gpu and not NVDIFFRAST_AVAILABLE:
            CONSOLE.print("[yellow]nvdiffrast not available, falling back to CPU rasterization")
        triangle_ids, bary_coords = rasterize_uv_cpu(uv_coords, uv_faces, texture_size)

    # Valid mask: pixels that belong to a triangle
    valid_mask = triangle_ids >= 0

    # Clamp triangle IDs for gathering (invalid will be masked out anyway)
    triangle_ids_clamped = triangle_ids.clamp(min=0)

    # Get vertex indices for each pixel's triangle
    pixel_faces = faces[triangle_ids_clamped]  # (H, W, 3)

    # Gather vertex positions and normals
    pixel_verts = vertices[pixel_faces]  # (H, W, 3, 3)
    pixel_norms = vertex_normals[pixel_faces]  # (H, W, 3, 3)

    # Interpolate using barycentric coordinates
    bary = bary_coords.unsqueeze(-1)  # (H, W, 3, 1)
    origins = (pixel_verts * bary).sum(dim=2)  # (H, W, 3)
    normals = (pixel_norms * bary).sum(dim=2)  # (H, W, 3)
    normals = torch.nn.functional.normalize(normals, dim=-1)

    CONSOLE.print("[green]Rasterization complete")

    return texture_uvs, origins, normals, valid_mask


def query_nerf_multidirection(
    pipeline: Pipeline,
    origins: Tensor,
    normals: Tensor,
    valid_mask: Tensor,
    num_directions: int = 6,
    ray_length: float = 0.1,
) -> Tensor:
    """Query NeRF from multiple directions per texel and average results.

    Args:
        pipeline: NeRF pipeline
        origins: Surface positions (H, W, 3)
        normals: Surface normals (H, W, 3)
        valid_mask: Valid pixel mask (H, W)
        num_directions: Number of ray directions per texel
        ray_length: Length of rays to cast

    Returns:
        rgb: Texture colors (H, W, 3)
    """
    device = pipeline.device
    H, W = origins.shape[:2]

    # Generate multiple directions per texel (pointing into surface / away from camera)
    # We pass -normals so hemisphere points into the surface
    directions = generate_hemisphere_directions(-normals, num_directions)  # (H, W, num_dirs, 3)

    # Ensure directions are normalized
    directions = torch.nn.functional.normalize(directions, dim=-1)

    # Offset origins along ray direction (move back from surface along ray)
    ray_origins = origins.unsqueeze(-2) - 0.5 * ray_length * directions  # (H, W, num_dirs, 3)

    # Flatten valid pixels for batched inference
    flat_origins = ray_origins[valid_mask]  # (N, num_dirs, 3)
    flat_dirs = directions[valid_mask]  # (N, num_dirs, 3)

    N = flat_origins.shape[0]
    if N == 0:
        return torch.zeros(H, W, 3, device=device)

    CONSOLE.print(f"Querying NeRF: {N} valid texels x {num_directions} directions")

    # Query NeRF for each direction
    # We process each direction separately, letting get_outputs_for_camera_ray_bundle
    # handle the internal chunking for memory efficiency
    all_rgb = []

    progress = get_progress(f"Rendering {num_directions} directions")
    with torch.no_grad(), progress:
        for d in progress.track(range(num_directions), total=num_directions):
            dir_origins = flat_origins[:, d, :]  # (N, 3)
            dir_dirs = flat_dirs[:, d, :]  # (N, 3)

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

            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle, progress=False)
            # Output shape is (1, N, 3), flatten to (N, 3)
            all_rgb.append(outputs["rgb"].reshape(N, 3))

    # Average RGB across directions
    stacked_rgb = torch.stack(all_rgb, dim=1)  # (N, num_dirs, 3)
    avg_rgb = stacked_rgb.mean(dim=1)  # (N, 3)

    # Reconstruct full texture
    rgb = torch.zeros(H, W, 3, device=device)
    rgb[valid_mask] = avg_rgb.to(device)

    return rgb


def write_textured_mesh_fast(
    vertices: Tensor,
    faces: Tensor,
    vertex_normals: Tensor,
    texture_uvs: Tensor,
    texture_image: np.ndarray,
    output_dir: Path,
    mesh_name: str = "mesh",
) -> None:
    """Write textured mesh using trimesh for fast I/O.

    Args:
        vertices: Mesh vertices (V, 3)
        faces: Mesh faces (F, 3)
        vertex_normals: Vertex normals (V, 3)
        texture_uvs: Per-face UV coordinates (F, 3, 2)
        texture_image: Texture image (H, W, 3) as numpy array [0, 1]
        output_dir: Output directory
        mesh_name: Base name for output files
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

    # Flip V coordinate for OBJ convention
    expanded_uvs[:, 1] = 1.0 - expanded_uvs[:, 1]

    # Convert texture to uint8 image
    texture_uint8 = (np.clip(texture_image, 0, 1) * 255).astype(np.uint8)
    texture_pil = PIL.Image.fromarray(texture_uint8)

    # Save texture image
    texture_path = output_dir / f"{mesh_name}_texture.png"
    texture_pil.save(texture_path)

    # Create trimesh with texture
    material = trimesh.visual.material.SimpleMaterial(  # type: ignore
        image=texture_pil,
        diffuse=[255, 255, 255, 255],
    )

    visuals = trimesh.visual.TextureVisuals(  # type: ignore
        uv=expanded_uvs,
        material=material,
    )

    mesh = trimesh.Trimesh(
        vertices=expanded_verts,
        faces=expanded_faces,
        vertex_normals=expanded_normals,
        visual=visuals,
        process=False,  # Don't merge vertices
    )

    # Export OBJ
    obj_path = output_dir / f"{mesh_name}.obj"
    mesh.export(obj_path, file_type="obj")

    # Also export GLB for convenience
    glb_path = output_dir / f"{mesh_name}.glb"
    mesh.export(glb_path, file_type="glb")

    CONSOLE.print(f"[green]Mesh exported to {obj_path} and {glb_path}")


def export_textured_mesh_v2(
    mesh: Mesh,
    pipeline: Pipeline,
    output_dir: Path,
    texture_size: int = 2048,
    num_directions: int = 6,
    use_gpu_rasterization: bool = True,
) -> None:
    """Export textured mesh using improved pipeline (Path A).

    Args:
        mesh: Input mesh
        pipeline: Trained NeRF/SDF pipeline
        output_dir: Output directory
        texture_size: Texture resolution
        num_directions: Number of ray directions per texel for averaging
        use_gpu_rasterization: Whether to use GPU rasterization
    """
    device = pipeline.device
    output_dir = Path(output_dir)

    vertices = mesh.vertices.to(device)
    faces = mesh.faces.to(device)
    vertex_normals = mesh.normals.to(device)

    CONSOLE.print(f"[bold]Texturing mesh: {len(vertices)} vertices, {len(faces)} faces")

    # Step 1: UV unwrap and rasterize
    texture_uvs, origins, normals, valid_mask = unwrap_mesh_with_xatlas_v2(
        vertices,
        faces,
        vertex_normals,
        texture_size=texture_size,
        use_gpu=use_gpu_rasterization,
    )

    # Step 2: Compute ray length from mesh scale
    face_verts = vertices[faces]
    edge_lengths = torch.norm(face_verts[:, 1] - face_verts[:, 0], dim=-1)
    ray_length = 2.0 * edge_lengths.mean().item()
    CONSOLE.print(f"Ray length: {ray_length:.4f}")

    # Step 3: Query NeRF with multi-direction averaging
    rgb = query_nerf_multidirection(
        pipeline,
        origins,
        normals,
        valid_mask,
        num_directions=num_directions,
        ray_length=ray_length,
    )

    # Step 4: Write mesh with trimesh
    texture_image = rgb.cpu().numpy()
    write_textured_mesh_fast(vertices, faces, vertex_normals, texture_uvs, texture_image, output_dir, mesh_name="mesh")

    # Also save texture separately
    import mediapy as media  # type: ignore

    media.write_image(str(output_dir / "texture.png"), texture_image)

    CONSOLE.print("[bold green]Texture export complete!")


# =============================================================================
# Path B: Render-and-Reproject (Stubs)
# =============================================================================


def generate_camera_poses_on_sphere(
    center: Tensor,
    radius: float,
    num_views: int = 30,
    elevation_range: tuple[float, float] = (-30, 60),
) -> tuple[Tensor, Tensor]:
    """Generate camera poses on a sphere looking at center.

    TODO: Implement camera pose generation
    - Distribute cameras on sphere using fibonacci spiral or similar
    - Compute look-at matrices
    - Return intrinsics and extrinsics

    Args:
        center: Center point to look at (3,)
        radius: Distance from center
        num_views: Number of camera views
        elevation_range: Min/max elevation angles in degrees

    Returns:
        intrinsics: Camera intrinsics (num_views, 3, 3)
        extrinsics: Camera extrinsics/poses (num_views, 4, 4)
    """
    # TODO: Implement fibonacci sphere sampling
    # TODO: Compute look-at matrices
    # TODO: Create reasonable intrinsics (FOV ~60 degrees)
    raise NotImplementedError("Path B: generate_camera_poses_on_sphere not yet implemented")


def render_views_from_nerf(
    pipeline: Pipeline,
    intrinsics: Tensor,
    extrinsics: Tensor,
    image_size: tuple[int, int] = (1024, 1024),
) -> list[Tensor]:
    """Render synthetic views from NeRF at given camera poses.

    TODO: Implement NeRF rendering at arbitrary camera poses
    - Create camera objects from intrinsics/extrinsics
    - Render RGB images
    - Optionally render depth for occlusion handling

    Args:
        pipeline: Trained NeRF pipeline
        intrinsics: Camera intrinsics (N, 3, 3)
        extrinsics: Camera extrinsics (N, 4, 4)
        image_size: Output image resolution (H, W)

    Returns:
        images: List of rendered RGB images (N,) each (H, W, 3)
    """
    # TODO: Create Cameras object from intrinsics/extrinsics
    # TODO: Generate rays for each camera
    # TODO: Batch render through pipeline
    # TODO: Return list of RGB images
    raise NotImplementedError("Path B: render_views_from_nerf not yet implemented")


def project_views_to_texture_open3d(
    mesh: Mesh,
    images: list[Tensor],
    intrinsics: Tensor,
    extrinsics: Tensor,
    texture_size: int = 2048,
) -> Tensor:
    """Project rendered views onto mesh texture using Open3D.

    TODO: Implement view projection using Open3D's project_images_to_albedo
    - Convert mesh to Open3D tensor mesh
    - Compute UV atlas if not present
    - Project images with blending

    Args:
        mesh: Input mesh
        images: Rendered RGB images
        intrinsics: Camera intrinsics (N, 3, 3)
        extrinsics: Camera extrinsics (N, 4, 4)
        texture_size: Output texture resolution

    Returns:
        texture: Projected texture (H, W, 3)
    """
    if not OPEN3D_AVAILABLE:
        raise RuntimeError("Open3D not available for Path B")

    # TODO: Convert mesh to o3d.t.geometry.TriangleMesh
    # TODO: Compute UV atlas with compute_uvatlas()
    # TODO: Convert images and cameras to Open3D format
    # TODO: Call project_images_to_albedo()
    # TODO: Extract and return texture
    raise NotImplementedError("Path B: project_views_to_texture_open3d not yet implemented")


def export_textured_mesh_multiview(
    mesh: Mesh,
    pipeline: Pipeline,
    output_dir: Path,
    texture_size: int = 2048,
    num_views: int = 30,
) -> None:
    """Export textured mesh using render-and-reproject approach (Path B).

    This approach:
    1. Generates camera poses around the object
    2. Renders synthetic views from the NeRF
    3. Projects/blends views onto UV-mapped mesh texture

    TODO: Implement full pipeline

    Args:
        mesh: Input mesh
        pipeline: Trained NeRF/SDF pipeline
        output_dir: Output directory
        texture_size: Texture resolution
        num_views: Number of synthetic views to render
    """
    _ = pipeline.device  # Will be used when implemented
    output_dir = Path(output_dir)
    _ = output_dir  # Will be used when implemented

    # TODO: Step 1 - Compute mesh bounding sphere for camera placement
    # vertices = mesh.vertices.to(device)
    # center = vertices.mean(dim=0)
    # radius = (vertices - center).norm(dim=-1).max().item() * 2.0

    # TODO: Step 2 - Generate camera poses
    # intrinsics, extrinsics = generate_camera_poses_on_sphere(center, radius, num_views)

    # TODO: Step 3 - Render views from NeRF
    # images = render_views_from_nerf(pipeline, intrinsics, extrinsics)

    # TODO: Step 4 - Project views to texture
    # texture = project_views_to_texture_open3d(mesh, images, intrinsics, extrinsics, texture_size)

    # TODO: Step 5 - Write mesh
    # write_textured_mesh_fast(...)

    raise NotImplementedError("Path B: export_textured_mesh_multiview not yet implemented")
