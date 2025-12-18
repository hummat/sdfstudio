# PBR Export Integration Plan

Goal: Get usable **metal–roughness PBR materials** (basecolor, normal, roughness, metallic) from SDFStudio without major new research—leverage existing Ref-NeRF-style features during training and optional off-the-shelf post-processing for refinement.

---

## Reality Check (What This Plan Does / Doesn’t Guarantee)

- Training-time “diffuse/specular/tint/roughness” in `SDFField` is a **learned factorization of radiance**, not a physically constrained BRDF + explicit lighting decomposition.
- `enable_pred_roughness` produces a **proxy roughness** used to mix view vs reflection encodings; it can be a useful texture for downstream look-dev, but it is not guaranteed to match GGX roughness under unknown illumination.
- Color space: current `SDFField` outputs (`rgb`, `diffuse`, `specular`, `tint`) are produced via `linear_to_srgb(...).clamp(0,1)` (already sRGB-ish), while `roughness` is in `[0,1]` and should be treated as linear. This matters for glTF wiring and for any refinement step.

If the target is “true PBR from real video under unknown lighting”, this plan should be viewed as:
1) a strong *initialization / proxy-PBR export*, and
2) an on-ramp to Stage 3 (inverse rendering) when needed.

---

## Practical “Minimal Real PBR” Target (Metal–Rough)

Minimum maps (typical glTF/Unreal pipelines):
- **Basecolor / Albedo** (sRGB)
- **Normal** (linear)
- **Roughness** (linear)
- **Metallic** (linear; often constant)

Notes:
- For dielectrics (wood/plastic/stone/paint/skin), **metallic ≈ 0 everywhere** → prefer a constant `metallicFactor=0` unless you truly have mixed materials.
- Many pipelines assume dielectric **F0 ≈ 0.04**; keep specular fixed initially for stability.

## Overview

| Stage | Effort | Description |
|-------|--------|-------------|
| **Training-time** | Low | Enable existing diffuse/specular/roughness outputs in SDFField |
| **Export** | Medium | glTF/GLB with proper PBR material wiring |
| **BRDF-ish Baseline** | Medium–High | In-repo UV-space Lambert → GGX fitting (fixed geometry) |
| **Refinement A** | Medium | Optional nvdiffrec inverse rendering |
| **Refinement B** | Medium | Optional generative per-view prediction (Material Palette, MatFusion) |

---

## Stage 1: Training-Time (Existing Features)

SDFField already has Ref-NeRF-style factorization. Enable it:

### Already implemented in `sdfstudio/fields/sdf_field.py`

| Config flag | Effect |
|-------------|--------|
| `use_diffuse_color` | View-independent diffuse head → better albedo proxy |
| `use_specular_tint` | Learned RGB specular tint (metals) |
| `use_reflections` | Reflection-direction encoding for specular |
| `use_n_dot_v` | Explicit incidence angle to color MLP |
| `enable_pred_roughness` | Roughness ∈ [0,1] mixing view/reflection dirs |

### Current status

- `bakedsdf` and `bakedsdf-mlp` enable diffuse/specular/tint/reflections/n_dot_v
- **`enable_pred_roughness` is not used in any config** ← enable this

### TODO

- [ ] Add config variant or document CLI override (note: roughness output is only emitted when `use_diffuse_color=True`):
  ```bash
  --pipeline.model.sdf-field.enable-pred-roughness True
  ```
- [ ] Verify roughness output flows through `base_surface_model.py` → texture export (branch `pbr-low-hanging-fruit` already added this passthrough)
- [ ] Test with a real scene, inspect diffuse/roughness maps

---

## Stage 2: glTF/GLB Export with PBR Material

Current texture export saves raw PNGs (`diffuse_0.png`, `roughness_0.png`, etc.) but doesn't wire them into a proper material.

### TODO

- [ ] Add glTF/GLB export function that creates a metal-rough material:
  - `baseColorTexture` ← diffuse output (already sRGB-ish; ensure glTF baseColor is treated as sRGB by consumers)
  - `normalTexture` ← baked normals (tangent-space, ensure conventions)
  - `metallicRoughnessTexture` ← pack roughness into G channel, metallic in B (or separate)
  - `metallicFactor` = 0 default (dielectric), allow override
- [ ] Standardize output naming: `basecolor.png`, `normal.png`, `roughness.png`, `metallic.png`
- [ ] Document color space conventions (optimize in linear, export basecolor as sRGB, others linear)
  - [ ] Confirm whether exporters write `diffuse` as sRGB or linear and document it (today it’s sRGB-ish from the field).

### Implementation notes

- Use `pygltflib` or similar for glTF writing
- Can extend existing `export_textured_mesh_v2` or add new function
- Tangent-space normal baking needs correct tangent frame from xatlas UVs
  - Many viewers can compute tangents if missing, but quality varies; ideally compute tangents (MikkTSpace) during export.

---

## Stage 2.5: In-Repo “BRDF-ish” Baseline (UV-Space Lambert → GGX)

This stage reintroduces the “physics baseline” from the original video-notes: given **known geometry + cameras** (from SDFStudio) and **real video frames**, fit an explicit forward shading model in UV space.

Goal: a buildable baseline that produces better “true PBR” maps than proxy exports, without trying to solve hard unsupervised inverse rendering end-to-end.

### 2.5A) Freeze geometry + define conventions

- [ ] Define a canonical mesh contract for fitting:
  - [ ] mesh triangles
  - [ ] UV atlas
  - [ ] normals (and ideally tangents/bitangents for tangent-space normal maps)
- [ ] Document texture-space convention (V flip expectations) and normal map convention (OpenGL vs DirectX Y).

### 2.5B) Build UV-space observation dataset (real frames)

Deliverable: a reusable “texel observations” dataset where each texel `(u,v)` has multi-view samples:
`{rgb_i, viewdir_i, frame_idx, weight_i, mask_i}`.

TODOs:
- [ ] Add a new script (tentative): `scripts/pbr_uv_observations.py`
- [ ] Inputs:
  - [ ] mesh (+ UVs if present; otherwise unwrap)
  - [ ] per-frame RGB images
  - [ ] per-frame cameras (intrinsics + c2w)
  - [ ] optional foreground masks
- [ ] Per-frame association:
  - [ ] rasterize the mesh in the frame to map visible pixels → UV texels (store RGB + viewdir + confidence)
  - [ ] aggregate samples per texel for fitting
- [ ] Outlier hooks (minimum viable):
  - [ ] downweight grazing angles
  - [ ] mask saturated pixels (simple threshold)
  - [ ] record UV coverage stats to drive regularization defaults

### 2.5C) Lighting model (simple and stable first)

- [ ] Option A (recommended): **global low-order SH** + **per-frame exposure scalar**
- [ ] Option B: per-frame SH (more flexible, more degenerate)

### 2.5D) Forward shading model (Lambert → GGX)

- [ ] Lambertian baseline:
  - parameters: `A(u,v)` (albedo), SH lighting, per-frame exposure
  - optimize in **linear** color space (linearize input frames)
- [ ] GGX microfacet next (dielectric first):
  - add `r(u,v)` (roughness)
  - keep `metallic=0` and `F0≈0.04` fixed initially

### 2.5E) Minimal objective (from the original notes)

Let:
- `A(u,v) ∈ [0,1]^3` (linear albedo)
- `r(u,v) ∈ (0,1]` (roughness)
- `m(u,v) ∈ [0,1]` (optional metallic; default constant 0)
- per-frame lighting `θ_i` (e.g., SH) and exposure `e_i`

```
min_{A, r, (m), {θ_i}, {e_i}}  Σ_{(u,v)} Σ_{i∈V(u,v)}  ρ( I_i(u,v) - Render(A, r, m, N, V_i, θ_i, e_i) )
                             + λ_TV · (TV(A) + TV(r) + TV(m))
                             + λ_prior · Prior(A, r, m)
```

Implementation notes:
- [ ] Use robust `ρ` (Charbonnier/Huber) and stage the optimization (Lambert first, then GGX).
- [ ] Enforce bounds: optimize roughness/metallic in logit space and clamp where needed.

### 2.5F) Validation

- [ ] Re-render frames with fitted maps (albedo should be less contaminated by shadows/highlights).
- [ ] Quick relighting sanity check (swap env lighting).

---

## Stage 3A: Optional Inverse Rendering (nvdiffrec)

For higher quality PBR when training-time outputs aren't sufficient.

**nvdiffrec** (NVIDIA, MIT license) does joint geometry + BRDF + environment lighting optimization. We can use it with frozen geometry to refine materials only.

### Integration approach

1. Export from SDFStudio:
   - Mesh + UVs (OBJ)
   - Training images + cameras (transforms.json or similar)
   - Initial material maps as starting point (optional)

2. Run nvdiffrec:
   - Configure to freeze/heavily-regularize geometry
   - Optimize materials + environment lighting
   - Outputs: basecolor, roughness, metallic, normal maps

3. Import results back (or use directly)

### TODO

- [ ] Add exporter for nvdiffrec input format (mesh + images + cameras)
- [ ] Document nvdiffrec setup and recommended config for material-only refinement
- [ ] Optionally: wrap as a script that shells out to nvdiffrec
  - [ ] Keep it optional (no hard dependency in `pyproject.toml`).

### Pros/cons

- ✓ Physically grounded, handles lighting properly
- ✓ Proven, actively maintained
- ✗ Another dependency (PyTorch3D, nvdiffrast)
- ✗ Slower than generative approach

---

## Stage 3B: Optional Generative Per-View Prediction

Alternative to inverse rendering: use a learned single-image SVBRDF predictor on rendered views, aggregate in UV space.

**Candidates:**
- **Material Palette** (CVPR 2024) — diffusion-based, extracts PBR from single image
- **MatFusion** (SIGGRAPH Asia 2023) — generative diffusion for SVBRDF

### Integration approach

1. Render N synthetic views from trained model (already have this in v2 multiview export)
2. Run predictor on each view → per-view PBR maps
3. Reproject to UV space, aggregate with confidence weighting:
   - Weight by view angle (prefer frontal)
   - Weight by predictor confidence if available
   - Median or robust mean to handle outliers

Alternative input: run the predictor on the **real video frames** (often preferable if you’re explicitly targeting “from video” rather than “from neural render”).

### TODO

- [ ] Add integration for Material Palette or MatFusion (pick one to start)
- [ ] Implement UV-space aggregation from per-view predictions
- [ ] Handle view-inconsistency (predictor may give different answers per view)

### Pros/cons

- ✓ Leverages strong learned priors (real material manifold)
- ✓ Faster than optimization-based inverse rendering
- ✓ Can work with fewer views
- ✗ Less physically grounded
- ✗ View inconsistency requires careful aggregation
- ✗ Another model to download/run

---

## Stage 4: Nice-to-Have (Future)

Lower priority, only after core pipeline works:

- [ ] Roughness-from-variance heuristic as zero-dependency fallback
- [ ] AO baking from geometry (raycast occlusion)
- [ ] Height/displacement from SDF detail
- [ ] Per-texel confidence/quality map for downstream tools

---

## “Things That Bite” Checklist (From the Original Notes)

- [ ] **Color space:** linearize input images for fitting; export basecolor as sRGB; keep roughness/metallic/normal as linear.
- [ ] **Normal conventions:** tangent basis + channel conventions (OpenGL vs DirectX normal Y).
- [ ] **Masking/outliers:** specular highlights, saturation, motion blur → robust loss and masking.
- [ ] **UV coverage:** sparse observation regions need stronger regularization (and should output a coverage metric).
- [ ] **Exposure changes:** per-frame exposure scalar often matters on real video.
- [ ] **Parameter bounds:** roughness/metallic constrained (logit), avoid degenerate “roughness explains everything”.

---

## File Touchpoints

| File | Role |
|------|------|
| `sdfstudio/fields/sdf_field.py` | Ref-NeRF features (already implemented) |
| `sdfstudio/models/base_surface_model.py` | Roughness passthrough (done in branch) |
| `sdfstudio/exporter/texture_utils.py` | Legacy export, saves PBR maps as PNG |
| `sdfstudio/exporter/texture_utils_v2.py` | v2 export, handles diffuse/roughness/etc. |
| `scripts/texture.py` | CLI for texture export |
| **NEW** `sdfstudio/exporter/gltf_export.py` | glTF/GLB with PBR material |
| **NEW** `scripts/pbr_uv_observations.py` | Build UV-space observations from real video |
| **NEW** `scripts/pbr_fit_uv.py` | UV-space Lambert/GGX fitting baseline |
| **NEW** `scripts/pbr_refine.py` | Optional nvdiffrec / generative refinement |

---

## References

### Inverse Rendering
- **nvdiffrec**: Munkberg et al., CVPR 2022 — "Extracting Triangular 3D Models, Materials, and Lighting From Images"
- **nvdiffrecmc**: Hasselgren et al., SIGGRAPH Asia 2022 — Monte Carlo version

### Generative SVBRDF
- **Material Palette**: Lopes et al., CVPR 2024 — "Material Palette: Extraction of Materials from a Single Image"
- **MatFusion**: Sartor & Peers, SIGGRAPH Asia 2023 — "MatFusion: a Generative Diffusion Model for SVBRDF Capture"

### Neural Field Factorization
- **Ref-NeRF**: Verbin et al., CVPR 2022 — reflection direction encoding, diffuse/specular split
- **NeRFactor**: Zhang et al., SIGGRAPH Asia 2021 — neural factorization of shape and reflectance
