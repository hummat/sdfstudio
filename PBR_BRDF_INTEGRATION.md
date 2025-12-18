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

---

## BRDF / Shading Effects Scope (What We Model When)

This section keeps the project self-contained by listing the relevant shading effects explicitly and how they map to our staged plan.

### MVP (should be “engineering-stable”)

- **Diffuse / albedo separation**: view-independent `A(x)` / `diffuse`.
- **Smooth lighting**: low-order environment lighting (SH) + per-image exposure scalar.
- **Basic angular cues**: `n·v` available as an input feature (helps Fresnel-like ramps and silhouette behavior).
- **Roughness proxy**: a per-point/per-sample roughness parameter in `[0,1]` (even if it starts as a heuristic/proxy).

### Next (improves “PBR-ness” for indoor scans)

- **Explicit specular term**:
  - start with a simple dielectric spec model (fixed `F0≈0.04`) before full microfacet GGX
  - optionally keep a small residual/spec term during a warmup
- **GGX microfacet** (Cook–Torrance) with:
  - roughness-driven lobe width
  - Fresnel (Schlick) and a masking–shadowing approximation (e.g., Smith)

### Later / optional (likely “research-y” for robustness)

- **Hard visibility / cast shadows** (beyond what smooth lighting can explain).
- **Indirect illumination / interreflections** (especially noticeable indoors).
- **Per-frame lighting variation** beyond exposure (risk: degeneracy where lighting explains material).

### Out of scope (for now)

- **Transmission/refraction** (glass, thin plastics).
- **Subsurface scattering** (skin, wax, marble).
- **Anisotropy** (brushed metals) unless there’s a strong need.

## Overview

| Stage | Effort | Description |
|-------|--------|-------------|
| **1: Training-time (existing)** | Low | Enable existing Ref-NeRF-ish outputs (diffuse/spec/tint/roughness proxy) |
| **2: Export (engineering)** | Medium | glTF/GLB with correct metal-rough wiring + Blender-friendly conventions |
| **1.5: NeuSFactor-lite (new method)** | High | New forward renderer + SH lighting + training schedule (warmup/blend) |
| **2.5: UV-space Lambert→GGX (new tool)** | High | UV observation builder + differentiable shader + optimization loop |
| **3A: Refinement (external, optimization)** | Medium | Optional nvdiffrec-style material refinement (mostly integration) |
| **3B: Refinement (external, generative)** | Medium | Optional per-view SVBRDF prediction + UV aggregation (integration + glue) |

Interpretation:
- Stages **1** and **2** are the “low-hanging fruit” path (export better maps/materials using what we already predict).
- Stages **1.5** and **2.5** are the main bespoke engineering projects (bounded, but substantial).

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
| `enable_pred_roughness` | Roughness proxy ∈ [0,1] mixing view/reflection dirs |
| `roughness_blend_space` | Where roughness mixes view vs reflection (“encoding” vs “direction”) |
| `use_roughness_in_color_mlp` | Append roughness scalar to the color MLP input (requires `enable_pred_roughness`) |
| `use_fresnel_term` | Append a Schlick-style Fresnel scalar to the color MLP input |

Notes:
- In `SDFField`, ray `directions` are camera→point; view vector is point→camera = `-directions`. `use_n_dot_v` and `use_fresnel_term` use `n·v` with that sign convention.
- `roughness` is only emitted in field outputs when `use_diffuse_color=True` (because the “diffuse/specular/roughness” tuple outputs only exist in that mode).
- If Stage 1 exports are later used as initialization for Stage 2.5 fitting: `diffuse/specular/tint/rgb` are already
  sRGB-ish (via `linear_to_srgb` in the field). Linearize them back before any physically-based fitting/optimization.

### Current status

- `bakedsdf` and `bakedsdf-mlp` enable diffuse/specular/tint/reflections/n_dot_v
- **`enable_pred_roughness` is not used in any config** ← enable this (and optionally the new knobs below)

### TODO

- [ ] Add config variant or document CLI override (note: roughness output is only emitted when `use_diffuse_color=True`):
  ```bash
  --pipeline.model.sdf-field.enable-pred-roughness
  # optional:
  --pipeline.model.sdf-field.roughness-blend-space direction
  --pipeline.model.sdf-field.use-roughness-in-color-mlp
  --pipeline.model.sdf-field.use-fresnel-term
  ```
  - Tyro boolean flags are typically presence/absence (`--flag` / `--no-flag`), not `--flag True`.
- [ ] Verify roughness output flows through `base_surface_model.py` → texture export (branch `pbr-low-hanging-fruit` already added this passthrough)
- [ ] Test with a real scene, inspect diffuse/roughness maps

### Suggested starting recipes (Stage 1 only)

Dielectric indoor smartphone object scan (e.g., LEGO/plastic):
- Start with: `use_diffuse_color`, `use_reflections`, `use_n_dot_v` (or `use_fresnel_term`), keep `use_specular_tint` off.
- If highlights look unstable / too sharp / “wrong”: add `enable_pred_roughness`.
- If roughness doesn’t seem to control appearance enough: add `use_roughness_in_color_mlp`.
- Only then try `roughness_blend_space=direction` (often a smaller effect than adding scalar conditioning).

Metallic / colored specular (jewelry, coins, brass):
- Add `use_specular_tint` (and consider `use_fresnel_term`).

Quick ablation (to see what mattered):
- With `enable_pred_roughness` on, test `use_roughness_in_color_mlp` alone first, then `use_fresnel_term`, then switch `roughness_blend_space`.

---

## Stage 1.5: NeuSFactor-lite (Training-Time “BRDF-ish” Rendering)

This stage targets “works somewhat better than today for indoor smartphone object scans” by making the *training-time* image formation more physically structured than a pure radiance MLP, without trying to solve full inverse rendering.

Key idea:
- Predict **material** (start with dielectric albedo + roughness).
- Use an **explicit lighting model** (start with global SH + per-image exposure).
- Render via a **simple forward shader** (Lambert first, then “GGX-ish”).
- Keep a small **residual view-dependent term** early on (anneal down) to avoid destabilizing geometry training.

### 1.5A) Minimum viable assumptions (to keep it engineering)

- Geometry comes from NeuS/NeuS-facto as usual (no special capture rig).
- Lighting is “approximately static in world” and reasonably smooth (typical indoor scans).
- Start **dielectric-only**:
  - `metallic=0`
  - fixed `F0≈0.04`

### 1.5B) Model outputs and parameters

Per sample (or per surface hit) predict:
- [ ] `albedo A(x)` (view-independent, linear)
- [ ] `roughness r(x)` (view-independent, linear in `(0,1]`)
- [ ] optional: `specular_tint(x)` / `metallic(x)` later

Global / per-image latent parameters:
- [ ] global low-order SH lighting coefficients (RGB)
- [ ] per-image exposure scalar (and optionally WB) as a small embedding

### 1.5C) Forward renderer (start simple)

- [ ] Lambert baseline:
  - `rgb_linear = exposure_i * ( A(x) * Irradiance_SH(N(x)) )`
- [ ] Add “GGX-ish” later:
  - keep it simple initially (dielectric only, fixed F0)
  - consider starting with a heuristic spec term before full GGX to reduce implementation risk

### 1.5D) Training schedule (avoid breaking geometry)

To prevent the structured renderer from underfitting early and harming geometry:
- [ ] Warmup with the existing radiance head for `N_warmup` steps.
- [ ] Then blend losses:
  - `L = w_radiance(step) * L_rgb(radiance_rgb, gt) + w_brdf(step) * L_rgb(brdf_rgb, gt)`
  - ramp `w_brdf ↑` and (optionally) `w_radiance ↓`.
- [ ] Keep a small “residual” branch (view-dependent) early; regularize/anneal it down.

Note:
- Annealing `w_radiance` all the way to 0 can be risky if the BRDF+lighting model is too constrained to explain
  real data (interreflections, cast shadows, exposure drift, etc.). A common compromise is to keep a small non-zero
  radiance term throughout training as a “catch-all” residual.

### 1.5E) What “success” looks like

- Diffuse/basecolor becomes less contaminated by highlights/shadows on average.
- Roughness maps become more spatially stable and correlate with perceived gloss.
- Rendered outputs remain visually competitive (even if not perfect PBR).

---

## Stage 2: glTF/GLB Export with PBR Material

The v2 exporter already writes a viewer-friendly `mesh.glb` with a metal–rough PBR material, plus canonical texture PNGs.

### Current status (what exists today)

- `sdfstudio/exporter/texture_utils_v2.py`:
  - writes `basecolor.png` (sRGB-ish) and also `texture.png` (OBJ/MTL compatibility)
  - writes optional linear basecolor artifacts for downstream fitting/refinement (see “Expected outputs”)
  - writes `normal.png` (tangent-space normal map) when model normals are available, and wires it as glTF `normalTexture`
  - writes `roughness.png` (roughness proxy) and packs `orm.png` (AO=R, Roughness=G, Metallic=B) when roughness exists
  - exports `mesh.obj`/`mesh.mtl` and `mesh.glb`
  - best-effort wires glTF PBR fields via `trimesh.visual.material.PBRMaterial` when available
- Not yet fully covered:
  - robust, version-stable glTF writing (trimesh API varies)
  - tangent-space normal baking (MikkTSpace) and documented conventions

### How to export today (end-to-end)

1) Train with the signals you want to export (example flags only):
```bash
# Ensure diffuse/specular outputs exist and roughness can be exported.
--pipeline.model.sdf-field.use-diffuse-color
--pipeline.model.sdf-field.use-reflections
--pipeline.model.sdf-field.enable-pred-roughness
```

2) Texture an extracted mesh:
```bash
sdf-texture-mesh \
  --load-config outputs/<exp>/<method>/<timestamp>/config.yml \
  --input-mesh-filename meshes/<mesh>.ply \
  --output-dir meshes/textured \
  --method gpu \
  --num-pixels-per-side 2048
```

Expected outputs (v2 cpu/gpu):
- `mesh.glb`, `mesh.obj`, `mesh.mtl`
- `basecolor.png` (and `texture.png` alias for OBJ wiring)
- `basecolor_linear.npy` (linearized from basecolor; useful as a starting point for fitting; still clipped if the field clipped)
- `normal.png` (tangent-space normal map; used by GLB when available)
- `roughness.png`, `orm.png` (if roughness is available)
- `rgb.png`, `specular.png`, `tint.png` (if available)
- `normal_object.png` (proxy only; not a tangent-space normal map)

### TODO

- [ ] Upgrade normal map baking to MikkTSpace (match Blender) and validate `normal.png` on real exports.
- [ ] Ensure the GLB material uses:
  - `baseColorTexture` ← `basecolor.png` / `texture.png` (sRGB)
  - `metallicRoughnessTexture` ← `orm.png` (linear; roughness in G, metallic in B; dielectric defaults)
- [ ] Document color space conventions explicitly:
  - basecolor: sRGB texture
  - normal/roughness/metallic/ORM: linear (non-color)
- [ ] Decide whether to keep trimesh-based GLB export or add a dedicated glTF writer for reproducibility across environments.

### Implementation notes

- Current implementation uses trimesh for GLB export; this is convenient but not always stable across trimesh versions.
- A dedicated writer (e.g., via `pygltflib`) is still attractive if we want deterministic output and full control over normal maps, samplers, and color-space metadata.
- Tangent-space normal baking needs correct tangent frame from xatlas UVs
  - Many viewers can compute tangents if missing, but results can differ subtly; for reproducibility, compute tangents
    explicitly (ideally MikkTSpace to match Blender).

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
    - Practical default: weight by something like `w = clamp(n·v, 0.1, 1.0)` (or a smooth falloff) rather than
      `max(n·v, 0)`; grazing views tend to be noisy/unreliable and can otherwise dominate.
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
- **DualMat** (2025) — coherent dual-path diffusion for PBR material estimation

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

These are “go-to” references to build working knowledge and to consult during implementation.

### Blender workflow (practical correctness)

- Blender Manual — **Principled BSDF**: https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/principled.html
- Blender Manual — **Normal Map** node (OpenGL/Y+ convention; set image to Non-Color): https://docs.blender.org/manual/en/latest/render/shader_nodes/vector/normal_map.html
- Blender Manual — **Color Management** (Filmic/OCIO; how display transforms affect what you see): https://docs.blender.org/manual/en/latest/render/color_management.html

### File formats / conventions (export correctness)

- Khronos — **glTF 2.0 Specification** (PBR metallic-roughness; color spaces; ORM packing): https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html

### Books (best for fundamentals)

- Pharr, Jakob, Humphreys — **Physically Based Rendering: From Theory To Implementation (PBRT), 4th ed.**: https://pbr-book.org/
- Akenine-Möller, Haines, Hoffman — **Real-Time Rendering, 4th ed.**: https://www.realtimerendering.com/

### Practical shading notes (high signal for “metal–rough”)

- Karis — **Real Shading in Unreal Engine 4** (SIGGRAPH “Physically Based Shading” course notes/slides):
  - course hub: https://blog.selfshadow.com/publications/s2013-shading-course/
  - notes PDF: https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
- Burley — **Physically-Based Shading at Disney** (SIGGRAPH 2012 course notes):
  - https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf

### Microfacet, Fresnel, and related “core equations”

- Schlick (1994) — **An Inexpensive BRDF Model for Physically-based Rendering** (Graphics Gems IV):
  - (one canonical listing) https://dblp.org/rec/books/el/94/Schlick94.html
- Walter et al. (2007) — **Microfacet Models for Refraction through Rough Surfaces** (EGSR):
  - (Cornell mirror) https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
- Heitz (2014) — **Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs** (JCGT):
  - (reference listing) https://www.scitepress.org/publishedPapers/2014/46514/pdf/index.html

### Lighting representations (for SH / env lighting baselines)

- Ramamoorthi & Hanrahan (2001) — **An Efficient Representation for Irradiance Environment Maps**:
  - https://graphics.stanford.edu/papers/envmap/
- Debevec (1998) — **Rendering Synthetic Objects into Real Scenes: Bridging Traditional and Image-based Graphics with Global Illumination and High Dynamic Range Photography**:
  - https://www.pauldebevec.com/Research/IBL/

### Tangent space (to match Blender)

- **MikkTSpace** (tangent basis reference implementation; Blender uses MikkTSpace):
  - https://github.com/mmikk/MikkTSpace

### Neural-field factorization context (“NeuSFactor-lite” lineage)

- Verbin et al. (2022) — **Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields**:
  - https://arxiv.org/abs/2112.03907
- Zhang et al. (2021) — **NeRFactor: Neural Factorization of Shape and Reflectance under an Unknown Illumination**:
  - https://arxiv.org/abs/2106.01970
- Boss et al. (2021) — **NeRD: Neural Reflectance Decomposition from Image Collections**:
  - https://arxiv.org/abs/2012.03918

### External refinement (optional)

- Munkberg et al. (CVPR 2022) — **Extracting Triangular 3D Models, Materials, and Lighting From Images** (“nvdiffrec”)
- Hasselgren et al. (SIGGRAPH Asia 2022) — Monte Carlo variant (“nvdiffrecmc”)
- Lopes et al. (CVPR 2024) — **Material Palette: Extraction of Materials from a Single Image**
- Sartor & Peers (SIGGRAPH Asia 2023) — **MatFusion: a Generative Diffusion Model for SVBRDF Capture**
- **DualMat: PBR Material Estimation via Coherent Dual-Path Diffusion** (arXiv 2025): https://arxiv.org/abs/2508.05060
