# Documentation

This is a short documentation of SDFStudio, organized as follows:

- [Methods](#Methods)
- [Representations](#Representations)
- [Supervision](#Supervision)

# Methods

SDF Studio implements multiple neural implicit surface reconstruction methods in one common framework. More specifically, SDF Studio builds on [UniSurf](https://github.com/autonomousvision/unisurf), [VolSDF](https://github.com/lioryariv/volsdf), and [NeuS](https://github.com/Totoro97/NeuS). The main difference of these methods is in how the points along the ray are sampled and how the SDF is used during volume rendering. For more details of these methods, please check the corresponding paper. Here we explain these methods shortly and provide examples on how to use them in the following.

## Method Registry Overview

The table below summarizes all methods exposed via `ns-train` / `method_configs`, with their main implementation file and the original paper or project.

| Method | Code (primary) | Paper / Project |
| --- | --- | --- |
| `bakedangelo` | `sdfstudio/models/bakedangelo.py` | [BakedSDF](https://bakedsdf.github.io/) + [Neuralangelo](https://research.nvidia.com/labs/dir/neuralangelo/) |
| `neuralangelo` | `sdfstudio/models/neuralangelo.py` | [Neuralangelo](https://research.nvidia.com/labs/dir/neuralangelo/) |
| `bakedsdf` | `sdfstudio/models/bakedsdf.py` | [BakedSDF](https://bakedsdf.github.io/) |
| `bakedsdf-mlp` | `sdfstudio/models/bakedsdf.py` | BakedSDF large-MLP variant (no separate paper) |
| `neus-facto-angelo` | `sdfstudio/models/neus_facto.py` | NeuS-facto + Neuralangelo-style schedules (no dedicated paper) |
| `neus-facto` | `sdfstudio/models/neus_facto.py` | NeuS with Nerfacto / mip-NeRF360-style proposal sampling (no dedicated paper) |
| `neus-facto-bigmlp` | `sdfstudio/models/neus_facto.py` | Large-MLP NeuS-facto variant (no dedicated paper) |
| `geo-volsdf` | `sdfstudio/models/volsdf.py` | [VolSDF](https://arxiv.org/abs/2106.12052) + [Geo-NeuS](https://github.com/GhiXu/Geo-Neus) patch warping |
| `monosdf` | `sdfstudio/models/monosdf.py` | [MonoSDF](https://arxiv.org/abs/2302.12276) |
| `volsdf` | `sdfstudio/models/volsdf.py` | [VolSDF](https://arxiv.org/abs/2106.12052) |
| `geo-neus` | `sdfstudio/models/neus.py` | [Geo-NeuS](https://github.com/GhiXu/Geo-Neus) |
| `mono-neus` | `sdfstudio/models/neus.py` | MonoSDF-style monocular cues on [NeuS](https://arxiv.org/abs/2106.10689) |
| `neus` | `sdfstudio/models/neus.py` | [NeuS](https://arxiv.org/abs/2106.10689) |
| `unisurf` | `sdfstudio/models/unisurf.py` | [UniSurf](https://arxiv.org/abs/2104.00400) |
| `mono-unisurf` | `sdfstudio/models/unisurf.py` | MonoSDF-style monocular cues on UniSurf (no separate paper) |
| `geo-unisurf` | `sdfstudio/models/unisurf.py` | Geo-NeuS-style patch warping on UniSurf (no separate paper) |
| `dto` | `sdfstudio/models/dto.py` | Internal occupancy-field method (“density guided sampling”, no external paper) |
| `neusW` | `sdfstudio/models/neuralreconW.py` | [NeuralRecon-W](https://github.com/zju3dv/NeuralRecon-W) |
| `neus-acc` | `sdfstudio/models/neus_acc.py` | NeuS with occupancy-grid acceleration (no separate paper) |
| `nerfacto` | `sdfstudio/models/nerfacto.py` | [Nerfstudio / nerfacto](https://arxiv.org/abs/2302.04264) |
| `mipnerf` | `sdfstudio/models/mipnerf.py` | [Mip-NeRF](https://arxiv.org/abs/2103.13415) |
| `semantic-nerfw` | `sdfstudio/models/semantic_nerfw.py` | [Semantic-NeRF](https://shuaifengzhi.com/Semantic-NeRF/) + [NeRF in the Wild](https://nerf-w.github.io/) |
| `vanilla-nerf` | `sdfstudio/models/vanilla_nerf.py` | [NeRF](https://arxiv.org/abs/2003.08934) |
| `tensorf` | `sdfstudio/models/tensorf.py` | [TensoRF](https://arxiv.org/abs/2203.09517) |
| `dnerf` | `sdfstudio/models/dnerf.py` | [D-NeRF](https://arxiv.org/abs/2011.13961) |
| `phototourism` | `sdfstudio/models/nerfacto.py` | Nerfacto on PhotoTourism / NeRF-W-style data (no separate paper) |

## UniSurf

UniSurf first finds the intersection of the surface and sample points around the surface. The sampling range starts from a large range and progressively decreases to a small range during training. When no surface is found for a ray, UniSurf samples uniformly according to the near and far value of the ray. To train a UniSurf model, run the following command:

```
ns-train unisurf --pipeline.model.sdf-field.inside-outside False sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65
```

## VolSDF

VolSDF uses an error-bound sampler [see paper for details] and converts the SDF value to a density value and then uses regular volume rendering as in NeRF. To train a VolSDF model, run the following command:

```
ns-train volsdf --pipeline.model.sdf-field.inside-outside False sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65
```

## NeuS

NeuS uses hierarchical sampling with multiple steps and converts the SDF value to an alpha value based on a sigmoid function [see paper for details]. To train a NeuS model, run the following command:

```
ns-train neus --pipeline.model.sdf-field.inside-outside False sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65
```

## MonoSDF

[MonoSDF](https://github.com/autonomousvision/monosdf) builds on VolSDF and proposes to use monocular depth and normal cues as additional supervision. This is particularly helpful in sparse settings (little views) and in indoor scenes. To train a MonoSDF model for an indoor scene, run the following command:

```
ns-train monosdf --pipeline.model.sdf-field.inside-outside True sdfstudio-data --data data/sdfstudio-demo-data/replica-room0 --include-mono-prior True

```

## Mono-UniSurf

Similar to MonoSDF, Mono-UniSurf uses monocular depth and normal cues as additional supervision for UniSurf. To train a Mono-UniSurf model, run the following command:

```
ns-train mono-unisurf --pipeline.model.sdf-field.inside-outside True sdfstudio-data --data data/sdfstudio-demo-data/replica-room0 --include-mono-prior True
```

## Mono-NeuS

Similar to MonoSDF, Mono-NeuS uses monocular depth and normal cues as additional supervision for NeuS. To train a Mono-NeuS model, run the following command:

```
ns-train mono-neus --pipeline.model.sdf-field.inside-outside True sdfstudio-data --data data/sdfstudio-demo-data/replica-room0 --include-mono-prior True
```

## Geo-NeuS

[Geo-NeuS](https://github.com/GhiXu/Geo-Neus) builds on NeuS and proposes a multi-view photometric consistency loss. To train a Geo-NeuS model on the DTU dataset, run the following command:

```
ns-train geo-neus --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan24 --load-pairs True
```

## Geo-UniSurf

The idea of Geo-NeuS can also be applied to UniSurf, which we call Geo-UniSurf. To train a Geo-UniSurf model on the DTU dataset, run the following command:

```
ns-train geo-unisurf --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan24 --load-pairs True
```

## Geo-VolSDF

Similarly, we can apply the idea of Geo-NeuS to VolSDF, which we call Geo-VolSDF. To train a Geo-VolSDF model on the DTU dataset, run the following command:

```
ns-train geo-volsdf --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan24 --load-pairs True
```

## NeuS-acc

NeuS-acc maintains an occupancy grid for empty space skipping during point sampling along the ray. This significantly reduces the number of samples required during training and hence speeds up training. To train a NeuS-acc model on the DTU dataset, run the following command:

```
ns-train neus-acc --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan65

```

## NeuS-facto

NeuS-facto is inspired by [nerfacto](https://github.com/nerfstudio-project/nerfstudio) in nerfstudio, where the proposal network from [mip-NeRF360](https://jonbarron.info/mipnerf360/) is used for sampling points along the ray. We apply this idea to NeuS to speed up the sampling process and reduce the number of samples for each ray. To train a NeuS-facto model on the DTU dataset, run the following command:

```
ns-train neus-facto --pipeline.model.sdf-field.inside-outside False sdfstudio-data -data data/dtu/scan65
```

## NeuralReconW

[NeuralReconW](https://github.com/zju3dv/NeuralRecon-W) is specifically designed for heritage scenes and hence can only be applied to these scenes. Specifically, it uses sparse point clouds from colmap to create a coarse occupancy grid. Using this occupancy grid, the near and far plane for each ray can be determined. Points are sampled uniformly along the ray within the near and far plane. Further, NeuralReconW also uses surface guided sampling, by sampling points in a small range around the predicted surface. To speed up sampling, it uses a high-resolution grid to cache the SDF field such that no network queries are required to find the surface intersection. The SDF cache is regularly updated during training (every 5K iterations). To train a NeuralReconW model on the DTU dataset, run the following command:

```
ns-train neusW --pipeline.model.sdf-field.inside-outside False heritage-data --data data/heritage/brandenburg_gate
```

# Representations

The representation stores geometry and appearance. The geometric mapping takes a 3D position as input and outputs an SDF value, a normal vector, and a geometric feature vector. The color mapping (implemented as an MLP) takes a 3D position and view direction together with the normal vector and the geometry feature vector from the geometry mapping as input and outputs an RGB color vector.

We support three representations for the geometric mapping: MLPs, Multi-Res. Feature Grids from [iNGP](https://github.com/NVlabs/instant-ngp), and Tri-plane from [ConvONet](https://github.com/autonomousvision/convolutional_occupancy_networks) or [EG3D](https://github.com/NVlabs/eg3d). We now explain these representations in more detail:

## MLPs

The 3D position is encoded using a positional encoding as in NeRF and passed to a multi-layer perceptron (MLP) network to predict an SDF value, normal, and geometry feature. To train VolSDF with an MLP with 8 layers and 512 hidden dimensions, run the following command:

```
ns-train volsdf --pipeline.model.sdf-field.use-grid-feature False sdfstudio-data --data YOUR_DATA
```

## Multi-res Feature Grids

The 3D position is first mapped to a multi-resolution feature grid, using tri-linear interpolation to retrieve the corresponding feature vector. This feature vector is then used as input to an MLP to predict SDF, normal, and geometry features. To train a VolSDF model with Multi-Res Feature Grid representation with 2 layers and 256 hidden dimensions, run the following command:

```
ns-train volsdf --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.encoding-type hash sdfstudio-data --data YOUR_DATA
```

## Tri-plane

The 3D position is first mapped to three orthogonal planes, using bi-linear interpolation to retrieve a feature vector for each plane which are concatenated as input to the MLP. To use a tri-plane representation on VolSDF, run the following command:

```
ns-train volsdf --pipeline.model.sdf-field.use-grid-feature True  --pipeline.model.sdf-field.encoding-type tri-plane sdfstudio-data --data YOUR_DATA
```

## Geometry Initialization

Proper initialization is very important to obtain good results. By default, SDF Studio initializes the SDF as a sphere. For example, for the DTU dataset, you can initialize the network with the following command:

```
ns-train volsdf  --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.bias 0.5 --pipeline.model.sdf-field.inside-outside False
```

For indoor scenes, please initialize the model using the following command:

```
ns-train volsdf --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.bias 0.8 --pipeline.model.sdf-field.inside-outside True
```

Note that for indoor scenes the cameras are inside the sphere, so we set `inside-outside` to `True` such that the points inside the sphere will have positive SDF values and points outside the sphere will have negative SDF values.

## Color Network

The color network is an MLPs, similar to the geometry MLP. It can be config using the following command:

```
ns-train volsdf --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.hidden-dim-color 512
```

## Improving Geometry Under Adverse Conditions

This section summarizes practical knobs that help when geometry quality is poor in challenging scenes (weak texture,
glossy materials, noisy poses). It is intentionally high level; see method papers for theory.

### 1. Fix Poses Before Anything Else

If Structure-from-Motion (SfM) poses are wrong, geometry will be wrong regardless of BRDF settings.

- Use a stronger SfM pipeline (e.g. exhaustive matching, Glomap, HLoc, VGGSfM) appropriate for your data.
- Where available, enable camera optimization in the SDFStudio config to refine poses during training, e.g.:

```bash
ns-train neus sdfstudio-data --data YOUR_DATA \
  --pipeline.datamanager.camera-optimizer.mode SO3xR3 \
  --pipeline.datamanager.camera-optimizer.optimizer.lr 1e-4 \
  --pipeline.datamanager.camera-optimizer.scheduler.lr-final 1e-5 \
  --pipeline.datamanager.camera-optimizer.scheduler.max-steps 5000
```

### 2. Robust BRDF Settings for SDF Fields

For SDF-based methods (`neus`, `neus-facto`, `volsdf`, `geo-neus`, etc.), the following `SDFFieldConfig` flags often
improve geometry on reflective or weakly textured surfaces by giving the color MLP better cues:

- **Baseline for most scenes (even mostly diffuse)**:

```bash
--pipeline.model.sdf-field.use-diffuse-color True
--pipeline.model.sdf-field.use-n-dot-v True
```

- **Scenes with clear gloss / specular highlights**:

```bash
--pipeline.model.sdf-field.use-diffuse-color True
--pipeline.model.sdf-field.use-specular-tint True
--pipeline.model.sdf-field.use-reflections True
--pipeline.model.sdf-field.use-n-dot-v True
```

- **Very shiny / metallic objects (strong mirror-like reflections)**:

```bash
--pipeline.model.sdf-field.enable-pred-roughness True
```

Notes:

- These flags only affect the *appearance* model; SDF geometry is still learned from color + regularizers, but the
  gradient signal is often more geometry-friendly.
- `use-n-dot-v` is cheap and generally safe to keep enabled whenever you care about good geometry.

### 3. Turn Up Existing Geometry Priors

Before inventing new losses, consider using the priors that already exist in SDFStudio:

- **Patch warping / Geo-NeuS-style multi-view consistency**  
  For methods that support it (e.g. `geo-neus`, `geo-volsdf`), increase `patch_warp_loss_mult` and use a moderate
  patch size / top-k:

  ```bash
  --pipeline.model.patch-warp-loss-mult 0.1 \
  --pipeline.model.patch-size 11 \
  --pipeline.model.topk 4
  ```

- **Monocular priors (MonoSDF-style)**  
  When you have precomputed mono normals / depth, enable `mono_normal_loss_mult` and/or `mono_depth_loss_mult` to
  stabilize large, weakly textured regions.

- **Sparse SfM point constraints**  
  For scenes with reliable sparse points, use `sparse_points_sdf_loss_mult > 0` so the SDF is directly constrained at
  those locations.

Start with small multipliers; over-regularization can flatten real detail.

### 4. Orientation and Distortion Losses

Two additional losses can be useful to stabilize geometry under adverse conditions:

- **Orientation loss (Ref-NeRF-style)**  
  Encourages visible normals to face towards the camera. Available for:
  - `nerfacto` (via `orientation_loss_mult` in `NerfactoModelConfig`).
  - All SDF models inheriting `SurfaceModel` (via `orientation_loss_mult` in `SurfaceModelConfig`).

  This is most helpful when normals are noisy or flipped in low-texture regions.

- **Distortion loss (Mip-NeRF 360-style)**  
  Encourages compact, non-overlapping depth distributions along each ray. Available for:
  - Proposal-based SDF methods (`neus-facto`, `bakedsdf`, `bakedangelo`) via `distortion_loss_mult` on
    `weights_list` / `ray_samples_list`.
  - Default `neus` via `distortion_loss_mult` and `nerfstudio_distortion_loss` on the main sampling hierarchy.

  A small `distortion_loss_mult` can reduce “double walls” and smeared geometry, especially in cluttered or
  unbounded scenes.

### 5. When It’s Still Not Enough

If geometry remains poor after the above:

- Adjust the **capture setup**: matte spray or washable paint for transparent/metallic objects, add texture in the
  background, use higher-quality images (less motion blur, better exposure).
- Only then consider **heavier models** (explicit lighting / environment estimation, microfacet BRDF heads, refractive
  components), which typically require research-level effort and are outside the scope of SDFStudio’s default methods.

# Supervision

## RGB Loss

We use the L1 loss for the RGB loss to supervise the volume rendered color at each ray. This is the default for all models.

## Mask Loss

The (optional) mask loss can be helpful to separate the foreground object from the background. However, it requires additional masks as inputs. For example, in NeuralReconW, a segmentation network can be used to predict the sky region and the sky segmentation can be used as a label for the mask loss. The mask loss is used by default if masks are provided in the dataset. You can change the weight for the mask loss via:

```
--pipeline.model.fg-mask-loss-mult 0.001
```

## Eikonal Loss

The Eikonal loss is used for all SDF-based methods to regularize the SDF field to properly represent SDFs. It is not used for UniSurf which uses an occupancy field. You can change the weight of eikonal loss with the following command:

```
--pipeline.model.eikonal-loss-mult 0.01
```

## Orientation Loss

The orientation loss (proposed in Ref-NeRF) encourages surface normals to roughly face the camera where there is
non-zero density. It is available in:

- `nerfacto` via `orientation_loss_mult` and predicted normals.
- All SDF models that use `SurfaceModel` via `orientation_loss_mult` in `SurfaceModelConfig`, using SDF-derived normals.

Example:

```bash
ns-train neus sdfstudio-data --data YOUR_DATA \
  --pipeline.model.orientation-loss-mult 1e-4
```

Use a small multiplier; this is a soft prior, not a hard constraint.

## Distortion Loss

The distortion loss (Mip-NeRF 360 style) penalizes stretched or multi-modal depth distributions along a ray. It helps
avoid “double walls” and smeared geometry, especially in unbounded or cluttered scenes.

It is available for:

- Proposal-based SDF methods such as `neus-facto`, `bakedsdf`, and `bakedangelo` via `distortion_loss_mult`, operating
  on `weights_list` / `ray_samples_list`.
- Default `neus` via `distortion_loss_mult`, using `nerfstudio_distortion_loss` on the main sampling hierarchy.

Example:

```bash
ns-train neus-facto sdfstudio-data --data YOUR_DATA \
  --pipeline.model.distortion-loss-mult 0.002
```

As with other regularizers, start with a small value and increase only if you see clear benefits.

## Smoothness Loss

The smoothness loss encourages smooth surfaces. This loss is used in UniSurf and encourages the normal of a surface point and the normal of a point sampled in its neighborhood to be similar. The weight for the smoothness loss can be changed with the following command:

```
--pipeline.model.smooth-loss-multi 0.01
```

## Monocular Depth Consistency

The monocular depth consistency loss is proposed in MonoSDF and uses depth predicted by a pretrained monocular depth network as additional constraint per image. This is particularly helpful in sparse settings (little views) and in indoor scenes. The weight for monocular depth consistency loss can be changed with the following command:

```
--pipeline.model.mono-depth-loss-mult 0.1
```

## Monocular Normal Consistency

The monocular normal consistency loss is proposed in MonoSDF and uses normals predicted by a pretrained monocular normal network as additional constraint during training. This is particularly helpful in sparse settings (little views) and in indoor scenes. The weight for monocular normal consistency loss can be changed with the following command:

```
--pipeline.model.mono-normal-loss-mult 0.05
```

## Multi-view Photometric Consistency

Encouraging multi-view photometric consistency is proposed in Geo-NeuS. For each ray, we seek the intersection with the surface and use the corresponding homography to warp patches from the source views to the target views and comparing those patches using normalized cross correlation (NCC). The weight for the multi-view photometric consistency loss can be changed with the following command:

```
--pipeline.model.patch-size 11 --pipeline.model.patch-warp-loss-mult 0.1 --pipeline.model.topk 4
```

where topk denotes the number of nearby views which have the smallest NCC error. Only those patches are used for supervision, effectively ignoring outliers, e.g., due to occlusion.

## Sensor Depth Loss

RGBD data is useful for high-quality surface reconstruction. [Neural RGB-D Surface Reconstruction](https://github.com/dazinovic/neural-rgbd-surface-reconstruction) propose two different loss functions: free space loss and sdf loss. Free space loss enforces the network to predict large SDF values between the camera origin and the truncation region of the observed surface. SDF loss enforces the network to predict approximate SDF values converted from depth observations. We further support L1 loss which enforce the consistency between volume rendered depth and sensor depth. The truncation value and the weights for sensor depth loss can be changed with the following command:

```bash
# truncation is set to 5cm with a rough scale value 0.3 (0.015 = 0.05 * 0.3)
--pipeline.model.sensor-depth-truncation 0.015 --pipeline.model.sensor-depth-l1-loss-mult 0.1 --pipeline.model.sensor-depth-freespace-loss-mult 10.0 --pipeline.model.sensor-depth-sdf-loss-mult 6000.0
```
