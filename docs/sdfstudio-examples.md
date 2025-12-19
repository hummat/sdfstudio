# Reproduce project page results

Here, we provide commands to reproduce reconstruction results on our project page. Please download the corresponding dataset before you run the following commands.

## BRDF “proxy PBR” recipes (practical, copy/paste)

These are lightweight, training-time Ref-NeRF-ish settings intended to produce nicer diffuse/specular separation and
useful proxy roughness maps for export (see `PBR_BRDF_INTEGRATION.md`).

All examples below assume the standard SDFStudio data parser:

```bash
sdf-train neus-facto --vis viewer sdfstudio-data --data YOUR_DATA \
  # ... add flags from the recipes below
```

### Plastic / LEGO (dielectric, neutral highlights)

```bash
sdf-train neus-facto --vis viewer sdfstudio-data --data YOUR_DATA \
  --pipeline.model.sdf-field.use-diffuse-color True \
  --pipeline.model.sdf-field.specular-exclude-geo-features True \
  --pipeline.model.sdf-field.use-reflections True \
  --pipeline.model.sdf-field.use-n-dot-v True \
  --pipeline.model.sdf-field.enable-pred-roughness True \
  --pipeline.model.sdf-field.use-roughness-gated-specular True
```

### Glossy dielectric (ceramic, varnish, lacquer)

```bash
sdf-train neus-facto --vis viewer sdfstudio-data --data YOUR_DATA \
  --pipeline.model.sdf-field.use-diffuse-color True \
  --pipeline.model.sdf-field.specular-exclude-geo-features True \
  --pipeline.model.sdf-field.use-reflections True \
  --pipeline.model.sdf-field.use-n-dot-v True \
  --pipeline.model.sdf-field.enable-pred-roughness True \
  --pipeline.model.sdf-field.use-roughness-gated-specular True \
  --pipeline.model.sdf-field.learned-specular-scale True
```

### Mixed materials (matte + shiny)

```bash
sdf-train neus-facto --vis viewer sdfstudio-data --data YOUR_DATA \
  --pipeline.model.sdf-field.use-diffuse-color True \
  --pipeline.model.sdf-field.use-reflections True \
  --pipeline.model.sdf-field.use-n-dot-v True \
  --pipeline.model.sdf-field.enable-pred-roughness True \
  --pipeline.model.sdf-field.use-roughness-in-color-mlp True \
  --pipeline.model.sdf-field.use-roughness-gated-specular True
```

### Metals (colored specular)

```bash
sdf-train neus-facto --vis viewer sdfstudio-data --data YOUR_DATA \
  --pipeline.model.sdf-field.use-diffuse-color True \
  --pipeline.model.sdf-field.use-specular-tint True \
  --pipeline.model.sdf-field.use-reflections True \
  --pipeline.model.sdf-field.use-n-dot-v True \
  --pipeline.model.sdf-field.enable-pred-roughness True
```

### Auxiliary losses / regularizers (optional add-ons)

These are scene-dependent. Start small and only add what addresses a failure mode you actually see.

- Orientation (Ref-NeRF-style; helps flipped/noisy normals): `--pipeline.model.orientation-loss-mult 1e-4`
- Distortion (Mip-NeRF 360-style; helps “double walls” / floaters): `--pipeline.model.distortion-loss-mult 1e-3`
- Roughness sparsity (can collapse roughness→0; see warnings in `docs/sdfstudio-methods.md`): `--pipeline.model.roughness-sparsity-loss-mult 1e-4`
- Geo-NeuS patch warping (multi-view consistency): `--pipeline.model.patch-warp-loss-mult 0.1 --pipeline.model.patch-size 11 --pipeline.model.topk 4`
- Monocular priors (when you have precomputed mono depth/normals): `--pipeline.model.mono-depth-loss-mult ... --pipeline.model.mono-normal-loss-mult ...`
- RGB-D (when you have sensor depth): `--pipeline.model.sensor-depth-l1-loss-mult ...` (and freespace/SDF variants)
- Sparse SfM points (when reliable): `--pipeline.model.sparse-points-sdf-loss-mult ...`

## NeuS-facto on the heritage dataset

```bash
sdf-train neus-facto-bigmlp --pipeline.model.sdf-field.use-grid-feature False --pipeline.model.sdf-field.use-appearance-embedding True --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False --pipeline.model.sdf-field.bias 0.3 --pipeline.model.sdf-field.beta-init 0.3 --pipeline.model.eikonal-loss-mult 0.0001 --pipeline.model.num-samples-outside 4 --pipeline.model.background-model grid --trainer.steps-per-eval-image 5000 --vis wandb --experiment-name neus-facto-bigmlp-gate --machine.num-gpus 8 heritage-data --data data/heritage/brandenburg_gate
```

## BakedSDF on the mipnerf360 dataset

```
# training
sdf-train bakedsdf-mlp --vis wandb --output-dir outputs/bakedsdf-mlp --trainer.steps-per-eval-batch 5000 --trainer.steps-per-eval-image 5000 --trainer.steps-per-eval-all-images 50000 --trainer.max-num-iterations 250001 --experiment-name bakedsdf-mlp-garden --pipeline.model.sdf-field.bias 1.5 --pipeline.model.sdf-field.inside-outside True --pipeline.model.eikonal-loss-mult 0.01 --pipeline.model.num-neus-samples-per-ray 32 --machine.num-gpus 4 --pipeline.model.scene-contraction-norm l2 mipnerf360-data --data data/nerfstudio-data-mipnerf360/garden

# mesh extraction
sdf-extract-mesh --load-config outputs/XXX/config.yml --output-path meshes/bakedsdf-mlp-garden-4096.ply --bounding-box-min -2.0 -2.0 -2.0 --bounding-box-max 2.0 2.0 2.0 --resolution 4096 --marching-cube-threshold 0.001 --create-visibility-mask True

# rendering
sdf-render-mesh --meshfile meshes/bakedsdf-mlp-garden-4096.ply --traj ellipse --fps 60 --num-views 480 --output-path renders/garden.mp4 mipnerf360-data --data data/nerfstudio-data-mipnerf360/garden
```

## Unisurf, VolSDF, and NeuS with multi-res. grids on the DTU dataset

```bash
# unisurf
sdf-train unisurf --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.num-layers 2 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding False --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False  --pipeline.model.sdf-field.bias 0.5 --trainer.steps-per-eval-image 5000 --pipeline.datamanager.train-num-rays-per-batch 2048 --pipeline.model.background-model none --vis wandb --experiment-name unisurf-dtu122  sdfstudio-data --data data/dtu/scan122

# volsdf
sdf-train volsdf --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.num-layers 2 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding False --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False  --pipeline.model.sdf-field.bias 0.5 --pipeline.model.sdf-field.beta-init 0.1 --trainer.steps-per-eval-image 5000 --pipeline.datamanager.train-num-rays-per-batch 2048 --pipeline.model.background-model none --vis wandb --experiment-name volsdf-dtu106  sdfstudio-data --data data/dtu/scan106

# neus
sdf-train neus --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.num-layers 2 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding False --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False  --pipeline.model.sdf-field.bias 0.5 --pipeline.model.sdf-field.beta-init 0.3 --trainer.steps-per-eval-image 5000 --pipeline.datamanager.train-num-rays-per-batch 2048 --pipeline.model.background-model none --vis wandb --experiment-name neus-dtu114  sdfstudio-data --data data/dtu/scan114
```

## Geo-Unisurf, Geo-VolSDF, and Geo-NeuS with MLP on the DTU dataset

```bash
# geo-unisurf
sdf-train geo-unisurf --pipeline.model.sdf-field.use-grid-feature False --pipeline.model.sdf-field.num-layers 8 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.use-appearance-embedding False --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False  --pipeline.model.sdf-field.bias 0.5 --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 0 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --vis wandb --experiment-name geo-unisurf-dtu110 --pipeline.datamanager.train-num-rays-per-batch 4096 sdfstudio-data --data data/dtu/scan110 --load-pairs True

# geo-volsdf
sdf-train geo-volsdf --pipeline.model.sdf-field.use-grid-feature False --pipeline.model.sdf-field.num-layers 8 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.use-appearance-embedding False --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False  --pipeline.model.sdf-field.bias 0.5 --pipeline.model.sdf-field.beta-init 0.1 --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 0 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --vis wandb --experiment-name geo-volsdf-dtu97 --pipeline.model.eikonal-loss-mult 0.1 --pipeline.datamanager.train-num-rays-per-batch 4096 sdfstudio-data --data data/dtu/scan97 --load-pairs True

#geo-neus
sdf-train geo-neus --pipeline.model.sdf-field.use-grid-feature False --pipeline.model.sdf-field.num-layers 8 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.use-appearance-embedding False --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside False  --pipeline.model.sdf-field.bias 0.5 --pipeline.model.sdf-field.beta-init 0.3 --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 0 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --vis wandb --experiment-name geo-volsdf-dtu24 --pipeline.model.eikonal-loss-mult 0.1 --pipeline.datamanager.train-num-rays-per-batch 4096 sdfstudio-data --data data/dtu/scan24 --load-pairs True

```

## MonoSDF on the Tanks and Temples dataset

```bash
sdf-train monosdf --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.num-layers 2 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding True --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside True  --pipeline.model.sdf-field.bias 0.8 --pipeline.model.sdf-field.beta-init 0.1 --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 0 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --vis wandb --experiment-name monosdf-htnt-scan1 --pipeline.model.mono-depth-loss-mult 0.001 --pipeline.model.mono-normal-loss-mult 0.01 --pipeline.datamanager.train-num-rays-per-batch 2048 --machine.num-gpus 8 sdfstudio-data --data data/tanks-and-temple-highres/scan1 --include_mono_prior True --skip_every_for_val_split 30
```

## NeuS-facto-bigmlp on the Tanks and Temples dataset with monocular prior (Mono-NeuS)

```bash
sdf-train neus-facto-bigmlp --pipeline.model.sdf-field.use-grid-feature False --pipeline.model.sdf-field.use-appearance-embedding True --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside True  --pipeline.model.sdf-field.bias 0.8 --pipeline.model.sdf-field.beta-init 0.3 --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 0 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --vis wandb --experiment-name neus-facto-bigmlp-tnt2 --pipeline.model.mono-depth-loss-mult 0.1 --pipeline.model.mono-normal-loss-mult 0.05 --pipeline.model.eikonal-loss-mult 0.01 --pipeline.datamanager.train-num-rays-per-batch 4096 --machine.num-gpus 8 sdfstudio-data --data data/tanks-and-temple/scan2 --include_mono_prior True --skip_every_for_val_split 30
```

## NeuS-acc with monocular prior on the Replica dataset

```bash
sdf-train neus-acc --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.num-layers 2 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding False --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside True --pipeline.model.sdf-field.bias 0.8 --pipeline.model.sdf-field.beta-init 0.3 --pipeline.model.eikonal-loss-mult 0.1 --pipeline.datamanager.train-num-images-to-sample-from 1 --pipeline.datamanager.train-num-times-to-repeat-images 0 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --pipeline.model.mono-depth-loss-mult 0.1 --pipeline.model.mono-normal-loss-mult 0.05 --pipeline.datamanager.train-num-rays-per-batch 2048 --vis wandb --experiment-name neus-acc-replica1 sdfstudio-data --data data/replica/scan1 --include_mono_prior True
```

## NeuS-RGBD on the synthetic Neural-rgbd dataset

```bash
#kitchen
sdf-train neus --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.num-layers 2 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding True --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside True  --pipeline.model.sdf-field.bias 0.8 --pipeline.model.sdf-field.beta-init 0.3 --pipeline.datamanager.train-num-images-to-sample-from -1 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --vis wandb --experiment-name kitchen_sensor_depth-neus --pipeline.model.sensor-depth-l1-loss-mult 0.1 --pipeline.model.sensor-depth-freespace-loss-mult 10.0 --pipeline.model.sensor-depth-sdf-loss-mult 6000.0 --pipeline.model.mono-normal-loss-mult 0.05 --pipeline.datamanager.train-num-rays-per-batch 2048 --machine.num-gpus 1 sdfstudio-data --data data/neural_rgbd/kitchen_sensor_depth --include_sensor_depth True --skip_every_for_val_split 30

# breadfast-room
sdf-train neus --pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.hidden-dim 256 --pipeline.model.sdf-field.num-layers 2 --pipeline.model.sdf-field.num-layers-color 2 --pipeline.model.sdf-field.use-appearance-embedding True --pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.inside-outside True  --pipeline.model.sdf-field.bias 0.8 --pipeline.model.sdf-field.beta-init 0.3 --pipeline.datamanager.train-num-images-to-sample-from -1 --trainer.steps-per-eval-image 5000 --pipeline.model.background-model none --vis wandb --experiment-name breakfast_room_sensor_depth-neus --pipeline.model.sensor-depth-l1-loss-mult 0.1 --pipeline.model.sensor-depth-freespace-loss-mult 10.0 --pipeline.model.sensor-depth-sdf-loss-mult 6000.0 --pipeline.model.mono-normal-loss-mult 0.05 --pipeline.datamanager.train-num-rays-per-batch 2048 --machine.num-gpus 1 sdfstudio-data --data data/neural_rgbd/breakfast_room_sensor_depth --include_sensor_depth True --skip_every_for_val_split 30
```
