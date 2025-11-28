# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SDFStudio is a unified framework for neural implicit surface reconstruction built on top of nerfstudio. It implements multiple SDF-based reconstruction methods (UniSurf, VolSDF, NeuS, MonoSDF, etc.) in a modular architecture that makes it easy to transfer ideas between methods.

## Development Commands

### Installation

```bash
# Create conda environment
conda create --name sdfstudio -y python=3.8
conda activate sdfstudio

# Install PyTorch with CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install SDFStudio in editable mode
pip install -e .

# Install CLI tab completion
ns-install-cli
```

### Testing

```bash
# Run all tests with pytest (uses 4 parallel workers, coverage enabled)
pytest

# Run specific test file
pytest tests/test_train.py

# Run tests without coverage
pytest --no-cov

# Run with specific number of workers
pytest -n 2
```

### Code Quality

```bash
# Format code with ruff
ruff format .

# Lint with ruff
ruff check .

# Type checking with pyright
pyright sdfstudio

# Legacy linting (if needed)
black sdfstudio/ scripts/ tests/
pylint sdfstudio tests scripts
```

### Training

```bash
# Basic training command structure:
# ns-train <method> [method-args] <dataparser> --data <path> [dataparser-args]

# Train NeuS-facto on DTU dataset
ns-train neus-facto --pipeline.model.sdf-field.inside-outside False --vis viewer sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65

# Resume from checkpoint
ns-train neus-facto --trainer.load-dir outputs/experiment-name/neus-facto/XXX/sdfstudio_models sdfstudio-data --data data/path

# View without training
ns-train neus-facto --viewer.start-train False --trainer.load-dir outputs/.../sdfstudio_models sdfstudio-data --data data/path
```

### Mesh Extraction and Rendering

```bash
# Extract mesh from trained model
ns-extract-mesh --load-config outputs/experiment-name/neus-facto/XXX/config.yml --output-path meshes/output.ply

# Render mesh to video
ns-render-mesh --meshfile meshes/output.ply --traj interpolate --output-path renders/output.mp4 sdfstudio-data --data data/path
```

## Architecture

### Core Components

SDFStudio follows a hierarchical architecture with clear separation of concerns:

**Pipeline** (`sdfstudio/pipelines/`)
- Orchestrates the entire training/inference workflow
- Manages interaction between DataManager and Model
- Handles distributed training setup (DDP)
- Entry point for Trainer to fetch losses and visualizations

**DataManager** (`sdfstudio/data/datamanagers/`)
- Manages dataset loading and batching
- Provides train/eval data iterators
- Types: VanillaDataManager, FlexibleDataManager, SemanticDataManager, VariableResDataManager

**DataParser** (`sdfstudio/data/dataparsers/`)
- Parses different dataset formats into SDFStudio format
- Handles camera poses, intrinsics, and scene metadata
- Available parsers: SDFStudioDataParser, BlenderDataParser, NerfstudioDataParser, HeritageDataParser, etc.

**Model** (`sdfstudio/models/`)
- Implements specific reconstruction methods
- Base classes: `Model` → `SurfaceModel` → specific implementations (NeuSModel, VolSDFModel, etc.)
- Handles forward pass, loss computation, and metric calculation
- Key models: neus.py, volsdf.py, unisurf.py, monosdf.py, neus_facto.py, neus_acc.py, neuralangelo.py, bakedsdf.py

**Field** (`sdfstudio/fields/`)
- Neural network implementations for geometry and appearance
- `SDFField`: Core SDF representation with various encodings (MLP, hash grid, tri-plane)
- Outputs: SDF values, normals, geometry features, RGB colors

**Field Components** (`sdfstudio/field_components/`)
- Reusable building blocks: encodings, MLPs, spatial distortions
- Supports positional encoding, hash encoding (iNGP), tri-plane encoding

**Model Components** (`sdfstudio/model_components/`)
- Shared utilities: losses, renderers, ray samplers, scene colliders
- Ray samplers: error-bound sampling (VolSDF), hierarchical sampling (NeuS), proposal networks (NeuS-facto)
- Renderers: RGB, depth, normal, accumulation

**Engine** (`sdfstudio/engine/`)
- Training loop, optimizers, schedulers, callbacks
- Supports multiple optimizers (Adam, RAdam, AdamW) and custom schedulers

### Method-Specific Architectures

**NeuS-facto** (Recommended for most use cases)
- Uses proposal network from mip-NeRF360 for efficient sampling
- Significantly faster training than vanilla NeuS
- Hybrid representation: hash grid + small MLP

**NeuS-acc**
- Maintains occupancy grid for empty space skipping
- Speeds up training by reducing number of samples per ray

**MonoSDF/Mono-NeuS/Mono-UniSurf**
- Incorporates monocular depth and normal priors
- Particularly useful for indoor scenes and sparse views
- Requires preprocessing to extract monocular cues

**BakedSDF/BakedAngelo/Neuralangelo**
- Focus on high-quality reconstruction
- Neuralangelo: numerical gradients + progressive training
- BakedSDF: hybrid representation for faster rendering

**NeuralReconW**
- Designed for heritage/outdoor scenes
- Uses sparse point clouds from COLMAP for occupancy grid
- Voxel-surface guided sampling with SDF caching

### Configuration System

All models/pipelines/datamanagers are configured via dataclasses using `tyro` for CLI parsing:
- Method configs defined in `sdfstudio/configs/method_configs.py`
- Hierarchical config structure: `Config` contains `TrainerConfig`, `PipelineConfig`, `DataManagerConfig`, `ModelConfig`
- CLI arguments cascade: `ns-train <method> [method-args] <dataparser> [dataparser-args]`
- **Important**: Order matters! Arguments apply to the preceding subcommand

### Key Design Patterns

**Inside-Outside Convention**
- `inside-outside=False`: For object-centric scenes (DTU). Negative SDF inside object, positive outside.
- `inside-outside=True`: For indoor scenes (Replica). Cameras inside scene volume. Positive SDF inside, negative outside.
- Set via `--pipeline.model.sdf-field.inside-outside True/False`

**Geometric Initialization**
- Default: SDF initialized as sphere
- Object-centric: `--pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.bias 0.5`
- Indoor: `--pipeline.model.sdf-field.geometric-init True --pipeline.model.sdf-field.bias 0.8`

**Representations**
- MLP: `--pipeline.model.sdf-field.use-grid-feature False`
- Hash Grid: `--pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.encoding-type hash`
- Tri-plane: `--pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.encoding-type tri-plane`

## Data Format

SDFStudio uses a JSON-based metadata format (`meta_data.json`) that includes:
- Camera model (currently only OPENCV supported)
- Image dimensions
- Per-frame data: RGB path, camera-to-world transform, intrinsics
- Optional: monocular depth/normal priors, sensor depth, masks
- Scene bounding box (AABB) and near/far planes
- Collider type: "near_far", "box", or "sphere"

See `docs/sdfstudio-data.md` for complete format specification.

## Common Loss Functions

Configure loss weights via CLI:
- RGB Loss (L1): Always enabled
- Eikonal: `--pipeline.model.eikonal-loss-mult 0.1` (regularizes SDF field)
- Foreground Mask: `--pipeline.model.fg-mask-loss-mult 0.01` (requires mask data)
- Mono Depth: `--pipeline.model.mono-depth-loss-mult 0.1` (requires monocular priors)
- Mono Normal: `--pipeline.model.mono-normal-loss-mult 0.05` (requires monocular priors)
- Multi-view Patch Warp: `--pipeline.model.patch-warp-loss-mult 0.1 --pipeline.model.topk 4`
- Sensor Depth: `--pipeline.model.sensor-depth-l1-loss-mult 0.1` (for RGB-D data)

## Type Annotations

Recent updates use builtin generics (PEP 585):
- Use `list[T]` instead of `List[T]`
- Use `dict[K, V]` instead of `Dict[K, V]`
- Use `tuple[...]` instead of `Tuple[...]`
- Requires Python 3.10+ (see pyproject.toml)

## Testing Notes

- Test configuration in `pyproject.toml`: pytest runs with 4 workers, typeguard, coverage enabled
- Tests located in `tests/` directory mirroring `sdfstudio/` structure
- Key test: `tests/test_train.py` - validates training setup
