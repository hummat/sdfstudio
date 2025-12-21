# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SDFStudio is a unified framework for neural implicit surface reconstruction built on top of nerfstudio. It implements multiple SDF-based reconstruction methods (UniSurf, VolSDF, NeuS, MonoSDF, etc.) in a modular architecture that makes it easy to transfer ideas between methods.

## Project Structure

```
sdfstudio/
├── sdfstudio/           # Core library
│   ├── cameras/         # Camera models, optimizers, ray generation
│   ├── configs/         # Method configs and base configuration
│   ├── data/            # Data pipeline (dataparsers, datamanagers, datasets)
│   ├── engine/          # Training loop, optimizers, schedulers, callbacks
│   ├── exporter/        # Mesh and texture export utilities
│   ├── fields/          # Neural field implementations (SDF, density, etc.)
│   ├── field_components/# Encodings, MLPs, embeddings, activations
│   ├── model_components/# Losses, renderers, ray samplers, colliders
│   ├── models/          # Method implementations (NeuS, VolSDF, etc.)
│   ├── pipelines/       # Training pipeline orchestration
│   ├── process_data/    # Dataset preprocessing converters
│   ├── utils/           # General utilities (math, marching cubes, etc.)
│   └── viewer/          # Web viewer (Tornado backend + React frontend)
├── scripts/             # CLI entry points
├── tests/               # Unit tests (mirrors sdfstudio/ structure)
├── docs/                # Sphinx documentation
├── omnidata/            # Vendored external tools (do not modify)
└── colab/               # Google Colab notebooks (do not modify)
```

## Development Commands

### Installation

```bash
# Create conda environment (Python 3.10+ required)
conda create --name sdfstudio -y python=3.10
conda activate sdfstudio

# Install PyTorch with CUDA (check pytorch.org for current versions)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install tiny-cuda-nn (optional, for hash grid encoding)
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install SDFStudio in editable mode
pip install -e .

# Install with dev dependencies
pip install -e .[dev]

# Install CLI tab completion
sdf-install-cli
```

### CLI Commands

All commands use the `sdf-` prefix:

| Command | Purpose |
|---------|---------|
| `sdf-train` | Train a model |
| `sdf-eval` | Evaluate a trained model |
| `sdf-render` | Render novel views |
| `sdf-export` | Export model to various formats |
| `sdf-extract-mesh` | Extract mesh from SDF |
| `sdf-render-mesh` | Render extracted mesh |
| `sdf-texture-mesh` | Apply texture to mesh |
| `sdf-process-data` | Preprocess raw data to SDFStudio format |
| `sdf-download-data` | Download demo datasets |
| `sdf-dev-test` | Run full dev testing pipeline (lint, format, tests, docs) |
| `sdf-bridge-server` | Start viewer backend server |
| `sdf-install-cli` | Install CLI tab completion |

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

# Full dev test suite (lint, format, tests, docs)
sdf-dev-test
```

### Training

```bash
# Basic training command structure:
# sdf-train <method> [method-args] <dataparser> --data <path> [dataparser-args]

# Train NeuS-facto on DTU dataset
sdf-train neus-facto --pipeline.model.sdf-field.inside-outside False --vis viewer sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65

# Resume from checkpoint
sdf-train neus-facto --trainer.load-dir outputs/experiment-name/neus-facto/XXX/sdfstudio_models sdfstudio-data --data data/path

# View without training
sdf-train neus-facto --viewer.start-train False --trainer.load-dir outputs/.../sdfstudio_models sdfstudio-data --data data/path

# Download demo data
sdf-download-data sdfstudio --dataset-name sdfstudio-demo-data
```

### Mesh Extraction and Rendering

```bash
# Extract mesh from trained model
sdf-extract-mesh --load-config outputs/experiment-name/neus-facto/XXX/config.yml --output-path meshes/output.ply

# Render mesh to video
sdf-render-mesh --meshfile meshes/output.ply --traj interpolate --output-path renders/output.mp4 sdfstudio-data --data data/path
```

## Method Registry

All 27 methods registered in `sdfstudio/configs/method_configs.py`:

### SDF-Based Methods (20)

| Method | Description |
|--------|-------------|
| `neus-facto` | **Recommended** - NeuS with proposal networks, fast training |
| `neus-facto-bigmlp` | NeuS-facto with larger MLP |
| `neus-facto-angelo` | NeuS-facto with Neuralangelo-style progressive training |
| `neus` | Vanilla NeuS |
| `neus-acc` | NeuS with occupancy grid acceleration |
| `neus2` | NeuS with hash grids and analytic curvature via tcnn double backward |
| `volsdf` | Volume rendering with SDF |
| `geo-volsdf` | VolSDF + patch warping loss |
| `geo-neus` | NeuS + patch warping loss |
| `unisurf` | UniSurf surface extraction |
| `mono-unisurf` | UniSurf + monocular depth/normal priors |
| `geo-unisurf` | UniSurf + patch warping loss |
| `monosdf` | MonoSDF with monocular priors |
| `mono-neus` | NeuS + monocular priors |
| `bakedsdf` | BakedSDF with multi-res hash grids |
| `bakedsdf-mlp` | BakedSDF with large MLPs |
| `neuralangelo` | Neuralangelo with numerical gradients |
| `bakedangelo` | BakedSDF + Neuralangelo |
| `neusW` | NeuralReconW for heritage/outdoor scenes |
| `dto` | DTO occupancy field |

### NeRF-Based Methods (7)

| Method | Description |
|--------|-------------|
| `nerfacto` | Recommended general NeRF method |
| `mipnerf` | Mip-NeRF |
| `vanilla-nerf` | Original NeRF |
| `semantic-nerfw` | Semantic NeRF + NeRF-in-the-wild |
| `tensorf` | TensoRF |
| `dnerf` | Dynamic NeRF |
| `phototourism` | Nerfacto on PhotoTourism data |

## Architecture

### Core Components

SDFStudio follows a hierarchical architecture with clear separation of concerns:

**Cameras** (`sdfstudio/cameras/`)
- Camera models, intrinsics, and extrinsics
- Camera optimizers for pose refinement
- Ray generation and camera paths
- Lie group utilities for SE(3) operations

**Pipeline** (`sdfstudio/pipelines/`)
- Orchestrates the entire training/inference workflow
- Manages interaction between DataManager and Model
- Handles distributed training setup (DDP)
- Entry point for Trainer to fetch losses and visualizations

**DataManager** (`sdfstudio/data/datamanagers/`)
- Manages dataset loading and batching
- Provides train/eval data iterators
- Types: VanillaDataManager, SemanticDataManager, VariableResDataManager
- `DataManager.next_train` returns `(RayBundle, batch)`; batch may include extras (`depth`, `normal`, `sensor_depth`, `pairs`, `sparse_sfm_points`)

**DataParser** (`sdfstudio/data/dataparsers/`)
- Parses different dataset formats into SDFStudio format
- Handles camera poses, intrinsics, and scene metadata
- Available parsers: SDFStudioDataParser, BlenderDataParser, NerfstudioDataParser, HeritageDataParser, MonoSDFDataParser, Record3DDataParser, etc.
- Flow: Dataparser → `DataparserOutputs` → `InputDataset`/`GeneralizedDataset`

**Model** (`sdfstudio/models/`)
- Implements specific reconstruction methods
- Base classes: `Model` → `SurfaceModel` → specific implementations (NeuSModel, VolSDFModel, etc.)
- Handles forward pass, loss computation, and metric calculation
- Models combine fields, samplers (`ray_samplers`), colliders, and renderers

**Field** (`sdfstudio/fields/`)
- Neural network implementations for geometry and appearance
- `SDFField`: Core SDF representation with various encodings (MLP, hash grid, tri-plane)
- Also: DensityField, NerfactoField, SemanticNeRFField, TensoRFField, VanillaNeRFField
- Outputs: SDF values, normals, geometry features, RGB colors

**Field Components** (`sdfstudio/field_components/`)
- Reusable building blocks: encodings, MLPs, spatial distortions
- Supports positional encoding, hash encoding (iNGP), tri-plane encoding
- Embeddings for appearance and temporal variations
- Field heads for density, SDF, color outputs

**Model Components** (`sdfstudio/model_components/`)
- Shared utilities: losses, renderers, ray samplers, scene colliders
- Ray samplers: error-bound sampling (VolSDF), hierarchical sampling (NeuS), proposal networks (NeuS-facto)
- Renderers: RGB, depth, normal, accumulation

**Engine** (`sdfstudio/engine/`)
- Training loop, optimizers, schedulers, callbacks
- Supports multiple optimizers (Adam, RAdam, AdamW) and custom schedulers

**Process Data** (`sdfstudio/process_data/`)
- Dataset conversion utilities for COLMAP, Record3D, Polycam, Metashape, RealityCapture, ODM, Insta360, etc.
- Equirectangular image processing
- HLOC (Hierarchical Localization) utilities

**Exporter** (`sdfstudio/exporter/`)
- Mesh export utilities
- Texture baking and UV mapping
- TSDF fusion utilities

**Viewer** (`sdfstudio/viewer/`)
- Web-based visualization (Tornado server backend + React/TypeScript frontend)
- Enabled with `--vis viewer`; for CI/tests use `--vis wandb` or `--vis tensorboard`

### End-to-End Training Flow

```
CLI (scripts/train.py)
  → Config (from method_configs)
    → Trainer
      → Pipeline
        → DataManager + Model
          → Fields, Samplers, Colliders, Renderers
```

### Configuration System

All models/pipelines/datamanagers are configured via dataclasses using `tyro` for CLI parsing:
- Method configs defined in `sdfstudio/configs/method_configs.py`
- Hierarchical config structure: `Config` contains `TrainerConfig`, `PipelineConfig`, `DataManagerConfig`, `ModelConfig`
- CLI arguments cascade: `sdf-train <method> [method-args] <dataparser> [dataparser-args]`
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

## Coding Style & Naming

- Python 3.10+, 4-space indentation, type hints for public APIs
- Formatting: `black` (line length 120) + `ruff` + `pylint` + `pyright` (configured in `pyproject.toml`)
- Use builtin generics (PEP 585): `list[T]`, `dict[K, V]`, `tuple[...]` instead of `List`, `Dict`, `Tuple`
- Prefer dataclasses for configs; use `config_utils.to_immutable_dict` for default dicts
- Keep names descriptive (`VolSDFModel`, `SDFFieldConfig`); avoid unnecessary abbreviations

## Adding New Methods

1. Create `*ModelConfig` + `*Model` in `sdfstudio/models/`
2. Register a `Config` entry in `method_configs.py` with trainer, pipeline, and optimizers
3. Decide placement: data (dataparser/datamanager), geometry (field/sampler), or training (callbacks/losses)
4. Update docs in `docs/sdfstudio-methods.md` if it affects user-facing configs
5. Add at least a smoke test similar to `tests/test_train.py`

## Testing Guidelines

- Tests live under `tests/` and follow `test_*.py` naming
- Mirror the package structure; e.g., new sampler → `tests/model_components/test_<name>.py`
- Test configuration in `pyproject.toml`: pytest runs with 4 workers, typeguard, coverage enabled
- Ensure `pytest` and `sdf-dev-test` pass before opening a PR
- For validation, run at least one short training with demo data and a fast method (`neus-facto` or `nerfacto`)

## Commit & PR Guidelines

- Commits are short, imperative summaries, e.g. `Refactor typing annotations` or `Add roughness gating to SDF field`
- Group related changes; avoid large, unrelated refactors in a single commit
- PRs should describe the change, motivation, and any API/config differences
- Link to issues or papers when relevant
- Include output snippets or screenshots for viewer/visual changes

## Performance Notes

- High-res marching cubes (`sdf-extract-mesh` with `resolution >= 2048`) is GPU/CPU intensive; avoid in unit tests
- `omnidata/` and dataset conversion scripts are optional; don't assume they run in constrained environments
- For method reference, see `docs/sdfstudio-methods.md` for mapping from `method_configs` to implementations and papers

## Paper References (PaperPipe)

This project implements methods from scientific papers. Papers are managed via `papi` (paperpipe).

### Paper Database Location

Default database root is `~/.paperpipe/`, but it may be overridden (e.g. via `PAPER_DB_PATH`).
Prefer discovering the active location with:

```bash
papi path
```

Per-paper files live at: `<paper_db>/papers/{paper}/`

- `meta.json` — metadata + tags
- `summary.md` — coding-context overview
- `equations.md` — key equations + explanations (best for implementation verification)
- `source.tex` — full LaTeX (if available)
- `paper.pdf` — PDF (used by PaperQA2)

### When to Use What

| Task | Best source |
|------|-------------|
| “Does my code match the paper?” | Read `{paper}/equations.md` (and/or `{paper}/source.tex`) |
| “What’s the high-level approach?” | Read `{paper}/summary.md` |
| “Find the exact formulation / definitions” | Read `{paper}/source.tex` |
| “Which papers discuss X?” | Run `papi search "X"` (fast) or `papi ask "X"` (PaperQA2) |
| “Compare methods across papers” | Load multiple `{paper}/equations.md` files |

### Useful Commands

```bash
# List papers and tags
papi list
papi tags

# Search by title, tag, or content
papi search "sdf loss"

# Export equations/summaries into the repo for a coding session
papi export neuralangelo neus --level equations --to ./paper-context/

# Add papers (arXiv) / regenerate; use --no-llm to avoid LLM calls
papi add 2303.13476 --name neuralangelo
papi regenerate neuralangelo --no-llm
```

### Code Verification Workflow

1. Identify the referenced paper(s) (comments, function names, README, etc.)
2. Read `{paper}/equations.md` and compare symbol-by-symbol with the implementation
3. If ambiguous, confirm definitions/assumptions in `{paper}/source.tex`
4. If the question is broad or spans multiple papers, run `papi ask "..."` (requires PaperQA2)
