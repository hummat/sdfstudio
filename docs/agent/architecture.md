# Architecture

## Layout

```
sdfstudio/
├── AGENTS.md                 # Agent instructions (CLAUDE.md, GEMINI.md symlink here)
├── pyproject.toml            # Project config, dependencies, tool settings
├── sdfstudio/                # Core library (NOT src-layout)
│   ├── __init__.py           # Package root, version
│   ├── cameras/              # Camera models, ray generation, SE(3) utilities
│   ├── configs/              # Method configs (method_configs.py) + base config
│   ├── data/                 # DataParsers, DataManagers, Datasets
│   ├── engine/               # Trainer, optimizers, schedulers, callbacks
│   ├── exporter/             # Mesh export, texture baking, TSDF fusion
│   ├── fields/               # Neural fields (SDFField, DensityField, etc.)
│   ├── field_components/     # Encodings, MLPs, embeddings, activations
│   ├── model_components/     # Losses, renderers, ray samplers, colliders
│   ├── models/               # Method implementations (NeuS, VolSDF, etc.)
│   ├── pipelines/            # Training pipeline orchestration
│   ├── process_data/         # Dataset conversion (COLMAP, Record3D, etc.)
│   ├── utils/                # Math, marching cubes, general utilities
│   └── viewer/               # Tornado backend + React frontend
├── scripts/                  # CLI entry points (train.py, eval.py, etc.)
├── tests/                    # Unit tests mirror sdfstudio/ structure
│   ├── conftest.py           # Shared fixtures
│   └── test_*.py             # Test files
├── docs/                     # Sphinx docs + agent guides
├── omnidata/                 # Vendored (do not modify)
└── colab/                    # Notebooks (do not modify)
```

## Conventions

- **Flat layout**: Package code directly under `sdfstudio/`, not `src/`
- **Entry points**: CLI via `scripts/*.py`; registered in `pyproject.toml [project.scripts]`
- **Config**: Tool config in `pyproject.toml` (ruff, pyright, pytest, coverage)
- **Dependencies**: Managed via `uv`; lock file is `uv.lock`. Also supports pip/conda.

## Package Manager

```bash
uv sync                       # Install deps (incl. dev) from lock
uv add <pkg>                  # Add dependency
uv add --optional dev <pkg>   # Add dev dependency
uv run <cmd>                  # Run command in venv

# Alternative: pip
pip install -e .[dev]         # Editable install with dev deps
```

## Adding New Modules

1. Create `sdfstudio/<subpackage>/<module>.py`
2. Create corresponding `tests/<subpackage>/test_<module>.py`
3. Export public API in `__init__.py` if needed
4. For new methods: register in `sdfstudio/configs/method_configs.py`

## Key Extension Points

| What | Where |
|------|-------|
| New reconstruction method | `sdfstudio/models/` + `method_configs.py` |
| New field encoding | `sdfstudio/field_components/encodings.py` |
| New ray sampler | `sdfstudio/model_components/ray_samplers.py` |
| New loss function | `sdfstudio/model_components/losses.py` |
| New dataset format | `sdfstudio/data/dataparsers/` |
| New data converter | `sdfstudio/process_data/` |
