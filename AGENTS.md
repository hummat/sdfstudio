# Repository Guidelines

This document is a brief contributor guide for `sdfstudio`, optimized for both humans and code agents.

## Project Structure & Modules

- Core library: `sdfstudio/` (configs, data pipeline, models, engine, viewer).
- Scripts & CLIs: `scripts/` (`sdf-train`, `sdf-eval`, `sdf-render`, `sdf-extract-mesh`, `sdf-export`, etc.).
- Tests: `tests/` mirrors the package layout (`test_train.py`, `model_components/`, `data/`, …).
- Docs: `docs/` (Sphinx site, method and data docs).
- Vendor / external tools: `omnidata/` and `colab/` should not be modified unless you know their upstream.
- Method registry: `sdfstudio/configs/method_configs.py` defines all training presets (`neus-facto`, `bakedsdf`, `nerfacto`, etc.).
- Training engine: `sdfstudio/engine/` (trainer, optimizers, schedulers, callbacks).
- Data pipeline: `sdfstudio/data/` (dataparsers, datamanagers, datasets, pixel samplers).
- Rendering & geometry: `sdfstudio/fields/`, `sdfstudio/model_components/`, `sdfstudio/utils/marching_cubes.py`.

## Build, Test, and Dev Commands

- Install (dev): `pip install -e .[dev]`
- Run unit tests: `pytest` (parallel by default via `pytest.ini`).
- Run full code checks + docs build: `sdf-dev-test`  
  (wraps lint, format, tests, and `docs/` build).
- Build docs only: `cd docs && make html`
- Download example data: `sdf-download-data sdfstudio --dataset-name sdfstudio-demo-data`
- Quick training sanity check: e.g.  
  `sdf-train neus-facto sdfstudio-data --data data/sdfstudio-demo-data/dtu-scan65 --vis viewer`

## Coding Style & Naming

- Python 3.10+, 4‑space indentation, type hints for public APIs and helpers.
- Formatting: `black` (line length 120) + `ruff` + `pylint` + `pyright` (configured in `pyproject.toml`).
- Prefer dataclasses for configs; use `config_utils.to_immutable_dict` for default dicts.
- Keep names descriptive (`VolSDFModel`, `SDFFieldConfig`); avoid unnecessary abbreviations.
- New methods should:
  - Add a `*ModelConfig` + `*Model` in `sdfstudio/models/`.
  - Register a `Config` entry in `method_configs` with trainer, pipeline, and optimizers wired up.

## Testing Guidelines

- Tests live under `tests/` and follow `test_*.py` naming.
- Mirror the package structure; e.g., new sampler → `tests/model_components/test_<name>.py`.
- Ensure `pytest` and `sdf-dev-test` pass before opening a PR.
- For new methods, add at least a smoke test similar to `tests/test_train.py`.

## Commit & PR Guidelines

- Commits are short, imperative summaries, e.g. `Refactor typing annotations` or `Adjust crop factor handling`.
- Group related changes; avoid large, unrelated refactors in a single commit.
- PRs should:
  - Describe the change, motivation, and any API/config differences.
  - Link to issues or papers when relevant.
  - Include output snippets or screenshots for viewer/visual changes.

## Architecture & Workflow Notes (for Agents)

- End-to-end training flow:
  - CLI (`scripts/train.py`) → `Config` (from `method_configs`) → `Trainer` → `Pipeline` → `DataManager` + `Model`.
  - Models combine fields (e.g. `SDFField`), samplers (`ray_samplers`), colliders, and renderers.
- Data flow:
  - Dataparser (`sdfstudio/data/dataparsers`) → `DataparserOutputs` → `InputDataset`/`GeneralizedDataset`.
  - `DataManager.next_train` returns `(RayBundle, batch)`; batch may include extras (`depth`, `normal`, `sensor_depth`, `pairs`, `sparse_sfm_points`).
- Viewer:
  - Viewer is wired via `sdfstudio/viewer/server/viewer_utils.py` and enabled when `Config.vis == "viewer"`.
  - For automated tests or CI, prefer `vis="wandb"` or `"tensorboard"` to avoid long-lived viewer loops.
- Heavy operations:
  - High-res marching cubes (`sdf-extract-mesh` with `resolution >= 2048`) and large datasets are GPU/CPU intensive; avoid wiring them into unit tests.
  - `omnidata/` and dataset conversion scripts under `scripts/datasets/` are optional; do not assume they run in constrained environments.
  - Method reference: see `docs/sdfstudio-methods.md#method-registry-overview` for a mapping from `method_configs` entries to model implementations and original papers.

## Agent-Specific Guidelines

- When modifying behavior:
  - Prefer changing configs in `method_configs` or dedicated `*ModelConfig` fields over inlining constants.
  - Keep new helpers small and local; reuse existing abstractions (`Sampler`, `Field`, `DataManager`, `TrainingCallback`).
- When adding a new feature:
  - Decide whether it belongs in data (dataparser/datamanager), geometry (field/sampler), or training (callbacks/losses).
  - Update documentation in `docs/sdfstudio-methods.md` or `docs/sdfstudio-data.md` if it affects user-facing configs.
- Default validation for agent changes:
  - Run `pytest` plus at least one short training run using a small dataset (e.g. `sdfstudio-demo-data`) and a fast method (`neus-facto` or `nerfacto`).
