# Repository Guidelines

This file provides guidance to AI coding agents when working with code in this repository.

## Conventions

Read relevant `docs/agent/` files before proceeding:
- `workflow.md` — **read before starting any feature** (issues, branching, PRs)
- `code_conventions.md` — **read before writing code** (style, typing, minimal diffs)
- `architecture.md` — read before adding modules/restructuring
- `testing_patterns.md` — read before writing tests

**REQUIRED: Read `docs/agent/workflow.md` before implementing, updating, fixing, or changing anything.**

---

## Project Overview

- **Type:** Python library + CLI for neural implicit surface reconstruction
- **CLI entry points:** `sdf-train`, `sdf-eval`, `sdf-export` (defined in `pyproject.toml`)
- **Goal:** Unified framework for SDF-based methods (NeuS, VolSDF, MonoSDF, etc.) built on nerfstudio

## Commands

```bash
# Dev
uv sync                          # Install deps (incl. dev) from lock
uv run pytest                    # Run tests
uv run pyright sdfstudio         # Type check
uv run ruff check . && uv run ruff format .  # Lint + format
sdf-dev-test                     # Full validation

# Training
sdf-train neus-facto sdfstudio-data --data <path>
```

## Key Files

| File | Purpose |
|------|---------|
| `sdfstudio/configs/method_configs.py` | All 27 method registrations |
| `sdfstudio/models/*.py` | Method implementations |
| `sdfstudio/fields/sdf_field.py` | Core SDF field |
| `sdfstudio/model_components/ray_samplers.py` | Sampling strategies |
| `pyproject.toml` | All tool config (ruff, pyright, pytest) |

## Method Selection

| Use Case | Method |
|----------|--------|
| General SDF (fast) | `neus-facto` |
| High quality | `neus-facto-bigmlp`, `neuralangelo` |
| With depth/normal priors | `monosdf`, `mono-neus` |
| Indoor scenes | Set `--pipeline.model.sdf-field.inside-outside True` |
| Object-centric | Set `--pipeline.model.sdf-field.inside-outside False` |

## Adding New Methods

1. Create `*ModelConfig` + `*Model` in `sdfstudio/models/`
2. Register in `method_configs.py`
3. Add smoke test in `tests/`
4. Update `docs/sdfstudio-methods.md` if user-facing

## Code Workflow

1. **Before editing**: read files first; understand existing code
2. **After code changes**: run the CI validation pipeline:
   ```bash
   uv run ruff format --check . && uv run ruff check . && uv run pyright sdfstudio && uv run pytest -m "not integration and not slow"
   ```
   If format check fails, run `uv run ruff format .` to fix, then re-run the pipeline.
3. **Doc check**: verify if docs need updating

If `uv run` fails (sandbox/offline): fall back to `.venv/bin/*` or set `UV_CACHE_DIR=$PWD/.uv-cache`.
