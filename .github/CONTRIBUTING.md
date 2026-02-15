# Contributing to sdfstudio

Thanks for your interest in contributing! This document covers development setup and guidelines.

## Development Setup

### Prerequisites

- Python 3.9-3.11
- CUDA-compatible GPU (recommended)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Quick Start

```bash
# Clone the repository
git clone https://github.com/autonomousvision/sdfstudio.git
cd sdfstudio

# Install development dependencies with CUDA PyTorch
uv sync --extra cuda

# Or for CPU-only (CI, no GPU):
# uv sync --extra cpu

# Run all checks
sdf-dev-test
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/ -k test_name -v

# Run with coverage
uv run pytest --cov=sdfstudio --cov-report=term-missing
```

## Code Style

### Python

- Python 3.9+ compatible
- 120-character line limit (see `[tool.ruff]` in pyproject.toml)
- Type hints for function signatures
- Run `ruff format .` and `ruff check .` before committing
- Run `pyright sdfstudio` for type checking

### Workflow

1. Read files before editing — understand existing code
2. After changes: `uv run ruff format .` → `uv run ruff check .` → `uv run pyright sdfstudio` → `uv run pytest`
3. Check if docs need updating

## Architecture

Before making changes, read the architecture docs:

- `docs/agent/architecture.md` — Package structure and method organization
- `docs/agent/code_conventions.md` — Detailed style guide
- `docs/agent/testing_patterns.md` — Testing approach

### Adding New Methods

1. Create `*ModelConfig` + `*Model` in `sdfstudio/models/`
2. Register in `sdfstudio/configs/method_configs.py`
3. Add smoke test in `tests/`
4. Update `docs/sdfstudio-methods.md` if user-facing

## Issues & Project Board

- **Issue templates**: Use the [bug report](https://github.com/hummat/sdfstudio/issues/new?template=bug_report.yml) or [feature request](https://github.com/hummat/sdfstudio/issues/new?template=feature_request.yml) form
- **Labels**: Area and priority labels are defined in `.github/labels.yml`; area labels are auto-applied from issue form dropdowns
- **Project board**: [3D Reconstruction Pipeline](https://github.com/users/hummat/projects/4) — cross-repo board covering mini-mesh, sdfstudio, and dependencies

## Pull Request Process

1. **Create an issue first** for non-trivial changes
2. **Fork and branch** from `main`
3. **Make your changes** following the style guide
4. **Run `sdf-dev-test`** — all checks must pass
5. **Update documentation** if adding/changing methods or CLI flags
6. **Submit PR** using the template

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
type(scope): description

feat: Add new export format
fix(sampler): Handle edge case in ray marching
docs: Update installation guide
refactor(fields): Simplify SDF computation
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`, `ci`
**Scope:** Optional component name in parentheses
**Breaking changes:** Add `!` after type, e.g., `feat!: Remove legacy API`

See `docs/agent/releases.md` for full guidelines.

## What to Contribute

### Good First Issues

- Documentation improvements
- Adding test coverage
- Bug fixes with clear reproduction steps

### Feature Ideas

- New SDF methods
- Export format options
- Viewer improvements

### Before Starting Large Features

Please open an issue first to discuss the approach. This helps avoid duplicate work and ensures the feature aligns with project goals.

## Questions?

- Open a [Discussion](https://github.com/autonomousvision/sdfstudio/discussions) for questions
- Check existing [Issues](https://github.com/autonomousvision/sdfstudio/issues) for known problems
