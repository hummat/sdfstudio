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

# Install development dependencies (includes CUDA PyTorch)
uv sync

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

## Pull Request Process

1. **Create an issue first** for non-trivial changes
2. **Fork and branch** from `main`
3. **Make your changes** following the style guide
4. **Run `sdf-dev-test`** — all checks must pass
5. **Update documentation** if adding/changing methods or CLI flags
6. **Submit PR** using the template

### Commit Messages

- Use present tense: "Add feature" not "Added feature"
- Keep the first line under 72 characters
- Reference issues: "Fix mesh export (#42)"
- Optionally prefix with type: `feat:`, `fix:`, `docs:`, `refactor:`

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
