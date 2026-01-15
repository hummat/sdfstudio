# Testing Patterns

## Toolchain

- **Runner**: pytest (4 parallel workers default, typeguard enabled)
- **Coverage**: pytest-cov (threshold in `pyproject.toml`)
- **Types**: pyright
- **Lint/format**: ruff

## Commands

```bash
# Basic
uv run pytest                         # Run all tests
uv run pytest tests/test_train.py     # Run single file
uv run pytest -k "test_name"          # Run by name pattern
uv run pytest --no-cov                # Skip coverage
uv run pytest -n 2                    # Custom worker count

# Quality checks
uv run pyright sdfstudio              # Type check
uv run ruff check .                   # Lint
uv run ruff format .                  # Format

# Full validation
sdf-dev-test                          # Lint + format + tests + docs
```

## Test Organization

- Tests mirror `sdfstudio/` structure in `tests/`
- File naming: `test_<module>.py`
- Function naming: `test_<behavior>_<scenario>()`
- Shared fixtures in `tests/conftest.py`

## Writing Tests

- One assertion per test when practical
- Use fixtures for shared setup; define in `conftest.py`
- Prefer real objects over mocks; mock only external I/O
- Test behavior, not implementation
- For new methods: add smoke test similar to `tests/test_train.py`

## Running Tests (Agent Guidance)

- During development: run single test file, not full suite
- After changes: affected tests → types → lint
- Before commit: full suite with coverage (`uv run pytest`)
- For validation: short training with demo data + fast method (`neus-facto` or `nerfacto`)

## Performance Considerations

- High-res marching cubes (`resolution >= 2048`) is GPU/CPU intensive; avoid in unit tests
- `omnidata/` and dataset converters are optional; don't assume available in constrained environments
- Use `--vis tensorboard` or `--vis wandb` in CI (not `--vis viewer`)

## Acceptance Criteria Pattern

Before editing behavior/API:
1. State 1–3 acceptance criteria
2. Implement change
3. Run: `uv run pytest <relevant_test>` + `uv run pyright sdfstudio`
4. Report: command + pass/fail + key error if failed
