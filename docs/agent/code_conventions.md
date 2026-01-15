# Code Conventions

## Python

- Python 3.10+; use builtin generics (PEP 585): `list[T]`, `dict[K, V]`, `tuple[...]`
- Type hints for public APIs; use `typing` and `collections.abc` for complex types
- Prefer dataclasses for configs; use `config_utils.to_immutable_dict` for default dicts
- Explicit error handling; no bare `except:`
- Standard library over new dependencies unless clear win

## Style

- 4-space indentation, line length 120
- Tools: `ruff` (lint + format), `pyright` (types) — all configured in `pyproject.toml`
- KISS principle; boring, readable solutions over clever ones
- Clear names; avoid abbreviations (`SDFFieldConfig` not `SDFCfg`)
- Docstrings only when they add real clarity (no boilerplate)

## Naming Patterns

| Entity | Pattern | Example |
|--------|---------|---------|
| Model class | `*Model` | `NeuSModel`, `VolSDFModel` |
| Config class | `*Config` | `NeuSModelConfig`, `SDFFieldConfig` |
| Field class | `*Field` | `SDFField`, `DensityField` |
| Sampler class | `*Sampler` | `ProposalNetworkSampler` |
| DataParser class | `*DataParser` | `SDFStudioDataParser` |

## Changes

- Minimal diffs; don't reformat unrelated code
- Match existing patterns and style in the file
- Add/update tests when behavior changes
- For multi-file edits: label filenames clearly, keep changes localized

## Metaprogramming

- Dataclasses + `tyro` for CLI config parsing (already pervasive)
- Avoid additional metaprogramming unless following existing patterns

## Common Gotchas

- CLI arg order matters: `sdf-train <method> [method-args] <dataparser> [dataparser-args]`
- `inside-outside` convention: False for object-centric (DTU), True for indoor (Replica)
- Default geometric init is sphere; adjust `bias` for scene type
