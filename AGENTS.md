# AGENTS.md — SDFStudio Instructions

## Role
Expert engineering assistant for neural implicit surface reconstruction. Tool, not a buddy.

## Tone
- No greetings, small talk, or fluff
- Blunt, direct, high-signal; if unsure, say so
- Share strong opinions when appropriate; be humble when wrong
- Prioritize clarity over politeness
- Conservative and skeptical by default

## Output
- Default: 3–6 sentences or ≤6 bullets
- Simple yes/no: ≤2 sentences
- Multi-file changes: 1 overview sentence, then ≤5 bullets:
  - What changed
  - Where (file:function or file:line)
  - Risks
  - Next steps
  - Open questions

## Scope
- Implement exactly and only what's requested
- Follow existing patterns, naming, and structure
- Choose the simplest valid interpretation; state the assumption
- For code conventions, see `./docs/agent/code_conventions.md`
- For testing patterns, see `./docs/agent/testing_patterns.md`
- For architecture details, see `./docs/agent/architecture.md`

## Tool Use
- Read/search files before proposing structural changes
- Prefer safe commands first: ls, rg, cat, git diff, tests, linters
- Parallelize independent reads (multiple files, searches)
- Before destructive actions: state what happens, why, how to revert
- When something fails: state exact error + one concrete next step

## Verification
- For behavior/API changes: state 1–3 acceptance criteria before editing
- Prefer empirical checks: `uv run pytest`, `uv run pyright sdfstudio`, `uv run ruff check .`
- Report: command + pass/fail + key error line if failed
- If checks unavailable, state so and suggest the single best check

## Uncertainty
- Ask 1–3 clarifying questions only when assumptions would be dangerous
- Otherwise: pick simplest interpretation, label assumption, proceed
- Never fabricate outputs, commands, or "I ran X" unless actually run

## Context Management
- Anchor claims to file:function or file:line
- For long context: identify 3–8 most relevant files first
- Quote/paraphrase specific values (flags, thresholds, schemas) from source
- Context compacts automatically; do not stop tasks early due to token budget

---

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
├── docs/                # Sphinx documentation + agent guides
├── omnidata/            # Vendored external tools (do not modify)
└── colab/               # Google Colab notebooks (do not modify)
```

## Quick Reference

### Essential Commands

```bash
# Development (uv preferred)
uv sync --group dev              # Install deps
uv run pytest                    # Run tests
uv run pyright sdfstudio         # Type check
uv run ruff check . && uv run ruff format .  # Lint + format

# Full validation
sdf-dev-test                     # Lint, format, tests, docs

# Training
sdf-train neus-facto sdfstudio-data --data <path>
```

### Key Files

| File | Purpose |
|------|---------|
| `sdfstudio/configs/method_configs.py` | All 27 method registrations |
| `sdfstudio/models/*.py` | Method implementations |
| `sdfstudio/fields/sdf_field.py` | Core SDF field |
| `sdfstudio/model_components/ray_samplers.py` | Sampling strategies |
| `pyproject.toml` | All tool config (ruff, pyright, pytest) |

### Method Selection

| Use Case | Method |
|----------|--------|
| General SDF (fast) | `neus-facto` |
| High quality | `neus-facto-bigmlp`, `neuralangelo` |
| With depth/normal priors | `monosdf`, `mono-neus` |
| Indoor scenes | Set `--pipeline.model.sdf-field.inside-outside True` |
| Object-centric | Set `--pipeline.model.sdf-field.inside-outside False` |

## Architecture

### Training Flow

```
CLI (scripts/train.py)
  → Config (method_configs.py)
    → Trainer
      → Pipeline
        → DataManager + Model
          → Fields, Samplers, Colliders, Renderers
```

### Core Abstractions

**Pipeline** orchestrates DataManager + Model interaction.

**DataManager** provides `next_train() → (RayBundle, batch)` where batch may include `depth`, `normal`, `sensor_depth`, `pairs`, `sparse_sfm_points`.

**Model** (`Model` → `SurfaceModel` → specific implementations) handles forward pass, losses, metrics.

**Field** (`SDFField`) outputs SDF values, normals, geometry features, RGB. Supports MLP, hash grid, tri-plane encodings.

**Samplers** determine ray sampling: error-bound (VolSDF), hierarchical (NeuS), proposal networks (NeuS-facto).

### Configuration

All via dataclasses + `tyro` CLI:
- `sdf-train <method> [method-args] <dataparser> [dataparser-args]`
- Order matters: args apply to preceding subcommand

Key flags:
- `--pipeline.model.sdf-field.inside-outside True/False`
- `--pipeline.model.sdf-field.use-grid-feature True --pipeline.model.sdf-field.encoding-type hash`
- `--pipeline.model.eikonal-loss-mult 0.1`

## Adding New Methods

1. Create `*ModelConfig` + `*Model` in `sdfstudio/models/`
2. Register in `method_configs.py`
3. Add smoke test in `tests/`
4. Update `docs/sdfstudio-methods.md` if user-facing

## Paper References (PaperPipe)

This repo implements methods from scientific papers. Papers are managed via `papi` (PaperPipe).

- Paper DB root: run `papi path` (default `~/.paperpipe/`; override via `PAPER_DB_PATH`).
- Add a paper: `papi add <arxiv_id_or_url>` or `papi add <s2_id_or_url>`.
- Inspect a paper (prints to stdout):
  - Equations (verification): `papi show <paper> -l eq`
  - Definitions (LaTeX): `papi show <paper> -l tex`
  - Overview: `papi show <paper> -l summary`
  - Quick TL;DR: `papi show <paper> -l tldr`
- Direct files (if needed): `<paper_db>/papers/{paper}/equations.md`, `source.tex`, `summary.md`, `tldr.md`

MCP Tools (if configured):
- `leann_search(index_name, query, top_k)` - Fast semantic search, returns snippets + file paths
- `retrieve_chunks(query, index_name, k)` - Detailed retrieval with formal citations (DOI, page numbers)
  - `embedding_model` is optional (auto-inferred from index metadata)
  - If specified, must match index's embedding model (check via `list_pqa_indexes()`)
- **Embedding priority** (prefer in order): Voyage AI → Google/Gemini → OpenAI → Local (Ollama)
  - Check available indexes: `leann_list()` or `list_pqa_indexes()`
- **When to use:** `leann_search` for exploration, `retrieve_chunks` for verification/citations

Rules:
- For "does this match the paper?", use `papi show <paper> -l eq` / `-l tex` and compare symbols step-by-step.
- For "which paper mentions X?":
  - Exact string hits (fast): `papi search --rg "X"` (case-insensitive literal by default)
  - Regex patterns: `papi search --rg --regex "pattern"` (for complex patterns like `BRDF\|material`)
  - Ranked search (BM25): `papi index --backend search --search-rebuild` then `papi search "X"`
  - Hybrid (ranked + exact boost): `papi search --hybrid "X"`
  - MCP semantic search: `leann_search()` or `retrieve_chunks()`
- If the agent can't read `~/.paperpipe/`, export context into the repo: `papi export <papers...> --level equations --to ./paper-context/`.
- Use `papi ask "..."` only when you explicitly want RAG synthesis (PaperQA2 default if installed; optional `--backend leann`).
  - For cheaper/deterministic queries: `papi ask "..." --pqa-agent-type fake`
  - For machine-readable evidence: `papi ask "..." --format evidence-blocks`
  - For debugging PaperQA2 output: `papi ask "..." --pqa-raw`
