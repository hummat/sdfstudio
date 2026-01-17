# Repository Guidelines

This file provides guidance to AI coding agents when working with code in this repository.

## Conventions

Read relevant `docs/agent/` files before proceeding:
- `code_conventions.md` — **read before writing code** (style, typing, minimal diffs)
- `architecture.md` — read before adding modules/restructuring
- `testing_patterns.md` — read before writing tests

---

## Project Overview

- **Type:** Python library + CLI for neural implicit surface reconstruction
- **CLI entry points:** `sdf-train`, `sdf-eval`, `sdf-export` (defined in `pyproject.toml`)
- **Goal:** Unified framework for SDF-based methods (NeuS, VolSDF, MonoSDF, etc.) built on nerfstudio

## Commands

```bash
# Dev
uv sync --group dev              # Install deps from lock
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
2. **After code changes**: `uv run ruff format .` → `uv run ruff check .` → `uv run pyright sdfstudio` → `uv run pytest` (order matters)
3. **Doc check**: verify if docs need updating

If `uv run` fails (sandbox/offline): fall back to `.venv/bin/*` or set `UV_CACHE_DIR=$PWD/.uv-cache`.

## Paper References (PaperPipe)

This repo implements methods from scientific papers. Papers are managed via `papi` (PaperPipe).

- Paper DB root: run `papi path` (default `~/.paperpipe/`; override via `PAPER_DB_PATH`).
- Add a paper: `papi add <arxiv_id_or_url>` or `papi add <s2_id_or_url>`.
- Inspect a paper (prints to stdout):
  - Equations (verification): `papi show <paper> -l eq`
  - Definitions (LaTeX): `papi show <paper> -l tex`
  - Overview: `papi show <paper> -l summary`
  - Quick TL;DR: `papi show <paper> -l tldr`
- Direct files (if needed): `<paper_db>/papers/{paper}/equations.md`, `source.tex`, `summary.md`, `tldr.md`, `figures/`

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
