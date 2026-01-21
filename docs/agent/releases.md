# Releases

## Prerequisites

```bash
# Check if git-cliff is installed
git-cliff --version || cargo install git-cliff  # or: brew install git-cliff
```

## Creating a Release

```bash
# Generate changelog for new version
git cliff --tag v0.9.0 -o CHANGELOG.md

# Or preview unreleased changes
git cliff --unreleased

# Tag and push
git tag v0.9.0
git push && git push --tags

# Create GitHub release with generated notes
gh release create v0.9.0 --notes "$(git cliff --latest --strip header)"
```

## Version Numbering

Pre-1.0: `0.x.y` where x = breaking/feature, y = bugfix/internal.
Post-1.0: Semantic versioning (major.minor.patch).

## Conventional Commits

**Required format:** `type(scope): description`

| Type | Purpose | Version Impact |
|------|---------|----------------|
| `feat:` | New feature | Minor bump |
| `fix:` | Bug fix | Patch bump |
| `docs:` | Documentation only | Patch bump |
| `refactor:` | Code restructure, no behavior change | Patch bump |
| `perf:` | Performance improvement | Patch bump |
| `test:` | Adding/updating tests | Patch bump |
| `chore:` | Build, CI, tooling | Patch bump |
| `ci:` | CI configuration | Patch bump |
| `revert:` | Revert previous commit | Depends |

**Scope** (optional): Component affected, e.g., `fix(sampler):`, `feat(cli):`

**Breaking changes:** Add `!` after type or include `BREAKING CHANGE:` in body:
```
feat!: Remove deprecated config option
feat(api)!: Change return type of get_field()
```

## Examples

```
feat: Add cuda-build extra for tinycudann
fix(ci): Set TCNN_CUDA_ARCHITECTURES for uv resolution
docs: Update README with modern install instructions
chore: Bump version to 0.8.0
refactor(fields): Simplify SDF field initialization
feat!: Remove legacy dataparser support
```

## Key Files

- `cliff.toml` — git-cliff changelog config
- `pyproject.toml` — version source of truth
