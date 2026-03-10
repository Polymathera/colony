# Publishing & CI/CD

This guide walks through the full setup for GitHub Actions CI, PyPI publishing, and GitHub Pages documentation deployment.

## Overview

Colony uses three GitHub Actions workflows:

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `ci.yml` | Push to `main`, PRs | Lint (ruff), test (pytest), build docs |
| `publish.yml` | Push a `v*` tag | Build and publish to TestPyPI → PyPI |
| `docs.yml` | Push to `main` (docs/src changes) | Deploy docs to GitHub Pages |

---

## 1. GitHub Actions CI

**File**: `.github/workflows/ci.yml`

This runs automatically on every PR and push to `main`. No setup needed — it works out of the box once the repo is on GitHub.

### What it runs

- **Lint**: `ruff check src/` and `ruff format --check src/` — catches style issues and import ordering
- **Test**: `pytest` on Python 3.11 and 3.12, with a Redis service container for integration tests
- **Docs**: `mkdocs build` — ensures documentation compiles without errors

### Running locally

```bash
# Lint (same as CI)
ruff check src/
ruff format --check src/

# Auto-fix lint issues
ruff check --fix src/
ruff format src/

# Run tests
pytest src/ --timeout=120 -x -q

# Build docs
mkdocs build
```

---

## 2. PyPI Publishing

### 2.1 One-time setup: Trusted Publishing (OIDC)

PyPI supports "trusted publishing" — your GitHub Actions workflow authenticates directly with PyPI using OpenID Connect. No API tokens to manage or rotate.

#### Step A: Create accounts

1. Go to [https://pypi.org/account/register/](https://pypi.org/account/register/) and create an account
2. Go to [https://test.pypi.org/account/register/](https://test.pypi.org/account/register/) and create an account (separate registration)
3. Enable 2FA on both (required for publishing)

#### Step B: Register as a trusted publisher on TestPyPI

This is done **before** your first publish — you're telling TestPyPI to trust your GitHub repo.

1. Go to [https://test.pypi.org/manage/account/publishing/](https://test.pypi.org/manage/account/publishing/)
2. Under "Add a new pending publisher", fill in:
    - **PyPI project name**: `polymathera-colony`
    - **Owner**: `polymathera` (your GitHub org/username)
    - **Repository**: `colony`
    - **Workflow name**: `publish.yml`
    - **Environment name**: `testpypi`
3. Click "Add"

#### Step C: Register as a trusted publisher on PyPI

Same process on production PyPI:

1. Go to [https://pypi.org/manage/account/publishing/](https://pypi.org/manage/account/publishing/)
2. Fill in:
    - **PyPI project name**: `polymathera-colony`
    - **Owner**: `polymathera`
    - **Repository**: `colony`
    - **Workflow name**: `publish.yml`
    - **Environment name**: `pypi`
3. Click "Add"

#### Step D: Create GitHub environments

1. Go to your repo on GitHub → **Settings** → **Environments**
2. Create environment `testpypi`:
    - No special protection rules needed (optional: add reviewers for extra safety)
3. Create environment `pypi`:
    - **Recommended**: Add a required reviewer (yourself) so you can approve before publishing to production PyPI
    - This means the workflow will pause after TestPyPI succeeds and wait for your manual approval

### 2.2 Publishing a release

Once the above setup is done, publishing is a two-step process:

```bash
# 1. Bump the version in pyproject.toml
#    Edit: version = "0.1.0" → version = "0.2.0" (or whatever)

# 2. Commit the version bump
git add pyproject.toml
git commit -m "Release v0.2.0"

# 3. Create and push a tag
git tag v0.2.0
git push origin main --tags
```

This triggers the `publish.yml` workflow:

1. **Build** — `poetry build` creates `dist/polymathera_colony-0.2.0.tar.gz` and `dist/polymathera_colony-0.2.0-py3-none-any.whl`
2. **TestPyPI** — uploads to [test.pypi.org](https://test.pypi.org/project/polymathera-colony/). Verify at `https://test.pypi.org/project/polymathera-colony/0.2.0/`
3. **PyPI** — if TestPyPI succeeded (and reviewer approved, if configured), uploads to [pypi.org](https://pypi.org/project/polymathera-colony/)

### 2.3 Verifying a release

After publishing, test the install from both indices:

```bash
# From TestPyPI (may need --extra-index-url for dependencies)
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ polymathera-colony

# From production PyPI
pip install polymathera-colony
```

### 2.4 Troubleshooting

| Problem | Solution |
|---------|----------|
| "The trusted publisher doesn't match" | Check that the workflow filename, environment name, owner, and repo match exactly between PyPI config and your workflow |
| "File already exists" | You can't overwrite a version on PyPI. Bump the version number. For test iterations, use pre-release versions: `0.2.0a1`, `0.2.0a2`, etc. |
| Build fails | Run `poetry build` locally to debug. Check that `packages = [{include = "polymathera", from = "src"}]` is correct |
| TestPyPI succeeds but PyPI fails | The `pypi` environment likely needs reviewer approval. Check the Actions tab for a pending deployment |

---

## 3. GitHub Pages Documentation

### 3.1 One-time setup

1. Go to your repo on GitHub → **Settings** → **Pages**
2. Under "Build and deployment", set **Source** to **GitHub Actions**
3. That's it — the `docs.yml` workflow handles the rest

### 3.2 How it works

Every push to `main` that changes files in `docs/`, `mkdocs.yml`, or `src/` triggers a docs rebuild and deployment. The site is published at:

```
https://polymathera.github.io/colony/
```

### 3.3 Manual deployment

If you need to deploy docs without pushing to main:

1. Go to **Actions** → **Deploy Docs** → **Run workflow**
2. Select the `main` branch
3. Click "Run workflow"

(To enable manual trigger, add `workflow_dispatch:` under the `on:` key in `docs.yml`.)

---

## 4. Version Management

Colony follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking API changes
- **MINOR** (0.2.0): New features, backward-compatible
- **PATCH** (0.1.1): Bug fixes

For pre-release versions during development:

```toml
# In pyproject.toml
version = "0.2.0a1"   # Alpha
version = "0.2.0b1"   # Beta
version = "0.2.0rc1"  # Release candidate
```

---

## 5. Complete release checklist

- [ ] All CI checks pass on `main`
- [ ] Update version in `pyproject.toml`
- [ ] Update CHANGELOG (if you maintain one)
- [ ] Commit: `git commit -m "Release vX.Y.Z"`
- [ ] Tag: `git tag vX.Y.Z`
- [ ] Push: `git push origin main --tags`
- [ ] Verify on TestPyPI: `https://test.pypi.org/project/polymathera-colony/`
- [ ] Approve PyPI deployment (if reviewer required)
- [ ] Verify on PyPI: `https://pypi.org/project/polymathera-colony/`
- [ ] Test install: `pip install polymathera-colony==X.Y.Z`
- [ ] Verify docs updated: `https://polymathera.github.io/colony/`
