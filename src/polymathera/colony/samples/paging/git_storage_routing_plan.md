# Plan: Route Codebase Through GitStorage Instead of Raw Local Path

## Context

`FileGrouperContextPageSource` currently receives a raw `repo_path` (e.g., `/mnt/shared/codebase`) and opens the git repo directly. The `FileGrouper` normalizes file paths via `polymathera.normalize_file_path()` and `denormalize_file_path()`, which assume paths are under the `GitStorage` prefix. For paths outside `GitStorage`, `denormalize(normalize(path))` is not idempotent — it produces mangled paths like `/mnt/shared/filesystem/<hash>/mnt/shared/codebase/foo.py`, causing zero blob matches in `_get_group_blobs` and zero shards.

The fix: clone the repo through `GitFileStorage.clone_or_retrieve_repository()` so it lands under the correct prefix. Pass `origin_url`/`branch`/`commit` through the chain instead of `repo_path`.

**Secondary fix:** `FileGraphCache` only supports JSON serialization but inherits the default `CacheConfig` which uses `pickle`, causing `Unsupported serialization format: pickle` errors.

## Changes

### 1. `colony/python/colony/cli/polymath.py`

**CLI options:** Replace the `codebase_path` positional arg with three options. `--origin-url` and `--local-repo` are mutually exclusive — exactly one must be provided:

```python
origin_url: Optional[str] = typer.Option(
    None,
    "--origin-url",
    help="Git repository URL (HTTPS) for the codebase to analyze.",
),
local_repo: Optional[str] = typer.Option(
    None,
    "--local-repo",
    help="Path to a local git repository. Equivalent to --origin-url file://<path>.",
),
branch: str = typer.Option(
    "main",
    "--branch",
    help="Git branch to check out.",
),
commit: str = typer.Option(
    "HEAD",
    "--commit",
    help="Git commit SHA to check out (defaults to branch HEAD).",
),
```

Validation: exactly one of `--origin-url` or `--local-repo` must be provided. If `--local-repo` is given, set `origin_url = f"file://{local_repo}"`.

**`TestConfig`:** Add `origin_url`, `branch`, `commit` fields (settable via YAML or CLI):

```python
@dataclass
class TestConfig:
    origin_url: str = ""       # Required: git repo URL (https:// or file://)
    branch: str = "main"
    commit: str = "HEAD"
    # ... existing fields ...
```

**`run_integration_test()`:** Remove `codebase_path` parameter. Pass `origin_url`/`branch`/`commit` to `mmap_application_scope`:

```python
mmap_result = await vcm_handle.mmap_application_scope(
    scope_id=config.repo_id,
    group_id=config.repo_id,
    tenant_id=config.tenant_id,
    source_type=BuilInContextPageSourceType.FILE_GROUPER.value,
    config=mmap_config,
    origin_url=config.origin_url,
    branch=config.branch,
    commit=config.commit,
)
```

### 2. `colony/python/colony/cli/deploy/providers/compose.py`

**`run()` method:** Extract git info from the local repo and pass as CLI args to `polymath run`:

```python
import git as gitpython

codebase = Path(codebase_path).resolve()
repo = gitpython.Repo(codebase)

# Extract git remote URL; use --local-repo for local-only repos
try:
    origin_url = repo.remotes.origin.url
except (ValueError, IndexError):
    origin_url = None  # No remote — will use --local-repo

branch = "HEAD" if repo.head.is_detached else repo.active_branch.name
commit = repo.head.commit.hexsha

# Copy codebase to shared volume (needed for file:// URLs and as
# fallback when the remote is unreachable from Docker)
await self._exec("docker", "cp", f"{codebase}/.", f"{head}:/mnt/shared/codebase/")

cmd.extend(["python", "-m", "colony.cli.polymath", "run"])
if origin_url:
    cmd.extend(["--origin-url", origin_url])
else:
    cmd.extend(["--local-repo", "/mnt/shared/codebase"])
cmd.extend(["--branch", branch, "--commit", commit])
```

### 3. `colony/python/colony/samples/paging/file_grouper_page_source.py`

**Constructor:** Replace `repo_path: str` with `origin_url: str`, `branch: str`, `commit: str`.

**`_build_and_persist_page_graph()`:** Clone via `GitStorage` before opening the repo. The `origin_url` may be `https://...` or `file://...` — `GitStorage` handles both transparently:

```python
polymathera = get_polymathera()
storage = await polymathera.get_storage()

# Clone into GitStorage's managed directory so normalize/denormalize work.
# Call git_storage directly — the Storage wrapper's _validate_repo_url()
# uses HttpUrl() which rejects file:// URLs. Auth is a no-op stub.
repo_path = await storage.git_storage.clone_or_retrieve_repository(
    origin_url=self.origin_url,
    branch=self.branch,
    commit=self.commit,
    vmr_id=self.group_id,
)
repo = git.Repo(str(repo_path))
```

### 4. `colony/python/colony/samples/paging/sharding/file_grouping.py`

**`FileGraphCache.initialize()`:** Override serialization format to JSON (the only format `node_link_data()` supports):

```python
if self.config.serialization_format != "json":
    self.config = self.config.model_copy(update={"serialization_format": "json"})
```

### 5. `colony/python/colony/samples/paging/sharding/strategy.py`

**Revert `_get_group_blobs`** to its original form (path mismatch is fixed at the source now).

## Files Modified

| File | Change |
|------|--------|
| `colony/python/colony/cli/polymath.py` | Replace `codebase_path` with `--origin-url`/`--local-repo`/`--branch`/`--commit` |
| `colony/python/colony/cli/deploy/providers/compose.py` | Extract git info from local repo, pass as CLI args |
| `colony/python/colony/samples/paging/file_grouper_page_source.py` | Accept `origin_url`/`branch`/`commit`, clone via `GitStorage` |
| `colony/python/colony/samples/paging/sharding/file_grouping.py` | Force JSON serialization for `FileGraphCache` |
| `colony/python/colony/samples/paging/sharding/strategy.py` | Revert `_get_group_blobs` to original |

## Key Design Decisions

1. **`--origin-url` vs `--local-repo`:** Two mutually exclusive options. `--origin-url` takes an HTTPS git URL. `--local-repo` takes a local filesystem path and translates it to `file://<path>`. Both `--branch` and `--commit` work with either. Downstream code (`FileGrouperContextPageSource`, `GitStorage`) handles `https://` and `file://` URLs agnostically.

2. **Call `git_storage` directly** (not `Storage` wrapper): `Storage._validate_repo_url()` uses `HttpUrl()` which rejects `file://` URLs. `Auth` is a no-op stub anyway.

3. **`compose.py` still copies codebase:** The `docker cp` step is retained — it's needed for `file://` URLs (the repo must exist on the shared volume for `file://` to work from inside the container). If the local repo has a remote, `compose.py` passes `--origin-url` instead; the copy is still done as a fallback.

## Verification

1. Rebuild Docker image: `colony-env build`
2. Run integration test: `colony-env run /path/to/codebase --config test.yaml`
3. Check logs for:
   - `clone_or_retrieve_repository` completing successfully
   - Shards being created (non-zero count)
   - File paths under `GitStorage` prefix in `_get_group_blobs`
   - No `Unsupported serialization format: pickle` errors
