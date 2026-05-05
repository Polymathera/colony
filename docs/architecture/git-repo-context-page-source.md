# `GitRepoContextPageSource`

Pages a git repository (or a subtree of one) into VCM. Backing
implementation is `GitRepoShardingStrategy` + `FileGrouperWithGraph`,
both of which the source composes; the source itself is responsible
for cloning, sub-tree restriction, ignore-file handling, and watcher
composition.

## Minimal example

```python
from polymathera.colony.samples.paging import GitRepoContextPageSource
from polymathera.colony.vcm.models import MmapConfig

source = GitRepoContextPageSource(
    scope_id="my-repo",
    mmap_config=MmapConfig(),
    origin_url="https://github.com/example/code.git",
    branch="main",
    commit="HEAD",
)
await source.initialize()
```

That's the equivalent of the historical default — the whole tree, no
filtering, only `GitRemoteWatcher` (no `LocalFsWatcher`).

## Map a single subtree, skip the build directory

```python
GitRepoContextPageSource(
    scope_id="my-repo:src",
    mmap_config=MmapConfig(),
    origin_url="https://github.com/example/code.git",
    start_dir="src/",
    exclude_globs=["**/build/**", "**/__pycache__/**"],
)
```

Patterns are gitignore-style (`pathspec` with `GitWildMatchPattern`).
A `.colonyignore` (or `.gitignore`) at the sub-tree root augments
`exclude_globs` automatically.

## Frozen-commit context (no watcher)

```python
GitRepoContextPageSource(
    scope_id="external-foo",
    mmap_config=MmapConfig(),
    origin_url="https://github.com/upstream/foo.git",
    commit="<sha>",
    static=True,
    watch_remote=False,
)
```

`static=True` tells the convergence runtime that the page graph is
frozen at this commit; `watch_remote=False` skips the `GitRemoteWatcher`.

## Reference

| Argument | Default | Effect |
|---|---|---|
| `scope_id` | required | VCM scope identifier; `mmap_application_scope` keys on this. |
| `mmap_config` | required | Memory-mapped page-graph config (flush thresholds, locality policy). |
| `origin_url` | required | `https://`, `file://`, or any URL `git clone` accepts. |
| `branch` | `"main"` | Branch tracked by the watcher. |
| `commit` | `"HEAD"` | Commit pinned at clone time; combine with `static=True` for a frozen snapshot. |
| `start_dir` | `None` | Repo-relative directory to walk; `None` walks the repo root. |
| `include_globs` | `None` | Gitignore-style include patterns; `None` includes everything not excluded. |
| `exclude_globs` | `None` | Gitignore-style exclude patterns. |
| `ignore_files` | `(".gitignore", ".colonyignore")` | Filenames inside the repo whose patterns are merged into the exclude set. Pass `()` to disable. |
| `binary_policy` | `"skip"` | `"skip"` drops blobs whose first 8 KB contain a NUL byte; `"include"` keeps them (the literature source uses this). |
| `watch_remote` | `True` | Subscribe to `GitRemoteWatcher` events. |
| `watch_local` | `False` | Subscribe to `LocalFsWatcher` events. Off by default — the VCM mapping is the global read-only view of `branch`. |
| `static` | `False` | `True` produces a frozen-commit instance — no events, no watchers. |

## Failure modes

- **Subtree empty**: when `start_dir` resolves to an empty directory
  the source yields zero pages and logs a warning. Subsequent reads
  return empty mappings.
- **Binary files in a code subtree**: with the default `binary_policy="skip"`,
  PDFs and images are silently dropped. Use `LiteratureContextPageSource`
  for those.
- **Replica without a working tree**: when `git clone` fails on a
  replica, `_repo_path` stays `None` and `watch()` returns
  immediately. The convergence runtime handles the empty iterator
  cleanly.
