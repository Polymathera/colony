# Context Page Sources

A `ContextPageSource` is a mapping from an *application-level* record set
(files in a git repo, entries on a blackboard scope, chunks of a PDF) to
*VCM pages*. The VCM treats every page identically once mapped — the
source is what knows where the underlying data lives, how to chunk it,
and how to detect changes upstream.

## When to use which subclass

| Subclass | Backing store | Live updates | Typical use |
|---|---|---|---|
| [`GitRepoContextPageSource`](git-repo-context-page-source.md) | git repository (clone) | `GitRemoteWatcher` | code repos, documentation repos, frozen submodules |
| [`LiteratureContextPageSource`](literature-context-page-source.md) | git repo containing PDFs / `.md` / `.txt` | `GitRemoteWatcher` | papers, design literature, mixed-format corpora |
| `BlackboardContextPageSource` | `EnhancedBlackboard` scope | embedded event stream (no watcher) | live agent records (memories, decisions, hypotheses) |

Pick `GitRepoContextPageSource` for code; pick `LiteratureContextPageSource`
for binary documents that need text extraction; pick
`BlackboardContextPageSource` for content that mutates inside the colony
itself rather than being pushed by an external commit.

## Multi-source mapping for one repo

A single design monorepo will usually combine sources — code under
`tools/`, papers under `literature/`, frozen external submodules. The
[`.colony/repo_map.yaml`](repo-map.md) file declares one source per row
and the materialiser issues one VCM mapping per row, so all three of
the entries in the table above can coexist for the same repo.

## Common contract

Every source declares (in its constructor):

- `static: bool` — whether the backing store is frozen. The convergence
  runtime refuses to subscribe to a static source as a live input.
- `watch()` — async iterator yielding `PageChangeEvent` once the
  source has been initialised. Static sources leave the default
  (raises `NotImplementedError`); live sources override.

Capabilities and the convergence runtime read both fields through the
`ContextPageSource` ABC; they never depend on a specific subclass.

## Custom sources

Register your own subclass with the factory so the
[repo map](repo-map.md) can reference it by name:

```python
from polymathera.colony.vcm.sources import (
    ContextPageSource,
    ContextPageSourceFactory,
)

@ContextPageSourceFactory.register_new_source_type("my_source")
class MyContextPageSource(ContextPageSource):
    ...
```

The factory propagates the registered module path to Ray workers via
`POLYMATH_PAGE_SOURCE_MODULES`, so the same code path that imports
your driver also imports it on every replica.
