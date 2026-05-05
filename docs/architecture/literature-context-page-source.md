# `LiteratureContextPageSource`

Pages PDFs and plain-text papers into VCM at *chunk* granularity:
each chunk produced by `ProseChunker` becomes one
`VirtualContextPage`, so cache-aware inference operates at the
right granularity for retrieval-heavy literature work.

The class reuses the same backends as the curator side of the
[knowledge trio](knowledge-capabilities.md) — `PdfReader` from
`knowledge.readers.pdf`, `ProseChunker` from `knowledge.chunking` —
which means a paper paged into VCM and the same paper curated into
the knowledge base carry identical chunk shapes.

## Minimal example

```python
from polymathera.colony.samples.paging import LiteratureContextPageSource
from polymathera.colony.vcm.models import MmapConfig

source = LiteratureContextPageSource(
    scope_id="design-literature",
    mmap_config=MmapConfig(),
    origin_url="https://github.com/example/design-monorepo.git",
    branch="main",
    # walk literature/, default include globs already cover *.pdf/*.txt/*.md
)
await source.initialize()
```

The defaults walk `literature/` for `*.pdf`, `*.txt`, `*.md`. Each
PDF is split into chunks of ~800 tokens with 80 tokens of overlap.

## Tune chunking

```python
LiteratureContextPageSource(
    scope_id="design-literature",
    mmap_config=MmapConfig(),
    origin_url=...,
    chunk_target_tokens=600,
    chunk_overlap_tokens=60,
)
```

Chunk parameters propagate to `ProseChunker(ChunkerConfig(...))`. The
chunker is paragraph-aware, so chunks rarely cut mid-sentence.

## Where the chunks live

Each chunk:

- stores its text in a `VirtualContextPage` keyed by `chunk.chunk_id`,
- carries `metadata.file = "<rel>"` so any page traces back to its
  source PDF,
- is added as a node in the source's page graph.

The `record_id` for the abstract `ContextPageSource` API is the
chunk id; `get_record_ids_for_page(page_id)` returns `[page_id]`
(one chunk per page).

## Reference

| Argument | Default | Effect |
|---|---|---|
| `scope_id` | required | VCM scope identifier. |
| `mmap_config` | required | Memory-mapped page-graph config. |
| `origin_url` | required | Git repo holding the literature directory. |
| `branch` | `"main"` | Branch tracked. |
| `commit` | `"HEAD"` | Pinned commit. |
| `start_dir` | `"literature"` | Repo-relative literature directory; pass `None` to walk root. |
| `include_globs` | `("**/*.pdf", "**/*.txt", "**/*.md")` | File patterns to include. |
| `exclude_globs` | `None` | File patterns to drop. |
| `ignore_files` | `(".gitignore", ".colonyignore")` | Ignore-file names merged into exclude set. |
| `chunk_target_tokens` | `800` | Target token budget per chunk. |
| `chunk_overlap_tokens` | `80` | Sliding-window overlap. |
| `watch_remote` | `True` | Subscribe to `GitRemoteWatcher` events. |
| `static` | `False` | `True` produces a frozen-commit literature snapshot. |

## Failure modes

- **Unsupported extension**: dropped silently. Only `.pdf`, `.txt`,
  `.md` are handled. To extend, subclass and override
  `_extract_chunks`.
- **Encrypted PDF**: `pypdf` raises during extraction; the source
  logs and skips the file.
- **Tiny `.txt` notes**: chunks below `ChunkerConfig.min_tokens` are
  dropped to avoid polluting the page graph with single-sentence
  pages.
