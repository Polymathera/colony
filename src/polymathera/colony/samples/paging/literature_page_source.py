"""Literature page source — pages PDFs / plain-text papers into VCM.

Sibling to :class:`GitRepoContextPageSource`. Both clone a git
repository through ``GitFileStorage``; the difference is that this
source treats *content* as literature (PDFs, ``.txt``, ``.md``) and
emits one VCM page per :class:`Chunk` (token-bounded retrieval-sized
chunk).

Reuse:

- :func:`polymathera.colony.samples.paging._walk.walk_repo` for tree
  traversal + include/exclude/binary filtering.
- :class:`polymathera.colony.knowledge.readers.PdfReader` /
  :class:`PlainTextReader` / :class:`MarkdownReader` for parsing.
- :class:`polymathera.colony.knowledge.chunking.ProseChunker` for
  chunking. The same chunker the knowledge ingestor uses, so a PDF
  paged into VCM and the same PDF curated into the KB carry the
  identical chunk-token shape.
- :class:`polymathera.colony.vcm.watchers.GitRemoteWatcher` for live
  updates. ``LocalFsWatcher`` is intentionally not composed: the VCM
  mapping is the global read-only view of ``branch``.

The source is non-static by default; per-instance ``static=True``
gives a frozen-commit literature snapshot.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
from overrides import override

from polymathera.colony.distributed import get_polymathera
from polymathera.colony.vcm.models import (
    ContextPageId,
    MmapConfig,
    VirtualContextPage,
)
from polymathera.colony.vcm.page_events import PageChangeEvent
from polymathera.colony.vcm.page_storage import PageStorage
from polymathera.colony.vcm.sources import (
    BuiltInContextPageSourceType,
    ContextPageSource,
    ContextPageSourceFactory,
)
from polymathera.colony.vcm.watchers import (
    CompositeWatcher,
    GitRemoteWatcher,
    GitRemoteWatcherConfig,
)

from ._walk import PathFilter, walk_repo


logger = logging.getLogger(__name__)


_DEFAULT_INCLUDE_GLOBS: tuple[str, ...] = (
    "**/*.pdf",
    "**/*.txt",
    "**/*.md",
)


@ContextPageSourceFactory.register_new_source_type(
    BuiltInContextPageSourceType.LITERATURE.value,
)
class LiteratureContextPageSource(ContextPageSource):
    """Pages a literature subdirectory of a git repo into VCM.

    Each parsed-section chunk becomes one :class:`VirtualContextPage`;
    the page id is the chunk id and ``record_id == chunk_id``. The
    ``page_to_file`` map (chunk_id → relative file path) lets the VCM
    + agent capabilities trace any page back to its source PDF /
    document.
    """

    def __init__(
        self,
        *,
        scope_id: str,
        mmap_config: MmapConfig,
        origin_url: str,
        branch: str = "main",
        commit: str = "HEAD",
        start_dir: str | None = "literature",
        include_globs: list[str] | None = None,
        exclude_globs: list[str] | None = None,
        ignore_files: tuple[str, ...] = (".gitignore", ".colonyignore"),
        chunk_target_tokens: int = 800,
        chunk_overlap_tokens: int = 80,
        watch_remote: bool = True,
        static: bool = False,
    ):
        """Initialize the literature page source.

        Args:
            scope_id: Scope identifier (typically ``"literature"`` or a
                colony-specific scope).
            mmap_config: Memory-mapped page graph config.
            origin_url: Git repo URL holding the literature.
            branch: Branch tracked.
            commit: Commit pinned at clone time.
            start_dir: Repo-relative directory containing literature.
                Defaults to ``"literature"``; pass ``None`` to walk the
                repo root.
            include_globs: Gitignore-style include patterns. Defaults to
                ``("**/*.pdf", "**/*.txt", "**/*.md")``.
            exclude_globs: Gitignore-style exclude patterns.
            ignore_files: Filenames whose patterns are merged into the
                exclude set when found inside the walked subtree.
            chunk_target_tokens: Target token budget per chunk (page).
            chunk_overlap_tokens: Sliding-window overlap between
                successive chunks.
            watch_remote: Subscribe to upstream-branch commits.
            static: ``False`` (default) for live-watched literature;
                ``True`` for a frozen-commit snapshot.
        """
        super().__init__(
            scope_id=scope_id, mmap_config=mmap_config, static=static,
        )
        self.origin_url = origin_url
        self.branch = branch
        self.commit = commit
        # binary_policy is forced to "include" — PDFs are binary by
        # construction. Operators do not get to override here because
        # the literature reader cannot produce sections from skipped
        # binaries; if you want PDFs gone, exclude them explicitly.
        self._path_filter = PathFilter(
            start_dir=start_dir,
            include_globs=tuple(include_globs) if include_globs is not None
                          else _DEFAULT_INCLUDE_GLOBS,
            exclude_globs=tuple(exclude_globs or ()),
            ignore_files=ignore_files,
            binary_policy="include",
        )
        self._chunk_target_tokens = chunk_target_tokens
        self._chunk_overlap_tokens = chunk_overlap_tokens
        self._watch_remote = watch_remote

        self.page_storage: PageStorage | None = None
        # record_id (== chunk_id) → page_id (also == chunk_id).
        self.chunk_to_page: dict[str, str] = {}
        # page_id → relative source file path (one source file per
        # page; multiple pages per file).
        self.page_to_file: dict[str, str] = {}
        # Reverse: file rel-path → chunk_ids, used by ``watch()`` to
        # invalidate the right pages on a remote file change.
        self._file_to_chunks: dict[str, list[str]] = defaultdict(list)

        self._repo_path: Path | None = None
        self._composite_watcher: CompositeWatcher | None = None

    @override
    async def initialize(self) -> None:
        """Clone the repo (idempotent), walk the literature subdir, and
        materialise one VCM page per chunk."""

        from polymathera.colony.vcm.page_storage import PageStorageConfig
        from polymathera.colony.system import get_vcm

        if self.page_storage is not None and self.chunk_to_page:
            return

        vcm_handle = await get_vcm()
        config: PageStorageConfig | None = await vcm_handle.get_page_storage_config()
        if not config:
            raise ValueError("Missing PageStorageConfig in VCM")
        self.page_storage = PageStorage(
            backend_type=config.backend_type,
            storage_path=config.storage_path,
            s3_bucket=config.s3_bucket,
        )
        await self.page_storage.initialize()

        # Reuse existing graph if persisted (replica restart fast-path).
        page_graph = await self.page_storage.retrieve_page_graph()
        if page_graph and page_graph.number_of_nodes() > 0:
            self.chunk_to_page = await self.page_storage.retrieve_page_graph_level_data(
                data_key="chunk_to_page",
            ) or {}
            self.page_to_file = await self.page_storage.retrieve_page_graph_level_data(
                data_key="page_to_file",
            ) or {}
            self._rebuild_file_index()
            self._repo_path = await self._resolve_repo_path()
            return

        self._repo_path = await self._resolve_repo_path()
        if self._repo_path is None:
            logger.warning(
                "LiteratureContextPageSource[%s]: working tree unavailable; "
                "no pages built on this replica.", self.scope_id,
            )
            return

        files = walk_repo(str(self._repo_path), self._path_filter)
        page_graph = nx.DiGraph()
        for abs_path in files:
            try:
                chunks = await self._extract_chunks(Path(abs_path))
            except Exception:  # noqa: BLE001
                logger.exception(
                    "LiteratureContextPageSource[%s]: failed to extract %s",
                    self.scope_id, abs_path,
                )
                continue
            rel_path = str(Path(abs_path).relative_to(self._repo_path))
            for chunk in chunks:
                page = VirtualContextPage(
                    page_id=chunk.chunk_id,
                    tokens=[],
                    text=chunk.text,
                    size=chunk.token_count or max(1, len(chunk.text) // 4),
                    metadata={
                        "source": LiteratureContextPageSource.get_source_metadata(
                            self.scope_id,
                        ),
                        "file": rel_path,
                        "section_path": chunk.section_path,
                        "data_type": chunk.data_type,
                    },
                    scope_id=self.scope_id,
                    group_id=None,
                )
                await self.page_storage.store_page(page)
                self.chunk_to_page[chunk.chunk_id] = chunk.chunk_id
                self.page_to_file[chunk.chunk_id] = rel_path
                self._file_to_chunks[rel_path].append(chunk.chunk_id)
                page_graph.add_node(chunk.chunk_id)

        await self.page_storage.store_page_graph(graph_data=page_graph)
        await self.page_storage.store_page_graph_level_data(
            data_key="chunk_to_page", graph_data=self.chunk_to_page,
        )
        await self.page_storage.store_page_graph_level_data(
            data_key="page_to_file", graph_data=self.page_to_file,
        )
        logger.info(
            "LiteratureContextPageSource[%s]: %d pages from %d files",
            self.scope_id, len(self.chunk_to_page), len(files),
        )

    async def _resolve_repo_path(self) -> Path | None:
        """Clone-or-retrieve the repo through ``GitFileStorage``. Returns
        ``None`` only when the storage layer raises (e.g., a replica
        without write access on this node)."""
        try:
            polymathera = get_polymathera()
            storage = await polymathera.get_storage()
            repo_path = await storage.git_storage.clone_or_retrieve_repository(
                origin_url=self.origin_url,
                branch=self.branch,
                commit=self.commit,
                vmr_id=self.syscontext.colony_id,
            )
            return Path(str(repo_path))
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "LiteratureContextPageSource[%s]: clone_or_retrieve "
                "failed (%s); pages will not be built on this replica.",
                self.scope_id, e,
            )
            return None

    async def _extract_chunks(self, abs_path: Path) -> Sequence:
        """Reader → ``ProseChunker`` pipeline for a single file. Returns
        an empty sequence for unsupported extensions."""

        from polymathera.colony.knowledge.chunking import ChunkerConfig, ProseChunker
        from polymathera.colony.knowledge.models import (
            CitationSpan,
            KnowledgeFormat,
            ParsedSection,
            RawDocument,
        )
        from polymathera.colony.knowledge.readers.pdf import PdfReader

        chunker = ProseChunker(
            ChunkerConfig(
                target_tokens=self._chunk_target_tokens,
                overlap_tokens=self._chunk_overlap_tokens,
            ),
        )

        suffix = abs_path.suffix.lower()
        sections: Sequence[ParsedSection]
        if suffix == ".pdf":
            doc = RawDocument(
                source_uri=f"file://{abs_path}",
                detected_format=KnowledgeFormat.PDF,
                payload=abs_path.read_bytes(),
            )
            sections = PdfReader().read(doc)
        elif suffix in {".txt", ".md"}:
            text = abs_path.read_text(encoding="utf-8", errors="replace")
            sections = (
                ParsedSection(
                    section_path="1",
                    heading=abs_path.name,
                    text=text,
                    citation=CitationSpan(
                        source_uri=f"file://{abs_path}",
                        section_path="1",
                        char_start=0,
                        char_end=len(text),
                    ),
                ),
            )
        else:
            return ()

        chunks: list = []
        for section in sections:
            chunks.extend(chunker.chunk(section))
        return chunks

    def _rebuild_file_index(self) -> None:
        self._file_to_chunks = defaultdict(list)
        for chunk_id, file_path in self.page_to_file.items():
            self._file_to_chunks[file_path].append(chunk_id)

    @override
    async def claim_orphaned_events(self) -> None:
        """No replica-state to recover — events come from polling
        ``GitRemoteWatcher`` and are idempotent against the page graph."""

    @override
    async def shutdown(self) -> None:
        if self._composite_watcher is not None:
            self._composite_watcher.stop()
            self._composite_watcher = None
        self.page_storage = None

    @override
    async def watch(self) -> AsyncIterator[PageChangeEvent]:
        """Yield events from :class:`GitRemoteWatcher`.

        Events are emitted at file-path granularity (the watcher's
        native shape, with ``extra["relative_path"]`` carrying the
        path). The convergence runtime + ``DesignMonorepoCapability``
        decide how to react — typically by invalidating every chunk
        whose ``page_to_file`` matches the changed path and
        re-extracting on the next read. Per-chunk re-extraction is
        intentionally NOT done in this method so the watcher stays a
        thin polling loop.
        """

        if self._repo_path is None or not self._watch_remote:
            return
        source_uri = f"git:{self.origin_url}:{self.branch}:{self.commit}"
        watcher = GitRemoteWatcher(
            repo_path=self._repo_path,
            scope_id=self.scope_id,
            source_uri=source_uri,
            config=GitRemoteWatcherConfig(
                branch=self.branch,
                data_type="literature_file",
            ),
        )
        composite = CompositeWatcher((watcher,), scope_id=self.scope_id)
        self._composite_watcher = composite
        try:
            async for event in composite.watch():
                yield event
        finally:
            self._composite_watcher = None

    @override
    async def get_page_id_for_record(self, record_id: str) -> ContextPageId | None:
        return self.chunk_to_page.get(record_id)

    @override
    async def get_record_ids_for_page(self, page_id: ContextPageId) -> list[str]:
        # 1 chunk per page; record_id == page_id when we know the page.
        return [page_id] if page_id in self.chunk_to_page else []

    @override
    async def get_all_mapped_records(self) -> dict[str, ContextPageId]:
        return dict(self.chunk_to_page)

    @override
    async def get_all_mapped_pages(self) -> dict[ContextPageId, list[str]]:
        return {pid: [pid] for pid in self.chunk_to_page}
