"""Tests for ``materialize_knowledge_sources`` and the per-source
filtering / paging-override logic in ``materialize_vcm_sources``.

The VCM round-trip is exercised by the CLI integration test; here we
only verify that the orchestrators (a) skip rows whose names are not
in ``enabled_sources`` and (b) layer per-source paging overrides
onto the base :class:`MmapConfig` before invoking the VCM handle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polymathera.colony.design_monorepo.materialize import (
    DesignContextMaterialisationReport,
    materialize_design_context_sources,
    materialize_knowledge_sources,
    materialize_vcm_sources,
)
from polymathera.colony.design_monorepo.repo_map import (
    AcquirerSpec,
    DesignContextSource,
    KnowledgeSource,
    RepoMap,
    VcmSource,
)
from polymathera.colony.knowledge.acquirers import (
    AcquiredSource,
    AcquirerRegistry,
    AcquirerStrategy,
)
from polymathera.colony.knowledge.deps import (
    get_default_ingestor,
    reset_knowledge_deps,
)
from polymathera.colony.vcm.models import MmapConfig


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Each test gets a fresh in-memory ``Ingestor`` so writes from
    one case do not leak into another."""

    reset_knowledge_deps()
    yield
    reset_knowledge_deps()


def _seed_corpus(repo_root: Path) -> None:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "literature" / "curated").mkdir(parents=True)
    (repo_root / "literature" / "promoted").mkdir(parents=True)
    (repo_root / "literature" / "curated" / "a.txt").write_text(
        "Curated paper A.\n", encoding="utf-8",
    )
    (repo_root / "literature" / "curated" / "b.txt").write_text(
        "Curated paper B.\n", encoding="utf-8",
    )
    (repo_root / "literature" / "promoted" / "c.txt").write_text(
        "Paper C.\n", encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_knowledge_sources_ingest_matching_files(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    _seed_corpus(repo_root)

    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        knowledge_sources=[
            KnowledgeSource(
                name="curated",
                paths=["literature/curated/**/*.txt"],
                profile="scientific_paper",
            ),
        ],
    )
    report = await materialize_knowledge_sources(
        repo_map=repo_map, repo_root=repo_root,
    )
    sources = sorted(r.source_uri for r in report.records)
    assert len(report.records) == 2
    assert all("literature/curated" in s for s in sources)
    assert report.acquisitions == ()


@pytest.mark.asyncio
async def test_enabled_sources_filters_knowledge_rows(tmp_path: Path) -> None:
    """``enabled_sources={"a"}`` skips row ``b`` even though both rows
    match files on disk."""

    repo_root = tmp_path / "r"
    _seed_corpus(repo_root)

    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        knowledge_sources=[
            KnowledgeSource(name="a", paths=["literature/curated/a.txt"]),
            KnowledgeSource(name="b", paths=["literature/curated/b.txt"]),
        ],
    )
    report = await materialize_knowledge_sources(
        repo_map=repo_map, repo_root=repo_root, enabled_sources={"a"},
    )
    sources = [r.source_uri for r in report.records]
    assert len(report.records) == 1
    assert any("a.txt" in s for s in sources)


def _stub_storage_chain(repo_root: Path):
    """Build the (polymathera → storage → git_storage) async stub chain
    that ``materialize_vcm_sources`` walks before reading the repo map.

    Returns a context manager that patches all four globals at once.
    """

    git_storage = MagicMock()
    git_storage.clone_or_retrieve_repository = AsyncMock(return_value=str(repo_root))
    storage = MagicMock(); storage.git_storage = git_storage
    polymathera = MagicMock()
    polymathera.get_storage = AsyncMock(return_value=storage)

    return (
        patch(
            "polymathera.colony.design_monorepo.materialize.get_polymathera",
            return_value=polymathera,
        ),
        patch(
            "polymathera.colony.design_monorepo.materialize.serving.get_colony_id",
            return_value="c-test",
        ),
    )


@pytest.mark.asyncio
async def test_enabled_sources_filters_rows_in_repo_map(tmp_path: Path) -> None:
    """``enabled_sources={"a", "c"}`` skips row ``b`` even though it
    sits between two enabled rows in source order."""

    repo_root = tmp_path / "r"
    (repo_root / ".colony").mkdir(parents=True)
    (repo_root / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - { name: a, type: git_repo }\n"
        "  - { name: b, type: git_repo }\n"
        "  - { name: c, type: git_repo }\n",
        encoding="utf-8",
    )

    vcm_handle = MagicMock()
    vcm_handle.mmap_application_scope = AsyncMock(return_value=MagicMock(status="mapped"))

    storage_patch, colony_patch = _stub_storage_chain(repo_root)
    with storage_patch, colony_patch:
        await materialize_vcm_sources(
            vcm_handle=vcm_handle,
            origin_url="https://x.test/r.git",
            branch="main", commit="HEAD",
            base_scope_id="repo",
            mmap_config=MmapConfig(),
            enabled_sources={"a", "c"},
        )

    called_names = [
        call.kwargs["scope_id"].split(":", 1)[1]
        for call in vcm_handle.mmap_application_scope.await_args_list
    ]
    assert called_names == ["a", "c"]


@pytest.mark.asyncio
async def test_per_source_paging_overrides_layer_onto_base_config(
    tmp_path: Path,
) -> None:
    """A row that sets ``flush_threshold`` overrides the base
    ``MmapConfig`` for that one row only; rows that don't override
    receive the base config unchanged. Pinned, token budget, and
    threshold layer independently — confirms ``model_copy(update=...)``
    semantics.
    """

    repo_root = tmp_path / "r"
    (repo_root / ".colony").mkdir(parents=True)
    (repo_root / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - { name: bare, type: git_repo }\n"
        "  - name: tuned\n"
        "    type: git_repo\n"
        "    flush_threshold: 99\n"
        "    pinned: true\n",
        encoding="utf-8",
    )

    vcm_handle = MagicMock()
    vcm_handle.mmap_application_scope = AsyncMock(return_value=MagicMock(status="mapped"))

    storage_patch, colony_patch = _stub_storage_chain(repo_root)
    base = MmapConfig(flush_threshold=20, flush_token_budget=4096, pinned=False)
    with storage_patch, colony_patch:
        await materialize_vcm_sources(
            vcm_handle=vcm_handle,
            origin_url="https://x.test/r.git",
            branch="main", commit="HEAD",
            base_scope_id="repo",
            mmap_config=base,
        )

    bare_call, tuned_call = vcm_handle.mmap_application_scope.await_args_list
    assert bare_call.kwargs["config"] is base  # unchanged → same object
    tuned_cfg = tuned_call.kwargs["config"]
    assert tuned_cfg is not base
    assert tuned_cfg.flush_threshold == 99
    assert tuned_cfg.flush_token_budget == 4096   # base preserved
    assert tuned_cfg.pinned is True


@pytest.mark.asyncio
async def test_empty_knowledge_sources_is_a_noop(tmp_path: Path) -> None:
    repo_root = tmp_path / "r"
    _seed_corpus(repo_root)

    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        knowledge_sources=[],
    )
    report = await materialize_knowledge_sources(
        repo_map=repo_map, repo_root=repo_root,
    )
    assert report.records == ()
    assert report.acquisitions == ()
    _ = get_default_ingestor  # silence unused-import in some linters


# ---- Acquirer-shaped rows + sidecar interaction ----------------------


class _FixturePathAcquirer(AcquirerStrategy):
    """Acquirer that writes a fixed payload into ``destination_dir``.

    Stands in for the eventual arXiv / DOI / HTTP acquirers — exercises
    the materialiser's acquire → ingest path without external network
    or a TODO stub's ``NotImplementedError``."""

    METHOD = "fixture_path"

    def __init__(self, payload: bytes, basename: str) -> None:
        self._payload = payload
        self._basename = basename
        self.calls = 0

    @property
    def method(self) -> str:
        return self.METHOD

    async def acquire(
        self, *, args, destination_dir,
    ) -> AcquiredSource:
        self.calls += 1
        target = destination_dir / self._basename
        target.write_bytes(self._payload)
        return AcquiredSource(
            local_path=target,
            cached=False,
            fetched_bytes=len(self._payload),
            metadata={"source_uri": args.get("source_uri", "")},
        )


def _registry_for_acquirer(strategy: AcquirerStrategy) -> AcquirerRegistry:
    registry = AcquirerRegistry()
    registry.register(strategy)
    return registry


@pytest.mark.asyncio
async def test_acquirer_row_writes_to_destination_and_ingests(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "r"
    repo_root.mkdir()
    strategy = _FixturePathAcquirer(
        payload=b"# Title\n\nHello.\n", basename="paper.md",
    )

    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        knowledge_sources=[
            KnowledgeSource(
                name="fixture",
                acquirer=AcquirerSpec(
                    method=_FixturePathAcquirer.METHOD,
                    args={"source_uri": "fixture:paper"},
                ),
                destination="kb/literature/",
            ),
        ],
    )
    report = await materialize_knowledge_sources(
        repo_map=repo_map,
        repo_root=repo_root,
        acquirer_registry=_registry_for_acquirer(strategy),
    )

    assert strategy.calls == 1
    assert (repo_root / "kb" / "literature" / "paper.md").is_file()
    assert len(report.records) == 1
    assert len(report.acquisitions) == 1
    assert report.acquisitions[0].outcome == "acquired"
    assert report.acquisitions[0].name == "fixture"
    assert report.records[0].source_uri == "fixture:paper"


@pytest.mark.asyncio
async def test_acquirer_unknown_method_lands_in_acquisitions(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "r"
    repo_root.mkdir()

    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        knowledge_sources=[
            KnowledgeSource(
                name="phantom",
                acquirer=AcquirerSpec(method="not_registered", args={}),
                destination="elsewhere/",
            ),
        ],
    )
    report = await materialize_knowledge_sources(
        repo_map=repo_map,
        repo_root=repo_root,
        acquirer_registry=AcquirerRegistry(),  # empty
    )
    assert report.records == ()
    assert len(report.acquisitions) == 1
    assert report.acquisitions[0].outcome == "unsupported_method"


@pytest.mark.asyncio
async def test_paths_walk_skips_sidecar_subtree(tmp_path: Path) -> None:
    """A previous ingestion's ``.ingested/`` outputs must NOT match a
    ``paths`` glob — otherwise extracted markdown would be re-ingested
    as primary content alongside the source it came from."""

    repo_root = tmp_path / "r"
    (repo_root / "lit").mkdir(parents=True)
    (repo_root / "lit" / "real.txt").write_text(
        "Real source content.\n", encoding="utf-8",
    )
    # Pre-seed a sidecar from a hypothetical previous run.
    sidecar = repo_root / "lit" / ".ingested" / "real"
    sidecar.mkdir(parents=True)
    (sidecar / "extracted.md").write_text(
        "Stale cached extraction.\n", encoding="utf-8",
    )

    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        knowledge_sources=[
            KnowledgeSource(
                name="all", paths=["lit/**/*"], profile="paper_section",
            ),
        ],
    )
    report = await materialize_knowledge_sources(
        repo_map=repo_map, repo_root=repo_root,
    )
    sources = [r.source_uri for r in report.records]
    assert len(report.records) == 1
    assert "real.txt" in sources[0]
    assert all(".ingested" not in s for s in sources)


# ---------------------------------------------------------------------------
# materialize_design_context_sources — Phase 1, path 2 (VCM mapping)
# ---------------------------------------------------------------------------


def _make_repo_root_with_docs(tmp_path: Path) -> Path:
    """Build a tiny on-disk fixture with markdown the materialiser
    can count via ``_iter_design_context_files``."""

    repo_root = tmp_path / "r"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "docs" / "objectives.md").write_text(
        "# Objectives\n", encoding="utf-8",
    )
    (repo_root / "docs" / "constraints.md").write_text(
        "# Constraints\n", encoding="utf-8",
    )
    (repo_root / "src" / "code").mkdir(parents=True)
    (repo_root / "src" / "code" / "foo.py").write_text(
        "# not design context\n", encoding="utf-8",
    )
    return repo_root


@pytest.mark.asyncio
async def test_design_context_empty_block_is_noop(tmp_path: Path) -> None:
    """A RepoMap with no ``design_context_sources`` rows produces an
    empty report and never calls the VCM handle."""

    repo_root = _make_repo_root_with_docs(tmp_path)
    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
    )
    vcm_handle = MagicMock()
    vcm_handle.mmap_application_scope = AsyncMock()

    report = await materialize_design_context_sources(
        vcm_handle=vcm_handle,
        repo_map=repo_map,
        repo_root=repo_root,
        base_scope_id="dm:test",
        origin_url="https://x.test/r.git",
        branch="main",
        commit="abc123",
        mmap_config=MmapConfig(),
    )
    assert isinstance(report, DesignContextMaterialisationReport)
    assert report.rows == ()
    vcm_handle.mmap_application_scope.assert_not_awaited()


@pytest.mark.asyncio
async def test_design_context_maps_each_row_as_literature(
    tmp_path: Path,
) -> None:
    """Per row: one ``mmap_application_scope`` call with
    ``source_type='literature'`` (prose chunker) and the row's globs
    as ``include_globs``."""

    repo_root = _make_repo_root_with_docs(tmp_path)
    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        design_context_sources=[
            DesignContextSource(name="docs", paths=["docs/**/*.md"]),
            DesignContextSource(name="codedocs", paths=["src/**/*.md"]),
        ],
    )
    vcm_handle = MagicMock()
    vcm_handle.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )

    report = await materialize_design_context_sources(
        vcm_handle=vcm_handle,
        repo_map=repo_map,
        repo_root=repo_root,
        base_scope_id="dm:test",
        origin_url="https://x.test/r.git",
        branch="main",
        commit="abc123",
        mmap_config=MmapConfig(),
        # No Kuzu in this test — focus is purely on the VCM path.
        include_kuzu=False,
    )

    calls = vcm_handle.mmap_application_scope.await_args_list
    assert len(calls) == 2
    # Both rows mapped as literature with the right glob list.
    for call, expected_name, expected_globs in zip(
        calls,
        ["docs", "codedocs"],
        [["docs/**/*.md"], ["src/**/*.md"]],
        strict=True,
    ):
        kw = call.kwargs
        assert kw["source_type"] == "literature"
        assert kw["scope_id"] == f"dm:test:design_context.{expected_name}"
        assert kw["origin_url"] == "https://x.test/r.git"
        assert kw["branch"] == "main"
        assert kw["commit"] == "abc123"
        assert kw["include_globs"] == expected_globs

    # Report mirrors the calls: one vcm row + one kuzu row (skipped) per source.
    vcm_rows = list(report.vcm_rows)
    assert [r.source_name for r in vcm_rows] == ["docs", "codedocs"]
    docs_row = vcm_rows[0]
    assert docs_row.status == "mapped"
    assert docs_row.pinned is False
    assert docs_row.num_files == 2  # 2 .md files under docs/
    assert report.mapped_count == 2
    assert report.pinned_count == 0
    assert report.failed_count == 0
    # Kuzu rows are still emitted (with status='skipped') even when
    # include_kuzu=False, so the caller's blackboard emission stays
    # symmetric across invocations.
    kuzu_rows = list(report.kuzu_rows)
    assert [r.source_name for r in kuzu_rows] == ["docs", "codedocs"]
    assert all(r.status == "skipped" for r in kuzu_rows)
    assert all(r.num_claims == 0 for r in kuzu_rows)
    assert report.total_claims_extracted == 0


@pytest.mark.asyncio
async def test_design_context_enabled_sources_filter(tmp_path: Path) -> None:
    """``enabled_sources`` restricts which rows materialise."""

    repo_root = _make_repo_root_with_docs(tmp_path)
    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        design_context_sources=[
            DesignContextSource(name="a", paths=["docs/**/*.md"]),
            DesignContextSource(name="b", paths=["docs/**/*.md"]),
            DesignContextSource(name="c", paths=["docs/**/*.md"]),
        ],
    )
    vcm_handle = MagicMock()
    vcm_handle.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )

    report = await materialize_design_context_sources(
        vcm_handle=vcm_handle,
        repo_map=repo_map,
        repo_root=repo_root,
        base_scope_id="dm",
        origin_url="u",
        branch="main",
        commit="c",
        mmap_config=MmapConfig(),
        enabled_sources={"a", "c"},
        include_kuzu=False,  # focus is the filter, not path-1
    )

    vcm_names = [r.source_name for r in report.vcm_rows]
    assert vcm_names == ["a", "c"]
    assert vcm_handle.mmap_application_scope.await_count == 2


@pytest.mark.asyncio
async def test_design_context_pin_calls_lock_and_registers_renewer(
    tmp_path: Path,
) -> None:
    """``pin_in_vcm=True`` triggers ``get_pages_for_scope`` +
    ``lock_page`` per page and registers with the renewer."""

    repo_root = _make_repo_root_with_docs(tmp_path)
    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        design_context_sources=[
            DesignContextSource(
                name="pinned",
                paths=["docs/**/*.md"],
                pin_in_vcm=True,
                pin_lock_duration_days=3,
            ),
        ],
    )
    vcm_handle = MagicMock()
    vcm_handle.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )
    vcm_handle.get_pages_for_scope = AsyncMock(
        return_value=[
            {"page_id": "p1", "size": 100, "group_id": "g"},
            {"page_id": "p2", "size": 100, "group_id": "g"},
        ],
    )
    vcm_handle.lock_page = AsyncMock()
    renewer = MagicMock()
    renewer.register = AsyncMock()

    report = await materialize_design_context_sources(
        vcm_handle=vcm_handle,
        repo_map=repo_map,
        repo_root=repo_root,
        base_scope_id="dm",
        origin_url="u", branch="main", commit="c",
        mmap_config=MmapConfig(),
        renewer=renewer,
        include_kuzu=False,  # focus is pin behaviour, not path-1
    )

    # Two pages → two lock_page calls with the right locked_by tag
    # and a 3-day duration in seconds.
    assert vcm_handle.lock_page.await_count == 2
    expected_duration_s = 3 * 86400.0
    for call, expected_page_id in zip(
        vcm_handle.lock_page.await_args_list, ["p1", "p2"], strict=True,
    ):
        kw = call.kwargs
        assert kw["page_id"] == expected_page_id
        assert kw["locked_by"] == "design_context.pinned"
        assert kw["lock_duration_s"] == expected_duration_s
        assert "pinned" in kw["reason"]

    # Renewer registration with matching duration so refreshes
    # outrun expiry.
    renewer.register.assert_awaited_once()
    reg = renewer.register.await_args
    assert reg.kwargs["source_name"] == "pinned"
    assert reg.kwargs["scope_id"] == "dm:design_context.pinned"
    assert reg.kwargs["lock_duration_s"] == expected_duration_s

    # Outcome flags the pin (vcm row only — kuzu was opted-out).
    vcm_rows = list(report.vcm_rows)
    assert len(vcm_rows) == 1
    assert vcm_rows[0].pinned is True
    assert report.pinned_count == 1


@pytest.mark.asyncio
async def test_design_context_pin_with_no_renewer_logs_and_skips_pin(
    tmp_path: Path,
) -> None:
    """If pin_in_vcm=True but no renewer was passed, mmap still
    happens — the pin step is skipped + flagged in the row's error
    so the caller (and the planner) can see the misconfiguration."""

    repo_root = _make_repo_root_with_docs(tmp_path)
    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        design_context_sources=[
            DesignContextSource(
                name="pinned", paths=["docs/**/*.md"], pin_in_vcm=True,
            ),
        ],
    )
    vcm_handle = MagicMock()
    vcm_handle.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )
    vcm_handle.lock_page = AsyncMock()

    report = await materialize_design_context_sources(
        vcm_handle=vcm_handle,
        repo_map=repo_map,
        repo_root=repo_root,
        base_scope_id="dm",
        origin_url="u", branch="main", commit="c",
        mmap_config=MmapConfig(),
        renewer=None,
        include_kuzu=False,
    )

    # mmap ran, but lock_page never did.
    vcm_handle.mmap_application_scope.assert_awaited_once()
    vcm_handle.lock_page.assert_not_awaited()
    vcm_row = report.vcm_rows[0]
    assert vcm_row.pinned is False
    assert "no renewer provided" in vcm_row.error


@pytest.mark.asyncio
async def test_design_context_mmap_failure_on_one_row_does_not_block_others(
    tmp_path: Path,
) -> None:
    """First row raises; subsequent row still maps."""

    repo_root = _make_repo_root_with_docs(tmp_path)
    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        design_context_sources=[
            DesignContextSource(name="broken", paths=["docs/**/*.md"]),
            DesignContextSource(name="ok", paths=["docs/**/*.md"]),
        ],
    )
    vcm_handle = MagicMock()

    async def _mmap_side_effect(**kw):
        if kw["scope_id"].endswith("broken"):
            raise RuntimeError("simulated VCM failure")
        return MagicMock(status="mapped")

    vcm_handle.mmap_application_scope = AsyncMock(side_effect=_mmap_side_effect)

    report = await materialize_design_context_sources(
        vcm_handle=vcm_handle,
        repo_map=repo_map,
        repo_root=repo_root,
        base_scope_id="dm",
        origin_url="u", branch="main", commit="c",
        mmap_config=MmapConfig(),
        include_kuzu=False,
    )

    statuses = [(r.source_name, r.status) for r in report.vcm_rows]
    assert statuses == [("broken", "error"), ("ok", "mapped")]
    assert "simulated VCM failure" in report.vcm_rows[0].error
    assert report.mapped_count == 1
    assert report.failed_count == 1


# ---------------------------------------------------------------------------
# Path-1 (Kuzu KG) materialisation — Phase P3a
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_design_context_kuzu_path_ingests_each_file_with_design_context_uri(
    tmp_path: Path,
) -> None:
    """Each matching file is fed to ``Ingestor.ingest_file`` with the
    canonical ``design_context://{source_name}/{rel_path}`` URI
    scheme so downstream KG queries can filter on it."""

    from polymathera.colony.knowledge.models import (
        CorpusTier, IngestionRecord, IngestionStatus, KnowledgeFormat,
    )

    repo_root = _make_repo_root_with_docs(tmp_path)
    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        design_context_sources=[
            DesignContextSource(name="docs", paths=["docs/**/*.md"]),
        ],
    )
    vcm_handle = MagicMock()
    vcm_handle.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )

    captured: list[tuple[str, CorpusTier]] = []

    async def _ingest(path, *, tier, source_uri, **_kw):
        captured.append((source_uri, tier))
        return IngestionRecord(
            source_uri=source_uri,
            detected_format=KnowledgeFormat.MARKDOWN,
            tier=tier,
            status=IngestionStatus.COMPLETED,
            chunks_produced=1,
            claims_extracted=2,
            document_hash="sha",
        )

    fake_ingestor = MagicMock()
    fake_ingestor.ingest_file = AsyncMock(side_effect=_ingest)

    report = await materialize_design_context_sources(
        vcm_handle=vcm_handle,
        repo_map=repo_map,
        repo_root=repo_root,
        base_scope_id="dm",
        origin_url="u", branch="main", commit="c",
        mmap_config=MmapConfig(),
        include_kuzu=True,
        ingestor=fake_ingestor,
    )

    # Both files ingested with the canonical scheme + tier_1.
    assert fake_ingestor.ingest_file.await_count == 2
    uris = sorted(uri for uri, _ in captured)
    assert uris == [
        "design_context://docs/docs/constraints.md",
        "design_context://docs/docs/objectives.md",
    ]
    assert all(tier == CorpusTier.TIER_1_FOUNDATIONS for _, tier in captured)

    kuzu_rows = list(report.kuzu_rows)
    assert len(kuzu_rows) == 1
    row = kuzu_rows[0]
    assert row.source_name == "docs"
    assert row.status == "completed"
    assert row.num_files == 2
    assert row.num_claims == 4  # 2 files × 2 claims
    assert report.ingested_count == 1
    assert report.total_claims_extracted == 4


@pytest.mark.asyncio
async def test_design_context_kuzu_skipped_when_include_kuzu_false_emits_row(
    tmp_path: Path,
) -> None:
    """``include_kuzu=False`` still emits one kuzu row per source
    with ``status='skipped'`` (so the caller's blackboard emission
    stays symmetric across invocations) and never touches the
    ingestor."""

    repo_root = _make_repo_root_with_docs(tmp_path)
    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        design_context_sources=[
            DesignContextSource(name="docs", paths=["docs/**/*.md"]),
        ],
    )
    vcm_handle = MagicMock()
    vcm_handle.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )
    fake_ingestor = MagicMock()
    fake_ingestor.ingest_file = AsyncMock()

    report = await materialize_design_context_sources(
        vcm_handle=vcm_handle,
        repo_map=repo_map,
        repo_root=repo_root,
        base_scope_id="dm",
        origin_url="u", branch="main", commit="c",
        mmap_config=MmapConfig(),
        include_kuzu=False,
        ingestor=fake_ingestor,
    )

    fake_ingestor.ingest_file.assert_not_awaited()
    kuzu_rows = list(report.kuzu_rows)
    assert len(kuzu_rows) == 1
    assert kuzu_rows[0].status == "skipped"
    assert kuzu_rows[0].num_claims == 0
    assert report.ingested_count == 0


@pytest.mark.asyncio
async def test_design_context_kuzu_path_partial_on_per_file_exception(
    tmp_path: Path,
) -> None:
    """Per-file ingest exceptions degrade the row to 'partial' (when
    some files succeed) or 'error' (when all fail), with the failing
    files listed in the row's error string."""

    from polymathera.colony.knowledge.models import (
        IngestionRecord, IngestionStatus, KnowledgeFormat,
    )

    repo_root = tmp_path / "r"
    (repo_root / "docs").mkdir(parents=True)
    (repo_root / "docs" / "a.md").write_text("# a\n", encoding="utf-8")
    (repo_root / "docs" / "b.md").write_text("# b\n", encoding="utf-8")
    (repo_root / "docs" / "c.md").write_text("# c\n", encoding="utf-8")
    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        design_context_sources=[
            DesignContextSource(name="docs", paths=["docs/**/*.md"]),
        ],
    )

    vcm_handle = MagicMock()
    vcm_handle.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )

    async def _ingest(path, *, tier, source_uri, **_kw):
        if str(path).endswith("b.md"):
            raise RuntimeError("simulated ingest failure")
        return IngestionRecord(
            source_uri=source_uri,
            detected_format=KnowledgeFormat.MARKDOWN,
            tier=tier,
            status=IngestionStatus.COMPLETED,
            chunks_produced=1,
            claims_extracted=1,
            document_hash="sha",
        )

    fake_ingestor = MagicMock()
    fake_ingestor.ingest_file = AsyncMock(side_effect=_ingest)

    report = await materialize_design_context_sources(
        vcm_handle=vcm_handle,
        repo_map=repo_map,
        repo_root=repo_root,
        base_scope_id="dm",
        origin_url="u", branch="main", commit="c",
        mmap_config=MmapConfig(),
        include_kuzu=True,
        ingestor=fake_ingestor,
    )

    kuzu = report.kuzu_rows[0]
    assert kuzu.status == "partial"
    assert kuzu.num_files == 3
    assert kuzu.num_claims == 2  # a + c
    assert "b.md" in kuzu.error
    assert "simulated ingest failure" in kuzu.error


@pytest.mark.asyncio
async def test_design_context_kuzu_path_handles_zero_matching_files(
    tmp_path: Path,
) -> None:
    """An empty glob → 0 files → kuzu row with status='completed' +
    num_files=0 (technically nothing failed; the operator just
    declared an empty corpus)."""

    repo_root = tmp_path / "r"
    repo_root.mkdir()
    (repo_root / ".init").write_text("init", encoding="utf-8")
    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        design_context_sources=[
            DesignContextSource(name="empty", paths=["nonexistent/*.md"]),
        ],
    )
    vcm_handle = MagicMock()
    vcm_handle.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )
    fake_ingestor = MagicMock()
    fake_ingestor.ingest_file = AsyncMock()

    report = await materialize_design_context_sources(
        vcm_handle=vcm_handle,
        repo_map=repo_map,
        repo_root=repo_root,
        base_scope_id="dm",
        origin_url="u", branch="main", commit="c",
        mmap_config=MmapConfig(),
        include_kuzu=True,
        ingestor=fake_ingestor,
    )

    fake_ingestor.ingest_file.assert_not_awaited()
    kuzu = report.kuzu_rows[0]
    assert kuzu.status == "completed"
    assert kuzu.num_files == 0
    assert kuzu.num_claims == 0


# ---------------------------------------------------------------------------
# Fix B — progress_callback per-source emission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_progress_callback_emits_start_and_per_source(
    tmp_path: Path,
) -> None:
    """Fix B regression pin: when ``progress_callback`` is provided,
    the materializer emits one ``stage='start'`` message before the
    fan-out and one ``stage='source_done'`` message per source as it
    completes — completion order, not insertion order. Closes the
    chat-silence gap observed in run7 (16 min between 'loading
    design context' and the next status update)."""

    repo_root = _make_repo_root_with_docs(tmp_path)
    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        design_context_sources=[
            DesignContextSource(name="docs", paths=["docs/**/*.md"]),
            DesignContextSource(name="codedocs", paths=["src/**/*.md"]),
        ],
    )
    vcm_handle = MagicMock()
    vcm_handle.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )
    emissions: list[tuple[str, dict[str, Any]]] = []

    async def _callback(message: str, details: dict[str, Any]) -> None:
        emissions.append((message, details))

    await materialize_design_context_sources(
        vcm_handle=vcm_handle,
        repo_map=repo_map,
        repo_root=repo_root,
        base_scope_id="dm",
        origin_url="u", branch="main", commit="c",
        mmap_config=MmapConfig(),
        include_kuzu=False,
        progress_callback=_callback,
    )

    # 1 start + 1 per source.
    assert len(emissions) == 3
    start_msg, start_details = emissions[0]
    assert start_details["stage"] == "start"
    assert start_details["total"] == 2
    assert start_details["completed"] == 0
    # Per-source emissions advance the completed counter and name
    # the source.
    rest = emissions[1:]
    completed_counters = [d["completed"] for _, d in rest]
    assert completed_counters == [1, 2]
    source_names = {d["source_name"] for _, d in rest}
    assert source_names == {"docs", "codedocs"}
    for msg, details in rest:
        assert details["stage"] == "source_done"
        assert details["vcm_status"] in ("mapped", "already_mapped")


@pytest.mark.asyncio
async def test_progress_callback_failure_does_not_abort_ingest(
    tmp_path: Path,
) -> None:
    """Recovery contract: a misbehaving progress_callback raises in
    the per-source path, but the materializer still completes the
    ingest and returns a normal report. Progress is best-effort."""

    repo_root = _make_repo_root_with_docs(tmp_path)
    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        design_context_sources=[
            DesignContextSource(name="docs", paths=["docs/**/*.md"]),
        ],
    )
    vcm_handle = MagicMock()
    vcm_handle.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )

    async def _exploding_cb(message: str, details: dict[str, Any]) -> None:
        raise RuntimeError("synthetic UI relay failure")

    report = await materialize_design_context_sources(
        vcm_handle=vcm_handle,
        repo_map=repo_map,
        repo_root=repo_root,
        base_scope_id="dm",
        origin_url="u", branch="main", commit="c",
        mmap_config=MmapConfig(),
        include_kuzu=False,
        progress_callback=_exploding_cb,
    )

    # Report still good — the ingest itself was not poisoned.
    assert report.mapped_count == 1
    assert report.failed_count == 0


@pytest.mark.asyncio
async def test_progress_callback_none_preserves_legacy_path(
    tmp_path: Path,
) -> None:
    """When ``progress_callback`` is omitted (``None``), behaviour
    matches the pre-Fix-B path: no emissions, no per-source wrapper
    overhead beyond a single ``if is None`` check. Pins the
    backwards-compat contract for callers (tests, CLIs) that don't
    want streamed status."""

    repo_root = _make_repo_root_with_docs(tmp_path)
    repo_map = RepoMap(
        vcm_sources=[VcmSource(name="default", type="git_repo")],
        design_context_sources=[
            DesignContextSource(name="docs", paths=["docs/**/*.md"]),
        ],
    )
    vcm_handle = MagicMock()
    vcm_handle.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )

    # No progress_callback passed — should behave exactly as before.
    report = await materialize_design_context_sources(
        vcm_handle=vcm_handle,
        repo_map=repo_map,
        repo_root=repo_root,
        base_scope_id="dm",
        origin_url="u", branch="main", commit="c",
        mmap_config=MmapConfig(),
        include_kuzu=False,
    )

    assert report.mapped_count == 1
