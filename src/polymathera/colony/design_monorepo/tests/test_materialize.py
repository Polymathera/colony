"""Tests for ``materialize_knowledge_sources`` and the per-source
filtering / paging-override logic in ``materialize_vcm_sources``.

The VCM round-trip is exercised by the CLI integration test; here we
only verify that the orchestrators (a) skip rows whose names are not
in ``enabled_sources`` and (b) layer per-source paging overrides
onto the base :class:`MmapConfig` before invoking the VCM handle.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polymathera.colony.design_monorepo.materialize import (
    materialize_knowledge_sources,
    materialize_vcm_sources,
)
from polymathera.colony.design_monorepo.repo_map import (
    KnowledgeSource,
    RepoMap,
    VcmSource,
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
    records = await materialize_knowledge_sources(
        repo_map=repo_map, repo_root=repo_root,
    )
    sources = sorted(r.source_uri for r in records)
    assert len(records) == 2
    assert all("literature/curated" in s for s in sources)


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
    records = await materialize_knowledge_sources(
        repo_map=repo_map, repo_root=repo_root, enabled_sources={"a"},
    )
    sources = [r.source_uri for r in records]
    assert len(records) == 1
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
        "schema_version: 1\n"
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
        "schema_version: 1\n"
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
    records = await materialize_knowledge_sources(
        repo_map=repo_map, repo_root=repo_root,
    )
    assert records == []
    _ = get_default_ingestor  # silence unused-import in some linters
