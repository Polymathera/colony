"""Tests for the agent-facing capability surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from polymathera.colony.design_monorepo import (
    DesignCheckpointer,
    DesignMonorepoClient,
    RepoBootstrapSpec,
    RepoStateProvider,
    ToolBuilder,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def state_provider(bootstrapped_repo: DesignMonorepoClient) -> RepoStateProvider:
    cap = RepoStateProvider(
        agent=None, scope_id="test", working_dir=bootstrapped_repo.working_dir,
    )
    cap._client = bootstrapped_repo
    return cap


@pytest.fixture
def checkpointer(bootstrapped_repo: DesignMonorepoClient) -> DesignCheckpointer:
    cap = DesignCheckpointer(
        agent=None, scope_id="test", working_dir=bootstrapped_repo.working_dir,
    )
    cap._client = bootstrapped_repo
    return cap


@pytest.fixture
def tool_builder(bootstrapped_repo: DesignMonorepoClient) -> ToolBuilder:
    cap = ToolBuilder(
        agent=None, scope_id="test", working_dir=bootstrapped_repo.working_dir,
    )
    cap._client = bootstrapped_repo
    return cap


async def test_get_repo_state(state_provider: RepoStateProvider) -> None:
    st = await state_provider.get_repo_state()
    assert st.is_fresh is True
    assert st.current_branch == "main"


async def test_find_existing_tool_finds_after_bootstrap(
    state_provider: RepoStateProvider, tool_builder: ToolBuilder,
) -> None:
    spec = RepoBootstrapSpec(
        template="python_lib",
        target="subdir_in_monorepo:tools/shared/widgets",
        name="widget_engine",
        purpose="shared/widgets",
        license="MIT",
        capability="render_widget",
        description="Renders widgets.",
    )
    result = await tool_builder.bootstrap_repo(spec)
    assert result.tool_entry.capability == "render_widget"
    matches = await state_provider.find_existing_tool("render_widget")
    assert len(matches) == 1
    assert matches[0].entry.name == "widget_engine"
    assert matches[0].writable is True


async def test_find_existing_tool_partial_match(
    state_provider: RepoStateProvider, tool_builder: ToolBuilder,
) -> None:
    await tool_builder.bootstrap_repo(RepoBootstrapSpec(
        template="python_lib",
        target="subdir_in_monorepo:tools/shared/widgets",
        name="widget_engine",
        purpose="shared/widgets",
        license="MIT",
        capability="render_widget",
        description="renders widgets in 3D",
    ))
    matches = await state_provider.find_existing_tool("render")
    assert any(m.entry.name == "widget_engine" for m in matches)


async def test_ingest_repo_map_literature_walks_knowledge_sources(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
    monkeypatch,
) -> None:
    """The chat-callable ingestion action: write a ``repo_map.yaml``
    with two ``knowledge_sources`` rows, seed matching files, run the
    action with ``refresh=False`` (no remote to fetch from in this
    fixture). First run with no persisted selection (all rows ingest).
    Then patch the persisted selection to ``["curated"]`` and re-run —
    only the curated row's file should ingest.
    """

    from polymathera.colony.design_monorepo import capabilities as cap_mod
    from polymathera.colony.knowledge.deps import (
        get_default_ingestor, reset_knowledge_deps,
    )

    reset_knowledge_deps()
    try:
        repo_root = bootstrapped_repo.working_dir

        (repo_root / "literature" / "curated").mkdir(parents=True)
        (repo_root / "literature" / "promoted").mkdir(parents=True)
        (repo_root / "literature" / "curated" / "a.txt").write_text(
            "Curated paper A.\n", encoding="utf-8",
        )
        (repo_root / "literature" / "promoted" / "b.txt").write_text(
            "Promoted paper B.\n", encoding="utf-8",
        )

        # Write the repo map. ``refresh=False`` in the call below
        # skips ``git fetch`` so this works in the bootstrapped-repo
        # fixture which has no remote.
        (repo_root / ".colony" / "repo_map.yaml").write_text(
            "schema_version: 3\n"
            "vcm_sources:\n"
            "  - { name: default, type: git_repo }\n"
            "knowledge_sources:\n"
            "  - name: curated\n"
            "    paths: ['literature/curated/**/*.txt']\n"
            "    profile: scientific_paper\n"
            "  - name: promoted\n"
            "    paths: ['literature/promoted/**/*.txt']\n",
            encoding="utf-8",
        )

        # Stub the persisted-selection lookup. Tracks the value via
        # ``monkeypatch`` so the second arm can flip it without a
        # live Redis (which the unit-test process doesn't run).
        from polymathera.colony.design_monorepo import (
            source_selection as ss_mod,
        )
        persisted: dict[str, list[str] | None] = {"value": None}

        async def _stub_list(_colony_id: str) -> list[str] | None:
            return persisted["value"]

        monkeypatch.setattr(
            ss_mod, "list_enabled_knowledge_sources", _stub_list,
        )

        # Default (``persisted=None``) — both rows ingest.
        result = await state_provider.ingest_repo_map_literature(refresh=False)
        assert result["count"] == 2
        assert result["by_status"] == {"completed": 2}
        assert result["backend"]["vector_store"] == "InMemoryVectorStore"

        # Operator unticked ``promoted`` — only ``curated`` ingests.
        reset_knowledge_deps()
        persisted["value"] = ["curated"]
        result2 = await state_provider.ingest_repo_map_literature(refresh=False)
        assert result2["count"] == 1
        assert any("a.txt" in uri for uri in result2["ingested"])
        assert not any("b.txt" in uri for uri in result2["ingested"])
    finally:
        reset_knowledge_deps()


async def test_materialize_design_context_end_to_end(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
    monkeypatch,
) -> None:
    """End-to-end wiring test for ``materialize_design_context``:

    - Writes a ``repo_map.yaml`` with one pinned + one un-pinned
      ``design_context_sources`` row.
    - Stubs the VCM handle and colony blackboard so the test runs
      without a live cluster.
    - Verifies the action returns the right response shape, calls
      ``mmap_application_scope`` per row, calls ``lock_page`` only
      on the pinned row's pages, registers the pinned row with the
      renewer, and emits one ``DesignContextMappedProtocol`` event
      per row to the colony blackboard.
    - Verifies ``stop()`` cancels the renewer cleanly.
    """

    from unittest.mock import AsyncMock, MagicMock

    repo_root = bootstrapped_repo.working_dir

    # Seed the design-context markdown the materialiser will count.
    (repo_root / "docs").mkdir(parents=True, exist_ok=True)
    (repo_root / "docs" / "objectives.md").write_text(
        "# Objectives\n", encoding="utf-8",
    )
    (repo_root / "docs" / "constraints.md").write_text(
        "# Constraints\n", encoding="utf-8",
    )
    (repo_root / "hypotheses").mkdir(parents=True, exist_ok=True)
    (repo_root / "hypotheses" / "h1.md").write_text(
        "# Hypothesis 1\n", encoding="utf-8",
    )

    # Schema v3 with two rows: docs (pinned) + hypotheses (unpinned).
    (repo_root / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - { name: default, type: git_repo }\n"
        "design_context_sources:\n"
        "  - name: docs\n"
        "    paths: ['docs/**/*.md']\n"
        "    pin_in_vcm: true\n"
        "    pin_lock_duration_days: 5\n"
        "  - name: hypos\n"
        "    paths: ['hypotheses/**/*.md']\n",
        encoding="utf-8",
    )

    # Stub the VCM handle the action calls into.
    fake_vcm = MagicMock()
    fake_vcm.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )
    fake_vcm.get_pages_for_scope = AsyncMock(
        return_value=[
            {"page_id": "p1", "size": 100, "group_id": "g"},
            {"page_id": "p2", "size": 100, "group_id": "g"},
        ],
    )
    fake_vcm.lock_page = AsyncMock()
    fake_vcm.extend_page_lock = AsyncMock(return_value=True)

    async def _stub_get_vcm():
        return fake_vcm

    from polymathera.colony import _handles as handles_mod
    monkeypatch.setattr(handles_mod, "get_vcm", _stub_get_vcm)

    # Pre-set the cached colony blackboard so _get_colony_blackboard
    # short-circuits and returns our fake (avoids needing a live
    # serving cluster + ScopeUtils.get_colony_level_scope).
    fake_bb = MagicMock()
    fake_bb.write = AsyncMock()
    state_provider._colony_blackboard = fake_bb

    # ``_load_design_context_impl`` now builds the colony-prefixed
    # scope id via ``ScopeUtils.get_colony_level_scope()`` (so the
    # VCM's ``_local_scope_key`` validation accepts it). That helper
    # requires a live ``execution_context``; wrap the call so the
    # ContextVar resolves.
    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring, execution_context,
    )

    # Default include_kuzu=True — focus this test on the VCM path
    # by opting out of path-1; the path-1 wiring is exercised in
    # ``test_materialize_design_context_kuzu_path_ingests_into_graph``.
    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c1", session_id="s",
    ):
        result = await state_provider.materialize_design_context(
            refresh=False, include_kuzu=False,
        )

    # ---- response shape ----
    assert set(result["mapped"]) == {"docs", "hypos"}
    assert result["pinned"] == ["docs"]
    assert result["ingested"] == []  # include_kuzu=False
    assert result["total_claims"] == 0
    assert result["failed"] == []
    # 2 sources × 2 paths (vcm + kuzu-skipped) = 4 outcome rows.
    assert result["count"] == 4
    vcm_rows = [r for r in result["rows"] if r["path"] == "vcm"]
    rows_by_name = {r["source_name"]: r for r in vcm_rows}
    assert rows_by_name["docs"]["pinned"] is True
    assert rows_by_name["hypos"]["pinned"] is False
    assert rows_by_name["docs"]["status"] == "mapped"
    # Kuzu rows are present but skipped (include_kuzu=False).
    kuzu_rows = [r for r in result["rows"] if r["path"] == "kuzu"]
    assert {r["source_name"] for r in kuzu_rows} == {"docs", "hypos"}
    assert all(r["status"] == "skipped" for r in kuzu_rows)

    # ---- mmap calls ----
    mmap_calls = fake_vcm.mmap_application_scope.await_args_list
    assert len(mmap_calls) == 2
    for call in mmap_calls:
        assert call.kwargs["source_type"] == "literature"

    # ---- pinning: 2 lock_page calls for the pinned row only ----
    assert fake_vcm.lock_page.await_count == 2
    expected_duration_s = 5 * 86400.0
    for call in fake_vcm.lock_page.await_args_list:
        assert call.kwargs["locked_by"] == "design_context.docs"
        assert call.kwargs["lock_duration_s"] == expected_duration_s

    # ---- renewer was created + registered the pinned scope ----
    assert state_provider._design_context_renewer is not None
    assert state_provider._design_context_renewer.registered_scope_ids == [
        next(
            c.kwargs["scope_id"]
            for c in mmap_calls
            if c.kwargs["scope_id"].endswith("docs")
        ),
    ]

    # ---- blackboard events: one per outcome row (4 total) ----
    bb_calls = fake_bb.write.await_args_list
    assert len(bb_calls) == 4
    keys = [c.kwargs["key"] for c in bb_calls]
    assert all(k.startswith("design_context_mapped:") for k in keys)
    assert any("docs:vcm:" in k for k in keys)
    assert any("hypos:vcm:" in k for k in keys)
    assert any("docs:kuzu:" in k for k in keys)
    assert any("hypos:kuzu:" in k for k in keys)
    # Pinned row's vcm event carries the 'pinned' tag.
    docs_vcm = next(c for c in bb_calls if "docs:vcm:" in c.kwargs["key"])
    assert "pinned" in docs_vcm.kwargs["tags"]
    assert docs_vcm.kwargs["value"]["pinned"] is True
    assert docs_vcm.kwargs["value"]["num_files"] == 2  # objectives + constraints
    assert docs_vcm.kwargs["value"]["status"] == "mapped"
    hypos_vcm = next(c for c in bb_calls if "hypos:vcm:" in c.kwargs["key"])
    assert "pinned" not in hypos_vcm.kwargs["tags"]
    assert hypos_vcm.kwargs["value"]["pinned"] is False
    # Kuzu rows carry status='skipped' since include_kuzu=False.
    docs_kuzu = next(c for c in bb_calls if "docs:kuzu:" in c.kwargs["key"])
    assert docs_kuzu.kwargs["value"]["status"] == "skipped"
    assert docs_kuzu.kwargs["value"]["num_claims"] == 0

    # ---- stop() cancels the renewer cleanly ----
    await state_provider.stop()
    assert state_provider._design_context_renewer is None


async def test_materialize_design_context_kuzu_path_ingests_into_graph(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
    monkeypatch,
) -> None:
    """End-to-end: include_kuzu=True calls Ingestor.ingest_file per
    matching file with the design_context:// URI scheme, sums
    claims_extracted into the kuzu outcome row, and emits a kuzu
    blackboard event with ``status='completed'``."""

    from unittest.mock import AsyncMock, MagicMock

    from polymathera.colony.knowledge.models import (
        IngestionRecord, IngestionStatus, KnowledgeFormat,
    )

    repo_root = bootstrapped_repo.working_dir
    (repo_root / "docs").mkdir(parents=True, exist_ok=True)
    (repo_root / "docs" / "objectives.md").write_text(
        "# Objectives\n", encoding="utf-8",
    )
    (repo_root / "docs" / "constraints.md").write_text(
        "# Constraints\n", encoding="utf-8",
    )
    (repo_root / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - { name: default, type: git_repo }\n"
        "design_context_sources:\n"
        "  - name: docs\n"
        "    paths: ['docs/**/*.md']\n",
        encoding="utf-8",
    )

    # VCM handle — minimal stub; the focus is path-1.
    fake_vcm = MagicMock()
    fake_vcm.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )
    fake_vcm.get_pages_for_scope = AsyncMock(return_value=[])
    fake_vcm.lock_page = AsyncMock()
    async def _stub_get_vcm():
        return fake_vcm
    from polymathera.colony import _handles as handles_mod
    monkeypatch.setattr(handles_mod, "get_vcm", _stub_get_vcm)

    # Ingestor stub — returns a synthetic COMPLETED record with a
    # claim count per file. Replaces the singleton so no real Kuzu /
    # vector store / embedder is needed in this unit test.
    captured_uris: list[str] = []

    async def _stub_ingest(path, *, tier, source_uri, **_kw):
        captured_uris.append(source_uri)
        return IngestionRecord(
            source_uri=source_uri,
            detected_format=KnowledgeFormat.MARKDOWN,
            tier=tier,
            status=IngestionStatus.COMPLETED,
            chunks_produced=1,
            claims_extracted=3,
            document_hash="sha",
        )

    fake_ingestor = MagicMock()
    fake_ingestor.ingest_file = AsyncMock(side_effect=_stub_ingest)
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "get_default_ingestor", lambda: fake_ingestor,
    )

    fake_bb = MagicMock()
    fake_bb.write = AsyncMock()
    state_provider._colony_blackboard = fake_bb

    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring, execution_context,
    )
    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c1", session_id="s",
    ):
        result = await state_provider.materialize_design_context(
            refresh=False, include_kuzu=True,
        )

    # Each matching file got ingested with the design_context URI.
    assert fake_ingestor.ingest_file.await_count == 2
    assert sorted(captured_uris) == [
        "design_context://docs/docs/constraints.md",
        "design_context://docs/docs/objectives.md",
    ]
    # All ingest_file calls used tier_1_foundations (design context is
    # foundational, high retrieval weight per the master pipeline).
    from polymathera.colony.knowledge.models import CorpusTier
    for call in fake_ingestor.ingest_file.await_args_list:
        assert call.kwargs["tier"] == CorpusTier.TIER_1_FOUNDATIONS

    # Response shape: ingested + total_claims surface the kuzu outcome.
    assert result["ingested"] == ["docs"]
    assert result["total_claims"] == 6  # 2 files × 3 claims each
    kuzu_row = next(r for r in result["rows"] if r["path"] == "kuzu")
    assert kuzu_row["status"] == "completed"
    assert kuzu_row["num_files"] == 2
    assert kuzu_row["num_claims"] == 6
    assert kuzu_row["error"] == ""

    # Two blackboard events: one vcm, one kuzu.
    bb_calls = fake_bb.write.await_args_list
    paths_emitted = {
        c.kwargs["value"]["path"] for c in bb_calls
    }
    assert paths_emitted == {"vcm", "kuzu"}
    docs_kuzu = next(
        c for c in bb_calls
        if c.kwargs["value"]["path"] == "kuzu"
        and c.kwargs["value"]["source_name"] == "docs"
    )
    assert docs_kuzu.kwargs["value"]["status"] == "completed"
    assert docs_kuzu.kwargs["value"]["num_claims"] == 6
    assert "kuzu" in docs_kuzu.kwargs["tags"]


async def test_materialize_design_context_kuzu_partial_on_some_file_failures(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
    monkeypatch,
) -> None:
    """When some files fail to ingest but others succeed, the kuzu
    row's status degrades to 'partial' and the error string lists
    failing files."""

    from unittest.mock import AsyncMock, MagicMock

    from polymathera.colony.knowledge.models import (
        IngestionRecord, IngestionStatus, KnowledgeFormat,
    )

    repo_root = bootstrapped_repo.working_dir
    (repo_root / "docs").mkdir(parents=True, exist_ok=True)
    (repo_root / "docs" / "good.md").write_text("# good\n", encoding="utf-8")
    (repo_root / "docs" / "bad.md").write_text("# bad\n", encoding="utf-8")
    (repo_root / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n  - { name: default, type: git_repo }\n"
        "design_context_sources:\n"
        "  - name: docs\n    paths: ['docs/**/*.md']\n",
        encoding="utf-8",
    )

    fake_vcm = MagicMock()
    fake_vcm.mmap_application_scope = AsyncMock(
        return_value=MagicMock(status="mapped"),
    )
    fake_vcm.get_pages_for_scope = AsyncMock(return_value=[])
    fake_vcm.lock_page = AsyncMock()
    async def _stub_get_vcm():
        return fake_vcm
    from polymathera.colony import _handles as handles_mod
    monkeypatch.setattr(handles_mod, "get_vcm", _stub_get_vcm)

    async def _stub_ingest(path, *, tier, source_uri, **_kw):
        if "bad.md" in str(path):
            raise RuntimeError("simulated ingest failure")
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
    fake_ingestor.ingest_file = AsyncMock(side_effect=_stub_ingest)
    from polymathera.colony.knowledge import deps as kdeps_mod
    monkeypatch.setattr(
        kdeps_mod, "get_default_ingestor", lambda: fake_ingestor,
    )

    fake_bb = MagicMock()
    fake_bb.write = AsyncMock()
    state_provider._colony_blackboard = fake_bb

    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring, execution_context,
    )
    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c1", session_id="s",
    ):
        result = await state_provider.materialize_design_context(
            refresh=False, include_kuzu=True,
        )

    kuzu_row = next(r for r in result["rows"] if r["path"] == "kuzu")
    assert kuzu_row["status"] == "partial"
    assert kuzu_row["num_files"] == 2
    assert kuzu_row["num_claims"] == 2  # only good.md
    assert "bad.md" in kuzu_row["error"]
    # The 'docs' source still appears in 'ingested' (partial counts).
    assert result["ingested"] == ["docs"]
    # The failure also appears in 'failed' because error != "".
    assert any(
        f["source_name"] == "docs" and f["path"] == "kuzu"
        for f in result["failed"]
    )


async def test_materialize_design_context_no_rows_short_circuits(
    bootstrapped_repo: DesignMonorepoClient,
    state_provider: RepoStateProvider,
    monkeypatch,
) -> None:
    """Empty ``design_context_sources`` block returns a helpful
    message without touching VCM or the blackboard."""

    from unittest.mock import AsyncMock, MagicMock

    repo_root = bootstrapped_repo.working_dir
    (repo_root / ".colony" / "repo_map.yaml").write_text(
        "schema_version: 3\n"
        "vcm_sources:\n"
        "  - { name: default, type: git_repo }\n",
        encoding="utf-8",
    )

    fake_vcm = MagicMock()
    fake_vcm.mmap_application_scope = AsyncMock()
    async def _stub_get_vcm():
        return fake_vcm
    from polymathera.colony import _handles as handles_mod
    monkeypatch.setattr(handles_mod, "get_vcm", _stub_get_vcm)

    fake_bb = MagicMock()
    fake_bb.write = AsyncMock()
    state_provider._colony_blackboard = fake_bb

    result = await state_provider.materialize_design_context(refresh=False)

    assert result["count"] == 0
    assert result["mapped"] == []
    assert result["pinned"] == []
    assert "No ``design_context_sources:``" in result["message"]
    fake_vcm.mmap_application_scope.assert_not_awaited()
    fake_bb.write.assert_not_awaited()


async def test_checkpoint_state_round_trip(
    checkpointer: DesignCheckpointer, state_provider: RepoStateProvider,
) -> None:
    cp = await checkpointer.checkpoint_state("v1", "first")
    assert cp.label == "v1"
    cps = await checkpointer.list_checkpoints()
    assert any(c.checkpoint_id == cp.checkpoint_id for c in cps)
    st = await state_provider.get_repo_state()
    assert any(c.checkpoint_id == cp.checkpoint_id for c in st.checkpoints)


async def test_fork_design_creates_branch(
    checkpointer: DesignCheckpointer, state_provider: RepoStateProvider,
) -> None:
    fork = await checkpointer.fork_design("explore-A")
    assert fork.name == "fork/explore-A"
    st = await state_provider.get_repo_state()
    assert st.current_branch == "fork/explore-A"


async def test_initialize_repo_map_writes_repo_map_when_manifest_exists(
    bootstrapped_repo: DesignMonorepoClient,
    checkpointer: DesignCheckpointer,
) -> None:
    """When the manifest is already present (the
    ``bootstrapped_repo`` fixture writes one) but the repo_map is
    missing, ``initialize_repo_map`` writes only the repo_map and
    commits it. Pins that the action *does not* clobber the existing
    manifest and the resulting YAML parses cleanly.
    """

    from polymathera.colony.design_monorepo.manifest import (
        MANIFEST_RELATIVE_PATH,
    )
    from polymathera.colony.design_monorepo.repo_map import (
        REPO_MAP_DIR, REPO_MAP_FILENAME, RepoMap,
    )

    repo_root = bootstrapped_repo.working_dir
    target = repo_root / REPO_MAP_DIR / REPO_MAP_FILENAME
    assert not target.exists()
    manifest_before = (repo_root / MANIFEST_RELATIVE_PATH).read_text()

    result = await checkpointer.initialize_repo_map(push=False)
    assert result["status"] == "initialized"
    assert result["files_created"] == [f"{REPO_MAP_DIR}/{REPO_MAP_FILENAME}"]
    assert result["committed_sha"] is not None
    assert target.is_file()
    # Existing manifest was NOT touched.
    assert (repo_root / MANIFEST_RELATIVE_PATH).read_text() == manifest_before

    # Template parses through the repo-map schema with the default
    # ``git_repo`` source as the only active row.
    rm = RepoMap.load(repo_root)
    assert [s.name for s in rm.vcm_sources] == ["default"]
    assert rm.knowledge_sources == []

    # Working tree is clean — the action committed everything it wrote.
    assert not bootstrapped_repo.has_uncommitted_changes()


async def test_initialize_repo_map_bootstraps_empty_repo(
    tmp_path: Path,
) -> None:
    """The user-facing failure mode: a fresh-cloned repo with neither
    ``.colony/manifest.json`` nor ``.colony/repo_map.yaml``. Going
    through :class:`DesignMonorepoClient.open` would raise
    ``ManifestSchemaError`` because there's no manifest yet — the
    chicken-and-egg the action exists to break.

    The action MUST work on this case end-to-end: write both files
    with sensible defaults, commit them, and leave the working tree
    clean. Subsequent capability calls (which do go through
    ``Client.open``) then succeed.
    """

    import git

    from polymathera.colony.design_monorepo.manifest import (
        MANIFEST_RELATIVE_PATH, DesignMonorepoManifest,
    )
    from polymathera.colony.design_monorepo.repo_map import (
        REPO_MAP_DIR, REPO_MAP_FILENAME, RepoMap,
    )

    repo_root = tmp_path / "fresh"
    repo_root.mkdir()
    git.Repo.init(str(repo_root), initial_branch="main")

    cap = DesignCheckpointer(
        agent=None, scope_id="dm", working_dir=repo_root,
    )
    # The repo has no ``origin`` remote (just a local ``git init``)
    # so ``push=True`` records a clean push_error rather than failing.
    # We assert that path here too — the operator must be able to
    # bootstrap a local-only design monorepo without a remote.
    result = await cap.initialize_repo_map()

    assert result["status"] == "initialized"
    # ``.gitattributes`` joins manifest + repo_map.yaml in the
    # bootstrap commit because ``enable_lfs=True`` is the default.
    assert set(result["files_created"]) == {
        MANIFEST_RELATIVE_PATH,
        f"{REPO_MAP_DIR}/{REPO_MAP_FILENAME}",
        ".gitattributes",
    }
    assert result["committed_sha"] is not None
    assert result["pushed"] is False
    assert result["push_error"] is not None
    assert "origin" in result["push_error"].lower()

    # Both files are committed: working tree clean.
    repo = git.Repo(str(repo_root))
    assert not repo.is_dirty(untracked_files=True)

    # The manifest parses through its schema (defaults applied).
    manifest = DesignMonorepoManifest.load_path(repo_root)
    assert manifest.target_system == "unspecified"
    # The repo_map parses through its schema.
    rm = RepoMap.load(repo_root)
    assert [s.name for s in rm.vcm_sources] == ["default"]


async def test_initialize_repo_map_attribution_default_colony_co_author_user(
    tmp_path: Path,
) -> None:
    """The framework default: ``commit_principal=colony``,
    ``commit_co_author=user``. The commit's author/committer is
    ``colony:<id>`` and the message ends with a ``Co-Authored-By:``
    trailer naming the configured user.

    This is the user-visible change being shipped — the previous
    behaviour stamped the ephemeral agent identity on every commit;
    new default keeps the persistent collective identity as
    principal and surfaces the human via the trailer (so GitHub UI
    attribution and ``git log --grep`` both work).
    """

    from unittest.mock import MagicMock

    import git

    from polymathera.colony.design_monorepo.capabilities import (
        DesignCheckpointer,
    )
    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring, execution_context,
    )

    repo_root = tmp_path / "fresh"
    repo_root.mkdir()
    git.Repo.init(str(repo_root), initial_branch="main")

    agent = MagicMock()
    agent.agent_id = "agent-A"
    agent.colony_id = "c1"
    agent.metadata.parameters = {
        "git_attribution": {
            "commit_principal": "colony",
            "commit_co_author": "user",
        },
        "github_identity": {
            "git_user_name": "Ada Lovelace",
            "git_user_email": "ada@example.com",
        },
    }
    agent.metadata.role = "session_orchestrator"

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c1", session_id="s",
    ):
        cap = DesignCheckpointer(
            agent=agent, scope_id="dm", working_dir=repo_root,
        )
        await cap.initialize_repo_map(push=False)

    repo = git.Repo(str(repo_root))
    head = repo.head.commit
    assert head.author.name == "colony:c1"
    assert head.author.email == "c1@agent.colony.local"
    assert head.committer.name == "colony:c1"
    assert (
        "Co-Authored-By: Ada Lovelace <ada@example.com>"
        in head.message
    )


async def test_initialize_repo_map_attribution_user_principal_no_co_author(
    tmp_path: Path,
) -> None:
    """When the operator picks ``commit_principal=user`` with no
    co-author, the commit looks like a plain human commit — no
    trailer, real name/email as author/committer.
    """

    from unittest.mock import MagicMock

    import git

    from polymathera.colony.design_monorepo.capabilities import (
        DesignCheckpointer,
    )
    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring, execution_context,
    )

    repo_root = tmp_path / "fresh"
    repo_root.mkdir()
    git.Repo.init(str(repo_root), initial_branch="main")

    agent = MagicMock()
    agent.agent_id = "agent-A"
    agent.colony_id = "c1"
    agent.metadata.parameters = {
        "git_attribution": {
            "commit_principal": "user",
            "commit_co_author": None,
        },
        "github_identity": {
            "git_user_name": "Ada Lovelace",
            "git_user_email": "ada@example.com",
        },
    }
    agent.metadata.role = "session_orchestrator"

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c1", session_id="s",
    ):
        cap = DesignCheckpointer(
            agent=agent, scope_id="dm", working_dir=repo_root,
        )
        await cap.initialize_repo_map(push=False)

    repo = git.Repo(str(repo_root))
    head = repo.head.commit
    assert head.author.name == "Ada Lovelace"
    assert head.author.email == "ada@example.com"
    assert "Co-Authored-By:" not in head.message


async def test_initialize_repo_map_attribution_falls_back_when_user_unset(
    tmp_path: Path,
) -> None:
    """If ``commit_co_author=user`` is configured but no name/email
    is set on the colony, the action must still succeed — the
    trailer is dropped (with a warning) rather than blocking the
    commit. Operator can fix the config and re-run; partial
    attribution beats no commit.
    """

    from unittest.mock import MagicMock

    import git

    from polymathera.colony.design_monorepo.capabilities import (
        DesignCheckpointer,
    )
    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring, execution_context,
    )

    repo_root = tmp_path / "fresh"
    repo_root.mkdir()
    git.Repo.init(str(repo_root), initial_branch="main")

    agent = MagicMock()
    agent.agent_id = "agent-A"
    agent.colony_id = "c1"
    agent.metadata.parameters = {
        "git_attribution": {
            "commit_principal": "colony",
            "commit_co_author": "user",
        },
        "github_identity": {
            "git_user_name": None,
            "git_user_email": None,
        },
    }
    agent.metadata.role = "session_orchestrator"

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c1", session_id="s",
    ):
        cap = DesignCheckpointer(
            agent=agent, scope_id="dm", working_dir=repo_root,
        )
        await cap.initialize_repo_map(push=False)

    repo = git.Repo(str(repo_root))
    head = repo.head.commit
    # Principal still applied; trailer skipped silently.
    assert head.author.name == "colony:c1"
    assert "Co-Authored-By:" not in head.message


def test_lfs_patterns_from_gitattributes_extracts_filter_lfs_lines(
    tmp_path: Path,
) -> None:
    """``_lfs_patterns_from_gitattributes`` returns exactly the
    patterns that route through LFS. Skips comments, blank lines,
    and lines without ``filter=lfs``. Used by ``initialize_repo_map``
    when the operator opts into ``migrate_existing_to_lfs=True``."""

    from polymathera.colony.design_monorepo.capabilities import (
        _lfs_patterns_from_gitattributes,
    )

    p = tmp_path / ".gitattributes"
    p.write_text(
        "# header comment\n"
        "\n"
        "*.pdf       filter=lfs diff=lfs merge=lfs -text\n"
        "*.docx      filter=lfs diff=lfs merge=lfs -text\n"
        "*.md        text\n"  # NOT an LFS line
        "*.zip       filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    assert _lfs_patterns_from_gitattributes(p) == ["*.pdf", "*.docx", "*.zip"]


def test_lfs_patterns_from_gitattributes_returns_empty_when_missing(
    tmp_path: Path,
) -> None:
    from polymathera.colony.design_monorepo.capabilities import (
        _lfs_patterns_from_gitattributes,
    )
    assert _lfs_patterns_from_gitattributes(tmp_path / "absent") == []


async def test_initialize_repo_map_enables_lfs_by_default(
    tmp_path: Path,
) -> None:
    """``enable_lfs=True`` (the default): writes the bundled
    ``.gitattributes`` LFS template, flips the manifest's
    ``lfs.mode`` to ``"same_remote"``, and includes the file in the
    bootstrap commit. ``git lfs install --local`` is best-effort —
    if ``git-lfs`` isn't on the dev box the action logs a warning
    and continues, which is the contract this test pins.
    """

    import git

    from polymathera.colony.design_monorepo.capabilities import (
        DesignCheckpointer,
    )
    from polymathera.colony.design_monorepo.manifest import (
        DesignMonorepoManifest,
    )

    repo_root = tmp_path / "fresh"
    repo_root.mkdir()
    git.Repo.init(str(repo_root), initial_branch="main")

    cap = DesignCheckpointer(
        agent=None, scope_id="dm", working_dir=repo_root,
    )
    result = await cap.initialize_repo_map(push=False)

    assert result["status"] == "initialized"
    assert ".gitattributes" in result["files_created"]
    assert result["lfs_enabled"] is True
    assert result["migrated_to_lfs"] is False  # not opted in

    gitattrs = (repo_root / ".gitattributes").read_text(encoding="utf-8")
    # Spot-check a few representative LFS patterns from the template.
    assert "*.pdf" in gitattrs
    assert "filter=lfs" in gitattrs
    assert "*.zip" in gitattrs

    manifest = DesignMonorepoManifest.load_path(repo_root)
    assert manifest.lfs.mode == "same_remote"


async def test_initialize_repo_map_skip_lfs_when_disabled(
    tmp_path: Path,
) -> None:
    """``enable_lfs=False``: no ``.gitattributes`` written; manifest
    records ``lfs.mode="disabled"``. The opt-out exists for operators
    who don't want LFS at all (e.g. small monorepos with no large
    binaries, or air-gapped deployments without an LFS endpoint)."""

    import git

    from polymathera.colony.design_monorepo.capabilities import (
        DesignCheckpointer,
    )
    from polymathera.colony.design_monorepo.manifest import (
        DesignMonorepoManifest,
    )

    repo_root = tmp_path / "fresh"
    repo_root.mkdir()
    git.Repo.init(str(repo_root), initial_branch="main")

    cap = DesignCheckpointer(
        agent=None, scope_id="dm", working_dir=repo_root,
    )
    result = await cap.initialize_repo_map(push=False, enable_lfs=False)

    assert result["status"] == "initialized"
    assert ".gitattributes" not in result["files_created"]
    assert not (repo_root / ".gitattributes").exists()
    assert result["lfs_enabled"] is False

    manifest = DesignMonorepoManifest.load_path(repo_root)
    assert manifest.lfs.mode == "disabled"


async def test_initialize_repo_map_preserves_operator_edited_gitattributes(
    tmp_path: Path,
) -> None:
    """When ``.gitattributes`` already exists, ``initialize_repo_map``
    must NOT overwrite it — operator edits stay intact. Same
    no-clobber discipline as the manifest and repo_map.yaml.
    """

    import git

    from polymathera.colony.design_monorepo.capabilities import (
        DesignCheckpointer,
    )

    repo_root = tmp_path / "fresh"
    repo_root.mkdir()
    git.Repo.init(str(repo_root), initial_branch="main")

    operator_content = "# my custom rules\n*.weird filter=lfs diff=lfs merge=lfs -text\n"
    (repo_root / ".gitattributes").write_text(operator_content, encoding="utf-8")
    repo = git.Repo(str(repo_root))
    repo.config_writer().set_value("user", "email", "t@t").release()
    repo.config_writer().set_value("user", "name", "t").release()
    repo.index.add([".gitattributes"])
    repo.index.commit("operator: custom .gitattributes")

    cap = DesignCheckpointer(
        agent=None, scope_id="dm", working_dir=repo_root,
    )
    result = await cap.initialize_repo_map(push=False)

    assert result["status"] == "initialized"
    assert ".gitattributes" not in result["files_created"]
    assert (repo_root / ".gitattributes").read_text(encoding="utf-8") == operator_content


async def test_initialize_repo_map_flips_existing_manifest_lfs_mode(
    tmp_path: Path,
) -> None:
    """If a previously-initialised repo has ``lfs.mode="disabled"``
    in its manifest, calling ``initialize_repo_map(enable_lfs=True)``
    must flip the mode to ``"same_remote"`` and re-commit the
    manifest — that's how the operator opts an existing repo into
    LFS without re-creating it.
    """

    import json

    import git

    from polymathera.colony.design_monorepo.capabilities import (
        DesignCheckpointer,
    )
    from polymathera.colony.design_monorepo.manifest import (
        DesignMonorepoManifest, MANIFEST_RELATIVE_PATH,
    )

    repo_root = tmp_path / "fresh"
    repo_root.mkdir()
    repo = git.Repo.init(str(repo_root), initial_branch="main")
    repo.config_writer().set_value("user", "email", "t@t").release()
    repo.config_writer().set_value("user", "name", "t").release()

    # Pre-existing manifest with LFS disabled.
    manifest_path = repo_root / MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "tenant": "t", "colony": "c", "program": "p",
        "target_system": "ts", "topology": "external",
        "design_repo_url": "file://placeholder",
        "lfs": {"mode": "disabled", "separate_url": None, "credentials_ref": None},
    }) + "\n")
    repo.index.add([MANIFEST_RELATIVE_PATH])
    repo.index.commit("seed manifest with lfs disabled")

    cap = DesignCheckpointer(
        agent=None, scope_id="dm", working_dir=repo_root,
    )
    result = await cap.initialize_repo_map(push=False)

    assert result["status"] == "initialized"
    assert MANIFEST_RELATIVE_PATH in result["files_created"]

    manifest = DesignMonorepoManifest.load_path(repo_root)
    assert manifest.lfs.mode == "same_remote"


async def test_initialize_repo_map_is_idempotent(
    bootstrapped_repo: DesignMonorepoClient,
    checkpointer: DesignCheckpointer,
) -> None:
    """Second call when both files already exist must NOT overwrite
    operator edits — ``initialize`` is a one-shot scaffold, not a
    re-render. Returns ``status="already_initialized"`` so the
    planner can branch on it cleanly.
    """

    from polymathera.colony.design_monorepo.repo_map import (
        REPO_MAP_DIR, REPO_MAP_FILENAME,
    )

    await checkpointer.initialize_repo_map()
    target = bootstrapped_repo.working_dir / REPO_MAP_DIR / REPO_MAP_FILENAME
    target.write_text(
        "schema_version: 3\n"
        "sources:\n"
        "  - name: operator-edited\n"
        "    type: git_repo\n",
        encoding="utf-8",
    )

    result = await checkpointer.initialize_repo_map(push=False)
    assert result["status"] == "already_initialized"
    assert result["files_created"] == []
    assert result["committed_sha"] is None
    assert "operator-edited" in target.read_text(encoding="utf-8")


async def test_diff_design_against_checkpoint(
    bootstrapped_repo: DesignMonorepoClient,
    checkpointer: DesignCheckpointer,
    state_provider: RepoStateProvider,
) -> None:
    cp = await checkpointer.checkpoint_state("anchor", "")
    p = bootstrapped_repo.working_dir / "design" / "added.txt"
    p.write_text("x", encoding="utf-8")
    await checkpointer.commit_state(
        "add file",
        paths=["design/added.txt"],
    )
    diff = await state_provider.diff_against_checkpoint(cp.checkpoint_id)
    paths = {e.path for e in diff.entries}
    assert "design/added.txt" in paths


async def test_bootstrap_repo_unsupported_target(
    tool_builder: ToolBuilder,
) -> None:
    with pytest.raises(NotImplementedError):
        await tool_builder.bootstrap_repo(RepoBootstrapSpec(
            template="python_lib",
            target="new_standalone:https://example.com/x.git",
            name="x",
            purpose="p",
            license="MIT",
            capability="x",
        ))


async def test_get_branch_topology(
    checkpointer: DesignCheckpointer, state_provider: RepoStateProvider,
) -> None:
    await checkpointer.fork_design("alt1")
    topo = await state_provider.get_branch_topology()
    names = {b.name for b in topo.branches}
    assert "main" in names
    assert "fork/alt1" in names
