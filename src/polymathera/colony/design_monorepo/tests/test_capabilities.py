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
            "schema_version: 1\n"
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
        "schema_version: 1\n"
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
