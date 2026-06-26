"""Per-branch rehydrate: reads ``.colony/colony.kg.json`` from
``origin/<branch>`` and imports it into the shared GraphStore."""

from __future__ import annotations

from pathlib import Path

import git
import pytest

from polymathera.colony.design_monorepo.commit_hooks import (
    reset_pre_commit_registry,
)
from polymathera.colony.knowledge.models import Claim, CitationSpan
from polymathera.colony.knowledge.persistence import (
    KG_FILE_RELATIVE_PATH,
    KgFile,
    PersistedClaim,
    list_remote_branches,
    normalize_branch_name,
    rehydrate_branch_from_repo,
)
from polymathera.colony.knowledge.stores.graph import InMemoryGraphStore


def _claim(subject: str, obj: str, *, source: str = "lit:1") -> Claim:
    return Claim(
        subject=subject, predicate="links", object=obj, confidence=0.9,
        citation=CitationSpan(source_uri=source),
        provenance={"extractor": "deterministic@v1"},
    )


def _write_kg(repo: git.Repo, claims: list[Claim]) -> None:
    root = Path(repo.working_dir)
    path = root / KG_FILE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = KgFile(claims=[PersistedClaim.from_claim(c) for c in claims])
    path.write_text(payload.to_json(), encoding="utf-8")
    repo.git.add(str(path))
    repo.index.commit("seed kg")


@pytest.fixture
def isolated_deps(monkeypatch):
    from polymathera.colony.knowledge import deps as deps_mod

    reset_pre_commit_registry()
    monkeypatch.setattr(deps_mod, "_deps", None)
    monkeypatch.setattr(deps_mod, "_ingestor", None)
    store = InMemoryGraphStore()
    deps_mod.set_knowledge_deps(graph_store=store)
    yield store
    monkeypatch.setattr(deps_mod, "_deps", None)
    monkeypatch.setattr(deps_mod, "_ingestor", None)
    reset_pre_commit_registry()


@pytest.fixture
def origin_with_branches(tmp_path):
    """Bare upstream + a local clone with two branches (``main`` and
    ``feature``) each carrying its own ``.colony/colony.kg.json``."""

    upstream = tmp_path / "origin.git"
    git.Repo.init(upstream, initial_branch="main", bare=True)
    seed = tmp_path / "seed"
    seed_repo = git.Repo.clone_from(f"file://{upstream}", str(seed))
    seed_repo.git.checkout("-b", "main")
    seed_repo.config_writer().set_value("user", "email", "t@t").release()
    seed_repo.config_writer().set_value("user", "name", "t").release()
    _write_kg(seed_repo, [_claim("alpha", "beta")])
    seed_repo.git.push("origin", "main")
    seed_repo.git.checkout("-b", "feature")
    _write_kg(seed_repo, [_claim("gamma", "delta")])
    seed_repo.git.push("origin", "feature")

    clone = tmp_path / "clone"
    clone_repo = git.Repo.clone_from(f"file://{upstream}", str(clone))
    return clone_repo


def test_normalize_branch_name_strips_known_prefixes() -> None:
    assert normalize_branch_name("main") == "main"
    assert normalize_branch_name("origin/main") == "main"
    assert normalize_branch_name("refs/heads/feature/x") == "feature/x"
    assert normalize_branch_name("refs/remotes/origin/dev") == "dev"


@pytest.mark.asyncio
async def test_rehydrate_branch_from_repo_loads_origin_view(
    origin_with_branches, isolated_deps,
):
    result = await rehydrate_branch_from_repo(origin_with_branches, "main")
    assert result["branch"] == "main"
    assert result["claims_in_file"] == 1
    assert result["claims_newly_added"] == 1
    assert result["source_commit_sha"]
    n, e = await isolated_deps.count(branch_filter="main")
    assert (n, e) == (2, 1)


@pytest.mark.asyncio
async def test_rehydrate_branch_uses_origin_not_working_tree(
    origin_with_branches, isolated_deps, tmp_path,
):
    """The working tree starts on ``main``; rehydrating ``feature``
    must read the feature snapshot WITHOUT checking out feature."""

    head_before = origin_with_branches.head.commit.hexsha
    branch_before = origin_with_branches.active_branch.name
    result = await rehydrate_branch_from_repo(
        origin_with_branches, "feature",
    )
    assert result["claims_in_file"] == 1
    assert origin_with_branches.head.commit.hexsha == head_before
    assert origin_with_branches.active_branch.name == branch_before
    # Local Kùzu now carries the feature branch's claims.
    n, e = await isolated_deps.count(branch_filter="feature")
    assert (n, e) == (2, 1)
    # And the main snapshot's claims were NOT loaded (we rehydrated
    # only feature).
    n_main, e_main = await isolated_deps.count(branch_filter="main")
    assert (n_main, e_main) == (0, 0)


@pytest.mark.asyncio
async def test_rehydrate_strips_remote_prefix(
    origin_with_branches, isolated_deps,
):
    result = await rehydrate_branch_from_repo(
        origin_with_branches, "origin/main",
    )
    assert result["branch"] == "main"
    n, _ = await isolated_deps.count(branch_filter="main")
    assert n == 2


@pytest.mark.asyncio
async def test_rehydrate_is_idempotent(
    origin_with_branches, isolated_deps,
):
    first = await rehydrate_branch_from_repo(origin_with_branches, "main")
    second = await rehydrate_branch_from_repo(origin_with_branches, "main")
    assert first["claims_newly_added"] == 1
    assert second["claims_newly_added"] == 0
    assert second["claims_already_present"] == 1


@pytest.mark.asyncio
async def test_rehydrate_missing_file_returns_zero(tmp_path, isolated_deps):
    upstream = tmp_path / "empty.git"
    git.Repo.init(upstream, initial_branch="main", bare=True)
    seed = tmp_path / "seed"
    seed_repo = git.Repo.clone_from(f"file://{upstream}", str(seed))
    seed_repo.git.checkout("-b", "main")
    seed_repo.config_writer().set_value("user", "email", "t@t").release()
    seed_repo.config_writer().set_value("user", "name", "t").release()
    # Empty commit so origin/main resolves.
    (Path(seed) / "README").write_text("x", encoding="utf-8")
    seed_repo.git.add("README")
    seed_repo.index.commit("seed")
    seed_repo.git.push("origin", "main")
    clone = tmp_path / "clone"
    clone_repo = git.Repo.clone_from(f"file://{upstream}", str(clone))
    result = await rehydrate_branch_from_repo(clone_repo, "main")
    assert result["claims_in_file"] == 0
    n, e = await isolated_deps.count()
    assert (n, e) == (0, 0)


@pytest.mark.asyncio
async def test_list_remote_branches_enumerates_and_normalises(
    origin_with_branches,
):
    names = await list_remote_branches(origin_with_branches)
    assert set(names) >= {"main", "feature"}
    assert all(not n.startswith("origin/") for n in names)
