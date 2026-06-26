"""Pre-commit-callback contract: registry semantics + integration
with :func:`_commit_all_and_push` / :func:`_commit_paths_and_push`."""

from __future__ import annotations

import pytest

from polymathera.colony.design_monorepo.commit_hooks import (
    PreCommitContext,
    PreCommitRegistry,
    get_pre_commit_registry,
    reset_pre_commit_registry,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    reset_pre_commit_registry()
    yield
    reset_pre_commit_registry()


@pytest.mark.asyncio
async def test_register_rejects_duplicate_name() -> None:
    reg = PreCommitRegistry()

    async def _cb(ctx):  # noqa: ARG001
        return None

    reg.register("a", _cb)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("a", _cb)


@pytest.mark.asyncio
async def test_fire_all_runs_callbacks_in_registration_order() -> None:
    reg = PreCommitRegistry()
    order: list[str] = []

    async def _make(name: str):
        async def _cb(ctx):  # noqa: ARG001
            order.append(name)
        return _cb

    reg.register("first", await _make("first"))
    reg.register("second", await _make("second"))
    reg.register("third", await _make("third"))
    await reg.fire_all(_ctx())
    assert order == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_raising_callback_aborts_subsequent_callbacks() -> None:
    reg = PreCommitRegistry()
    later_fired = False

    async def _boom(ctx):  # noqa: ARG001
        raise RuntimeError("nope")

    async def _later(ctx):  # noqa: ARG001
        nonlocal later_fired
        later_fired = True

    reg.register("boom", _boom)
    reg.register("later", _later)
    with pytest.raises(RuntimeError, match="nope"):
        await reg.fire_all(_ctx())
    assert later_fired is False


@pytest.mark.asyncio
async def test_commit_all_and_push_fires_pre_commit_then_commits_then_pushes() -> None:
    """Callbacks fire BEFORE commit_with_identity, allowing them to
    write files into the working tree that ``git add -A`` then
    stages in the same commit."""

    from polymathera.colony.design_monorepo.capabilities import (
        _commit_all_and_push,
    )

    order: list[str] = []

    async def _pre(ctx: PreCommitContext) -> None:
        order.append("pre")

    get_pre_commit_registry().register("test.pre", _pre)

    class _Repo:
        class _Head:
            class _Commit:
                hexsha = "PRIOR"
            commit = _Commit()
        head = _Head()

    class _StubClient:
        _repo = _Repo()
        active_branch = "main"
        working_dir = None

        def commit_with_identity(self, identity, message, all_changes=False):  # noqa: ARG002
            order.append("commit")
            return "NEW"

        def push(self, *, branch=None, remote="origin", with_tags=False):  # noqa: ARG002
            order.append("push")

    sha, status = await _commit_all_and_push(_StubClient(), object(), "m")
    assert sha == "NEW" and status == "pushed"
    assert order == ["pre", "commit", "push"]


@pytest.mark.asyncio
async def test_pre_commit_failure_aborts_commit_helper() -> None:
    from polymathera.colony.design_monorepo.capabilities import (
        _commit_all_and_push,
    )

    async def _boom(ctx: PreCommitContext) -> None:  # noqa: ARG001
        raise RuntimeError("snapshot failed")

    get_pre_commit_registry().register("test.boom", _boom)

    commit_called = False

    class _Repo:
        class _Head:
            class _Commit:
                hexsha = "PRIOR"
            commit = _Commit()
        head = _Head()

    class _StubClient:
        _repo = _Repo()
        active_branch = "main"
        working_dir = None

        def commit_with_identity(self, identity, message, all_changes=False):  # noqa: ARG002
            nonlocal commit_called
            commit_called = True
            return "NEW"

        def push(self, *, branch=None, remote="origin", with_tags=False):  # noqa: ARG002
            return None

    with pytest.raises(RuntimeError, match="snapshot failed"):
        await _commit_all_and_push(_StubClient(), object(), "m")
    assert commit_called is False


@pytest.mark.asyncio
async def test_empty_registry_skips_pre_commit_step() -> None:
    """When nothing is registered the helper must not even construct
    a PreCommitContext (so callers without an active_branch / working_dir
    still work)."""

    from polymathera.colony.design_monorepo.capabilities import (
        _commit_all_and_push,
    )

    class _Repo:
        class _Head:
            class _Commit:
                hexsha = "PRIOR"
            commit = _Commit()
        head = _Head()

    class _MinimalClient:
        _repo = _Repo()

        def commit_with_identity(self, identity, message, all_changes=False):  # noqa: ARG002
            return "NEW"

        def push(self, *, branch=None, remote="origin", with_tags=False):  # noqa: ARG002
            return None

    sha, status = await _commit_all_and_push(_MinimalClient(), object(), "m")
    assert sha == "NEW" and status == "pushed"


def _ctx() -> PreCommitContext:
    from pathlib import Path
    return PreCommitContext(
        client=None, identity=None, message="m", branch="main",
        paths=None, working_dir=Path("/tmp"),
    )
