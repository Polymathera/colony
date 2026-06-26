"""R7-FIX-C regression: every share-purpose commit-producing action
must commit AND push to ``origin``. Run7 had ~30 PDFs ingested via
``ingest_repo_map_literature`` with sidecars persisted to the
agent's clone working dir BUT never pushed to github — the cache
goal (avoid re-paying for PDF OCR + LLM claim extraction on
subsequent sessions) was defeated when the agent's clone was GC'd.

The fix adds two helpers ``_commit_all_and_push`` +
``_commit_paths_and_push`` and applies them to the share-purpose
sites:

  - ``ingest_repo_map_literature``
  - ``checkpoint_state`` + the ``DesignCheckpointer._on_quiescence``
    auto-checkpoint
  - ``commit_state`` (inline + deferred-execution via
    ``_dispatch_protected_op``)

The helpers' contract is verified with a stub
:class:`DesignMonorepoClient` that records the call order
(commit-then-push) and returns synthetic SHAs. The action sites are
verified via source inspection — the same shape lint as
:mod:`test_r7_silent_swallow_removal` so a future refactor that
re-introduces local-only commits surfaces here.
"""

from __future__ import annotations

import inspect

import pytest


@pytest.fixture(autouse=True)
def _isolated_pre_commit_registry():
    """Reset the process-wide pre-commit registry between tests so a
    callback registered by another test (e.g. the KG snapshot hook)
    does not run against this file's stub clients (which omit fields
    real callbacks read)."""

    from polymathera.colony.design_monorepo.commit_hooks import (
        reset_pre_commit_registry,
    )

    reset_pre_commit_registry()
    yield
    reset_pre_commit_registry()


def test_commit_all_and_push_helper_exists() -> None:
    from polymathera.colony.design_monorepo.capabilities import (
        _commit_all_and_push,
        _commit_paths_and_push,
    )
    assert callable(_commit_all_and_push)
    assert callable(_commit_paths_and_push)


@pytest.mark.asyncio
async def test_commit_all_and_push_commits_then_pushes() -> None:
    """Stub client records call order; verify commit happens before
    push, push fires when the commit produced new content, and
    push_status reflects success."""

    from polymathera.colony.design_monorepo.capabilities import (
        _commit_all_and_push,
    )

    calls: list[str] = []

    class _Repo:
        class _Head:
            class _Commit:
                hexsha = "PRIOR_HEAD"
            commit = _Commit()
        head = _Head()

    class _StubClient:
        _repo = _Repo()
        active_branch = "main"
        working_dir = None

        def commit_with_identity(self, identity, message, all_changes=False):  # noqa: ARG002
            calls.append("commit")
            return "NEW_SHA"

        def push(self, *, branch=None, remote="origin", with_tags=False):  # noqa: ARG002
            calls.append("push")

    sha, status = await _commit_all_and_push(_StubClient(), object(), "msg")
    assert sha == "NEW_SHA"
    assert status == "pushed"
    assert calls == ["commit", "push"]


@pytest.mark.asyncio
async def test_commit_all_and_push_skips_push_when_no_changes() -> None:
    """When ``commit_with_identity`` returns the current HEAD (no-op
    commit), the helper MUST NOT push — there's nothing new to share."""

    from polymathera.colony.design_monorepo.capabilities import (
        _commit_all_and_push,
    )

    push_calls = 0

    class _Repo:
        class _Head:
            class _Commit:
                hexsha = "HEAD_SHA"
            commit = _Commit()
        head = _Head()

    class _StubClient:
        _repo = _Repo()
        active_branch = "main"
        working_dir = None

        def commit_with_identity(self, identity, message, all_changes=False):  # noqa: ARG002
            return "HEAD_SHA"

        def push(self, *, branch=None, remote="origin", with_tags=False):  # noqa: ARG002
            nonlocal push_calls
            push_calls += 1

    sha, status = await _commit_all_and_push(_StubClient(), object(), "msg")
    assert sha == "HEAD_SHA"
    assert status == "no_commit"
    assert push_calls == 0


@pytest.mark.asyncio
async def test_commit_all_and_push_surfaces_push_failure_in_status() -> None:
    """The commit is NOT rolled back on push failure (it's already
    in the local history). The helper returns the local sha + a
    ``push_failed:...`` status so the caller can log a warning + the
    operator can retry push out-of-band via ``push_remote``."""

    from polymathera.colony.design_monorepo.capabilities import (
        _commit_all_and_push,
    )

    class _Repo:
        class _Head:
            class _Commit:
                hexsha = "PRIOR_HEAD"
            commit = _Commit()
        head = _Head()

    class _StubClient:
        _repo = _Repo()
        active_branch = "main"
        working_dir = None

        def commit_with_identity(self, identity, message, all_changes=False):  # noqa: ARG002
            return "NEW_SHA"

        def push(self, *, branch=None, remote="origin", with_tags=False):  # noqa: ARG002
            raise RuntimeError("network unreachable")

    sha, status = await _commit_all_and_push(_StubClient(), object(), "msg")
    assert sha == "NEW_SHA"
    assert status.startswith("push_failed:")
    assert "RuntimeError" in status
    assert "network unreachable" in status


def test_ingest_repo_map_literature_uses_push_helper() -> None:
    """R7-FIX-C site #1: source inspection pins that the ingest
    action routes commits through the push helper (not the
    local-only ``_commit_all``)."""

    from polymathera.colony.design_monorepo.capabilities import (
        RepoStateProvider,
    )

    src = inspect.getsource(RepoStateProvider.ingest_repo_map_literature)
    assert "_commit_all_and_push" in src, (
        "ingest_repo_map_literature no longer routes through the "
        "push helper — run7 forensic R7-FIX-C regression."
    )
    # The returned envelope must include push_status so the LLM /
    # operator sees whether the cache reached the remote.
    assert "push_status" in src


def test_checkpoint_state_uses_push_helper() -> None:
    """R7-FIX-C site #2: checkpoints are share-purpose; local-only
    is wrong."""

    from polymathera.colony.design_monorepo.capabilities import (
        DesignCheckpointer,
    )

    src = inspect.getsource(DesignCheckpointer.checkpoint_state)
    assert "_commit_all_and_push" in src


def test_auto_checkpoint_on_quiescence_uses_push_helper() -> None:
    """R7-FIX-C site #3: auto-checkpoint at convergence quiescence is
    the same share-purpose contract."""

    from polymathera.colony.design_monorepo.capabilities import (
        DesignCheckpointer,
    )

    src = inspect.getsource(DesignCheckpointer._on_quiescence)
    assert "_commit_all_and_push" in src


def test_commit_state_inline_uses_paths_push_helper() -> None:
    """R7-FIX-C site #4: LLM-callable ``commit_state`` action lives
    on :class:`DesignCheckpointer` (sibling of ``checkpoint_state``)."""

    from polymathera.colony.design_monorepo.capabilities import (
        DesignCheckpointer,
    )

    src = inspect.getsource(DesignCheckpointer.commit_state)
    assert "_commit_paths_and_push" in src


def test_dispatch_protected_op_commit_state_uses_paths_push_helper(
) -> None:
    """R7-FIX-C site #5: deferred-execution path for ``commit_state``
    after operator approval. Once approved, the operator's intent is
    to share the commit — the deferred execution must push too."""

    from polymathera.colony.design_monorepo.capabilities import (
        DesignCheckpointer,
    )

    src = inspect.getsource(DesignCheckpointer._dispatch_protected_op)
    # The helper is only used in the commit_state branch of this
    # dispatcher — confirm via substring presence.
    assert "_commit_paths_and_push" in src
