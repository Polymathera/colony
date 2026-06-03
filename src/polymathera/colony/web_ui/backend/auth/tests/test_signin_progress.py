"""Tests for the in-memory sign-in progress tracker."""

from __future__ import annotations

import asyncio

import pytest

from polymathera.colony.web_ui.backend.auth import signin_progress


@pytest.fixture(autouse=True)
def _reset() -> None:
    signin_progress.reset_for_testing()
    yield
    signin_progress.reset_for_testing()


def test_get_returns_none_when_not_started() -> None:
    assert signin_progress.get("never-started") is None


def test_start_creates_entry_with_initial_message() -> None:
    p = signin_progress.start("nonce-1")
    assert p.nonce == "nonce-1"
    assert p.messages == ["Starting…"]
    assert p.done is False
    assert p.error is None
    # And it's retrievable.
    same = signin_progress.get("nonce-1")
    assert same is p


def test_emit_appends_messages() -> None:
    signin_progress.start("nonce-1")
    signin_progress.emit("nonce-1", "Step 1")
    signin_progress.emit("nonce-1", "Step 2")
    p = signin_progress.get("nonce-1")
    assert p is not None
    assert p.messages == ["Starting…", "Step 1", "Step 2"]


def test_emit_no_op_when_nonce_absent() -> None:
    """Walker emit after the entry was purged: silent no-op, not
    KeyError. Lets the walker continue even if the polling client
    abandoned long ago."""
    signin_progress.emit("missing", "should not crash")
    assert signin_progress.get("missing") is None


@pytest.mark.asyncio
async def test_emit_no_op_after_done() -> None:
    """Once ``done`` is True, further emits don't append — the
    polling client has likely already redirected; appending stale
    text would just confuse anyone who scrolls back."""
    signin_progress.start("nonce-1")
    signin_progress.emit("nonce-1", "Before done")
    signin_progress.mark_done("nonce-1")
    signin_progress.emit("nonce-1", "After done")
    p = signin_progress.get("nonce-1")
    assert p is not None
    assert p.messages == ["Starting…", "Before done"]


@pytest.mark.asyncio
async def test_mark_done_with_error() -> None:
    signin_progress.start("nonce-1")
    signin_progress.mark_done("nonce-1", error="walker crashed")
    p = signin_progress.get("nonce-1")
    assert p is not None
    assert p.done is True
    assert p.error == "walker crashed"


def test_register_task_holds_reference() -> None:
    """The asyncio task ref MUST live in the module-level dict so
    the GC doesn't cancel a background walker mid-run. Verify the
    dict holds it."""
    loop = asyncio.new_event_loop()
    try:
        async def _noop():
            await asyncio.sleep(0)
        task = loop.create_task(_noop())
        signin_progress.register_task("nonce-1", task)
        assert signin_progress._tasks["nonce-1"] is task
        loop.run_until_complete(task)
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_done_triggers_delayed_cleanup() -> None:
    """``mark_done`` schedules a cleanup task. Hard to test the
    actual 60s TTL without time-mocking; this test just verifies
    a cleanup task gets created so the dict isn't permanent."""
    # Patch the TTL to something tiny so this test runs fast.
    signin_progress._TTL_AFTER_DONE_S = 0.05  # noqa: SLF001
    try:
        signin_progress.start("nonce-1")
        signin_progress.mark_done("nonce-1")
        # Give the cleanup task time to fire.
        await asyncio.sleep(0.15)
        assert signin_progress.get("nonce-1") is None
    finally:
        signin_progress._TTL_AFTER_DONE_S = 60.0  # noqa: SLF001
