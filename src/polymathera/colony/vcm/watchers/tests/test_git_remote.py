"""Tests for ``GitRemoteWatcher`` using a real local git pair as the remote."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from polymathera.colony.vcm.watchers import (
    GitRemoteWatcher,
    GitRemoteWatcherConfig,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def remote_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    """Build a small bare-remote + working clone pair in tmp_path."""

    from git import Actor, Repo

    remote = tmp_path / "remote.git"
    Repo.init(str(remote), bare=True)

    # Build a working repo, push it to the bare remote.
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    repo = Repo.init(str(upstream), initial_branch="main")
    actor = Actor("Test", "test@example")
    (upstream / "a.txt").write_text("v0", encoding="utf-8")
    repo.git.add("-A")
    repo.index.commit("init", author=actor, committer=actor)
    repo.create_remote("origin", str(remote))
    repo.git.push("origin", "main")

    # Now clone into a local path that the watcher will track.
    clone = tmp_path / "clone"
    Repo.clone_from(str(remote), str(clone))

    return upstream, clone


async def test_detects_new_commit_added_file(
    remote_and_clone: tuple[Path, Path],
) -> None:
    from git import Actor

    upstream, clone = remote_and_clone

    watcher = GitRemoteWatcher(
        repo_path=clone, scope_id="prog",
        source_uri="git:test",
        config=GitRemoteWatcherConfig(
            poll_interval_s=0.1, branch="main",
        ),
    )

    async def push_change():
        await asyncio.sleep(0.2)
        from git import Repo
        upstream_repo = Repo(str(upstream))
        actor = Actor("Test", "test@example")
        (upstream / "b.txt").write_text("v0", encoding="utf-8")
        upstream_repo.git.add("-A")
        upstream_repo.index.commit("add b", author=actor, committer=actor)
        upstream_repo.git.push("origin", "main")

    push_task = asyncio.create_task(push_change())
    events: list = []

    async def collect():
        async for event in watcher.watch():
            events.append(event)
            if events:
                watcher.stop()
                break

    try:
        await asyncio.wait_for(collect(), timeout=3.0)
    finally:
        watcher.stop()
        await push_task

    paths = [e.extra["relative_path"] for e in events]
    assert "b.txt" in paths
    assert events[0].kind.value == "page_added"


async def test_no_change_no_events(
    remote_and_clone: tuple[Path, Path],
) -> None:
    """Without changes upstream, the watcher emits nothing."""

    _upstream, clone = remote_and_clone

    watcher = GitRemoteWatcher(
        repo_path=clone, scope_id="prog",
        source_uri="git:test",
        config=GitRemoteWatcherConfig(poll_interval_s=0.1, branch="main"),
    )

    events: list = []

    async def collect():
        async for event in watcher.watch():
            events.append(event)

    task = asyncio.create_task(collect())
    await asyncio.sleep(0.4)
    watcher.stop()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except asyncio.TimeoutError:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert events == []
