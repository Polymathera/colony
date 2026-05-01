"""Tests for ``LocalFsWatcher``.

These tests use the polling fallback (we set
``LocalFsWatcherConfig(poll_interval_s=0.05)``) by importing the
watcher and mocking out ``watchdog`` so we don't depend on the OS's
inotify quirks in CI.
"""

from __future__ import annotations

import asyncio
import builtins
from pathlib import Path

import pytest

from polymathera.colony.vcm.watchers import LocalFsWatcher, LocalFsWatcherConfig


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _force_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the polling fallback by hiding ``watchdog`` from imports
    inside the watcher module."""

    real_import = builtins.__import__

    def _import(name: str, *args, **kwargs):
        if name.startswith("watchdog"):
            raise ImportError("forced")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)


async def _drain(watcher, max_seconds: float = 1.5) -> list:
    events = []
    deadline = asyncio.get_running_loop().time() + max_seconds
    async for event in watcher.watch():
        events.append(event)
        if asyncio.get_running_loop().time() > deadline:
            watcher.stop()
            break
    return events


async def _collect_for(watcher, seconds: float) -> list:
    out = []
    task = asyncio.create_task(_drain(watcher, max_seconds=seconds))
    await asyncio.sleep(seconds)
    watcher.stop()
    return await task


async def test_detects_new_file(tmp_path: Path) -> None:
    config = LocalFsWatcherConfig(poll_interval_s=0.05, debounce_s=0.05)
    watcher = LocalFsWatcher(
        root=tmp_path, scope_id="s", source_uri="file://x", config=config,
    )

    async def writer():
        await asyncio.sleep(0.1)
        (tmp_path / "a.txt").write_text("hello", encoding="utf-8")

    writer_task = asyncio.create_task(writer())
    events = await _collect_for(watcher, 0.6)
    await writer_task
    paths = {e.extra["relative_path"] for e in events}
    assert "a.txt" in paths


async def test_detects_modification(tmp_path: Path) -> None:
    p = tmp_path / "b.txt"
    p.write_text("v0", encoding="utf-8")
    config = LocalFsWatcherConfig(poll_interval_s=0.05, debounce_s=0.05)
    watcher = LocalFsWatcher(
        root=tmp_path, scope_id="s", source_uri="file://x", config=config,
    )

    async def writer():
        await asyncio.sleep(0.15)
        # Write enough later that the mtime distinguishes; touch() is
        # not reliable across filesystems for this.
        import os, time
        time.sleep(0.05)
        p.write_text("v1", encoding="utf-8")
        os.utime(p, None)

    writer_task = asyncio.create_task(writer())
    events = await _collect_for(watcher, 0.7)
    await writer_task
    kinds = [(e.kind.value, e.extra["relative_path"]) for e in events]
    assert any(k == "page_replaced" and p == "b.txt" for k, p in kinds)


async def test_path_filter_excludes_dotfiles(tmp_path: Path) -> None:
    config = LocalFsWatcherConfig(poll_interval_s=0.05, debounce_s=0.05)
    watcher = LocalFsWatcher(
        root=tmp_path, scope_id="s", source_uri="file://x", config=config,
    )

    async def writer():
        await asyncio.sleep(0.1)
        (tmp_path / ".secret").write_text("v", encoding="utf-8")
        (tmp_path / "ok.txt").write_text("v", encoding="utf-8")

    writer_task = asyncio.create_task(writer())
    events = await _collect_for(watcher, 0.5)
    await writer_task
    paths = {e.extra["relative_path"] for e in events}
    assert "ok.txt" in paths
    assert ".secret" not in paths


async def test_root_must_exist(tmp_path: Path) -> None:
    watcher = LocalFsWatcher(
        root=tmp_path / "missing", scope_id="s", source_uri="file://x",
    )
    with pytest.raises(FileNotFoundError):
        async for _ in watcher.watch():
            break
