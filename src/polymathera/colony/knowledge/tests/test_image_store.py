"""Tests for the multimodal :class:`ImageStore` family.

Covers the contract every store must satisfy (idempotent put, miss
returns ``None`` rather than raising, atomic delete) plus the
filesystem-specific bits (sharded layout, sidecar metadata, atomic
rename, malformed-meta tolerance).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polymathera.colony.knowledge.stores.image import (
    URI_SCHEME,
    ImageStoreError,
    InMemoryImageStore,
    LocalFsImageStore,
    _build_uri,
    _ext_for_mime,
    _is_colony_image_uri,
    _sha_from_uri,
)


pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# URI helpers — round-trip + error cases
# ---------------------------------------------------------------------------


def test_uri_round_trip() -> None:
    uri = _build_uri("abc123")
    assert _is_colony_image_uri(uri)
    assert _sha_from_uri(uri) == "abc123"
    assert uri.startswith(f"{URI_SCHEME}://")


def test_sha_from_non_colony_uri_raises() -> None:
    with pytest.raises(ImageStoreError):
        _sha_from_uri("https://example.com/foo.png")


def test_ext_for_known_and_unknown_mimes() -> None:
    assert _ext_for_mime("image/png") == ".png"
    assert _ext_for_mime("image/JPEG") == ".jpg"
    assert _ext_for_mime("image/webp") == ".webp"
    assert _ext_for_mime("application/octet-stream") == ".bin"


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


async def test_in_memory_put_idempotent_and_round_trip() -> None:
    store = InMemoryImageStore()
    uri1 = await store.put(b"hello", mime="image/png")
    uri2 = await store.put(b"hello", mime="image/png")
    assert uri1 == uri2  # idempotent — same bytes, same URI
    assert await store.get(uri1) == b"hello"
    assert await store.has(uri1)


async def test_in_memory_distinct_payloads_distinct_uris() -> None:
    store = InMemoryImageStore()
    a = await store.put(b"alpha")
    b = await store.put(b"beta")
    assert a != b


async def test_in_memory_get_missing_returns_none() -> None:
    store = InMemoryImageStore()
    assert await store.get(_build_uri("deadbeef")) is None
    assert not await store.has(_build_uri("deadbeef"))


async def test_in_memory_stat_returns_size_mime_created() -> None:
    store = InMemoryImageStore()
    uri = await store.put(b"abcd", mime="image/jpeg")
    info = await store.stat(uri)
    assert info is not None
    assert info["size"] == 4
    assert info["mime"] == "image/jpeg"
    assert isinstance(info["created_at"], str) and "T" in info["created_at"]


async def test_in_memory_delete() -> None:
    store = InMemoryImageStore()
    uri = await store.put(b"x")
    assert await store.delete(uri) is True
    assert await store.delete(uri) is False  # second delete is a no-op
    assert await store.get(uri) is None


async def test_in_memory_rejects_non_bytes() -> None:
    store = InMemoryImageStore()
    with pytest.raises(ImageStoreError):
        await store.put("not bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Local-FS store
# ---------------------------------------------------------------------------


async def test_local_fs_put_round_trip(tmp_path: Path) -> None:
    store = LocalFsImageStore(root_dir=tmp_path)
    uri = await store.put(b"hello world", mime="image/png")
    assert await store.get(uri) == b"hello world"
    assert await store.has(uri)
    info = await store.stat(uri)
    assert info is not None
    assert info["size"] == 11
    assert info["mime"] == "image/png"


async def test_local_fs_sharded_layout(tmp_path: Path) -> None:
    """Files land under <root>/<sha[:2]>/<sha>.<ext>; sidecar next
    to the payload, not in a separate dir."""
    store = LocalFsImageStore(root_dir=tmp_path)
    uri = await store.put(b"shard-me", mime="image/jpeg")
    sha = _sha_from_uri(uri)
    payload_path = tmp_path / sha[:2] / f"{sha}.jpg"
    meta_path = tmp_path / sha[:2] / f"{sha}.jpg.meta.json"
    assert payload_path.is_file()
    assert meta_path.is_file()
    parsed = json.loads(meta_path.read_text(encoding="utf-8"))
    assert parsed["mime"] == "image/jpeg"
    assert parsed["size"] == 8


async def test_local_fs_idempotent_put_does_not_rewrite(tmp_path: Path) -> None:
    """Re-putting the same bytes is a no-op (no rename, no rewrite)."""
    store = LocalFsImageStore(root_dir=tmp_path)
    uri = await store.put(b"once")
    sha = _sha_from_uri(uri)
    payload = tmp_path / sha[:2] / f"{sha}.png"
    first_mtime = payload.stat().st_mtime_ns
    # Same bytes again — should short-circuit.
    uri2 = await store.put(b"once")
    assert uri == uri2
    second_mtime = payload.stat().st_mtime_ns
    assert first_mtime == second_mtime


async def test_local_fs_get_missing_returns_none(tmp_path: Path) -> None:
    store = LocalFsImageStore(root_dir=tmp_path)
    assert await store.get(_build_uri("aa" * 32)) is None
    assert not await store.has(_build_uri("aa" * 32))


async def test_local_fs_stat_falls_back_when_meta_missing(tmp_path: Path) -> None:
    """Operator deleted the sidecar by hand — stat still returns
    something useful instead of None."""
    store = LocalFsImageStore(root_dir=tmp_path)
    uri = await store.put(b"orphan", mime="image/png")
    sha = _sha_from_uri(uri)
    meta_path = tmp_path / sha[:2] / f"{sha}.png.meta.json"
    meta_path.unlink()
    info = await store.stat(uri)
    assert info is not None
    assert info["size"] == len(b"orphan")
    assert info["mime"] == "application/octet-stream"  # synthesised fallback


async def test_local_fs_stat_tolerates_corrupt_meta(tmp_path: Path) -> None:
    """A malformed sidecar is logged and stat falls back to file-stat."""
    store = LocalFsImageStore(root_dir=tmp_path)
    uri = await store.put(b"x", mime="image/png")
    sha = _sha_from_uri(uri)
    meta_path = tmp_path / sha[:2] / f"{sha}.png.meta.json"
    meta_path.write_text("not json", encoding="utf-8")
    info = await store.stat(uri)
    assert info is not None and info["size"] == 1


async def test_local_fs_delete(tmp_path: Path) -> None:
    store = LocalFsImageStore(root_dir=tmp_path)
    uri = await store.put(b"y", mime="image/png")
    assert await store.delete(uri) is True
    assert await store.get(uri) is None
    # Sidecar also gone
    sha = _sha_from_uri(uri)
    assert not (tmp_path / sha[:2] / f"{sha}.png.meta.json").exists()


async def test_local_fs_rejects_non_bytes(tmp_path: Path) -> None:
    store = LocalFsImageStore(root_dir=tmp_path)
    with pytest.raises(ImageStoreError):
        await store.put("not bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Blueprint cross-Ray pickle path
# ---------------------------------------------------------------------------


def test_local_fs_blueprint_pickles(tmp_path: Path) -> None:
    """The blueprint is picklable so the dashboard can ship it across
    the Ray boundary into capability constructors."""
    import pickle

    bp = LocalFsImageStore.bind(root_dir=str(tmp_path))
    raw = pickle.dumps(bp)
    bp2 = pickle.loads(raw)
    inst = bp2.local_instance()
    assert isinstance(inst, LocalFsImageStore)
    assert inst.root == tmp_path


def test_in_memory_blueprint_pickles() -> None:
    import pickle

    bp = InMemoryImageStore.bind()
    pickle.loads(pickle.dumps(bp)).local_instance()
