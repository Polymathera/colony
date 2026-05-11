"""``ImageStore`` ABC + two implementations.

The image store is the multimodal sibling of the corpus-wide
:class:`~polymathera.colony.knowledge.stores.vector.VectorStore`.
Layout-aware PDF readers (Marker, Docling, MinerU, Mistral OCR,
Anthropic native PDF) extract figure / table / diagram bytes
alongside section text; the store persists those bytes
content-addressed so the rest of the pipeline can refer to them by
URI without dragging raw bytes through the chunk pipeline.

Two implementations:

- :class:`InMemoryImageStore` — in-process ``dict`` keyed by URI.
  Used by tests and single-process deployments.
- :class:`LocalFsImageStore` — writes to a sharded directory tree on
  the shared volume (``/mnt/shared/colony-images/<sha[:2]>/<sha>``)
  with a sidecar ``.json`` carrying mime + size. Picked when
  ``knowledge.image_dir`` is set in the operator
  YAML (see
  :func:`~polymathera.colony.knowledge.deps._default_image_store`).

The contract is intentionally narrow — ``put`` is the only
mutating call; everything else is pure read. The store is **not**
ACID and the operator is expected not to mutate the underlying
storage out from under it; ingestion is the only writer in the
designed flow.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...agents.blueprint import blueprint


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URI scheme
# ---------------------------------------------------------------------------
#
# The store uses a single URI scheme so consumers (chunkers, retrieval
# adapters, the KB router that resolves images for the dashboard) can
# round-trip references without knowing which backend is active. The
# format is intentionally flat:
#
#     colony-image://<sha256-hex>
#
# No path component, no query string. The mime type is recorded
# alongside the bytes — callers that need it ask via :meth:`stat`.

URI_SCHEME = "colony-image"


class ImageStoreError(RuntimeError):
    """Base error for the image store."""


def _is_colony_image_uri(uri: str) -> bool:
    return uri.startswith(f"{URI_SCHEME}://")


def _sha_from_uri(uri: str) -> str:
    if not _is_colony_image_uri(uri):
        raise ImageStoreError(
            f"not a colony-image URI: {uri!r} (expected scheme '{URI_SCHEME}')",
        )
    return uri[len(URI_SCHEME) + 3 :]  # +3 for "://"


def _build_uri(sha: str) -> str:
    return f"{URI_SCHEME}://{sha}"


# A tight allow-list — we only persist what readers actually emit.
# Adding a new mime is a one-line change here; refusing to silently
# accept an unknown mime keeps callers honest about extension/mime
# alignment.
_MIME_TO_EXT: Mapping[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/svg+xml": ".svg",
}


def _ext_for_mime(mime: str) -> str:
    """Return a canonical filename extension for a mime type.

    Falls back to ``.bin`` for shapes the store does not specially
    recognise — readers are encouraged to pass an exact mime, but a
    one-off datasheet schematic in a custom format should not crash
    ingestion.
    """

    return _MIME_TO_EXT.get(mime.lower(), ".bin")


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class ImageStore(ABC):
    """Content-addressed binary store for figure / table / diagram bytes."""

    @abstractmethod
    async def put(self, payload: bytes, *, mime: str = "image/png") -> str:
        """Persist ``payload`` and return its content-addressed URI.

        ``mime`` is recorded so :meth:`stat` and the dashboard's
        figure-resolve endpoint can round-trip the right
        ``Content-Type``. Calls with the same payload are idempotent —
        duplicate bytes return the same URI without rewriting the
        underlying file.
        """

    @abstractmethod
    async def get(self, uri: str) -> bytes | None:
        """Resolve a URI returned by :meth:`put` to its bytes.

        Returns ``None`` for missing URIs (rather than raising) so
        the dashboard can render a placeholder for an evicted /
        operator-deleted figure without crashing the chat panel.
        """

    @abstractmethod
    async def has(self, uri: str) -> bool: ...

    @abstractmethod
    async def delete(self, uri: str) -> bool:
        """Best-effort delete. Returns ``True`` iff bytes were removed."""

    @abstractmethod
    async def stat(self, uri: str) -> dict[str, Any] | None:
        """Return ``{"size": int, "mime": str, "created_at": iso8601}``
        or ``None`` for missing URIs."""


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------


@blueprint
class InMemoryImageStore(ImageStore):
    """Process-local image store. Drops bytes on shutdown."""

    def __init__(self) -> None:
        self._items: dict[str, bytes] = {}
        self._meta: dict[str, dict[str, Any]] = {}

    async def put(self, payload: bytes, *, mime: str = "image/png") -> str:
        if not isinstance(payload, (bytes, bytearray)):
            raise ImageStoreError(
                f"InMemoryImageStore.put: payload must be bytes, got "
                f"{type(payload).__name__}",
            )
        sha = hashlib.sha256(payload).hexdigest()
        uri = _build_uri(sha)
        if uri not in self._items:
            self._items[uri] = bytes(payload)
            self._meta[uri] = {
                "size": len(payload),
                "mime": mime,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        return uri

    async def get(self, uri: str) -> bytes | None:
        return self._items.get(uri)

    async def has(self, uri: str) -> bool:
        return uri in self._items

    async def delete(self, uri: str) -> bool:
        existed = uri in self._items
        self._items.pop(uri, None)
        self._meta.pop(uri, None)
        return existed

    async def stat(self, uri: str) -> dict[str, Any] | None:
        meta = self._meta.get(uri)
        return dict(meta) if meta is not None else None


# ---------------------------------------------------------------------------
# Local-FS store
# ---------------------------------------------------------------------------


@blueprint
class LocalFsImageStore(ImageStore):
    """Image store backed by a sharded local filesystem tree.

    Layout::

        <root>/
          <sha[:2]>/
            <sha>.png
            <sha>.png.meta.json    # {"size": int, "mime": str, "created_at": iso}

    The two-character sha prefix shards bytes across at most 256
    directories so a single ``ls`` does not hang on huge corpora.
    Writes are atomic via ``tempfile + os.rename`` (POSIX atomic
    rename within the same filesystem); duplicate puts short-circuit
    on the existence check rather than rewriting.

    The store is safe for one writer + many concurrent readers across
    processes (the typical Colony layout: ingest runs on a worker,
    the dashboard reads). Concurrent writers of the same payload
    are also safe — both produce the same target path and the rename
    is atomic. Concurrent writers of *different* payloads at the
    same path are impossible because the path is content-addressed.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)
        # Lazy mkdir on first put — keeps the constructor cheap and
        # avoids surprising the operator with a created directory
        # tree just because the store was instantiated.
        self._ready = False
        # asyncio.Lock is per-loop and can't safely cross loops, so
        # use a thread lock for the cheap "ensure_root" check; the
        # heavy work is in worker threads via to_thread anyway.
        self._init_lock = threading.Lock()

    @property
    def root(self) -> Path:
        return self._root

    def _ensure_root(self) -> None:
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return
            self._root.mkdir(parents=True, exist_ok=True)
            self._ready = True

    def _path_for(self, sha: str, mime: str) -> Path:
        ext = _ext_for_mime(mime)
        return self._root / sha[:2] / f"{sha}{ext}"

    def _meta_path_for(self, payload_path: Path) -> Path:
        return payload_path.with_name(payload_path.name + ".meta.json")

    def _resolve(self, uri: str) -> tuple[Path, Path] | None:
        """Locate the on-disk payload for ``uri``.

        Returns ``(payload_path, meta_path)`` if found, or ``None``.
        The mime type is read from the meta sidecar so we can pick
        the right extension; fall back to scanning the shard if the
        sidecar is missing (operator manual edit, partial write).
        """

        try:
            sha = _sha_from_uri(uri)
        except ImageStoreError:
            return None
        shard = self._root / sha[:2]
        if not shard.exists():
            return None
        # Fast path: any file whose stem equals the sha is the payload.
        for entry in shard.iterdir():
            if entry.name.startswith(sha + ".") and not entry.name.endswith(".meta.json"):
                return entry, self._meta_path_for(entry)
        return None

    async def put(self, payload: bytes, *, mime: str = "image/png") -> str:
        if not isinstance(payload, (bytes, bytearray)):
            raise ImageStoreError(
                f"LocalFsImageStore.put: payload must be bytes, got "
                f"{type(payload).__name__}",
            )

        sha = hashlib.sha256(payload).hexdigest()
        uri = _build_uri(sha)
        target = self._path_for(sha, mime)
        meta_target = self._meta_path_for(target)

        def _write_sync() -> None:
            self._ensure_root()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and meta_target.exists():
                return  # idempotent — same bytes already on disk
            # Atomic write via tempfile + rename. We write into the
            # shard directory itself (not /tmp) so the rename stays
            # within one filesystem and is atomic on POSIX.
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_bytes(bytes(payload))
            tmp.replace(target)
            meta_payload = {
                "size": len(payload),
                "mime": mime,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_tmp = meta_target.with_suffix(meta_target.suffix + ".tmp")
            meta_tmp.write_text(json.dumps(meta_payload), encoding="utf-8")
            meta_tmp.replace(meta_target)

        await asyncio.to_thread(_write_sync)
        return uri

    async def get(self, uri: str) -> bytes | None:
        def _read_sync() -> bytes | None:
            paths = self._resolve(uri)
            if paths is None:
                return None
            payload_path, _ = paths
            try:
                return payload_path.read_bytes()
            except FileNotFoundError:
                return None

        return await asyncio.to_thread(_read_sync)

    async def has(self, uri: str) -> bool:
        return await asyncio.to_thread(lambda: self._resolve(uri) is not None)

    async def delete(self, uri: str) -> bool:
        def _delete_sync() -> bool:
            paths = self._resolve(uri)
            if paths is None:
                return False
            payload_path, meta_path = paths
            existed = False
            try:
                payload_path.unlink()
                existed = True
            except FileNotFoundError:
                pass
            try:
                meta_path.unlink()
            except FileNotFoundError:
                pass
            return existed

        return await asyncio.to_thread(_delete_sync)

    async def stat(self, uri: str) -> dict[str, Any] | None:
        def _stat_sync() -> dict[str, Any] | None:
            paths = self._resolve(uri)
            if paths is None:
                return None
            payload_path, meta_path = paths
            if meta_path.exists():
                try:
                    return json.loads(meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "LocalFsImageStore.stat: malformed meta sidecar %s: %s",
                        meta_path, exc,
                    )
            # Sidecar missing or unreadable — synthesise minimal stats
            # from the file itself so the caller gets something
            # useful instead of None.
            try:
                size = payload_path.stat().st_size
            except FileNotFoundError:
                return None
            return {
                "size": size,
                "mime": "application/octet-stream",
                "created_at": None,
            }

        return await asyncio.to_thread(_stat_sync)


__all__ = (
    "URI_SCHEME",
    "ImageStore",
    "ImageStoreError",
    "InMemoryImageStore",
    "LocalFsImageStore",
)
