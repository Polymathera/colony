"""Tests for ``DataCurationCapability`` (dataset versioning + lineage +
content-hash audit). The federation surface was deleted in PR #1."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from polymathera.colony.agents.roles import (
    DataCurationCapability,
    UnknownDatasetVersionError,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def curator() -> DataCurationCapability:
    return DataCurationCapability(agent=None, scope_id="dc_test")


async def test_register_assigns_root_version(
    curator: DataCurationCapability,
) -> None:
    v = await curator.register_dataset(
        "papers", content_hash="abc", storage_uri="s3://x",
    )
    assert v.dataset_id == "papers"
    assert v.parent_version_id is None
    assert v.version_id == "papers@v1"


async def test_version_dataset_chains_lineage(
    curator: DataCurationCapability,
) -> None:
    v1 = await curator.register_dataset("ds", content_hash="a")
    v2 = await curator.version_dataset(
        v1.version_id, content_hash="b", transform_description="filtered",
    )
    assert v2.parent_version_id == v1.version_id
    edges = await curator.list_lineage(v2.version_id)
    assert len(edges) == 1
    assert edges[0].relation == "derived_from"


async def test_version_unknown_parent_raises(
    curator: DataCurationCapability,
) -> None:
    with pytest.raises(UnknownDatasetVersionError):
        await curator.version_dataset(
            "missing@v1", content_hash="x", transform_description="",
        )


async def test_list_versions(curator: DataCurationCapability) -> None:
    v1 = await curator.register_dataset("ds", content_hash="a")
    v2 = await curator.version_dataset(
        v1.version_id, content_hash="b", transform_description="t",
    )
    versions = await curator.list_versions("ds")
    assert [v.version_id for v in versions] == [v1.version_id, v2.version_id]


async def test_verify_content_hash_match(
    curator: DataCurationCapability, tmp_path: Path,
) -> None:
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello, world")
    digest = hashlib.sha256(b"hello, world").hexdigest()
    v = await curator.register_dataset(
        "x", content_hash=digest,
    )
    res = await curator.verify_content_hash(
        v.version_id, path=str(p),
    )
    assert res["ok"] is True
    assert res["expected"] == res["observed"]


async def test_verify_content_hash_mismatch(
    curator: DataCurationCapability, tmp_path: Path,
) -> None:
    p = tmp_path / "data.bin"
    p.write_bytes(b"different content")
    v = await curator.register_dataset("x", content_hash="ZERO")
    res = await curator.verify_content_hash(v.version_id, path=str(p))
    assert res["ok"] is False


