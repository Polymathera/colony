"""Tests for ``DesignMonorepoManifest`` IO + validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polymathera.colony.design_monorepo import (
    DEFAULT_SURFACE_DIRS,
    DesignMonorepoManifest,
    ExtensionsConfig,
    ImportedRemote,
    LFSConfig,
    MANIFEST_RELATIVE_PATH,
    MANIFEST_SCHEMA_VERSION,
    ManifestSchemaError,
    SurfaceConfig,
    WebhookConfig,
)


def _minimal_kwargs() -> dict[str, object]:
    return dict(
        tenant="acme",
        colony="acme-colony",
        program="prog-1",
        target_system="x",
        design_repo_url="https://example.com/repo.git",
    )


def test_round_trip_minimal(tmp_path: Path) -> None:
    m = DesignMonorepoManifest(**_minimal_kwargs())
    written = m.write_path(tmp_path)
    assert written == tmp_path / MANIFEST_RELATIVE_PATH
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert payload["program"] == "prog-1"
    assert payload["topology"] == "external"

    reloaded = DesignMonorepoManifest.load_path(tmp_path)
    assert reloaded == m


def test_imports_appended_idempotently() -> None:
    m = DesignMonorepoManifest(**_minimal_kwargs())
    extra = ImportedRemote(name="r1", url="https://x.test/a.git", ref="main")
    m1 = m.with_imports([extra])
    assert len(m1.imports_tooling_from) == 1
    m2 = m1.with_imports([extra])
    assert len(m2.imports_tooling_from) == 1


def test_topology_validates() -> None:
    with pytest.raises(Exception):
        DesignMonorepoManifest(topology="not-a-topology", **_minimal_kwargs())


def test_lfs_separate_url_required() -> None:
    with pytest.raises(Exception):
        LFSConfig(mode="separate", separate_url=None)
    cfg = LFSConfig(mode="separate", separate_url="https://lfs.example/")
    assert cfg.mode == "separate"


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestSchemaError):
        DesignMonorepoManifest.load_path(tmp_path)


def test_load_malformed_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / MANIFEST_RELATIVE_PATH
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not json}", encoding="utf-8")
    with pytest.raises(ManifestSchemaError):
        DesignMonorepoManifest.load_path(tmp_path)


def test_load_future_schema_raises(tmp_path: Path) -> None:
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION + 99,
        **_minimal_kwargs(),
    }
    p = tmp_path / MANIFEST_RELATIVE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestSchemaError):
        DesignMonorepoManifest.load_path(tmp_path)


def test_webhook_defaults() -> None:
    m = DesignMonorepoManifest(**_minimal_kwargs())
    assert isinstance(m.webhook, WebhookConfig)
    assert m.webhook.enabled is False
    assert m.webhook.endpoint is None


# ---------------------------------------------------------------------------
# L1-A: schema v2 ``extensions`` block — load v1 cleanly, round-trip v2.
# ---------------------------------------------------------------------------


def test_v1_manifest_loads_with_extensions_none(tmp_path: Path) -> None:
    """v1 on-disk manifests (pre-L1-A) MUST keep parsing — bumping the
    in-process schema_version to 2 does not invalidate stored v1
    manifests. ``extensions`` defaults to None for them."""
    payload = {"schema_version": 1, **_minimal_kwargs()}
    p = tmp_path / MANIFEST_RELATIVE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")

    loaded = DesignMonorepoManifest.load_path(tmp_path)
    assert loaded.schema_version == 1
    assert loaded.extensions is None


def test_v2_manifest_round_trip_with_default_extensions(tmp_path: Path) -> None:
    m = DesignMonorepoManifest(extensions=ExtensionsConfig(), **_minimal_kwargs())
    m.write_path(tmp_path)

    reloaded = DesignMonorepoManifest.load_path(tmp_path)
    assert reloaded.schema_version == MANIFEST_SCHEMA_VERSION == 2
    assert reloaded.extensions is not None
    assert reloaded.extensions.plugins.directory == DEFAULT_SURFACE_DIRS["plugins"]
    assert reloaded.extensions.agents.directory == DEFAULT_SURFACE_DIRS["agents"]
    assert reloaded.extensions.deployments.directory == DEFAULT_SURFACE_DIRS["deployments"]
    assert reloaded.extensions.tools.directory == DEFAULT_SURFACE_DIRS["tools"]
    assert reloaded.extensions.profiles.directory == DEFAULT_SURFACE_DIRS["profiles"]


def test_v2_manifest_per_surface_directory_override(tmp_path: Path) -> None:
    m = DesignMonorepoManifest(
        extensions=ExtensionsConfig(
            tools=SurfaceConfig(directory="vendor/tooling/"),
        ),
        **_minimal_kwargs(),
    )
    m.write_path(tmp_path)
    reloaded = DesignMonorepoManifest.load_path(tmp_path)
    assert reloaded.extensions.tools.directory == "vendor/tooling/"
    # Other surfaces keep defaults — overriding one does not strand the others.
    assert reloaded.extensions.plugins.directory == DEFAULT_SURFACE_DIRS["plugins"]


def test_extensions_block_rejects_unknown_surface(tmp_path: Path) -> None:
    """``extra='forbid'`` on ExtensionsConfig catches typos; future schema
    versions adding new surface kinds bump the schema_version."""
    payload = {
        "schema_version": 2,
        "extensions": {"plgins": {"directory": ".colony/plugins/"}},  # typo
        **_minimal_kwargs(),
    }
    p = tmp_path / MANIFEST_RELATIVE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception):  # pydantic ValidationError subclass
        DesignMonorepoManifest.load_path(tmp_path)


def test_surface_config_rejects_unknown_field() -> None:
    """Per-surface config locks unknown fields out — future per-surface
    knobs grow the SurfaceConfig schema explicitly."""
    with pytest.raises(Exception):
        SurfaceConfig.model_validate({"directory": ".colony/plugins/", "weird": 1})
