"""L1-A integration test against ``monorepo_opm_meg/``.

Runs the actual discovery + migration code against the real OPM-MEG
design monorepo checkout (not a fabricated fixture). Today the monorepo
is content-only (a v1 manifest plus ``kb/``); there are no
``.colony/<surface>/`` directories, so discovery returns empty
containers. The migration test runs on a copy so the real monorepo's
manifest is never mutated.

Skipped automatically when the monorepo isn't present (CI without the
sibling clone).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from polymathera.colony.design_monorepo import (
    MANIFEST_RELATIVE_PATH,
    MANIFEST_SCHEMA_VERSION,
    DesignMonorepoManifest,
)
from polymathera.colony.design_monorepo.extensions import discover_all
from polymathera.colony.design_monorepo.manifest_migrate import migrate_manifest


def _opm_meg_root() -> Path | None:
    """Resolve the OPM-MEG monorepo: env var first, then sibling-clone
    convention next to the colony repo. ``None`` when neither exists —
    the calling test skips."""
    env = os.environ.get("POLYMATHERA_OPM_MEG_MONOREPO")
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_dir() else None
    # Sibling layout: <workspace>/colony/ + <workspace>/monorepo_opm_meg/.
    # This file lives at colony/src/polymathera/colony/design_monorepo/tests/
    # — six ``..`` up reaches the colony repo root; one more reaches the
    # workspace containing colony/ and monorepo_opm_meg/ as siblings.
    here = Path(__file__).resolve()
    workspace = here.parents[6]
    candidate = workspace / "monorepo_opm_meg"
    return candidate if candidate.is_dir() else None


@pytest.fixture(scope="module")
def opm_meg_root() -> Path:
    root = _opm_meg_root()
    if root is None:
        pytest.skip(
            "monorepo_opm_meg/ not found — set POLYMATHERA_OPM_MEG_MONOREPO or "
            "clone it as a sibling of the colony repo to run this test."
        )
    return root


def test_manifest_loads_from_real_monorepo(opm_meg_root: Path) -> None:
    """The OPM-MEG manifest still loads after the v1→v2 schema bump."""
    manifest = DesignMonorepoManifest.load_path(opm_meg_root)
    # Manifest is v1 today; bump to v2 lives in the migration utility.
    assert manifest.schema_version in (1, MANIFEST_SCHEMA_VERSION)
    assert manifest.program  # non-empty identity fields


def test_discover_all_against_real_monorepo_is_empty(opm_meg_root: Path) -> None:
    """Today: no ``.colony/<surface>/`` dirs in the monorepo. Discovery
    returns every surface empty — the post-bootstrap baseline §2.4 of
    the alignment plan codifies. This test starts passing populated
    results as soon as the OPM-MEG L1-E acceptance run lands content."""
    snap = discover_all(opm_meg_root)
    assert snap.plugins == []
    assert snap.agents == {}
    assert snap.deployments == {}
    assert len(snap.tools) == 0
    assert snap.profiles == {}


def test_migrate_manifest_on_copy_bumps_to_v2(opm_meg_root: Path, tmp_path: Path) -> None:
    """Migrate against a copy — the real monorepo's manifest must NOT be
    mutated by the test suite. Verifies the v1→v2 bump end-to-end against
    a real manifest, not a fabricated one."""
    shutil.copytree(
        opm_meg_root / ".colony",
        tmp_path / ".colony",
    )
    result = migrate_manifest(tmp_path)
    # If the source was already v2 (someone pre-migrated), this is a
    # no-op; if it was v1, the migrator bumped it.
    if result.was_migrated:
        assert result.from_version == 1
        assert result.to_version == MANIFEST_SCHEMA_VERSION == 2
    reloaded = DesignMonorepoManifest.load_path(tmp_path)
    assert reloaded.schema_version == MANIFEST_SCHEMA_VERSION
    # Default extensions block populated regardless of which branch
    # (migrated or already-v2) we took.
    assert reloaded.extensions is not None
