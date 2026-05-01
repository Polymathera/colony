"""Shared fixtures for the design_monorepo tests."""

from __future__ import annotations

# Pytest's import-mode=importlib + namespace-package src layout can
# trigger a module to be loaded under two distinct dotted names (once
# as ``polymathera.colony.x`` and once as ``colony.x``). When the
# module registers a Prometheus collector at import time the second
# pass raises ``ValueError: Duplicated timeseries``. ``src/conftest.py``
# normally guards against this in ``pytest_configure``, but that hook
# fires *after* descendant conftests' imports — so the guard must be
# installed *before* this conftest pulls in the design-monorepo capability
# module (which transitively imports the autoscaler with the
# module-level Counter). Re-installing it module-locally is idempotent
# and matches the discipline in ``src/conftest.py``.
from prometheus_client import REGISTRY as _PROM_REGISTRY

_orig_register = _PROM_REGISTRY.register


def _safe_register(collector):
    try:
        _orig_register(collector)
    except ValueError:
        pass


_PROM_REGISTRY.register = _safe_register


# SQLAlchemy's MetaData rejects a duplicate Table[name] declaration. The
# same import-twice pattern triggers this for ``virtual_context_pages``
# and other models registered at module import time.
try:
    from sqlalchemy.sql.schema import Table as _SqlTable  # noqa: E402

    _orig_new = _SqlTable._new.__func__

    def _tolerant_new(cls, *args, **kw):
        if args:
            name = args[0]
            metadata = args[1] if len(args) > 1 else kw.get("metadata")
            if metadata is not None and name in metadata.tables:
                return metadata.tables[name]
        return _orig_new(cls, *args, **kw)

    _SqlTable._new = classmethod(_tolerant_new)
except Exception:  # noqa: BLE001 - sqlalchemy may not be importable
    pass


from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from polymathera.colony.design_monorepo import (  # noqa: E402
    AgentIdentity,
    DesignMonorepoManifest,
    bootstrap_design_monorepo,
)


@pytest.fixture
def manifest() -> DesignMonorepoManifest:
    return DesignMonorepoManifest(
        tenant="acme",
        colony="acme-colony",
        program="prog-test",
        target_system="test_system",
        topology="external",
        design_repo_url="file:///tmp/never-cloned-from",
    )


@pytest.fixture
def identity() -> AgentIdentity:
    return AgentIdentity(
        agent_id="agent_test_001",
        role="bootstrap",
        colony_id="acme-colony",
    )


@pytest.fixture
def fresh_repo_dir(tmp_path: Path) -> Path:
    return tmp_path / "design_repo"


@pytest.fixture
def bootstrapped_repo(
    manifest: DesignMonorepoManifest,
    identity: AgentIdentity,
    fresh_repo_dir: Path,
):
    """Yield a freshly-bootstrapped DesignMonorepoClient."""

    return bootstrap_design_monorepo(manifest, fresh_repo_dir, identity=identity)
