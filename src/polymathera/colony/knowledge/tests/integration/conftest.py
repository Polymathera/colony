"""Integration-test shared fixtures.

Mirrors the registry guards from the parent ``tests/conftest.py`` so
that running the integration subdirectory directly still applies the
same Prometheus / SQLAlchemy duplicate-import safety net.
"""

from __future__ import annotations

import os

import pytest

from prometheus_client import REGISTRY as _PROM_REGISTRY


_orig_register = _PROM_REGISTRY.register


def _safe_register(collector):
    try:
        _orig_register(collector)
    except ValueError:
        pass


_PROM_REGISTRY.register = _safe_register

try:
    from sqlalchemy.sql.schema import Table as _SqlTable

    _orig_new = _SqlTable._new.__func__

    def _tolerant_new(cls, *args, **kw):
        if args:
            name = args[0]
            metadata = args[1] if len(args) > 1 else kw.get("metadata")
            if metadata is not None and name in metadata.tables:
                return metadata.tables[name]
        return _orig_new(cls, *args, **kw)

    _SqlTable._new = classmethod(_tolerant_new)
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------------------
# Env-var-gated skip helpers
# ---------------------------------------------------------------------------


QDRANT_URL_ENV = "POLYMATHERA_QDRANT_URL"
GROBID_URL_ENV = "POLYMATHERA_GROBID_URL"


def qdrant_url() -> str | None:
    return os.environ.get(QDRANT_URL_ENV)


def grobid_url() -> str | None:
    return os.environ.get(GROBID_URL_ENV)


@pytest.fixture
def qdrant_url_or_skip() -> str:
    url = qdrant_url()
    if not url:
        pytest.skip(
            f"Set {QDRANT_URL_ENV} (e.g., 'http://localhost:6333') to run "
            "Qdrant integration tests.",
        )
    return url


@pytest.fixture
def grobid_url_or_skip() -> str:
    url = grobid_url()
    if not url:
        pytest.skip(
            f"Set {GROBID_URL_ENV} (e.g., 'http://localhost:8070') to run "
            "GROBID integration tests.",
        )
    return url
