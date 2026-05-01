"""Shared guards for the convergence-runtime tests.

Pytest's import-mode=importlib + namespace-package src layout can load
modules under two dotted names; modules with module-level Prometheus
``Counter()`` registration or SQLAlchemy ``Table`` declarations then
fail on the second import. ``src/conftest.py`` patches the registries
in ``pytest_configure``, but that hook fires after descendant
conftests' imports — so we install the same guards module-locally here.
"""

from __future__ import annotations

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
