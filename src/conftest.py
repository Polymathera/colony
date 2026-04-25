"""Root conftest for Colony test suite.

Handles Prometheus collector registry conflicts AND SQLAlchemy table
redefinition conflicts that both occur when a module with module-level
side effects gets imported via different import paths during test
collection (e.g., once as ``polymathera.colony.vcm.models`` and once as
``colony.vcm.models`` from the src-layout namespace package).
"""

from prometheus_client import REGISTRY


def pytest_configure(config):
    """Make Prometheus + SQLAlchemy global registries idempotent.

    Neither fix changes runtime behaviour: in a deployed process a model
    module is loaded exactly once. The tolerant path only matters during
    pytest collection where the same module file can appear under two
    import names.
    """
    _original_register = REGISTRY.register

    def _safe_register(collector):
        try:
            _original_register(collector)
        except ValueError:
            pass  # Duplicate collector, ignore

    REGISTRY.register = _safe_register

    try:
        from sqlalchemy.sql.schema import Table
    except ImportError:
        return

    # ``Table._new`` is a classmethod; ``Table._new.__func__`` is the raw
    # underlying function that takes ``cls`` as its first argument.
    _original_new_func = Table._new.__func__

    def _tolerant_new(cls, *args, **kw):
        # If a table with this name already exists on the MetaData,
        # return it instead of erroring. SQLAlchemy's own ``extend_existing``
        # flag does the same thing at the single-table level.
        if args:
            name = args[0]
            metadata = args[1] if len(args) > 1 else kw.get("metadata")
            if metadata is not None and name in metadata.tables:
                return metadata.tables[name]
        return _original_new_func(cls, *args, **kw)

    Table._new = classmethod(_tolerant_new)
