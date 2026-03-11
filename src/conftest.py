"""Root conftest for Colony test suite.

Handles Prometheus collector registry conflicts that occur when
modules with module-level Counter/Gauge definitions are imported
via different import paths during test collection.
"""

from prometheus_client import REGISTRY


def pytest_configure(config):
    """Patch Prometheus registry to silently skip duplicate registrations."""
    _original_register = REGISTRY.register

    def _safe_register(collector):
        try:
            _original_register(collector)
        except ValueError:
            pass  # Duplicate collector, ignore

    REGISTRY.register = _safe_register
