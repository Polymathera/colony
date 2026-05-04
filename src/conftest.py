"""Root conftest for Colony test suite.

Handles Prometheus collector registry conflicts AND SQLAlchemy table
redefinition conflicts that both occur when a module with module-level
side effects gets imported via different import paths during test
collection (e.g., once as ``polymathera.colony.vcm.models`` and once as
``colony.vcm.models`` from the src-layout namespace package).
"""

import os

from prometheus_client import REGISTRY


# Test-only defaults for env vars that ConfigComponents tag as required
# (``json_schema_extra={"env": "..."}`` without ``optional: True``).
# When any test transitively imports the registering module *and* later
# calls ``ConfigurationManager.initialize()`` without a YAML config path,
# ``PolymatheraConfig()`` eagerly instantiates every registered component
# and ``_read_env_vars`` raises if these are unset. Production deployments
# always set these (Docker compose env, EKS env), so these defaults only
# affect the test process. ``setdefault`` preserves whatever a developer
# already has exported.
_TEST_ENV_DEFAULTS = {
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "OBJECT_STORAGE_BACKEND": "disabled",
    "JSON_STORAGE_BACKEND": "local",
    "JSON_STORAGE_LOCAL_PATH": "/tmp/colony_test_json_storage",
    "GIT_COLD_STORAGE_ENABLED": "false",
    "DISTRIBUTED_FS_BACKEND": "local",
    "LOCAL_FS_ROOT_PATH": "/tmp/colony_test_local_fs",
    "AUTH_ENABLED": "false",
    "RELATIONAL_STORAGE_BACKEND": "local",
    "RDS_USER": "test",
    "RDS_PASSWORD": "test",
    "RDS_HOST": "localhost",
    "RDS_PORT": "5432",
    "RDS_DB_NAME": "test",
    "SLACK_ENABLED": "false",
    "GITHUB_TOKEN": "",
    "GITLAB_TOKEN": "",
}
for _k, _v in _TEST_ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)


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
