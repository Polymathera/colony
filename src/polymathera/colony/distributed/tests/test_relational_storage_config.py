"""The RelationalStorage config's password requiredness is *conditional*.

colony resolves the DB password at runtime (``RelationalStorage._resolve_db_password``):
it prefers ``RDS_PASSWORD`` but falls back to AWS Secrets Manager via
``RDS_SECRET_ARN``. For that fallback to be reachable, ``RDS_PASSWORD`` must be
allowed to stay UNSET — so the config loader marks ``db_password`` optional exactly
when it is not needed: non-RDS backends, or RDS with a secret ARN. It stays
required (the loader fails loud) only when backend=RDS and neither a password nor
a secret ARN is given.

These lock that ``optional`` predicate — the load-bearing part of the CPS serving
fleet's dedicated-RDS wiring, where the VCM's Postgres password lives only in
Secrets Manager — independent of the env-var loader that evaluates it.
"""

from __future__ import annotations

from polymathera.colony.distributed.configs import (
    RelationalStorageBackendType,
    RelationalStorageConfig,
)

_ARN = "arn:aws:secretsmanager:us-east-1:123:secret:VcmDbCredentials-xyz"


def _optional_predicate(field_name: str):
    """The loader's ``optional`` callable for ``field_name``. It returns True when
    the env var may be absent (loader skips), False when it is required (loader
    raises if unset)."""
    return RelationalStorageConfig.model_fields[field_name].json_schema_extra["optional"]


def test_password_optional_for_rds_with_secret_arn() -> None:
    # The dedicated-RDS path: the password comes from Secrets Manager, so
    # RDS_PASSWORD must be allowed to stay unset — otherwise _resolve_db_password's
    # Secrets Manager branch is unreachable (the bug this fix removes).
    cfg = RelationalStorageConfig(
        backend=RelationalStorageBackendType.RDS, db_password_secret_arn=_ARN,
    )
    assert _optional_predicate("db_password")(cfg) is True


def test_password_required_for_rds_without_secret_arn() -> None:
    # backend=RDS, no secret ARN (and no password) → the loader must fail loud.
    cfg = RelationalStorageConfig(backend=RelationalStorageBackendType.RDS)
    assert _optional_predicate("db_password")(cfg) is False


def test_password_optional_for_non_rds_backend() -> None:
    # LOCAL (in-memory / SQLite) needs no DB password at all.
    cfg = RelationalStorageConfig(backend=RelationalStorageBackendType.LOCAL)
    assert _optional_predicate("db_password")(cfg) is True


def test_secret_arn_required_only_for_rds() -> None:
    # The sibling predicate: RDS_SECRET_ARN is required exactly when backend=RDS
    # (that is what makes the password fetchable at runtime); optional otherwise.
    predicate = _optional_predicate("db_password_secret_arn")
    assert predicate(RelationalStorageConfig(backend=RelationalStorageBackendType.RDS)) is False
    assert predicate(RelationalStorageConfig(backend=RelationalStorageBackendType.LOCAL)) is True
