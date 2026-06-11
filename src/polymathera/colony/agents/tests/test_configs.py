"""Targeted tests for ``agents/configs.py`` validators."""

from __future__ import annotations

from polymathera.colony.agents.configs import (
    GitHubAuthConfig,
    MissionExecutionPolicy,
    resolve_effective_max_iterations,
)


def test_github_auth_config_normalises_escaped_pem() -> None:
    """Single-line PEMs with ``\\n`` escapes (the format docker-compose
    forces — see ``GitHubAuthConfig._normalize_pem``) are converted to
    real newlines so pyjwt sees a valid key."""

    escaped = (
        "-----BEGIN RSA PRIVATE KEY-----\\nBODY\\n"
        "-----END RSA PRIVATE KEY-----\\n"
    )
    cfg = GitHubAuthConfig(private_key_pem=escaped)
    assert cfg.private_key_pem == (
        "-----BEGIN RSA PRIVATE KEY-----\nBODY\n"
        "-----END RSA PRIVATE KEY-----\n"
    )


def test_github_auth_config_preserves_real_newlines_pem() -> None:
    """Multi-line PEMs (already-decoded form) pass through unchanged —
    idempotent."""

    real = (
        "-----BEGIN RSA PRIVATE KEY-----\nBODY\n"
        "-----END RSA PRIVATE KEY-----\n"
    )
    cfg = GitHubAuthConfig(private_key_pem=real)
    assert cfg.private_key_pem == real


def test_github_auth_config_leaves_empty_pem_alone() -> None:
    """Empty value (unconfigured deployment) is not touched."""

    cfg = GitHubAuthConfig(private_key_pem="")
    assert cfg.private_key_pem == ""


# ---------------------------------------------------------------------------
# resolve_effective_max_iterations — precedence rules
# ---------------------------------------------------------------------------


def test_caller_override_wins_over_policy() -> None:
    """An explicit ``max_iterations`` from the spawn caller (LLM
    planner or REST API) beats every other layer."""

    policy = MissionExecutionPolicy(max_iterations=50)
    assert resolve_effective_max_iterations(
        caller_override=7, policy=policy,
    ) == 7


def test_policy_used_when_caller_didnt_override() -> None:
    """No caller override → coordinator's declared
    ``MISSION_EXECUTION_POLICY.max_iterations`` wins. This is what
    makes ``ProjectPlanningCoordinator``'s 50 land in the spawned
    coordinator's metadata."""

    policy = MissionExecutionPolicy(max_iterations=50)
    assert resolve_effective_max_iterations(
        caller_override=None, policy=policy,
    ) == 50


def test_schema_default_used_when_neither_set() -> None:
    """No caller override + no policy override → fall back to the
    :class:`AgentMetadata` schema default of 20. Pre-policy behaviour."""

    policy = MissionExecutionPolicy()
    assert policy.max_iterations is None
    assert resolve_effective_max_iterations(
        caller_override=None, policy=policy,
    ) == 20


def test_custom_schema_default_is_honoured() -> None:
    """Callers can pass a non-20 ``schema_default`` to match a
    different metadata model (rare, but the layering supports it)."""

    policy = MissionExecutionPolicy()
    assert resolve_effective_max_iterations(
        caller_override=None, policy=policy, schema_default=99,
    ) == 99


def test_project_planning_coordinator_policy_carries_max_iterations() -> None:
    """End-to-end pin: the ``ProjectPlanningCoordinator`` ClassVar
    really does declare ``max_iterations=50`` so the spawn site can
    pick it up. Catches accidental removal of the override."""

    from polymathera.colony.agents.missions.project_planning.coordinator import (
        ProjectPlanningCoordinator,
    )

    assert ProjectPlanningCoordinator.MISSION_EXECUTION_POLICY.max_iterations == 50


def test_project_planning_mission_does_not_declare_repo_caller_param() -> None:
    """Pin: the project_planning mission MUST NOT re-grow a ``repo``
    caller_parameter. The action surface resolves ``owner/repo`` from
    the agent's ``design_monorepo_url`` metadata parameter — the LLM
    planner should not be threading it. See
    [[no-llm-facing-framework-state]]."""

    from polymathera.colony.agents.configs import _builtin_missions

    spec = _builtin_missions()["project_planning"]
    names = {p.name for p in spec.caller_parameters}
    assert "repo" not in names, (
        "project_planning mission re-grew a 'repo' caller_parameter — "
        "the action surface resolves owner/repo from agent metadata."
    )
