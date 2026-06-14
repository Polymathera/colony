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


def test_decompose_mode_self_concept_names_wait_for_next_event() -> None:
    """Decompose-mode self-concept text MUST list
    ``wait_for_next_event`` as an available primitive so the planner
    LLM can compose request→wait without bouncing off the guardrail
    block message. Regression for live run 3 Failure 1 (LLM poll-looped
    ``get_response`` for 5 minutes because the static self-concept
    omitted the wait primitive)."""

    from polymathera.colony.agents.configs import _builtin_missions

    spec = _builtin_missions()["project_planning"]
    decompose_blocks = [
        text for text in (spec.self_concept.goals or [])
        if "decompose" in text.lower()
    ]
    assert decompose_blocks, "no decompose-mode self_concept.goals block found"
    available_primitive_block = next(
        (text for text in decompose_blocks if "Available primitives" in text),
        None,
    )
    assert available_primitive_block is not None, (
        "decompose-mode block does not name the 'Available primitives' "
        "set — the planner has no enumerated surface to compose from."
    )
    assert "wait_for_next_event" in available_primitive_block, (
        "decompose-mode primitives list omits ``wait_for_next_event`` — "
        "the LLM has no named pause primitive and re-introduces the "
        "live-run-3 poll-loop pattern."
    )


def test_decompose_mode_self_concept_does_not_teach_polling() -> None:
    """The decompose-mode prompt MUST NOT instruct the LLM to keep
    polling ``get_response``. Regression for live run 3 — the prompt
    fix is the durable answer; structural counter-forces were
    withdrawn as band-aids."""

    from polymathera.colony.agents.configs import _builtin_missions

    spec = _builtin_missions()["project_planning"]
    for text in (spec.self_concept.goals or []):
        lowered = text.lower()
        assert "poll get_response" not in lowered, text
        assert "keep polling" not in lowered, text
        # The old "WAIT for the user's choice" recipe was the
        # ambiguous shape the LLM interpreted as "poll" — must be
        # replaced by an explicit wait_for_next_event mention.
        if "wait for the user's choice" in lowered:
            assert "wait_for_next_event" in text, text


def test_decompose_mode_self_concept_carries_no_python_class_attribute_refs() -> None:
    """LLM-facing self-concept text MUST NOT carry Python class-
    attribute references (``ClassName.UPPER_CASE_ATTR``). The LLM
    consumes the text verbatim — it cannot resolve a Python symbol at
    inference time, so a leak provides zero information AND violates
    [[no-llm-facing-framework-state]]: framework-known state must
    not leak into the LLM-facing surface.

    Regression for the original Q1 rewrite that referenced
    ``HumanApprovalCapability.RESPONSE_CONTEXT_KEY_PREFIX`` as a
    literal string in the decompose-mode goal block — the operator
    caught it during review and demanded the fix."""

    import re
    from polymathera.colony.agents.configs import _builtin_missions

    spec = _builtin_missions()["project_planning"]
    all_text = "\n".join(spec.self_concept.goals or [])
    # Looks for ``<CamelCase>.<UPPER_SNAKE>`` — the shape of a Python
    # ClassVar reference. Allows lowercase attribute references like
    # ``response.choice`` and ``mission_params['issue_numbers']`` which
    # are LLM-facing field references on values the LLM holds, not
    # Python class attributes.
    pattern = re.compile(r"\b[A-Z][A-Za-z0-9]+\.[A-Z][A-Z0-9_]+\b")
    hits = pattern.findall(all_text)
    assert not hits, (
        f"decompose-mode self_concept.goals leaks Python class-"
        f"attribute references into the LLM-facing prompt: {hits}. "
        f"Hoist the mechanical contract into the standing guardrail "
        f"advisory (single source of truth) and let the goal block "
        f"point at it by description, not by Python symbol."
    )


def test_decompose_mode_self_concept_names_four_choice_surface() -> None:
    """The Q1 four-choice surface (approve_once / approve_all /
    reject / abort) must be named in the decompose-mode prompt so the
    planner knows what choices to expect on the response binding."""

    from polymathera.colony.agents.configs import _builtin_missions

    spec = _builtin_missions()["project_planning"]
    all_text = "\n".join(spec.self_concept.goals or [])
    for choice in ("approve_once", "approve_all", "reject", "abort"):
        assert choice in all_text, (
            f"decompose-mode prompt does not name choice {choice!r}"
        )
    # Operator's required-on-reject/abort justification:
    assert "explanation" in all_text, (
        "decompose-mode prompt does not mention the operator's "
        "explanation field — the LLM won't know to read it on "
        "reject/abort responses."
    )
