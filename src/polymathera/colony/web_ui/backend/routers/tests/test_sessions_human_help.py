"""Lint-style tests pinning the SessionAgent's HumanHelp wiring.

The SessionAgent's blueprint list (in ``routers/sessions.py``) and its
``self_concept`` MISSION SPAWN PROTOCOL block are LLM-facing surfaces
— a regression that drops the ``HumanHelpCapability.bind(...)`` line
or rewrites the prose without the translation-layer framing would
silently revert the SessionAgent to the lazy-relay shape that
shipped through run7 (which passed bare ``{"mode": "decompose"}``
without translating the user's intent).

These tests are source-string asserts — coarse but mechanical. They
catch the most common regression: a dev removes the bind or rewrites
the prose without realising the LLM relies on specific framing.
"""

from __future__ import annotations

import inspect

import pytest


def _load_router_source() -> str:
    """Read the user-session blueprint factory source. PR1-B moved
    the blueprint construction (including HumanHelpCapability mount
    and the self_concept MISSION SPAWN PROTOCOL prose) out of
    ``routers/sessions.py:create_session`` into
    ``chat/user_session_factory.py`` so the same shape is used for
    create-session AND for the dashboard's lazy respawn. The pins
    follow the code to the factory; the router's call site is
    pinned by ``tests/test_pr1b_respawn.py``."""

    from polymathera.colony.web_ui.backend.chat import (
        user_session_factory,
    )
    return inspect.getsource(user_session_factory)


def test_session_agent_blueprint_mounts_human_help_capability() -> None:
    """The SessionAgent's blueprint list (constructed in
    ``create_session``) MUST include ``HumanHelpCapability.bind(...)``
    so the agent's planner can call ``request_help`` from the MISSION
    SPAWN PROTOCOL's completeness gate when user intent is missing,
    vague, or ambiguous relative to a required mission parameter.
    Without the mount, the planner has no way to clarify, and the
    completeness gate degrades to silent param fabrication."""

    source = _load_router_source()
    assert "HumanHelpCapability.bind(" in source, (
        "SessionAgent blueprint list no longer mounts "
        "HumanHelpCapability — the MISSION SPAWN PROTOCOL "
        "completeness gate would have no clarification primitive. "
        "Restore the bind line alongside HumanApprovalCapability + "
        "MissionStatusCapability."
    )


def test_session_agent_self_concept_has_translation_layer_framing() -> None:
    """The MISSION SPAWN PROTOCOL block in the SessionAgent's
    self_concept must articulate the translation-layer framing, the
    completeness gate using ``request_help``, the free-text
    translation step, and the multi-mission decomposition fallback.
    A regression that rewrites the prose without these four pieces
    sends the SessionAgent back to the lazy-relay shape that shipped
    ``{"mode": "decompose"}`` with no user-intent extraction."""

    source = _load_router_source()
    # Translation-layer framing — the LLM must understand it is NOT
    # a relay.
    assert "TRANSLATION LAYER" in source, (
        "MISSION SPAWN PROTOCOL has lost its 'TRANSLATION LAYER' "
        "framing — the LLM defaults back to verbatim relay."
    )
    # Translation as SEMANTIC, not lexical.
    assert "Translate intent" in source, (
        "MISSION SPAWN PROTOCOL has lost the Step 2 'translate intent "
        "to mission_params' instruction."
    )
    # Completeness gate uses request_help, not respond_to_user.
    assert "Completeness gate" in source
    assert "request_help" in source, (
        "MISSION SPAWN PROTOCOL has lost the request_help reference "
        "— the completeness gate has no clarification primitive."
    )
    # Multi-mission decomposition step.
    assert "Multi-mission decomposition" in source, (
        "MISSION SPAWN PROTOCOL has lost the multi-mission "
        "decomposition fallback (Step 5). Single-mission lock-in "
        "regression."
    )
    # Free-text translation on the reply.
    assert "human_help_response:" in source, (
        "MISSION SPAWN PROTOCOL has lost the reference to the "
        "human_help_response:{request_id} planner-context binding "
        "the LLM reads to translate the operator's free-text reply."
    )


def test_session_agent_self_concept_has_retry_halt_protocol() -> None:
    """The MISSION SPAWN PROTOCOL block must articulate the
    retry-halt rule for persistent identical errors. Without it, the
    SessionAgent's LLM loops on ``outcome='error'`` (the run8
    regression: 12 identical ``_app_name`` AttributeError responses
    burned the iteration budget while never surfacing the real cause
    to the operator). Pin the prose so a future rewrite that drops
    the halt signal surfaces here, not in another runaway loop."""

    source = _load_router_source()
    assert "RETRY HALT" in source, (
        "MISSION SPAWN PROTOCOL has lost the RETRY HALT section. "
        "Without it the LLM may loop on persistent identical "
        "spawn_mission errors (run8: 12 identical errors before "
        "max_iterations hit)."
    )
    # The rule's two load-bearing pieces: stop after repeated
    # identical errors, AND surface the verbatim error to the user.
    assert "IDENTICAL" in source or "identical" in source, (
        "RETRY HALT prose has lost the identical-error condition."
    )
    assert "verbatim error" in source, (
        "RETRY HALT prose has lost the verbatim-error reporting "
        "instruction; the operator must see the actual failure, "
        "not another 'spawning…' message."
    )


def test_human_help_router_is_registered_in_main() -> None:
    """The REST endpoint that accepts the operator's typed help
    response MUST be registered in the FastAPI app. Without
    registration, the chat UI's POST to
    /api/v1/sessions/{id}/human_help/{rid}/respond returns 404 and
    the planner-context binding never lands on the requesting
    agent."""

    from polymathera.colony.web_ui.backend import main as main_mod
    source = inspect.getsource(main_mod)
    assert "human_help.router" in source, (
        "human_help router not registered in main.py — REST endpoint "
        "is unreachable from the browser."
    )
    assert "human_help," in source, (
        "human_help router import missing from main.py's routers "
        "import block."
    )
