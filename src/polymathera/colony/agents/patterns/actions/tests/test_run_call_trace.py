"""Tests for :class:`RunCallTrace` — the typed boundary view over
``CodeGenerationActionPolicy._run_call_trace``.

The contract: any drift between the writers' dict shape (four append
sites in ``code_generation.py``) and a consumer's read expectations
surfaces as ``pydantic.ValidationError`` at view construction, NOT as
a silent miscount in a downstream condition. This module pins that
contract.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from polymathera.colony.agents.patterns.actions.run_call_trace import (
    RunCallTrace,
    RunCallTraceEntry,
)


# Representative samples mirroring each of the four writer sites in
# ``code_generation.py``.

def _action_dispatched_entry(
    call_index: int = 0,
    action_key: str = "DesignProcessCapability.create_decomposition",
    parameters: dict[str, Any] | None = None,
    success: bool = True,
) -> dict[str, Any]:
    return {
        "call_index": call_index,
        "action_key": action_key,
        "parameters": parameters or {"parent_number": 44, "child_titles": ["A", "B"]},
        "success": success,
        "error": None if success else "Boom",
        "output_preview": "" if success else "Boom",
        "blocked": False,
    }


def _guardrail_blocked_entry(call_index: int = 0) -> dict[str, Any]:
    return {
        "call_index": call_index,
        "action_key": "DesignProcessCapability.create_decomposition",
        "parameters": {"parent_number": 44, "dry_run": False},
        "success": False,
        "error": "Blocked: guardrail",
        "output_preview": "",
        "blocked": True,
    }


def _signal_completion_entry(
    call_index: int = 0, success: bool = True,
) -> dict[str, Any]:
    return {
        "call_index": call_index,
        "action_key": "signal_completion",
        "success": success,
        "error": None if success else "Rejected: backlog not drained",
        "output_preview": (
            "Completion accepted" if success else "must call create_decomposition"
        ),
        "blocked": False,
    }


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_view_validates_action_dispatch_entry() -> None:
    view = RunCallTrace([_action_dispatched_entry()])
    assert len(view) == 1
    entry = view[0]
    assert isinstance(entry, RunCallTraceEntry)
    assert entry.action_key.endswith("create_decomposition")
    assert entry.success is True
    assert entry.blocked is False
    assert entry.parameters == {
        "parent_number": 44,
        "child_titles": ["A", "B"],
    }


def test_view_validates_guardrail_blocked_entry() -> None:
    view = RunCallTrace([_guardrail_blocked_entry()])
    e = view[0]
    assert e.blocked is True
    assert e.success is False


def test_view_validates_signal_completion_entry_without_parameters() -> None:
    """``signal_completion`` entries omit ``parameters``; the view must
    accept the absence cleanly (modeled as ``None``, NOT defaulted to
    ``{}``)."""

    view = RunCallTrace([_signal_completion_entry(success=True)])
    e = view[0]
    assert e.action_key == "signal_completion"
    assert e.parameters is None
    assert e.success is True


def test_view_construction_raises_on_missing_action_key() -> None:
    """Loud-failure invariant: a dropped writer field surfaces here,
    not in a downstream consumer."""

    bad = _action_dispatched_entry()
    del bad["action_key"]
    with pytest.raises(ValidationError):
        RunCallTrace([bad])


def test_view_construction_raises_on_unknown_field() -> None:
    """``extra="forbid"`` so a new writer field (e.g. wall-time, tags)
    surfaces here rather than silently passing through. The view OWNS
    the read schema — adding a field is a one-line edit on
    :class:`RunCallTraceEntry`, but a silent shape drift is impossible."""

    bad = _action_dispatched_entry()
    bad["new_field_we_did_not_account_for"] = 42
    with pytest.raises(ValidationError):
        RunCallTrace([bad])


def test_view_construction_raises_on_type_mismatch() -> None:
    bad = _action_dispatched_entry()
    bad["success"] = "yes"  # str, not bool
    with pytest.raises(ValidationError):
        RunCallTrace([bad])


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


def test_calls_to_filters_by_exact_action_key() -> None:
    raw = [
        _action_dispatched_entry(
            call_index=0,
            action_key="DesignProcessCapability.classify_issues_decomposability",
        ),
        _action_dispatched_entry(
            call_index=1,
            action_key="DesignProcessCapability.create_decomposition",
        ),
        _action_dispatched_entry(
            call_index=2,
            action_key="DesignProcessCapability.create_decomposition",
            success=False,
        ),
    ]
    view = RunCallTrace(raw)
    create = view.calls_to("DesignProcessCapability.create_decomposition")
    assert len(create) == 2
    classify = view.calls_to(
        "DesignProcessCapability.classify_issues_decomposability",
    )
    assert len(classify) == 1


def test_successful_calls_to_excludes_failed_and_blocked() -> None:
    raw = [
        _action_dispatched_entry(call_index=0, success=True),
        _action_dispatched_entry(call_index=1, success=False),
        _guardrail_blocked_entry(call_index=2),
    ]
    view = RunCallTrace(raw)
    successful = view.successful_calls_to(
        "DesignProcessCapability.create_decomposition",
    )
    assert len(successful) == 1
    assert successful[0].call_index == 0


def test_parameters_of_successful_pulls_typed_values() -> None:
    raw = [
        _action_dispatched_entry(
            call_index=0,
            action_key="DesignProcessCapability.create_decomposition",
            parameters={"parent_number": 44, "dry_run": False},
        ),
        _action_dispatched_entry(
            call_index=1,
            action_key="DesignProcessCapability.create_decomposition",
            parameters={"parent_number": 45, "dry_run": False},
        ),
        _action_dispatched_entry(
            call_index=2,
            action_key="DesignProcessCapability.create_decomposition",
            parameters={"parent_number": 46},
            success=False,
        ),
    ]
    view = RunCallTrace(raw)
    parents = view.parameters_of_successful(
        "DesignProcessCapability.create_decomposition",
        "parent_number",
    )
    assert parents == (44, 45)


def test_parameters_of_successful_skips_missing_parameter_key() -> None:
    """When the writer recorded ``parameters`` but the key is absent,
    the read primitive treats it as meaningful absence — not a default
    paper-over."""

    raw = [
        _action_dispatched_entry(
            call_index=0,
            parameters={"parent_number": 44},
        ),
        _action_dispatched_entry(
            call_index=1,
            parameters={"unrelated": "value"},
        ),
    ]
    view = RunCallTrace(raw)
    parents = view.parameters_of_successful(
        "DesignProcessCapability.create_decomposition",
        "parent_number",
    )
    assert parents == (44,)


def test_parameters_of_successful_skips_signal_completion() -> None:
    """``signal_completion`` entries carry ``parameters=None``; the
    read primitive must skip them cleanly (no AttributeError on
    ``None.get``)."""

    raw = [
        _action_dispatched_entry(call_index=0, parameters={"parent_number": 44}),
        _signal_completion_entry(call_index=1),
    ]
    view = RunCallTrace(raw)
    parents = view.parameters_of_successful(
        "DesignProcessCapability.create_decomposition",
        "parent_number",
    )
    assert parents == (44,)


# ---------------------------------------------------------------------------
# Canonical action-key constants on DesignProcessCapability
# ---------------------------------------------------------------------------


def test_canonical_action_key_constants_match_method_names() -> None:
    """Single source of truth: the ClassVar string is the same as the
    method name (after the dispatch-key prefix the dispatcher prepends).
    A rename of the method without updating the constant fails this
    pin loudly."""

    from polymathera.colony.design_monorepo.process import (
        DesignProcessCapability,
    )

    assert (
        DesignProcessCapability.CREATE_DECOMPOSITION_ACTION_KEY
        == "create_decomposition"
    )
    assert (
        DesignProcessCapability.CLASSIFY_ISSUES_DECOMPOSABILITY_ACTION_KEY
        == "classify_issues_decomposability"
    )
    assert (
        DesignProcessCapability.PROPOSE_DECOMPOSITIONS_ACTION_KEY
        == "propose_decompositions"
    )
    # Method existence pin — a rename without updating the constant
    # MUST break here (the attribute disappears).
    assert hasattr(DesignProcessCapability, "create_decomposition")
    assert hasattr(DesignProcessCapability, "classify_issues_decomposability")
    assert hasattr(DesignProcessCapability, "propose_decompositions")
