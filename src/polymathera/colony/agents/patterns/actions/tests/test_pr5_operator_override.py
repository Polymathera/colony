"""Tests for the operator runtime override on
:class:`SemanticConstraintGuardrail`.

The guardrail reads the operator-disabled set LIVE from the session
blackboard at each ``.check()`` — there is no local mirror. The
dashboard's POST writes the same BB keys
(``operator_override:semantic_constraint:<id>``); ``.check()`` reads
them back.

Reading at check-time (instead of mirroring + subscribing) means:
- Overrides survive SessionAgent respawn for free.
- No in-memory state can drift from the BB.
- No event-handler / subscription glue to maintain.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from polymathera.colony.agents.blackboard.protocol import (
    OperatorOverrideProtocol,
)
from polymathera.colony.agents.patterns.actions.semantic_constraints import (
    ConstraintVerdict,
    PythonPredicateVerifier,
    SemanticConstraint,
    SemanticConstraintGuardrail,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _exec_ctx():
    """``get_scope_prefix(BlackboardScope.SESSION)`` reads the
    ambient execution context; every test in this module needs one
    set so both the production code under test AND the fake-owner's
    scope-strict assertion resolve the same prefix."""

    from polymathera.colony.distributed.ray_utils.serving.context import (
        Ring, execution_context,
    )
    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


def _always_block(name: str) -> SemanticConstraint:
    return SemanticConstraint(
        id=name,
        rule_nl="t",
        applies_to=["x"],
        verifier=PythonPredicateVerifier(
            predicate=lambda *_: ConstraintVerdict(
                satisfied=False, reason="no",
            ),
            description="d",
        ),
    )


class _FakeBB:
    """Minimal in-memory blackboard exposing only the surface the
    guardrail's ``_read_disabled_ids`` uses: ``query(namespace=...)``
    returning entries with ``.key`` and ``.value``."""

    def __init__(
        self,
        overrides: dict[str, dict[str, Any]] | None = None,
        *,
        raise_on_query: Exception | None = None,
    ) -> None:
        self._entries: list[SimpleNamespace] = []
        for cid, payload in (overrides or {}).items():
            key = OperatorOverrideProtocol.semantic_constraint_key(cid)
            self._entries.append(SimpleNamespace(key=key, value=payload))
        self._raise_on_query = raise_on_query

    async def query(self, *, namespace: str) -> list[SimpleNamespace]:
        if self._raise_on_query is not None:
            raise self._raise_on_query
        prefix = namespace.rstrip("*")
        return [e for e in self._entries if e.key.startswith(prefix)]


def _fake_owner(bb: _FakeBB | None) -> Any:
    """Owner stub exposing ``get_blackboard()`` only — that's the
    sole surface the guardrail reads from the agent.

    The stub REJECTS calls that don't request the SESSION scope —
    a scope mismatch (e.g. defaulting to AGENT scope) would silently
    return the wrong BB in production, so the test fake refuses it
    instead of papering over with the same ``bb`` for every scope."""

    from polymathera.colony.agents.scopes import (
        BlackboardScope, get_scope_prefix,
    )

    async def _get_bb(scope_id: str | None = None, **_kw: Any) -> _FakeBB | None:
        expected = get_scope_prefix(BlackboardScope.SESSION)
        if scope_id != expected:
            raise AssertionError(
                f"SemanticConstraintGuardrail asked for the wrong "
                f"blackboard scope: got {scope_id!r}, expected the "
                f"session-default scope {expected!r}. The dashboard "
                f"writes operator overrides at session scope; reading "
                f"at any other scope returns the empty set."
            )
        return bb

    return SimpleNamespace(
        get_blackboard=_get_bb, infer=None, agent_id="test-agent",
    )


async def test_disabled_constraint_skips_verifier() -> None:
    """A constraint whose id has ``disabled=True`` on the BB must
    not invoke its verifier or block."""

    invoked = {"count": 0}

    def pred(*_):
        invoked["count"] += 1
        return ConstraintVerdict(satisfied=False, reason="no")

    g = SemanticConstraintGuardrail([
        SemanticConstraint(
            id="r",
            rule_nl="t",
            applies_to=["x"],
            verifier=PythonPredicateVerifier(predicate=pred, description="d"),
        ),
    ])
    bb = _FakeBB(overrides={"r": {"disabled": True}})
    g.bind_agent(_fake_owner(bb))

    d = await g.check(
        action_key="x.go", params={"content": "any"}, call_history=[],
    )
    assert d.allowed
    assert invoked["count"] == 0


async def test_disabling_one_constraint_leaves_others_active() -> None:
    """Disabling rule A on the BB does not disable rule B."""

    g = SemanticConstraintGuardrail([_always_block("a"), _always_block("b")])
    bb = _FakeBB(overrides={"a": {"disabled": True}})
    g.bind_agent(_fake_owner(bb))

    d = await g.check(action_key="x.go", params={"content": "x"}, call_history=[])
    assert not d.allowed
    assert "[b]" in d.reason


async def test_disabled_false_payload_does_not_skip() -> None:
    """A stale BB entry with ``disabled=False`` (the dashboard's
    enable path writes this) must not skip the constraint — it's
    re-enabled, not disabled."""

    g = SemanticConstraintGuardrail([_always_block("r")])
    bb = _FakeBB(overrides={"r": {"disabled": False}})
    g.bind_agent(_fake_owner(bb))

    d = await g.check(action_key="x.go", params={}, call_history=[])
    assert not d.allowed


async def test_no_overrides_on_bb_runs_verifier() -> None:
    """Empty BB → no constraint is disabled → verifier runs normally."""

    g = SemanticConstraintGuardrail([_always_block("r")])
    g.bind_agent(_fake_owner(_FakeBB()))

    d = await g.check(action_key="x.go", params={}, call_history=[])
    assert not d.allowed


async def test_no_owner_treats_as_no_overrides() -> None:
    """Unbound guardrail (no ``bind_agent`` call yet, or owner is
    None) treats overrides as absent — strictly worse to block every
    action because we can't reach the BB."""

    g = SemanticConstraintGuardrail([_always_block("r")])
    # No bind_agent call. self._owner is None.
    d = await g.check(action_key="x.go", params={}, call_history=[])
    assert not d.allowed


async def test_bb_query_failure_degrades_open(caplog) -> None:
    """BB transient failure on the query → degrade OPEN (no
    overrides applied this iteration). Losing operator overrides
    during a BB outage is strictly less bad than blocking every
    guardrailed action because we can't read state."""

    g = SemanticConstraintGuardrail([_always_block("r")])
    bb = _FakeBB(raise_on_query=RuntimeError("redis down"))
    g.bind_agent(_fake_owner(bb))

    d = await g.check(action_key="x.go", params={}, call_history=[])
    # Verifier still runs → blocks. Override path didn't crash.
    assert not d.allowed


async def test_override_survives_guardrail_rebuild() -> None:
    """The single-source-of-truth property: a fresh guardrail bound
    to the same owner sees the same disabled set. This is what makes
    operator overrides survive SessionAgent respawn — no local state
    to rehydrate, the BB IS the state."""

    bb = _FakeBB(overrides={"r": {"disabled": True}})
    owner = _fake_owner(bb)

    g1 = SemanticConstraintGuardrail([_always_block("r")])
    g1.bind_agent(owner)
    assert (await g1.check(action_key="x.go", params={}, call_history=[])).allowed

    # Simulate respawn: brand-new guardrail instance, same BB / owner.
    g2 = SemanticConstraintGuardrail([_always_block("r")])
    g2.bind_agent(owner)
    assert (await g2.check(action_key="x.go", params={}, call_history=[])).allowed
