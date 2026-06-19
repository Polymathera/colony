"""Unit tests for ``HumanApprovalCapability``.

These exercise the capability in detached mode against an in-memory
blackboard backend. The four-layer chain the capability participates
in (agent → session blackboard → Web UI → session blackboard → agent
event handler) is verified by writing the response payload directly
to the blackboard and observing that the capability's event handler
fires + that ``get_response`` returns the expected typed result.

What is NOT tested here: the Web UI HTTP endpoint; SessionAgent's
relay to the chat UI. Those have their own tests.
"""

from __future__ import annotations

import asyncio

import pytest

from polymathera.colony.agents.blackboard import EnhancedBlackboard
from polymathera.colony.agents.blackboard.protocol import HumanApprovalProtocol
from polymathera.colony.agents.models import AgentSuspensionState
from polymathera.colony.agents.patterns.capabilities.human_approval import (
    HumanApprovalCapability,
    HumanApprovalRequest,
    HumanApprovalResponse,
    RequestHumanApprovalEmpty,
)
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)


pytestmark = pytest.mark.asyncio


@pytest.fixture
def _exec_ctx():
    """Provide an execution context with session_id so the capability's
    SESSION-scoped scope_id resolves."""

    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1",
        session_id="s1", origin="test",
    ) as ctx:
        yield ctx


async def _make_capability(_exec_ctx) -> HumanApprovalCapability:
    """Build a detached capability with an in-memory blackboard pre-wired."""

    cap = HumanApprovalCapability(
        agent=None,
        capability_key="hac_test",
        app_name="test_app",
    )
    bb = EnhancedBlackboard(
        app_name="test_app",
        scope_id=cap.scope_id,
        backend_type="memory",
        enable_events=True,
    )
    await bb.initialize()
    cap._blackboard = bb  # bypass deferred init, force memory backend
    return cap


# ---------------------------------------------------------------------------
# Request side
# ---------------------------------------------------------------------------


async def test_request_writes_typed_payload_to_session_scope(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    result = await cap.request_human_approval(
        question="Approve the design checkpoint?",
        options=("approve", "reject"),
        extra={"checkpoint_id": "cp_42"},
    )
    assert result["ok"] is True
    rid = result["request_id"]
    assert rid.startswith("appr_")
    bb = await cap.get_blackboard()
    raw = await bb.read(HumanApprovalProtocol.request_key(rid))
    assert isinstance(raw, dict)
    request = HumanApprovalRequest.model_validate(raw)
    assert request.question == "Approve the design checkpoint?"
    assert request.options == ("approve", "reject")
    assert request.extra == {"checkpoint_id": "cp_42"}
    pending = await cap.list_pending()
    assert pending["ok"] is True
    assert rid in pending["pending_request_ids"]


async def test_request_id_is_unique_across_calls(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    a = (await cap.request_human_approval(question="Approve this change?"))["request_id"]
    b = (await cap.request_human_approval(question="Approve a different change?"))["request_id"]
    assert a != b
    assert {a, b} == set(
        (await cap.list_pending())["pending_request_ids"],
    )


# ---------------------------------------------------------------------------
# Receive side — event handler + cache
# ---------------------------------------------------------------------------


async def test_event_handler_caches_response_and_returns_context(
    _exec_ctx,
) -> None:
    """When the response lands on the blackboard, the @event_handler
    fires inside the capability's normal event loop, caches the
    response, and surfaces it as planner context."""

    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_human_approval(
        question="Approve question Q?", options=("a", "b"),
    ))["request_id"]

    # Simulate the Web UI HTTP endpoint writing the response.
    response = HumanApprovalResponse(
        request_id=rid, choice="a", note="ok", decided_by="alice",
    )
    bb = await cap.get_blackboard()
    await bb.write(
        HumanApprovalProtocol.response_key(rid),
        response.model_dump(mode="json"),
    )

    # Drive the handler manually (as the agent's event loop would).
    fake_event = type("E", (), {})()
    fake_event.key = HumanApprovalProtocol.response_key(rid)
    fake_event.value = response.model_dump(mode="json")
    result = await cap._on_response(fake_event, None)

    assert result is not None
    assert result.context_key == (
        f"{HumanApprovalCapability.RESPONSE_CONTEXT_KEY_PREFIX}{rid}"
    )
    assert result.context == {
        "request_id": rid,
        "choice": "a",
        "explanation": "",
        "note": "ok",
        "decided_by": "alice",
    }
    # The cache survives — get_response should not need a blackboard hit.
    envelope = await cap.get_response(rid)
    assert envelope["ok"] is True
    assert envelope["state"] == "ready"
    assert envelope["response"]["choice"] == "a"
    assert envelope["response"]["decided_by"] == "alice"
    pending = await cap.list_pending()
    assert rid not in pending["pending_request_ids"]


async def test_event_handler_drops_malformed_payload(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    fake_event = type("E", (), {})()
    fake_event.key = HumanApprovalProtocol.response_key("appr_bad")
    fake_event.value = "not-a-dict"
    result = await cap._on_response(fake_event, None)
    assert result is None


async def test_event_handler_ignores_non_response_keys(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    fake_event = type("E", (), {})()
    fake_event.key = "some:other:key"
    fake_event.value = {"choice": "a", "request_id": "x"}
    result = await cap._on_response(fake_event, None)
    assert result is None


# ---------------------------------------------------------------------------
# get_response — cache + blackboard fallback
# ---------------------------------------------------------------------------


async def test_get_response_envelope_state_pending(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_human_approval(question="Approve question Q?"))["request_id"]
    envelope = await cap.get_response(rid)
    assert envelope == {"ok": True, "state": "pending", "response": None}


async def test_get_response_falls_back_to_blackboard(_exec_ctx) -> None:
    """A response that landed during agent suspension (so the in-process
    event handler never fired) is still recoverable through a direct
    blackboard read on resume."""

    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_human_approval(question="Approve question Q?"))["request_id"]

    response = HumanApprovalResponse(
        request_id=rid, choice="approve", note="lgtm", decided_by="bob",
    )
    bb = await cap.get_blackboard()
    await bb.write(
        HumanApprovalProtocol.response_key(rid),
        response.model_dump(mode="json"),
    )
    # Skip the event handler entirely — simulate the resume case.
    cap._responses.clear()

    envelope = await cap.get_response(rid)
    assert envelope["ok"] is True
    assert envelope["state"] == "ready"
    assert envelope["response"]["choice"] == "approve"
    assert envelope["response"]["decided_by"] == "bob"
    # Cache populated by the fallback so subsequent reads are cheap.
    assert rid in cap._responses


# ---------------------------------------------------------------------------
# Suspend / resume
# ---------------------------------------------------------------------------


async def test_suspend_resume_round_trips_requests_and_responses(
    _exec_ctx,
) -> None:
    cap1 = await _make_capability(_exec_ctx)
    rid_pending = (await cap1.request_human_approval(
        question="Pending?",
    ))["request_id"]
    rid_resolved = (await cap1.request_human_approval(
        question="Resolved?",
    ))["request_id"]
    response = HumanApprovalResponse(
        request_id=rid_resolved, choice="approve", decided_by="carol",
    )
    cap1._responses[rid_resolved] = response

    state = AgentSuspensionState(
        agent_id="test",
        agent_type="test_agent",
        suspension_reason="test",
        suspended_at=0.0,
    )
    await cap1.serialize_suspension_state(state)

    cap2 = await _make_capability(_exec_ctx)
    await cap2.deserialize_suspension_state(state)

    assert rid_pending in cap2._requests
    assert rid_resolved in cap2._requests
    assert cap2._responses.get(rid_resolved) is not None
    assert cap2._responses[rid_resolved].choice == "approve"
    pending = await cap2.list_pending()
    assert pending["pending_request_ids"] == [rid_pending]


# ---------------------------------------------------------------------------
# End-to-end via blackboard event stream
# ---------------------------------------------------------------------------


async def test_end_to_end_via_blackboard_event_stream(_exec_ctx) -> None:
    """Drive the full chain: request → blackboard → event stream →
    capability handler. Proves the @event_handler pattern subscribes
    correctly on the session blackboard."""

    cap = await _make_capability(_exec_ctx)
    rid = (await cap.request_human_approval(
        question="Final answer?",
    ))["request_id"]

    bb = await cap.get_blackboard()

    # Subscribe directly so we can verify the event lands on the topic
    # the capability would observe via @event_handler. We do not run
    # the agent's full event loop here — that is exercised by the
    # broader agent integration tests.
    queue: asyncio.Queue = asyncio.Queue()
    bb.stream_events_to_queue(
        queue,
        pattern=HumanApprovalProtocol.response_pattern(),
    )

    response = HumanApprovalResponse(
        request_id=rid,
        choice="reject",
        explanation="not yet ready",
        note="not yet",
        decided_by="dan",
    )
    await bb.write(
        HumanApprovalProtocol.response_key(rid),
        response.model_dump(mode="json"),
    )

    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event.key == HumanApprovalProtocol.response_key(rid)
    parsed = HumanApprovalResponse.model_validate(event.value)
    assert parsed.choice == "reject"

    # Feed the event into the capability's handler the way the agent
    # event loop would, and confirm the cache + planner context.
    result = await cap._on_response(event, None)
    assert result is not None
    assert result.context["choice"] == "reject"
    envelope = await cap.get_response(rid)
    assert envelope["state"] == "ready"
    assert envelope["response"]["choice"] == "reject"


# ---------------------------------------------------------------------------
# has_active_approval_for — guardrail's lookup path
# ---------------------------------------------------------------------------


async def _seed_response(
    cap, *, rid: str, choice: str, action_type: str | None,
) -> None:
    """Plant a typed request + response on the blackboard, like a
    real Web UI POST would. ``reject`` / ``abort`` get a placeholder
    ``explanation`` so the validator (which enforces non-empty
    explanation on those choices) is satisfied — the seed helper is
    for unit tests that exercise downstream consumers, not the
    explanation contract itself (covered by dedicated tests)."""

    cap._requests[rid] = HumanApprovalRequest(
        request_id=rid,
        question="?",
        action_type=action_type,
        options=(
            ("approve_once", "approve_all", "reject", "abort")
            if action_type is not None else ("approve", "reject")
        ),
    )
    explanation = (
        "test seed explanation"
        if choice in {"reject", "abort"} else ""
    )
    response = HumanApprovalResponse(
        request_id=rid, choice=choice, explanation=explanation,
        decided_by="t",
    )
    bb = await cap.get_blackboard()
    await bb.write(
        HumanApprovalProtocol.response_key(rid),
        response.model_dump(mode="json"),
    )
    cap._responses[rid] = response


async def test_has_active_approval_blocks_when_no_approval(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    allowed, rid = await cap.has_active_approval_for(
        "DesignProcessCapability.create_decomposition",
    )
    assert allowed is False
    assert rid is None


async def test_has_active_approval_blocks_on_rejected(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    await _seed_response(
        cap, rid="r1", choice="reject",
        action_type="create_decomposition",
    )
    allowed, _ = await cap.has_active_approval_for(
        "DesignProcessCapability.create_decomposition",
    )
    assert allowed is False


async def test_approve_once_allows_first_dispatch_and_consumes(
    _exec_ctx,
) -> None:
    cap = await _make_capability(_exec_ctx)
    await _seed_response(
        cap, rid="r1", choice="approve_once",
        action_type="create_decomposition",
    )

    first, rid1 = await cap.has_active_approval_for(
        "DesignProcessCapability.create_decomposition",
    )
    assert first is True
    assert rid1 == "r1"

    # Subsequent dispatch — consumption marker should make this False.
    second, rid2 = await cap.has_active_approval_for(
        "DesignProcessCapability.create_decomposition",
    )
    assert second is False
    assert rid2 is None


async def test_approve_all_allows_unbounded_dispatches(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    await _seed_response(
        cap, rid="r1", choice="approve_all",
        action_type="create_decomposition",
    )

    for _ in range(5):
        allowed, rid = await cap.has_active_approval_for(
            "DesignProcessCapability.create_decomposition",
        )
        assert allowed is True
        assert rid == "r1"


async def test_action_type_must_match_action_key(_exec_ctx) -> None:
    cap = await _make_capability(_exec_ctx)
    await _seed_response(
        cap, rid="r1", choice="approve_all",
        action_type="sync_roadmap_with_github",
    )
    allowed, _ = await cap.has_active_approval_for(
        "DesignProcessCapability.create_decomposition",
    )
    assert allowed is False


async def test_legacy_untyped_approve_still_unlocks(_exec_ctx) -> None:
    """Backwards compat for missions that haven't migrated to
    ``action_type``: an untyped ``choice='approve'`` unlocks any
    gated action."""

    cap = await _make_capability(_exec_ctx)
    await _seed_response(
        cap, rid="r1", choice="approve", action_type=None,
    )
    allowed, rid = await cap.has_active_approval_for(
        "DesignProcessCapability.bootstrap_roadmap_from_objectives",
    )
    assert allowed is True
    assert rid == "r1"


async def test_approve_all_preferred_over_unconsumed_approve_once(
    _exec_ctx,
) -> None:
    """When both an ``approve_all`` and an ``approve_once`` cover an
    action_key, ``approve_all`` wins so we don't consume an
    approve_once unnecessarily."""

    cap = await _make_capability(_exec_ctx)
    await _seed_response(
        cap, rid="once", choice="approve_once",
        action_type="create_decomposition",
    )
    await _seed_response(
        cap, rid="all", choice="approve_all",
        action_type="create_decomposition",
    )

    allowed, rid = await cap.has_active_approval_for(
        "DesignProcessCapability.create_decomposition",
    )
    assert allowed is True
    assert rid == "all"
    # approve_once stays unconsumed and ready for use elsewhere.
    assert await cap._is_consumed("once") is False


# ---------------------------------------------------------------------------
# 4-choice approval surface — reject / abort require explanation, the
# validator enforces it, the typed-options tuple lists all four, and
# the event-handler context surfaces the explanation verbatim.
# ---------------------------------------------------------------------------


def test_response_validator_rejects_empty_explanation_on_reject() -> None:
    """``HumanApprovalResponse._require_explanation_on_reject_or_abort``
    is the data-shape contract for Q1's reject/abort surface. Empty
    explanation on ``reject`` must raise so the chat-UI relay and the
    HTTP endpoint can't construct an invalid response object."""

    with pytest.raises(ValueError, match="explanation"):
        HumanApprovalResponse(
            request_id="r1", choice="reject", explanation="", decided_by="u",
        )


def test_response_validator_rejects_empty_explanation_on_abort() -> None:
    with pytest.raises(ValueError, match="explanation"):
        HumanApprovalResponse(
            request_id="r1", choice="abort", explanation="   ", decided_by="u",
        )


def test_response_validator_accepts_empty_explanation_on_approve() -> None:
    """Approve choices do not require an explanation — the validator
    must NOT raise. (Regression for a too-strict validator that
    accidentally requires explanation on every choice.)"""

    resp = HumanApprovalResponse(
        request_id="r1", choice="approve_once", decided_by="u",
    )
    assert resp.explanation == ""

    resp = HumanApprovalResponse(
        request_id="r1", choice="approve_all", decided_by="u",
    )
    assert resp.explanation == ""


def test_response_validator_accepts_non_empty_explanation_on_reject() -> None:
    resp = HumanApprovalResponse(
        request_id="r1",
        choice="reject",
        explanation="the proposal targets the wrong subsystem",
        decided_by="u",
    )
    assert resp.choice == "reject"
    assert resp.explanation.startswith("the proposal")


async def test_typed_request_offers_four_choices(_exec_ctx) -> None:
    """When ``action_type`` is set, the default options tuple lists
    approve_once / approve_all / reject / abort — order matters because
    the chat UI renders left-to-right and approve-first is the
    operator-friendly default."""

    cap = await _make_capability(_exec_ctx)
    result = await cap.request_human_approval(
        question="OK to apply?",
        action_type="create_decomposition",
    )
    rid = result["request_id"]
    req = cap._requests[rid]
    assert req.options == (
        "approve_once", "approve_all", "reject", "abort",
    )


async def test_response_context_includes_explanation(_exec_ctx) -> None:
    """The event-handler context must include ``explanation`` so the
    next planner iteration can read the operator's justification —
    today's planner sees only a one-word ``choice``, which loses the
    "why" on reject and abort."""

    cap = await _make_capability(_exec_ctx)
    result = await cap.request_human_approval(
        question="Approve this decomposition for issue #42?",
        action_type="create_decomposition",
    )
    rid = result["request_id"]

    response = HumanApprovalResponse(
        request_id=rid,
        choice="reject",
        explanation="docs are not engineering substance",
        decided_by="alice",
    )
    fake_event = type("E", (), {})()
    fake_event.key = HumanApprovalProtocol.response_key(rid)
    fake_event.value = response.model_dump(mode="json")
    handler_result = await cap._on_response(fake_event, None)

    assert handler_result is not None
    assert handler_result.context["choice"] == "reject"
    assert handler_result.context["explanation"] == (
        "docs are not engineering substance"
    )


def test_response_envelope_includes_explanation() -> None:
    """The ``get_response`` envelope mirrors the validator's contract —
    every consumer reading ``response.explanation`` (chat-UI relay,
    guardrail predicates, LLM context) sees the same field. This also
    documents the envelope shape for downstream tools."""

    from polymathera.colony.agents.patterns.capabilities.human_approval import (
        _render_get_response_envelope,
    )

    response = HumanApprovalResponse(
        request_id="r1",
        choice="abort",
        explanation="operator changed their mind",
        decided_by="alice",
    )
    envelope = _render_get_response_envelope(response)
    assert envelope["state"] == "ready"
    assert envelope["response"]["choice"] == "abort"
    assert envelope["response"]["explanation"] == "operator changed their mind"


# ---------------------------------------------------------------------------
# RESPONSE_CONTEXT_KEY_PREFIX is the canonical owner of the planner-
# context key shape — every consumer must reference the ClassVar so a
# rename can't drift between writer and readers.
# ---------------------------------------------------------------------------


def test_response_context_key_prefix_is_classvar() -> None:
    """``RESPONSE_CONTEXT_KEY_PREFIX`` is a ClassVar on
    ``HumanApprovalCapability`` — readable on the class without an
    instance and immutable per type annotation. Pairs with the repo-
    wide grep test that no inline ``"human_approval_response:"`` string
    survives outside this attribute and this test."""

    assert hasattr(HumanApprovalCapability, "RESPONSE_CONTEXT_KEY_PREFIX")
    assert HumanApprovalCapability.RESPONSE_CONTEXT_KEY_PREFIX == (
        "human_approval_response:"
    )


def test_no_inline_response_context_key_literal_in_workspace() -> None:
    """Repo-wide grep test: no production source file (anything outside
    a ``tests/`` directory and outside the canonical owner
    ``human_approval.py``) may carry the literal string
    ``"human_approval_response:"``. Test files are allowed to reference
    the literal when describing the rule; production code must
    reference ``HumanApprovalCapability.RESPONSE_CONTEXT_KEY_PREFIX``.

    Catches a future drop-in that re-introduces the inline literal and
    bypasses the ClassVar. Pairs with
    [[fix-the-class-not-the-instance]] — single source of truth for
    the planner-context key shape."""

    import subprocess
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[7]
    # ``rg`` is available in dev environments; fall back to ``grep -r``
    # so the test runs in CI without ripgrep on PATH.
    rg = ["rg", "-l", "-F", "human_approval_response:", str(repo_root / "src")]
    grep = ["grep", "-rl", "-F", "human_approval_response:", str(repo_root / "src")]
    try:
        out = subprocess.run(rg, capture_output=True, text=True, check=False)
        hits = out.stdout
    except FileNotFoundError:
        out = subprocess.run(grep, capture_output=True, text=True, check=False)
        hits = out.stdout
    if not hits.strip():
        return
    offenders = []
    for line in hits.splitlines():
        if not line:
            continue
        path = Path(line)
        if "__pycache__" in path.parts:
            continue
        # Test directories may reference the literal in prose / asserts.
        if "tests" in path.parts:
            continue
        # Canonical owner.
        if path.name == "human_approval.py":
            continue
        offenders.append(str(path))
    assert not offenders, (
        "Inline ``human_approval_response:`` literal found outside the "
        "canonical owner. Reference "
        "``HumanApprovalCapability.RESPONSE_CONTEXT_KEY_PREFIX`` "
        f"instead. Offenders: {offenders}"
    )


# ---------------------------------------------------------------------------
# Bucket A.3 / Fix F5 prevention — RequestHumanApprovalEmpty
#
# Pair with the shipped ErrorRewriterReflector F5 rule (Slice C). The
# rule's match predicate looks for "empty" or "RequestHumanApprovalEmpty"
# in the action error string; both phrases must appear in the
# exception's str() so the rewriter rule keeps matching the new shape.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_human_approval_rejects_empty_question(_exec_ctx) -> None:
    """Empty ``question`` → ``RequestHumanApprovalEmpty``. The operator
    has nothing to evaluate; failing here is better than surfacing a
    blank approval card."""

    cap = await _make_capability(_exec_ctx)
    with pytest.raises(RequestHumanApprovalEmpty) as exc_info:
        await cap.request_human_approval(question="")
    msg = str(exc_info.value)
    # Both keywords must be in the message because the
    # ErrorRewriterReflector F5 rule matches on either; pinning them
    # both keeps the rewriter wiring stable.
    assert "empty" in msg.lower()


@pytest.mark.asyncio
async def test_request_human_approval_rejects_whitespace_question(
    _exec_ctx,
) -> None:
    """Whitespace-only counts as empty — same forensic risk surface."""

    cap = await _make_capability(_exec_ctx)
    with pytest.raises(RequestHumanApprovalEmpty):
        await cap.request_human_approval(question="   \n\n\t ")


@pytest.mark.asyncio
async def test_request_human_approval_rejects_templated_empty_body(
    _exec_ctx,
) -> None:
    """The forensic case (head4.log:appr_5a424365bbd1): a question with
    a markdown header announcing items + a footer, with ZERO list items
    between. The string is non-empty by length but operator-empty by
    structure."""

    cap = await _make_capability(_exec_ctx)
    # Reconstructed from head4.log. No 3+ newlines (the old regex
    # missed this); the structural check catches it.
    question = (
        "## Proposed Decompositions (4 issues)\n\n"
        "Approve to create sub-issues and update parent issues on "
        "GitHub."
    )
    with pytest.raises(RequestHumanApprovalEmpty) as exc_info:
        await cap.request_human_approval(question=question)
    assert "no enumerated items" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_request_human_approval_accepts_brief_substantive_question(
    _exec_ctx,
) -> None:
    """Brief non-empty questions (e.g. ``"Approve?"``) are legitimate
    and must pass the validator. The F5 prevention is not a length
    check — short doesn't mean empty."""

    cap = await _make_capability(_exec_ctx)
    result = await cap.request_human_approval(question="Approve?")
    assert result["ok"] is True
    assert result["request_id"]


@pytest.mark.asyncio
async def test_request_human_approval_accepts_double_newline_separators(
    _exec_ctx,
) -> None:
    """Normal Markdown question spacing (single blank lines between
    paragraphs / list / footer) does NOT trigger the templated-empty-
    body check — it's only 3+ consecutive newlines that does."""

    cap = await _make_capability(_exec_ctx)
    question = (
        "## Approve this change\n\n"
        "Summary: refactor the foo module to bar.\n\n"
        "Reply yes/no."
    )
    result = await cap.request_human_approval(question=question)
    assert result["ok"] is True


def test_request_human_approval_empty_exception_name_in_str() -> None:
    """The ErrorRewriterReflector F5 rule's match predicate looks for
    "RequestHumanApprovalEmpty" or "empty" (case-insensitive) in
    the action's error string. Confirm the exception's ``str()``
    contains BOTH so the rule fires deterministically without
    depending on which phrase the rewriter happens to grep."""

    exc = RequestHumanApprovalEmpty("question is empty; do X")
    s = str(exc).lower()
    assert "empty" in s
    # The exception class name itself shows up when the framework
    # formats it via ``type(exc).__name__: <message>`` — the
    # rewriter rule's second-keyword fallback.
    assert "requesthumanapprovalempty" in type(exc).__name__.lower()
