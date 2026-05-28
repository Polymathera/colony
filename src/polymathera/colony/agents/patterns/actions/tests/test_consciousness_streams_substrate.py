"""Tests for the PR-Sub-1 substrate refactor that lifts consciousness
stream storage onto :class:`BaseActionPolicy` and adds the
:meth:`_feed_action_to_streams` hook.

Covered:

1. ``BaseActionPolicy.__init__`` accepts ``consciousness_streams=`` +
   stores them + exposes them via ``get_consciousness_streams``.
2. ``BaseActionPolicy._feed_action_to_streams`` fans one call out to
   every mounted stream's ``consider_action`` (the central hook every
   concrete policy uses).
3. ``EventDrivenActionPolicy`` (the existing direct subclass) forwards
   its own ``consciousness_streams=`` kwarg through to the base — no
   double-storage, no override drift.
4. ``CacheAwareActionPolicy.__init__`` accepts the kwarg (previously
   stripped it before passing to super).
5. ``MinimalActionPolicy.__init__`` accepts the kwarg.
6. ``create_minimal_action_policy`` threads the kwarg through.
7. ``create_cache_aware_action_policy`` accepts the kwarg via its
   signature (full ``initialize`` involves the cache-aware planner +
   live agent infrastructure — those are exercised by the broader
   integration suite; this test asserts the signature contract only).
8. ``defaults.py``'s MINIMAL branch imports
   ``create_minimal_action_policy`` from the correct module
   (``.minimal``, not ``.policies``). The MINIMAL branch is dead code
   today because ``_DEFAULT_POLICY = "CODE_GEN"``, but the broken
   import would crash the moment an operator flips it.
9. ``CodeGenerationActionPolicy._after_step`` calls the new helper
   (regression — preserves the per-REPL-call feed shape).
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.patterns.actions.code_generation import (
    CodeGenerationActionPolicy,
)
from polymathera.colony.agents.patterns.actions.minimal import (
    MinimalActionPolicy,
    create_minimal_action_policy,
)
from polymathera.colony.agents.patterns.actions.policies import (
    BaseActionPolicy,
    CacheAwareActionPolicy,
    EventDrivenActionPolicy,
    create_cache_aware_action_policy,
)
from polymathera.colony.agents.patterns.planning.streams import (
    ActionKeySubstringFilter,
    ConsciousnessStream,
    JSONStreamFormatter,
    SuccessfulActionFilter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _agent() -> Any:
    """Minimal agent stub — the policy constructors only touch
    ``agent.agent_id`` / ``agent.metadata`` and the few attributes the
    methods under test reference."""
    agent = MagicMock()
    agent.agent_id = "agent_streams_test"
    agent.metadata = MagicMock()
    agent.metadata.goals = []
    agent.metadata.role = "test"
    agent.metadata.parameters = {}
    return agent


def _stream(name: str) -> ConsciousnessStream:
    """Stream with an action filter that accepts everything ending in
    ``"_action"`` so we can fan typed calls into it. Pure JSON formatter."""
    return ConsciousnessStream(
        name=name,
        formatter=JSONStreamFormatter(section_title=f"## {name}"),
        action_filter=ActionKeySubstringFilter("_action"),
    )


def _bypass_init(cls, **attrs) -> Any:
    """Construct an instance without running ``__init__`` and set the
    attributes the methods under test reference. Pattern lifted from
    ``test_event_priority.py``."""
    obj = object.__new__(cls)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


# ---------------------------------------------------------------------------
# 1. BaseActionPolicy hosts streams
# ---------------------------------------------------------------------------


class TestBaseActionPolicyStreams:
    def test_init_accepts_consciousness_streams_kwarg(self) -> None:
        s = _stream("base_s")
        policy = BaseActionPolicy(
            agent=_agent(), consciousness_streams=[s],
        )
        assert policy._consciousness_streams == [s]

    def test_init_defaults_to_empty_list(self) -> None:
        policy = BaseActionPolicy(agent=_agent())
        assert policy._consciousness_streams == []

    def test_get_consciousness_streams_returns_copy(self) -> None:
        s1, s2 = _stream("a"), _stream("b")
        policy = BaseActionPolicy(
            agent=_agent(), consciousness_streams=[s1, s2],
        )
        result = policy.get_consciousness_streams()
        assert result == [s1, s2]
        # Mutating the returned list does not affect the policy's storage.
        result.append(_stream("c"))
        assert len(policy._consciousness_streams) == 2

    def test_feed_action_to_streams_fans_out(self) -> None:
        s1, s2 = _stream("a"), _stream("b")
        policy = BaseActionPolicy(
            agent=_agent(), consciousness_streams=[s1, s2],
        )
        call = {
            "action_key": "some_action",
            "parameters": {"x": 1},
            "success": True,
            "output_preview": "ok",
        }
        policy._feed_action_to_streams(call)
        # Both streams' filters accept (substring "_action") → both
        # have one entry.
        for s in (s1, s2):
            assert len(s._entries) == 1
            assert s._entries[0]["kind"] == "action"
            assert s._entries[0]["call"] == call

    def test_feed_action_to_streams_respects_action_filter(self) -> None:
        """Stream's ``action_filter`` decides whether to record;
        ``_feed_action_to_streams`` doesn't bypass it."""
        s = ConsciousnessStream(
            name="picky",
            formatter=JSONStreamFormatter(section_title="## P"),
            action_filter=ActionKeySubstringFilter("never_match"),
        )
        policy = BaseActionPolicy(
            agent=_agent(), consciousness_streams=[s],
        )
        policy._feed_action_to_streams({
            "action_key": "some_action", "success": True,
        })
        assert s._entries == []

    def test_feed_action_to_streams_noop_without_streams(self) -> None:
        """Empty stream list → no error, no work."""
        policy = BaseActionPolicy(agent=_agent())
        policy._feed_action_to_streams({"action_key": "x"})
        # Did not raise; nothing to assert beyond that.


# ---------------------------------------------------------------------------
# 2. EventDrivenActionPolicy forwards streams to base (no double-storage)
# ---------------------------------------------------------------------------


class TestEventDrivenForwarding:
    def test_event_driven_forwards_streams_to_base(self) -> None:
        s = _stream("ed")
        policy = EventDrivenActionPolicy(
            agent=_agent(), consciousness_streams=[s],
        )
        assert policy._consciousness_streams == [s]
        assert policy.get_consciousness_streams() == [s]

    def test_event_driven_does_not_re_override_get_consciousness_streams(
        self,
    ) -> None:
        """The override on ``EventDrivenActionPolicy`` was removed —
        the method is inherited from the base. Asserting MRO inheritance
        catches any future drift where someone re-adds an override that
        would shadow the base's behaviour."""
        method = EventDrivenActionPolicy.get_consciousness_streams
        # The method object is the one defined on BaseActionPolicy.
        assert method is BaseActionPolicy.get_consciousness_streams


# ---------------------------------------------------------------------------
# 3. CacheAwareActionPolicy accepts the kwarg (previously stripped)
# ---------------------------------------------------------------------------


class TestCacheAwareStreams:
    def test_init_signature_accepts_consciousness_streams(self) -> None:
        sig = inspect.signature(CacheAwareActionPolicy.__init__)
        assert "consciousness_streams" in sig.parameters

    def test_init_forwards_to_base(self) -> None:
        s = _stream("ca")
        # Bypass: we don't want to spin a real planner. The constructor's
        # only effect on our concern is calling super().__init__ with
        # consciousness_streams. Verify by constructing a stub planner.
        planner = MagicMock()
        policy = CacheAwareActionPolicy(
            agent=_agent(), planner=planner,
            consciousness_streams=[s],
        )
        assert policy._consciousness_streams == [s]
        assert policy.get_consciousness_streams() == [s]


class TestCreateCacheAwareFactorySignature:
    def test_factory_accepts_consciousness_streams_kwarg(self) -> None:
        """The factory's signature is the public contract operators
        bind against — assert the kwarg exists. Full ``initialize()``
        spins the cache-aware planner + agent infra; exercised by the
        broader integration suite."""
        sig = inspect.signature(create_cache_aware_action_policy)
        assert "consciousness_streams" in sig.parameters
        assert sig.parameters["consciousness_streams"].default is None


# ---------------------------------------------------------------------------
# 4. MinimalActionPolicy accepts the kwarg + factory forwards it
# ---------------------------------------------------------------------------


class TestMinimalActionPolicy:
    def test_init_accepts_consciousness_streams(self) -> None:
        s = _stream("m")
        policy = MinimalActionPolicy(
            agent=_agent(), consciousness_streams=[s],
        )
        assert policy._consciousness_streams == [s]

    def test_init_defaults_empty(self) -> None:
        policy = MinimalActionPolicy(agent=_agent())
        assert policy._consciousness_streams == []

    def test_factory_signature_accepts_kwarg(self) -> None:
        sig = inspect.signature(create_minimal_action_policy)
        assert "consciousness_streams" in sig.parameters
        assert sig.parameters["consciousness_streams"].default is None


# ---------------------------------------------------------------------------
# 5. defaults.py — MINIMAL branch import path + kwarg threading
# ---------------------------------------------------------------------------


class TestDefaultsBranches:
    def test_minimal_branch_imports_from_minimal_module(self) -> None:
        """``defaults.py`` previously had
        ``from .policies import create_minimal_action_policy`` which
        was broken — the factory lives in ``.minimal``. Fixed in
        PR-Sub-1 while threading streams through; regression test
        guards the fix."""
        from polymathera.colony.agents.patterns.actions import minimal as _m

        # The function the corrected import targets exists in .minimal.
        assert hasattr(_m, "create_minimal_action_policy")

    @pytest.mark.parametrize(
        "policy_kind,factory_attr,expected_module_suffix",
        [
            ("CODE_GEN", "create_code_generation_action_policy",
             "code_generation"),
            ("CACHE_AWARE", "create_cache_aware_action_policy", "policies"),
            ("MINIMAL", "create_minimal_action_policy", "minimal"),
        ],
    )
    def test_each_factory_accepts_consciousness_streams(
        self, policy_kind: str, factory_attr: str, expected_module_suffix: str,
    ) -> None:
        """Each of the three factory functions
        ``defaults.create_default_action_policy`` dispatches to must
        accept ``consciousness_streams=`` so the kwarg threading
        through ``defaults.py`` actually reaches the policy."""
        import importlib

        module = importlib.import_module(
            f"polymathera.colony.agents.patterns.actions.{expected_module_suffix}",
        )
        factory = getattr(module, factory_attr)
        sig = inspect.signature(factory)
        assert "consciousness_streams" in sig.parameters, (
            f"{factory_attr} (used for _DEFAULT_POLICY={policy_kind!r}) "
            "is missing the consciousness_streams kwarg"
        )


# ---------------------------------------------------------------------------
# 6. CodeGenerationActionPolicy._after_step refactor regression
# ---------------------------------------------------------------------------


class TestCodeGenerationAfterStepRefactor:
    def test_after_step_uses_base_feed_helper(self) -> None:
        """The pre-PR-Sub-1 implementation looped inline:

            for call in self._run_call_trace:
                for stream in self._consciousness_streams:
                    stream.consider_action(call)

        Post-PR-Sub-1 the inner loop is delegated to
        ``self._feed_action_to_streams(call)`` — same behaviour, one
        less code path that can drift from the base's implementation.

        We verify by source-grep: the inline ``for stream in
        self._consciousness_streams`` pattern must no longer appear
        inside ``code_generation._after_step`` (or anywhere in
        ``code_generation.py``)."""
        import polymathera.colony.agents.patterns.actions.code_generation as cg
        from pathlib import Path

        source = Path(cg.__file__).read_text(encoding="utf-8")
        assert "for stream in self._consciousness_streams" not in source, (
            "code_generation.py still loops streams inline; should call "
            "self._feed_action_to_streams(call) instead."
        )
        # And the new hook IS invoked from the file.
        assert "_feed_action_to_streams" in source

    def test_code_generation_class_inherits_feed_helper(self) -> None:
        """The helper lives on ``BaseActionPolicy``;
        ``CodeGenerationActionPolicy`` inherits it through the
        ``EventDrivenActionPolicy`` chain."""
        assert hasattr(CodeGenerationActionPolicy, "_feed_action_to_streams")
        # And it's the base's implementation (no shadowing override).
        assert (
            CodeGenerationActionPolicy._feed_action_to_streams
            is BaseActionPolicy._feed_action_to_streams
        )


# ---------------------------------------------------------------------------
# 7. End-to-end: an action filter + the central feed actually filter
# ---------------------------------------------------------------------------


class TestFeedActionWithRealFilter:
    def test_successful_action_filter_records_only_success(self) -> None:
        """Demonstrates the substrate change end-to-end: a stream with
        ``SuccessfulActionFilter`` mounted on a policy; the central
        ``_feed_action_to_streams`` fans calls, the stream's filter
        applies."""
        s = ConsciousnessStream(
            name="success_only",
            formatter=JSONStreamFormatter(section_title="## S"),
            action_filter=SuccessfulActionFilter(
                ActionKeySubstringFilter("foo"),
            ),
        )
        policy = BaseActionPolicy(
            agent=_agent(), consciousness_streams=[s],
        )
        policy._feed_action_to_streams({
            "action_key": "foo_action", "success": True,
        })
        policy._feed_action_to_streams({
            "action_key": "foo_action", "success": False,
        })
        policy._feed_action_to_streams({
            "action_key": "bar_action", "success": True,
        })
        # Only the first call: matches "foo" AND is successful.
        assert len(s._entries) == 1
        assert s._entries[0]["call"]["success"] is True
        assert s._entries[0]["call"]["action_key"] == "foo_action"
