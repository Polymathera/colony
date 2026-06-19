"""Regression tests for ``BaseActionPolicy._get_object_event_handlers``.

The handler-discovery code walks every attribute on every capability /
action-provider to find ``@event_handler``-decorated methods. The
implementation must:

1. Skip ``@property`` descriptors (and similar non-method descriptors)
   without evaluating them. Evaluating a property invokes its body,
   and capabilities with I/O properties (e.g.,
   ``DesignProcessCapability.current_branch`` which opens a git repo)
   can raise on missing filesystem state — collapsing the entire
   handler-discovery path. This was the silent failure surfaced via
   ``[Bus] handlers_block_raised`` in head5.log.
2. Continue past one bad ``getattr`` so a single attribute surprise
   (a ``__getattr__`` that raises on some names) does not poison the
   rest of the handler list.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.patterns.actions.policies import (
    EventDrivenActionPolicy,
)
from polymathera.colony.agents.patterns.events import event_handler


# A minimal stand-in that exposes the discovery method without booting
# the full policy/agent stack.
class _Discoverer:
    def __init__(self, agent):
        self.agent = agent

    # Same code as the live implementation — delegate so the test
    # exercises the production discovery path.
    _get_object_event_handlers = (
        EventDrivenActionPolicy._get_object_event_handlers
    )


class _CapabilityWithRaisingProperty:
    """Mimics a real capability shape: one @event_handler method
    alongside a property that does I/O and raises on missing state."""

    @property
    def current_branch(self) -> str:
        raise RuntimeError(
            "filesystem not initialized — would normally be a "
            "NoSuchPathError from gitpython on a missing clone"
        )

    @property
    def working_dir(self) -> str:
        # Defensive but also raises. The discovery path must not
        # evaluate either property.
        raise FileNotFoundError("/mnt/shared/...")

    @event_handler(pattern="x:*")
    async def on_x_event(self, event, repl):  # noqa: ARG002
        return None


class _CapabilityWithLandminedAttr:
    """Mimics a capability with a ``__getattr__`` that raises on a
    specific name — ensures one bad attribute can't kill discovery
    of the rest."""

    @event_handler(pattern="y:*")
    async def on_y_event(self, event, repl):  # noqa: ARG002
        return None

    def __getattr__(self, name):
        if name == "landmine_attr":
            raise RuntimeError("don't touch me")
        raise AttributeError(name)


@pytest.fixture
def discoverer():
    agent = MagicMock()
    agent.agent_id = "agent-test"
    return _Discoverer(agent)


def test_property_raising_does_not_break_discovery(discoverer):
    """The real-world bug from head5.log: a capability has a property
    that raises (e.g., touches a missing git clone). Discovery must
    skip the property and still find the @event_handler method."""

    cap = _CapabilityWithRaisingProperty()
    handlers = discoverer._get_object_event_handlers(cap)
    handler_names = [h.__name__ for h in handlers]
    assert handler_names == ["on_x_event"]


def test_property_is_not_evaluated_during_discovery(discoverer):
    """Even when a property would NOT raise, the discovery code MUST
    NOT call it — properties that touch external state shouldn't be
    invoked as a side effect of introspection. A counter on the
    property tracks invocations."""

    invocations = {"current_branch": 0}

    class _Cap:
        @property
        def current_branch(self) -> str:
            invocations["current_branch"] += 1
            return "main"

        @event_handler(pattern="x:*")
        async def on_x_event(self, event, repl):  # noqa: ARG002
            return None

    cap = _Cap()
    discoverer._get_object_event_handlers(cap)
    assert invocations["current_branch"] == 0


def test_one_bad_getattr_does_not_break_other_handlers(discoverer):
    """A capability with a __getattr__ that raises on a specific name
    must not block discovery of the legitimate handlers."""

    cap = _CapabilityWithLandminedAttr()
    # Force the landmine into ``dir()`` so the discovery walk encounters
    # it. Otherwise ``dir()`` only returns the class-level + bound
    # attrs and never asks for ``landmine_attr``.
    cap.__dict__["landmine_attr"] = (
        None  # makes ``dir()`` list it; getattr will hit __getattr__
    )
    handlers = discoverer._get_object_event_handlers(cap)
    handler_names = [h.__name__ for h in handlers]
    assert "on_y_event" in handler_names


def test_staticmethod_and_classmethod_are_skipped(discoverer):
    """staticmethod / classmethod descriptors are never event handlers
    and discovery should skip them without evaluating."""

    class _Cap:
        @staticmethod
        def helper_static():
            return 1

        @classmethod
        def helper_classmethod(cls):
            return 2

        @event_handler(pattern="z:*")
        async def on_z_event(self, event, repl):  # noqa: ARG002
            return None

    cap = _Cap()
    handlers = discoverer._get_object_event_handlers(cap)
    handler_names = [h.__name__ for h in handlers]
    assert handler_names == ["on_z_event"]
