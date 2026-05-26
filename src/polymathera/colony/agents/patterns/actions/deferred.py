"""Deferred-closure primitive — extract a serializable callable from
an ``@action_executor`` method without coupling the callable to the
agent process.

Three pieces:

- :class:`DeferredClosure` — abstract base for the serialisable
  callable. Authors subclass per action and bundle any context the
  callable needs (handles, sandbox, HTTP client, configs, …) via the constructor's
  ``**context`` kwargs. The capability itself is never stored; only
  its FQN + tool name + the agent's current run-id are snapshotted
  for provenance.
- :func:`is_eager_execution` / :func:`eager_execution` — ContextVar-
  backed toggle the framework flips when extracting a closure rather
  than invoking it.
- :func:`deferred` — decorator that wraps an ``@action_executor``
  method to hide the dual-mode boilerplate. In eager mode the
  decorator invokes the closure the body returned and surfaces the
  result; in deferred-extraction mode it returns the closure object
  so the caller can ship it elsewhere.

The pattern is an incremental extension of the existing
``@action_executor`` model: a tool author adds one decorator + a
sibling closure subclass per action that the framework should be
able to extract as a serialisable unit of work. Nothing else
changes; non-extractable actions stay regular ``@action_executor``
methods.

Use cases include any framework that wants to lift work out of an
agent's process and dispatch it elsewhere: deferred queues, worker
pools, replay systems, downstream task substrates.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Generic, TypeVar


R = TypeVar("R")


# ---------------------------------------------------------------------------
# Eager-execution context — flipped to False by the framework when
# extracting a closure rather than invoking it.
# ---------------------------------------------------------------------------


_EAGER: ContextVar[bool] = ContextVar(
    "polymathera_eager_execution", default=True,
)


def is_eager_execution() -> bool:
    """Return ``True`` if the current context is eagerly invoking
    ``@deferred`` actions (the default), ``False`` if the framework
    has temporarily switched to deferred-extraction mode."""
    return _EAGER.get()


@contextmanager
def eager_execution(value: bool):
    """Temporarily set the eager-execution flag.

    Used by downstream extractors that want to obtain the
    :class:`DeferredClosure` an ``@deferred`` action's body builds
    without invoking it. ContextVar-based so asyncio task spawns
    inherit the right value without manual plumbing.
    """
    tok = _EAGER.set(value)
    try:
        yield
    finally:
        _EAGER.reset(tok)


# ---------------------------------------------------------------------------
# DeferredClosure — the serialisable callable
# ---------------------------------------------------------------------------


class DeferredClosure(Generic[R], ABC):
    """Abstract base for the serialisable callable extracted from an
    ``@action_executor`` method.

    Lifecycle (TensorFlow-style build / compile / execute):

    1. **Build** — the action body constructs the closure via
       ``__init__``. Snapshots ``tool_name`` + ``capability_fqn`` +
       ``run_id`` from the capability and stashes a transient
       reference to ``capability`` for :meth:`compile` to use.
    2. **Compile** (:meth:`compile`) — resolves any sub-closures
       (composites override; leaves inherit the default no-op),
       clears the transient capability reference. Called
       automatically by the :func:`deferred` decorator after the
       action body returns. Idempotent.
    3. **Execute** (``__call__``) — runs the work using the
       (already-compiled) sub-closures + the static context.

    Authors subclass per action and:

    - declare the context items the callable needs as keyword args
      in ``__init__`` (or extract them in :meth:`compile`), and
    - implement ``async __call__`` to do the work using those items
      via :meth:`get_context_by_key`.
    - For composite closures (one closure invoking other closures),
      override :meth:`compile` to extract sub-closures from the
      capability under :func:`eager_execution` ``(False)``.

    Cloudpickle contract: every context value passed via the
    ``**context`` kwargs of :meth:`__init__` MUST be serialisable.
    Authors are responsible for satisfying that — the base does
    nothing to enforce it; downstream extractors enforce by failing
    at cloudpickle-dumps time if a closure can't be shipped. The
    transient ``capability`` reference is cleared by :meth:`compile`
    before any cloudpickle round-trip.

    Provenance snapshot: the capability instance is read for its
    :attr:`spec.name` (tool name) + the agent's current run-id, then
    the reference is held only until :meth:`compile` runs. The
    closure carries the snapshots as plain strings so it can build
    provenance metadata without reaching back to a (no longer
    available) parent agent.
    """

    def __init__(self, capability: Any, **context: Any) -> None:
        self.context: dict[str, Any] = dict(context)
        cls = type(capability)
        self.capability_fqn: str = f"{cls.__module__}.{cls.__qualname__}"
        # Prefer ToolSpec.name when present; fall back to class name.
        spec = getattr(cls, "spec", None)
        self.tool_name: str = getattr(spec, "name", cls.__name__)
        # run_id snapshot via the public surface: gate on
        # ``is_detached`` (public boolean), then read via the public
        # ``agent`` property + ``agent.metadata.run_id`` — the same
        # convention :class:`AgentTracingFacility` uses
        # (``getattr(self.agent.metadata, "run_id", None)`` at
        # observability/facility.py). Empty string when detached or
        # when no run_id has been set on the agent's metadata.
        self.run_id: str = ""
        if not getattr(capability, "is_detached", True):
            self.run_id = (
                getattr(capability.agent.metadata, "run_id", "") or ""
            )
        # Transient capability reference — used by :meth:`compile`
        # to extract sub-closures and/or read additional capability
        # state. Cleared by :meth:`compile` so the closure is
        # cloudpickle-safe afterwards. Single underscore: subclasses
        # may access via inheritance; not part of the cross-module
        # public surface.
        self._capability_pre_compile: Any = capability
        self._compiled: bool = False

    @property
    def is_compiled(self) -> bool:
        """``True`` once :meth:`compile` has run for this closure."""
        return self._compiled

    async def compile(self) -> None:
        """Resolve sub-closures + finalize the closure for
        cloudpickle transfer.

        Default: no-op (leaves the closure as-is, just drops the
        transient capability reference + marks compiled). Composite
        closures override to extract sub-closures from the parent
        capability under :func:`eager_execution` ``(False)``, e.g.::

            async def compile(self) -> None:
                if self.is_compiled:
                    return
                cap = self._capability_pre_compile
                with eager_execution(False):
                    self.mu_metal = await cap.model_mu_metal_hysteresis()
                    self.active   = await cap.synthesize_active_shielding()
                await super().compile()  # clears capability + marks compiled

        Composites do NOT recursively call ``await sub.compile()``
        themselves; the :func:`deferred` decorator auto-compiles
        every closure it returns, so sub-closures arrive
        already-compiled.

        Idempotent: re-calling :meth:`compile` on a compiled closure
        is a cheap no-op.

        Called automatically by the :func:`deferred` decorator after
        the action body returns. Tests that build closures directly
        (bypassing the decorator) must call ``await closure.compile()``
        manually before invoking the closure or shipping it through
        cloudpickle.
        """
        if self._compiled:
            return
        self._compiled = True
        self._capability_pre_compile = None

    def get_context_by_key(self, key: str) -> Any:
        """Look up a context value the closure was constructed with.

        Returns ``None`` for missing keys — authors that require a
        key should raise explicitly in ``__call__``.
        """
        return self.context.get(key)

    @abstractmethod
    async def __call__(self, *args: Any, **kwargs: Any) -> R: ...


# ---------------------------------------------------------------------------
# @deferred decorator — dual-mode dispatch for action methods
# ---------------------------------------------------------------------------


def deferred(
    action_method: Callable[..., Awaitable[DeferredClosure[R]]],
) -> Callable[..., Awaitable[Any]]:
    """Decorator: mark an ``@action_executor`` method as a deferred
    action — its body returns a :class:`DeferredClosure`, and the
    framework either invokes the closure (eager mode, default) or
    surfaces it for downstream extraction (deferred mode).

    Usage::

        @action_executor(planning_summary="...")
        @deferred
        async def do_work(
            self, *, target: str | None = None,
        ) -> DoWorkClosure:
            handle = await self._build_handle()
            return DoWorkClosure(capability=self, handle=handle)

    In eager mode (default agent dispatch via the action dispatcher),
    the decorator invokes the returned closure with the same kwargs
    the caller supplied and returns the closure's result.

    In deferred mode (set by an extractor that calls the action under
    :func:`eager_execution`\\ ``(False)``), the decorator returns the
    closure object directly so the caller can ship it elsewhere
    without invoking it.

    Type contract: the action body MUST return a
    :class:`DeferredClosure` instance. Returning anything else raises
    ``TypeError`` immediately (no silent passthrough). This is the
    correct-by-construction guarantee that bodies don't accidentally
    do the work themselves.

    Kwarg contract: the action's kwargs SHOULD all have defaults.
    Downstream extractors typically call the action with limited or
    no caller-supplied kwargs at extraction time; the action body's
    only job is to assemble context (handles, configs, …) from
    ``self`` + module state — it typically does not reference the
    kwargs at all. The runtime values flow into the closure's
    ``__call__`` later, where its signature can be strict.
    """

    @functools.wraps(action_method)
    async def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        closure = await action_method(self, *args, **kwargs)
        if not isinstance(closure, DeferredClosure):
            raise TypeError(
                f"@deferred action {action_method.__qualname__!r} must "
                f"return a DeferredClosure; got "
                f"{type(closure).__name__}. The action body's only "
                "job is to assemble context and return the closure "
                "object — never to do the work directly.",
            )
        # Auto-compile the closure before either invoking it
        # (eager mode) or surfacing it (deferred mode). compile()
        # extracts sub-closures (composites override) + clears the
        # transient capability reference, leaving a cloudpickle-safe
        # tree. Recursive: composite ``compile()`` extracts
        # sub-closures by awaiting their actions, which run through
        # this same decorator and arrive already-compiled.
        await closure.compile()
        if is_eager_execution():
            return await closure(*args, **kwargs)
        return closure

    _wrapped._is_deferred_action = True  # type: ignore[attr-defined]
    return _wrapped


__all__ = (
    "DeferredClosure",
    "deferred",
    "eager_execution",
    "is_eager_execution",
)
