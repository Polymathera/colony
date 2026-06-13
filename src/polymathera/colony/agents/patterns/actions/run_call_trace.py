"""Typed view over the per-iteration call trace.

The codegen action policy maintains ``self._run_call_trace`` as a
``list[dict[str, Any]]`` whose entries are appended at four sites in
``code_generation.py``: action dispatch (success / failure), guardrail
block, and the two ``signal_completion`` branches (allowed / rejected).
The storage is untyped because the writes were authored ad-hoc; readers
that ``dict.get("action_key")`` against this storage are silently
exposed to schema rot — a future rename or nested restructure breaks
every consumer at runtime, not at import.

This module provides the BOUNDARY VIEW: consumers (the decompose-
completion validator, future audit tools, signature pins) construct
:class:`RunCallTrace` from the raw list. The pydantic model enforces
shape at construction time; any drift in the writer's payload surfaces
loudly the moment a consumer instantiates the view, NOT in production
when a downstream condition silently misclassifies. The writers in
``code_generation.py`` are deliberately UNCHANGED — we add a read view,
not a migration.

See ``decompose_one_and_done_and_spinner_plan.md`` Change 4 for the
design rationale: this is the [[no-getattr-defaults]] principle
applied to dict reads — model absence explicitly via pydantic
validation, do not paper over with ``.get(default)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError


class RunCallTraceEntry(BaseModel):
    """One entry in ``self._run_call_trace``.

    Fields match the four append sites in ``code_generation.py``.
    ``parameters`` is optional because the ``signal_completion``
    branches do not record kwargs (only action dispatches do); every
    other field is mandatory so a missing one surfaces as a
    ``ValidationError`` at view construction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_index: int = Field(ge=0)
    action_key: str = Field(min_length=1)
    parameters: dict[str, Any] | None = Field(default=None)
    # Use ``StrictBool`` so a writer drift like ``"yes"`` /
    # ``1`` / ``None`` for ``success`` / ``blocked`` surfaces as a
    # ``ValidationError`` here, not as a silently-coerced True downstream.
    success: StrictBool
    error: str | None = Field(default=None)
    output_preview: str = Field(default="")
    blocked: StrictBool = Field(default=False)


class RunCallTrace:
    """Typed read view over the codegen policy's
    ``_run_call_trace`` list.

    Construct from the raw list at the consumer boundary; the
    underlying storage in ``CodeGenerationActionPolicy`` is unchanged.
    Construction validates every entry against
    :class:`RunCallTraceEntry`; a schema mismatch raises
    ``ValidationError`` — the consumer learns at the read site, not in
    a silent miscount downstream.

    The view exposes only READ-shaped helpers; no mutation, no order
    rewrites. Consumers compose the helpers (``calls_to``,
    ``successful_calls_to``) as needed; the view does NOT bake a
    fixed pipeline of "find then filter then count" — per
    [[primitives-not-pipelines]] the read primitives are orthogonal
    and the consumer chooses how to combine them.
    """

    __slots__ = ("_entries",)

    def __init__(self, raw: Iterable[dict[str, Any]]) -> None:
        # Materialise eagerly so validation fires at construction.
        # ``RunCallTraceEntry`` raises ``ValidationError`` on any
        # missing/extra field — the loud-failure mode we want for
        # schema drift detection.
        self._entries: tuple[RunCallTraceEntry, ...] = tuple(
            RunCallTraceEntry.model_validate(item) for item in raw
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[RunCallTraceEntry]:
        return iter(self._entries)

    def __getitem__(self, idx: int) -> RunCallTraceEntry:
        return self._entries[idx]

    def calls_to(self, action_key: str) -> tuple[RunCallTraceEntry, ...]:
        """Every entry whose ``action_key`` exactly matches.

        Use a canonical action-key constant from the owning capability
        (e.g. ``DesignProcessCapability.CREATE_DECOMPOSITION_ACTION_KEY``)
        rather than a string literal — string literals are the
        equivalent of ``getattr(default)`` for static identifiers per
        [[colony-scoped-params-propagation]] applied to action keys.
        """

        return tuple(e for e in self._entries if e.action_key == action_key)

    def successful_calls_to(
        self, action_key: str,
    ) -> tuple[RunCallTraceEntry, ...]:
        """``calls_to(action_key)`` filtered to ``success=True`` and
        ``blocked=False`` — the canonical "this action actually applied"
        predicate. Failed attempts and guardrail-blocked attempts are
        excluded.
        """

        return tuple(
            e for e in self.calls_to(action_key)
            if e.success and not e.blocked
        )

    def parameters_of_successful(
        self,
        action_key: str,
        parameter_key: str,
    ) -> tuple[Any, ...]:
        """Pull a specific parameter value from every successful call
        to ``action_key``. Skips entries whose ``parameters`` dict is
        absent or does not carry ``parameter_key`` — the absence is
        meaningful (e.g. ``signal_completion`` has no parameters dict),
        not a missing default to paper over.
        """

        out: list[Any] = []
        for entry in self.successful_calls_to(action_key):
            if entry.parameters is None:
                continue
            if parameter_key not in entry.parameters:
                continue
            out.append(entry.parameters[parameter_key])
        return tuple(out)


__all__ = ("RunCallTrace", "RunCallTraceEntry")
