"""Contract-drift reflector.

Single :class:`StreamReflector` that subsumes the prior pair
``ContractDriftDetector`` + ``ContractDriftAdvisor``. Watches each
iteration's :class:`IterationObservation`; when the LLM re-calls a
typed-discriminator action (today: ``spawn_mission``) that already
returned a terminal outcome in an earlier iteration, emits one
``typed_discriminator_drift`` advisory plus one ``contract_drift``
diagnostic naming the prior outcome + the existing agent_id.

The cross-iteration ``_prior_terminal`` map is the reflector's private
state, persisted via :meth:`serialize_state` so suspend/resume keeps
the drift signal correct across restarts."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, ClassVar

from ..models import (
    AdvisoryEntry,
    Diagnostic,
    IterationObservation,
    ReflectMoment,
    StreamReflection,
)
from ..reflection import (
    StreamReflector,
)

logger = logging.getLogger(__name__)


_TerminalKey = tuple[str, tuple]


class ContractRegistration:
    """One action's registration: which outcome values terminate retry,
    and how to extract the per-call identity the reflector matches
    across iterations.

    ``key_extractor(params)`` returns the tuple identifying "this kind
    of call" — for ``spawn_mission`` that's ``(mission_type,)``. Tuple
    keeps the value hashable for the prior-terminal map."""

    def __init__(
        self,
        *,
        action_key_suffix: str,
        terminal_outcomes: frozenset[str],
        key_extractor: Callable[[dict], tuple],
    ) -> None:
        self.action_key_suffix = action_key_suffix
        self.terminal_outcomes = terminal_outcomes
        self.key_extractor = key_extractor


def _spawn_mission_key(params: dict) -> tuple:
    mt = params.get("mission_type") if isinstance(params, dict) else None
    return (mt,) if mt else ()


_DEFAULT_REGISTRATIONS: tuple[ContractRegistration, ...] = (
    ContractRegistration(
        action_key_suffix="spawn_mission",
        terminal_outcomes=frozenset(
            {"spawned", "return_existing", "rejected"},
        ),
        key_extractor=_spawn_mission_key,
    ),
)


class ContractDriftReflector(StreamReflector):
    """Emits one advisory + diagnostic per ``(action_suffix, key)`` pair
    where the current iteration repeats an action that returned a
    terminal outcome in a PRIOR iteration."""

    name = "contract_drift"

    REFLECT_AT: ClassVar[frozenset[ReflectMoment]] = frozenset(
        {"iteration_boundary"},
    )

    def __init__(
        self,
        registrations: tuple[ContractRegistration, ...] | None = None,
    ) -> None:
        self._registrations: list[ContractRegistration] = list(
            registrations if registrations is not None
            else _DEFAULT_REGISTRATIONS
        )
        self._prior_terminal: dict[_TerminalKey, dict] = {}

    def register(self, registration: ContractRegistration) -> None:
        self._registrations.append(registration)

    def reflect(
        self,
        *,
        entries: list[dict[str, Any]],  # noqa: ARG002
        observation: IterationObservation | None,
        moment: ReflectMoment,  # noqa: ARG002
    ) -> StreamReflection:
        if observation is None:
            return StreamReflection()

        advisories: list[AdvisoryEntry] = []
        diagnostics: list[Diagnostic] = []
        # 1. Check current calls against PRIOR-iteration terminals.
        for rec in observation.actions_called:
            for reg in self._registrations:
                if not rec.action_key.endswith("." + reg.action_key_suffix):
                    continue
                key = reg.key_extractor(rec.params or {})
                prior = self._prior_terminal.get(
                    (reg.action_key_suffix, key),
                )
                if prior is None or prior["iter_index"] == observation.iter_index:
                    continue
                advisory, diagnostic = _build_drift_outputs(
                    rec=rec,
                    suffix=reg.action_key_suffix,
                    key=key,
                    prior=prior,
                    current_iter=observation.iter_index,
                )
                advisories.append(advisory)
                diagnostics.append(diagnostic)

        # 2. Record NEW terminal outcomes from THIS iteration so the
        #    next iteration's check sees them.
        for rec in observation.actions_called:
            if rec.status != "ok" or not isinstance(rec.result, dict):
                continue
            outcome = rec.result.get("outcome")
            if outcome is None:
                continue
            for reg in self._registrations:
                if not rec.action_key.endswith("." + reg.action_key_suffix):
                    continue
                if outcome not in reg.terminal_outcomes:
                    continue
                key = reg.key_extractor(rec.params or {})
                logger.info(
                    "[ContractDrift] iter=%d state_recorded: suffix=%s key=%s outcome=%s",
                    observation.iter_index,
                    reg.action_key_suffix,
                    key,
                    outcome,
                )
                self._prior_terminal[(reg.action_key_suffix, key)] = {
                    "iter_index": observation.iter_index,
                    "outcome": outcome,
                    "payload": {
                        "agent_id": rec.result.get("agent_id"),
                        "mission_type": rec.result.get("mission_type"),
                        "reason": rec.result.get("reason"),
                    },
                }

        return StreamReflection(
            advisories=advisories, diagnostics=diagnostics,
        )

    def serialize_state(self) -> dict[str, Any]:
        # JSON-shaped: a list of (suffix, key_tuple_as_json, payload).
        return {
            "prior_terminal": [
                {
                    "suffix": suffix,
                    "key": list(key),
                    "value": value,
                }
                for (suffix, key), value in self._prior_terminal.items()
            ],
        }

    def deserialize_state(self, state: dict[str, Any]) -> None:
        self._prior_terminal = {}
        for record in state.get("prior_terminal") or []:
            if not isinstance(record, dict):
                continue
            suffix = record.get("suffix")
            key_list = record.get("key") or []
            value = record.get("value")
            if not isinstance(suffix, str) or not isinstance(value, dict):
                continue
            self._prior_terminal[(suffix, tuple(key_list))] = value


def _build_drift_outputs(
    *,
    rec: Any,
    suffix: str,
    key: tuple,
    prior: dict[str, Any],
    current_iter: int,
) -> tuple[AdvisoryEntry, Diagnostic]:
    prior_outcome = prior.get("outcome") or "?"
    prior_iter = prior.get("iter_index")
    prior_payload = prior.get("payload") or {}
    agent_id = prior_payload.get("agent_id")
    mission_type = prior_payload.get("mission_type")
    reason = prior_payload.get("reason") or ""

    if suffix == "spawn_mission":
        body, next_code = _build_spawn_mission_drift(
            prior_iter=prior_iter,
            prior_outcome=prior_outcome,
            agent_id=agent_id,
            mission_type=mission_type,
            reason=reason,
        )
    else:
        body, next_code = _build_generic_drift(
            suffix=suffix,
            prior_iter=prior_iter,
            prior_outcome=prior_outcome,
        )

    advisory = AdvisoryEntry(
        source="contract_drift",
        kind="typed_discriminator_drift",
        body=body,
        next_action_code=next_code,
    )
    diagnostic = Diagnostic(
        kind="contract_drift",
        severity="warning",
        payload={
            "action_key": rec.action_key,
            "action_suffix": suffix,
            "key_params": list(key),
            "prior_iter": prior_iter,
            "prior_outcome": prior_outcome,
            "prior_payload": prior_payload,
            "current_iter": current_iter,
            "current_action_id": rec.action_id,
        },
    )
    return advisory, diagnostic


def _build_spawn_mission_drift(
    *,
    prior_iter,
    prior_outcome: str,
    agent_id: str | None,
    mission_type: str | None,
    reason: str,
) -> tuple[str, str | None]:
    mt = mission_type or "?"
    aid = agent_id or "<unknown>"
    reason_clause = f" Gate reason: {reason!r}." if reason else ""
    body = (
        f"You called `spawn_mission` for "
        f"`mission_type={mt!r}` again, but iter {prior_iter} already "
        f"received `outcome={prior_outcome!r}` for the same mission "
        f"type with `agent_id={aid!r}`.{reason_clause}\n\n"
        f"`{prior_outcome}` is a TERMINAL outcome — re-calling "
        f"`spawn_mission` returns the same envelope (or the gate's "
        f"rejection) and burns an iteration. The discriminated return "
        f"shape carries `outcome` precisely so you can branch on it "
        f"instead of retrying.\n\n"
        f"Recovery: use `agent_id={aid!r}` from the prior iteration "
        f"directly. If you need to verify the coordinator is alive, "
        f"call `get_agent_status` — do NOT spawn it again."
    )
    if agent_id:
        next_code = (
            f"# Use the agent_id from iter {prior_iter} directly.\n"
            f"agent_id = {aid!r}\n"
            f"# Optional: verify it's still alive.\n"
            f"await run(\n"
            f"    \"AgentPoolCapability."
            f"AgentPoolCapability.get_agent_status\",\n"
            f"    agent_ids=[agent_id],\n"
            f")"
        )
    else:
        next_code = (
            f"# Iter {prior_iter} rejected the spawn — do not retry "
            f"blindly. Tell the user the gate's reason; respect the "
            f"decision; consider whether the goal can be reached "
            f"another way.\n"
            f"await run(\n"
            f"    \"SessionOrchestratorCapability."
            f"SessionOrchestratorCapability.respond_to_user\",\n"
            f"    content=(\n"
            f"        \"The {mt} spawn was rejected: {reason!r}. \"\n"
            f"        \"I won't retry without a different approach.\"\n"
            f"    ),\n"
            f")"
        )
    return body, next_code


def _build_generic_drift(
    *,
    suffix: str,
    prior_iter,
    prior_outcome: str,
) -> tuple[str, str | None]:
    body = (
        f"You called `{suffix}` again, but iter {prior_iter} already "
        f"received `outcome={prior_outcome!r}` — a terminal "
        f"discriminator value. The action's typed return carries "
        f"`outcome` precisely so you can branch on it instead of "
        f"retrying. Read the prior iteration's result from your run "
        f"trace; use it directly."
    )
    return body, None


__all__ = (
    "ContractDriftReflector",
    "ContractRegistration",
)
