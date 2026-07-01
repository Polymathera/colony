"""``recording_grade`` lifts span input/output truncation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from polymathera.colony.agents.observability.facility import (
    AgentTracingFacility,
)
from polymathera.colony.distributed.observability import SpanKind, TracingConfig


class _Action:
    action_type = "do_thing"
    reasoning = "because"

    def __init__(self, params: dict[str, Any]) -> None:
        self.parameters = params


class _Result:
    """Minimal ActionResult stand-in (only ``model_dump`` is read)."""

    def __init__(self, output: str) -> None:
        self._output = output

    def model_dump(self, mode: str = "json") -> dict[str, Any]:  # noqa: ARG002
        return {"success": True, "output": self._output}


def _facility(*, recording_grade: bool) -> AgentTracingFacility:
    config = TracingConfig(recording_grade=recording_grade)
    agent = SimpleNamespace(agent_id="a1")
    return AgentTracingFacility(config, agent)  # type: ignore[arg-type]


def _ctx(action: _Action) -> Any:
    return SimpleNamespace(args=(action,), kwargs={})


def test_output_truncated_by_default() -> None:
    out = _facility(recording_grade=False).summarize_output(
        "dispatch", SpanKind.ACTION, _Result("y" * 5000),
    )
    assert out["output"].endswith("...")
    assert len(out["output"]) <= 1500 + 3


def test_output_full_under_recording_grade() -> None:
    out = _facility(recording_grade=True).summarize_output(
        "dispatch", SpanKind.ACTION, _Result("y" * 5000),
    )
    assert out["output"] == "y" * 5000


def test_input_param_value_truncated_by_default() -> None:
    summ = _facility(recording_grade=False).summarize_input(
        _ctx(_Action({"p": "z" * 5000})), SpanKind.ACTION,
    )
    assert summ["parameters"]["p"].endswith("...")


def test_input_full_and_all_params_under_recording_grade() -> None:
    params = {f"k{i}": "z" * 5000 for i in range(15)}
    summ = _facility(recording_grade=True).summarize_input(
        _ctx(_Action(params)), SpanKind.ACTION,
    )
    # Full values …
    assert summ["parameters"]["k0"] == "z" * 5000
    # … and the 10-key cap is lifted.
    assert len(summ["parameters"]) == 15
