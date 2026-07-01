"""Promotion gate + degeneracy guards + the flywheel orchestrator."""

from __future__ import annotations

from polymathera.colony.training.flywheel import run_flywheel
from polymathera.colony.training.promotion import (
    EvalSummary,
    collapse_guard,
    promotion_gate,
    reward_hack_guard,
    summarize_eval,
)


# ---- guards ---------------------------------------------------------------

def test_collapse_guard_fails_on_low_diversity() -> None:
    assert collapse_guard(EvalSummary(distinct_ratio=0.9)).passed
    bad = collapse_guard(EvalSummary(distinct_ratio=0.1))
    assert not bad.passed and "diversity" in bad.reason


def test_reward_hack_guard_fails_on_degenerate() -> None:
    assert reward_hack_guard(EvalSummary(degenerate_ratio=0.05)).passed
    bad = reward_hack_guard(EvalSummary(degenerate_ratio=0.5))
    assert not bad.passed and "degenerate" in bad.reason


# ---- gate -----------------------------------------------------------------

def _healthy(mean_reward: float) -> EvalSummary:
    return EvalSummary(mean_reward=mean_reward, num_samples=10, distinct_ratio=1.0, degenerate_ratio=0.0)


def test_gate_promotes_first_model_passing_guards() -> None:
    d = promotion_gate(_healthy(0.5))
    assert d.promote and "first promotion" in d.reason


def test_gate_requires_beating_incumbent() -> None:
    assert promotion_gate(_healthy(0.7), _healthy(0.5)).promote
    assert not promotion_gate(_healthy(0.5), _healthy(0.7)).promote


def test_gate_holds_on_failed_guard_even_if_reward_higher() -> None:
    # Reward-hacking candidate: huge reward, but degenerate outputs.
    hacked = EvalSummary(mean_reward=99.0, degenerate_ratio=0.9, distinct_ratio=0.05)
    d = promotion_gate(hacked, _healthy(0.5))
    assert not d.promote
    assert "degenerate" in d.reason or "diversity" in d.reason


def test_min_gain_threshold() -> None:
    assert not promotion_gate(_healthy(0.52), _healthy(0.5), min_gain=0.1).promote
    assert promotion_gate(_healthy(0.65), _healthy(0.5), min_gain=0.1).promote


# ---- summarize_eval -------------------------------------------------------

def test_summarize_eval_computes_diversity_and_degeneracy() -> None:
    completions = ["a good long answer", "another good long answer", "x", "x"]
    summary = summarize_eval(completions, [1.0, 1.0, 0.0, 0.0])
    assert summary.num_samples == 4
    assert summary.mean_reward == 0.5
    assert summary.distinct_ratio == 0.75  # 3 unique / 4
    assert summary.degenerate_ratio == 0.5  # two "x" are too short


# ---- orchestrator ---------------------------------------------------------

def test_run_flywheel_publishes_only_when_gate_promotes() -> None:
    published: list[str] = []

    def fake_train(base, dataset):
        return type("M", (), {"model_id": "cand-1"})()

    outcome = run_flywheel(
        base_model="base/llm",
        assemble=lambda: "dsv-abc",
        train=fake_train,
        evaluate=lambda m: _healthy(0.8),
        publish=lambda m: published.append("cand-1") or "cand-1",
        incumbent_eval=_healthy(0.5),
    )
    assert outcome.promoted and outcome.published_name == "cand-1"
    assert outcome.model_id == "cand-1" and outcome.dataset_version == "dsv-abc"
    assert published == ["cand-1"]


def test_run_flywheel_holds_and_does_not_publish() -> None:
    published: list[str] = []
    outcome = run_flywheel(
        base_model="base/llm",
        assemble=lambda: "dsv-abc",
        train=lambda b, d: type("M", (), {"model_id": "cand-2"})(),
        evaluate=lambda m: _healthy(0.3),          # worse than incumbent
        publish=lambda m: published.append("x") or "x",
        incumbent_eval=_healthy(0.9),
    )
    assert not outcome.promoted and outcome.published_name is None
    assert published == []
