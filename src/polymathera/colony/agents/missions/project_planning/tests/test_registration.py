"""Static checks for the ``project_planning`` mission registration.

These don't spin up a coordinator agent — they just confirm the
registry entry round-trips through :class:`MissionSpec`, the
coordinator FQN resolves to the real class, and the self-concept
encodes the propose-approve-apply invariant the planner must follow.
"""

from __future__ import annotations

import importlib

import pytest

from polymathera.colony.agents.configs import (
    MissionSpec,
    _BUILTIN_MISSIONS,
    _builtin_missions,
)
from polymathera.colony.agents.missions.project_planning.coordinator import (
    ProjectPlanningCoordinator,
)


def test_project_planning_entry_present() -> None:
    """The mission is registered in the colony-builtin dict."""

    assert "project_planning" in _BUILTIN_MISSIONS
    raw = _BUILTIN_MISSIONS["project_planning"]
    assert raw["label"] == "Project Planning"
    assert "bootstrap" in raw["description"].lower()


def test_project_planning_round_trips_through_missionspec() -> None:
    """The raw dict satisfies :class:`MissionSpec`'s strict schema —
    catches typo'd keys + missing required fields the planner relies
    on (label, description, coordinator_v2, self_concept)."""

    specs = _builtin_missions()
    assert "project_planning" in specs
    spec = specs["project_planning"]
    assert isinstance(spec, MissionSpec)
    assert spec.label == "Project Planning"
    assert spec.coordinator_v1 == spec.coordinator_v2
    assert "ProjectPlanningCoordinator" in spec.coordinator_v2


def test_project_planning_worker_is_empty() -> None:
    """The mission ships without a separate worker class — the
    coordinator drives the full flow over existing action surfaces.
    ``MissionSpec.worker`` defaults to ``""``; the spec exercises
    that path."""

    spec = _builtin_missions()["project_planning"]
    assert spec.worker == ""
    # The four readers in cli/polymath.py + session_agent.py all use
    # ``reg.get("worker", "")`` — empty is a clean code-path, not an
    # error trigger.


def test_project_planning_coordinator_class_is_importable() -> None:
    """The FQN in coordinator_v2 resolves to the actual class —
    catches typos in the registration string before they bite a
    chat-side spawn_mission call."""

    spec = _builtin_missions()["project_planning"]
    module_path, class_name = spec.coordinator_v2.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    assert cls is ProjectPlanningCoordinator


def test_project_planning_self_concept_encodes_dry_run_first_constraint() -> None:
    """The single hard constraint of this mission is: never apply
    without an explicit approve response. The self-concept MUST
    surface this so the planner doesn't skip the gate."""

    spec = _builtin_missions()["project_planning"]
    constraints_blob = "\n".join(spec.self_concept.constraints).lower()
    assert "dry_run=false" in constraints_blob
    assert "approve" in constraints_blob


def test_project_planning_self_concept_names_the_three_modes() -> None:
    """The planner picks the action based on ``mission_params['mode']``
    — bootstrap / refresh / assignments. The goals must enumerate
    all three so the planner doesn't silently default."""

    spec = _builtin_missions()["project_planning"]
    goals_blob = "\n".join(spec.self_concept.goals).lower()
    assert "bootstrap" in goals_blob
    assert "refresh" in goals_blob
    assert "assignments" in goals_blob
    # The matching action names too, so the planner picks the right
    # one without LLM hallucination.
    assert "bootstrap_roadmap_from_objectives" in goals_blob
    assert "sync_roadmap_with_github" in goals_blob
    assert "propose_task_assignments" in goals_blob


def test_project_planning_declares_expected_caller_parameters() -> None:
    """``caller_parameters`` documents the mission_params the
    coordinator's planner reads. ``mode`` is required (no default);
    the other four entries are optional with declared defaults so
    the planner can omit them when the colony-level resolution
    fits."""

    spec = _builtin_missions()["project_planning"]
    by_name = {p.name: p for p in spec.caller_parameters}
    assert set(by_name) == {
        "mode", "repo", "roadmap_path", "user_github_login", "direction",
    }
    # ``mode`` is the only required CALLER param.
    assert by_name["mode"].required is True
    # The other four carry declared defaults (Pydantic-style:
    # required iff no default of any kind).
    for optional in ("repo", "roadmap_path", "user_github_login", "direction"):
        assert by_name[optional].required is False


def test_project_planning_lists_action_surfaces_in_coordinator_capabilities() -> None:
    """``coordinator_capabilities`` is documentation-only but it MUST
    enumerate the action surfaces this mission drives over so the
    ``describe`` command shows operators what's mounted."""

    spec = _builtin_missions()["project_planning"]
    expected = {
        "DesignProcessCapability",
        "SystemDesignCapability",
        "GitHubCapability",
        "HumanApprovalCapability",
    }
    assert expected.issubset(set(spec.coordinator_capabilities))


def test_missionspec_worker_field_is_optional_with_empty_default() -> None:
    """A MissionSpec without a ``worker`` key validates cleanly with
    ``worker=""``. Guards the schema change made for P6."""

    spec = MissionSpec(
        label="x", description="y",
        coordinator_v1="pkg.mod:A", coordinator_v2="pkg.mod:A",
        self_concept={"description": "z"},
    )
    assert spec.worker == ""


def test_coordinator_module_binds_each_required_capability() -> None:
    """The coordinator's ``initialize()`` must add blueprints for
    every capability the mission's planner needs. A test that
    imports the module + inspects the source string for each
    ``.bind(`` is brittle but catches the most common regression:
    a dev removes the blueprint but keeps the import (or vice
    versa) and the planner silently loses access to a key action."""

    import inspect

    from polymathera.colony.agents.missions.project_planning import (
        coordinator as coord_mod,
    )
    source = inspect.getsource(coord_mod)
    # Each capability must be both imported AND bound — the .bind(
    # call site is the operational check, the import is the lexical
    # check that catches typos / stale imports.
    for needle in (
        "DesignProcessCapability.bind(",
        "SystemDesignCapability.bind(",
        "GitHubCapability.bind(",
        "HumanApprovalCapability.bind(",
        "design_monorepo_capability_blueprints(",
    ):
        assert needle in source, (
            f"ProjectPlanningCoordinator no longer binds {needle!r} — "
            "the planner would lose access to that action surface."
        )
