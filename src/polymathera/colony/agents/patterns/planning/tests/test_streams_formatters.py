"""Tests for the PR-Sub-3 stock formatter + filter combinator additions
to :mod:`polymathera.colony.agents.patterns.planning.streams`.

Covered:

1. ``EventLogFormatter`` — chronological generic; handles all kinds;
   ``max_entries_shown`` trims to the most-recent.
2. ``ToolResultFormatter`` — renders payload + units + provenance;
   skips non-tool_output entries; success vs failure marker.
3. ``VCMUpdateFormatter`` — renders mutation kind + page-source + page-id;
   skips non-vcm_update entries.
4. ``MonorepoCommitFormatter`` — short SHA + branch + truncated
   message + paths summary; skips non-commit entries.
5. ``DomainStateFormatter`` — filters to one state-machine when
   ``state_machine_name`` set; renders ``from_state → to_state`` arrows.
6. ``AnyOf`` / ``AllOf`` / ``Not`` combinators — short-circuit
   semantics + empty-input rejection on AnyOf/AllOf.
"""

from __future__ import annotations

import pytest

from polymathera.colony.agents.patterns.planning.streams import (
    AllOf,
    AnyOf,
    DomainStateFormatter,
    EventLogFormatter,
    MonorepoCommitFormatter,
    Not,
    ToolResultFormatter,
    VCMUpdateFormatter,
)


# ---------------------------------------------------------------------------
# 1. EventLogFormatter
# ---------------------------------------------------------------------------


class TestEventLogFormatter:
    def test_empty_returns_empty_string(self) -> None:
        assert EventLogFormatter(section_title="## L").format([]) == ""

    def test_renders_all_kinds_in_order(self) -> None:
        entries = [
            {"kind": "event", "timestamp": 0, "contexts": {"k": "v"}},
            {"kind": "action", "timestamp": 1,
             "call": {"action_key": "foo", "output_preview": "ok"}},
            {"kind": "tool_output", "timestamp": 2,
             "payload": {"action_key": "x", "success": True}},
            {"kind": "vcm_update", "timestamp": 3,
             "payload": {"kind": "added", "page_id": "p1"}},
        ]
        out = EventLogFormatter(section_title="## L").format(entries)
        assert out.startswith("## L\n\n")
        assert out.count("**event**") == 1
        assert out.count("**action**") == 1
        assert out.count("**tool_output**") == 1
        assert out.count("**vcm_update**") == 1

    def test_max_entries_shown_trims_to_most_recent(self) -> None:
        entries = [
            {"kind": "event", "timestamp": i, "contexts": {"i": i}}
            for i in range(5)
        ]
        out = EventLogFormatter(
            section_title="## L", max_entries_shown=2,
        ).format(entries)
        assert "\"i\": 3" in out
        assert "\"i\": 4" in out
        assert "\"i\": 0" not in out
        assert "\"i\": 1" not in out

    def test_value_truncation(self) -> None:
        entries = [
            {"kind": "event", "timestamp": 0,
             "contexts": {"long": "x" * 500}},
        ]
        out = EventLogFormatter(
            section_title="## L", max_value_chars=50,
        ).format(entries)
        assert "..." in out
        # The rendered line shouldn't be 500+ chars wide.
        for line in out.split("\n"):
            if line.startswith("- "):
                assert len(line) < 100


# ---------------------------------------------------------------------------
# 2. ToolResultFormatter
# ---------------------------------------------------------------------------


class TestToolResultFormatter:
    def test_empty_returns_empty_string(self) -> None:
        assert ToolResultFormatter().format([]) == ""

    def test_skips_non_tool_output_entries(self) -> None:
        out = ToolResultFormatter().format([
            {"kind": "event", "timestamp": 0, "contexts": {}},
            {"kind": "action", "timestamp": 1,
             "call": {"action_key": "x"}},
        ])
        assert out == ""

    def test_renders_payload_units_provenance(self) -> None:
        entries = [
            {"kind": "tool_output", "timestamp": 0, "payload": {
                "action_key": "compute_sensitivity",
                "success": True,
                "tool_result": {
                    "payload": {"sensitivity_fT_per_sqrt_Hz": 0.531},
                    "units": {"sensitivity_fT_per_sqrt_Hz": "fT/√Hz"},
                    "provenance": {"tool_name": "serf_simulator"},
                },
            }},
        ]
        out = ToolResultFormatter().format(entries)
        assert "compute_sensitivity" in out
        assert "serf_simulator" in out
        assert "0.531" in out
        assert "fT/√Hz" in out
        assert "✓" in out

    def test_failure_marker(self) -> None:
        entries = [
            {"kind": "tool_output", "timestamp": 0, "payload": {
                "action_key": "x", "success": False,
                "tool_result": {
                    "payload": {}, "units": {},
                    "provenance": {"tool_name": "t"},
                },
            }},
        ]
        out = ToolResultFormatter().format(entries)
        assert "✗" in out


# ---------------------------------------------------------------------------
# 3. VCMUpdateFormatter
# ---------------------------------------------------------------------------


class TestVCMUpdateFormatter:
    def test_empty_returns_empty_string(self) -> None:
        assert VCMUpdateFormatter().format([]) == ""

    def test_skips_non_vcm_entries(self) -> None:
        out = VCMUpdateFormatter().format([
            {"kind": "tool_output", "timestamp": 0, "payload": {}},
        ])
        assert out == ""

    def test_renders_mutation_summary(self) -> None:
        entries = [
            {"kind": "vcm_update", "timestamp": 0, "payload": {
                "kind": "added", "page_source": "literature",
                "page_id": "paper_kominis_2003", "scope_id": "scope_serf",
            }},
        ]
        out = VCMUpdateFormatter().format(entries)
        assert "added" in out
        assert "paper_kominis_2003" in out
        assert "literature" in out
        assert "scope_serf" in out

    def test_max_entries_shown_trims(self) -> None:
        entries = [
            {"kind": "vcm_update", "timestamp": i, "payload": {
                "kind": "added", "page_source": "l", "page_id": f"p_{i}",
            }}
            for i in range(5)
        ]
        out = VCMUpdateFormatter(max_entries_shown=2).format(entries)
        assert "p_4" in out
        assert "p_3" in out
        assert "p_0" not in out


# ---------------------------------------------------------------------------
# 4. MonorepoCommitFormatter
# ---------------------------------------------------------------------------


class TestMonorepoCommitFormatter:
    def test_renders_sha_branch_message_paths(self) -> None:
        entries = [
            {"kind": "monorepo_commit", "timestamp": 0, "payload": {
                "sha": "abcdef1234567890",
                "branch": "main",
                "message": "G-1 belief_registry: record Kominis 2003 belief",
                "paths": [
                    "design/beliefs/main/beliefs.json",
                    "design/beliefs/main/edges.json",
                ],
            }},
        ]
        out = MonorepoCommitFormatter().format(entries)
        assert "abcdef12" in out  # short SHA (first 8)
        assert "main" in out
        assert "G-1 belief_registry" in out
        assert "design/beliefs/main/beliefs.json" in out

    def test_truncates_message(self) -> None:
        entries = [
            {"kind": "monorepo_commit", "timestamp": 0, "payload": {
                "sha": "x", "branch": "main",
                "message": "long " * 200, "paths": [],
            }},
        ]
        out = MonorepoCommitFormatter(max_message_chars=40).format(entries)
        assert "..." in out

    def test_path_truncation(self) -> None:
        entries = [
            {"kind": "monorepo_commit", "timestamp": 0, "payload": {
                "sha": "x", "branch": "main", "message": "m",
                "paths": [f"path_{i}.txt" for i in range(10)],
            }},
        ]
        out = MonorepoCommitFormatter(max_paths_shown=3).format(entries)
        assert "path_0.txt" in out
        assert "+7 more" in out


# ---------------------------------------------------------------------------
# 5. DomainStateFormatter
# ---------------------------------------------------------------------------


class TestDomainStateFormatter:
    def test_filters_to_one_state_machine_when_named(self) -> None:
        entries = [
            {"kind": "domain_state", "timestamp": 0, "payload": {
                "state_machine": "hypothesis_game",
                "transition": "PROPOSE→CHALLENGE",
                "from_state": "PROPOSE", "to_state": "CHALLENGE",
            }},
            {"kind": "domain_state", "timestamp": 1, "payload": {
                "state_machine": "experiment_lifecycle",
                "transition": "start_study", "from_state": "", "to_state": "",
            }},
        ]
        out = DomainStateFormatter(
            section_title="## Game phase",
            state_machine_name="hypothesis_game",
        ).format(entries)
        assert "PROPOSE" in out and "CHALLENGE" in out
        assert "experiment_lifecycle" not in out

    def test_no_filter_renders_all_with_machine_prefix(self) -> None:
        entries = [
            {"kind": "domain_state", "timestamp": 0, "payload": {
                "state_machine": "hypothesis_game",
                "from_state": "PROPOSE", "to_state": "CHALLENGE",
            }},
            {"kind": "domain_state", "timestamp": 1, "payload": {
                "state_machine": "experiment_lifecycle",
                "from_state": "", "to_state": "", "transition": "start_study",
            }},
        ]
        out = DomainStateFormatter(section_title="## DS").format(entries)
        assert "[hypothesis_game]" in out
        assert "[experiment_lifecycle]" in out


# ---------------------------------------------------------------------------
# 6. Filter combinators
# ---------------------------------------------------------------------------


class TestAnyOf:
    def test_short_circuits(self) -> None:
        calls = []

        def f1(p):
            calls.append("f1")
            return True

        def f2(p):
            calls.append("f2")
            return True

        combo = AnyOf(f1, f2)
        assert combo({}) is True
        assert calls == ["f1"]  # short-circuit on first True

    def test_returns_false_when_all_reject(self) -> None:
        combo = AnyOf(lambda _: False, lambda _: False)
        assert combo({}) is False

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            AnyOf()


class TestAllOf:
    def test_short_circuits_on_first_false(self) -> None:
        calls = []

        def f1(p):
            calls.append("f1")
            return False

        def f2(p):
            calls.append("f2")
            return True

        combo = AllOf(f1, f2)
        assert combo({}) is False
        assert calls == ["f1"]  # short-circuit on first False

    def test_returns_true_when_all_accept(self) -> None:
        combo = AllOf(lambda _: True, lambda _: True)
        assert combo({}) is True

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            AllOf()


class TestNot:
    def test_inverts(self) -> None:
        assert Not(lambda _: True)({}) is False
        assert Not(lambda _: False)({}) is True


class TestCombinatorComposition:
    def test_any_of_all_of_nested(self) -> None:
        """Real-world shape: accept payloads where (success AND tool_name
        is one of {a, b}) OR (action_key startswith "propose_")."""
        from polymathera.colony.agents.patterns.planning.streams import (
            ConsciousnessStream,
            JSONStreamFormatter,
        )

        def success(p):
            return bool(p.get("success"))

        def tool_in_ab(p):
            tool = (p.get("tool_result") or {}).get(
                "provenance", {}
            ).get("tool_name", "")
            return tool in ("a", "b")

        def propose_action(p):
            return p.get("action_key", "").startswith("propose_")

        filt = AnyOf(AllOf(success, tool_in_ab), propose_action)
        stream = ConsciousnessStream(
            name="x",
            formatter=JSONStreamFormatter(section_title="## X"),
            tool_output_filter=filt,
        )
        stream.consider_tool_output({
            "action_key": "compute", "success": True,
            "tool_result": {"provenance": {"tool_name": "a"}},
        })
        stream.consider_tool_output({
            "action_key": "propose_factorial", "success": False,
            "tool_result": {"provenance": {"tool_name": "z"}},
        })
        stream.consider_tool_output({
            "action_key": "compute", "success": False,
            "tool_result": {"provenance": {"tool_name": "z"}},
        })
        assert len(stream._entries) == 2  # first two accepted, third rejected
