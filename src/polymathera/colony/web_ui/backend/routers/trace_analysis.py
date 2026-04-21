"""Trace analysis endpoints — structured views over raw span data.

Provides pre-processed codegen iteration data, prompt diffs, and FSM
state graphs for the dashboard's advanced trace analysis views.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import logging
import re
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, Depends, Query

from ..auth.middleware import require_auth
from ..dependencies import get_colony
from ..services.colony_connection import ColonyConnection

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Prompt section parsing
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)

# Sections that define agent "state" (stable across iterations).
# Volatile sections (execution history, errors, memories) change every step.
_STRUCTURAL_SECTIONS = frozenset({
    "Goals",
    "Constraints",
    "Current Mode: PLANNING",
    "Current Mode: EXECUTION",
    "Available Actions",
    "Rules",
    "Memory Architecture",
    "Namespace",
})


def _parse_prompt_sections(prompt: str) -> dict[str, str]:
    """Split a codegen prompt into named sections keyed by ``## `` headers."""
    sections: dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(prompt))

    if not matches:
        return {"_system_prompt": prompt}

    # Text before the first header is the system prompt
    if matches[0].start() > 0:
        sections["_system_prompt"] = prompt[: matches[0].start()].strip()

    for i, match in enumerate(matches):
        name = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        sections[name] = prompt[start:end].strip()

    return sections


def _extract_mode(prompt: str) -> str:
    """Extract the current mode from a codegen prompt."""
    if "## Current Mode: PLANNING" in prompt:
        return "planning"
    if "## Current Mode: EXECUTION" in prompt:
        return "execution"
    return "unknown"


def _fingerprint_prompt(prompt: str) -> str:
    """Compute a state fingerprint from the structural sections of a prompt.

    Two prompts with the same fingerprint represent the same "state" —
    same identity, goals, mode, and available actions.  Only volatile
    sections (execution history, errors, memories) differ.
    """
    sections = _parse_prompt_sections(prompt)
    canonical_parts: list[str] = []

    # Always include system prompt as part of state
    sys_prompt = sections.get("_system_prompt", "")
    if sys_prompt:
        canonical_parts.append(sys_prompt.strip())

    for name, content in sections.items():
        if name == "_system_prompt":
            continue
        # Match structural sections by prefix (handles "Current Mode: PLANNING" etc.)
        if any(name.startswith(s) for s in _STRUCTURAL_SECTIONS):
            canonical_parts.append(f"---{name}---\n{content.strip()}")

    blob = "\n".join(canonical_parts).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-run() call line number extraction
# ---------------------------------------------------------------------------


_TRACED_CALL_NAMES = {"run", "signal_completion"}


def _extract_traced_call_line_numbers(code: str) -> list[int]:
    """Extract line numbers of traced calls (``run()``, ``signal_completion()``) via AST.

    Returns line numbers in the order they appear in the source,
    matching the sequential order of trace entries.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    line_numbers: list[int] = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _TRACED_CALL_NAMES):
            line_numbers.append(node.lineno)
    return sorted(line_numbers)


# ---------------------------------------------------------------------------
# Span tree traversal
# ---------------------------------------------------------------------------


def _build_children_map(
    spans: list[dict[str, Any]],
) -> dict[str | None, list[dict[str, Any]]]:
    """Build a parent_span_id → children mapping."""
    children: dict[str | None, list[dict[str, Any]]] = {}
    for s in spans:
        pid = s.get("parent_span_id")
        children.setdefault(pid, []).append(s)
    return children


def _find_child(
    children_map: dict[str | None, list[dict[str, Any]]],
    parent_id: str,
    kind: str,
) -> dict[str, Any] | None:
    """Find the first span of a given kind under a parent (recursive).

    Searches children, then grandchildren, etc. This handles the
    case where INFER is nested under a child PLAN span rather than
    being a direct child of the top-level PLAN span.
    """
    for child in children_map.get(parent_id, []):
        if child.get("kind") == kind:
            return child
    # Recurse into children (e.g., PLAN → PLAN → INFER)
    for child in children_map.get(parent_id, []):
        found = _find_child(children_map, child["span_id"], kind)
        if found is not None:
            return found
    return None


def _extract_iterations(
    spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract ordered codegen iterations from a flat span list.

    Traverses the hierarchy: AGENT_STEP → PLAN → INFER + ACTION.
    """
    children_map = _build_children_map(spans)

    # Find AGENT_STEP spans, ordered chronologically
    agent_steps = sorted(
        [s for s in spans if s.get("kind") == "agent_step"],
        key=lambda s: s.get("start_wall", 0),
    )

    iterations: list[dict[str, Any]] = []
    for idx, step_span in enumerate(agent_steps):
        step_id = step_span["span_id"]
        plan_span = _find_child(children_map, step_id, "plan")
        if plan_span is None:
            continue

        plan_id = plan_span["span_id"]
        infer_span = _find_child(children_map, plan_id, "infer")
        action_span = _find_child(children_map, plan_id, "action")

        # Skip steps without an infer span (non-codegen iterations)
        if infer_span is None:
            continue

        input_summary = infer_span.get("input_summary") or {}
        output_summary = infer_span.get("output_summary") or {}
        action_output = (action_span.get("output_summary") or {}) if action_span else {}

        prompt = input_summary.get("prompt", "")
        generated_code = output_summary.get("response", "")

        # Extract per-run() call trace from ACTION span metadata
        run_call_trace = None
        if action_span:
            action_metadata = action_output.get("metadata", {})
            if isinstance(action_metadata, dict):
                raw_trace = action_metadata.get("run_call_trace")
                if raw_trace and isinstance(raw_trace, list):
                    line_numbers = _extract_traced_call_line_numbers(generated_code)
                    run_call_trace = []
                    for i, entry in enumerate(raw_trace):
                        enriched = dict(entry)
                        enriched["line_number"] = line_numbers[i] if i < len(line_numbers) else None
                        run_call_trace.append(enriched)

        iterations.append({
            "step_index": len(iterations),
            "agent_step_span_id": step_id,
            "infer_span_id": infer_span["span_id"],
            "action_span_id": action_span["span_id"] if action_span else None,
            "prompt": prompt,
            "generated_code": generated_code,
            "success": action_output.get("success", False) if action_span else None,
            "error": action_output.get("error"),
            "duration_ms": step_span.get("duration_ms"),
            "input_tokens": infer_span.get("input_tokens"),
            "output_tokens": infer_span.get("output_tokens"),
            "mode": _extract_mode(prompt),
            "start_wall": step_span.get("start_wall"),
            "run_call_trace": run_call_trace,
        })

    return iterations


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/traces/{trace_id}/codegen-iterations")
async def get_codegen_iterations(
    trace_id: str,
    agent_id: str = Query(..., description="Filter iterations by agent ID"),
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> list[dict[str, Any]]:
    """Extract the linearized codegen iteration sequence for one agent.

    Traverses the span tree (AGENT_STEP → PLAN → INFER/ACTION) and
    returns a flat list of iterations with prompt, code, and result.
    """
    store = colony.get_span_query_store()
    if store is None:
        return []

    try:
        # Fetch all spans for this agent in the trace
        all_spans = await store.get_spans(trace_id, limit=10000)
        agent_spans = [s for s in all_spans if s.get("agent_id") == agent_id]
        return _extract_iterations(agent_spans)
    except Exception as e:
        logger.warning("Failed to extract codegen iterations: %s", e)
        return []


@router.get("/traces/{trace_id}/prompt-diff")
async def get_prompt_diff(
    trace_id: str,
    agent_id: str = Query(...),
    step_a: int = Query(..., ge=0, description="First step index"),
    step_b: int = Query(..., ge=0, description="Second step index"),
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Compute a section-aware diff between two codegen prompts.

    Parses both prompts into ``## ``-delimited sections, diffs each
    section independently, and returns structured change data.
    """
    store = colony.get_span_query_store()
    if store is None:
        return {"error": "Store not available"}

    try:
        all_spans = await store.get_spans(trace_id, limit=10000)
        agent_spans = [s for s in all_spans if s.get("agent_id") == agent_id]
        iterations = _extract_iterations(agent_spans)

        if step_a >= len(iterations) or step_b >= len(iterations):
            return {"error": f"Step index out of range (have {len(iterations)} iterations)"}

        prompt_a = iterations[step_a]["prompt"]
        prompt_b = iterations[step_b]["prompt"]

        return _compute_section_diff(prompt_a, prompt_b, step_a, step_b)
    except Exception as e:
        logger.warning("Failed to compute prompt diff: %s", e)
        return {"error": str(e)}


def _compute_section_diff(
    prompt_a: str,
    prompt_b: str,
    step_a: int,
    step_b: int,
) -> dict[str, Any]:
    """Compute a structured, section-aware diff between two prompts."""
    sections_a = _parse_prompt_sections(prompt_a)
    sections_b = _parse_prompt_sections(prompt_b)

    # Union of all section names, preserving order from prompt_b
    all_names: list[str] = []
    seen: set[str] = set()
    for name in list(sections_a.keys()) + list(sections_b.keys()):
        if name not in seen:
            all_names.append(name)
            seen.add(name)

    diff_sections: list[dict[str, Any]] = []
    total_added = 0
    total_removed = 0

    for name in all_names:
        text_a = sections_a.get(name, "")
        text_b = sections_b.get(name, "")

        if text_a == text_b:
            continue  # Skip unchanged sections entirely

        lines_a = text_a.splitlines(keepends=True)
        lines_b = text_b.splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(
            lines_a, lines_b,
            fromfile=f"step {step_a}",
            tofile=f"step {step_b}",
            lineterm="",
        ))

        if not diff_lines:
            continue

        changes: list[dict[str, str]] = []
        for line in diff_lines:
            if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
                continue
            if line.startswith("+"):
                changes.append({"type": "added", "content": line[1:]})
                total_added += 1
            elif line.startswith("-"):
                changes.append({"type": "removed", "content": line[1:]})
                total_removed += 1
            else:
                changes.append({"type": "context", "content": line[1:] if line.startswith(" ") else line})

        if changes:
            display_name = name if name != "_system_prompt" else "System Prompt"
            diff_sections.append({"name": display_name, "changes": changes})

    return {
        "step_a": step_a,
        "step_b": step_b,
        "sections": diff_sections,
        "total_added": total_added,
        "total_removed": total_removed,
        "is_identical": total_added == 0 and total_removed == 0,
    }


@router.get("/traces/{trace_id}/fsm")
async def get_fsm_graph(
    trace_id: str,
    agent_id: str = Query(...),
    _user: dict = Depends(require_auth),
    colony: ColonyConnection = Depends(get_colony),
) -> dict[str, Any]:
    """Compute the finite-state machine graph for one agent's execution.

    States are identified by prompt fingerprint (structural sections).
    Transitions are the generated code at each step.
    """
    store = colony.get_span_query_store()
    if store is None:
        return {"states": [], "transitions": [], "loops": []}

    try:
        all_spans = await store.get_spans(trace_id, limit=10000)
        agent_spans = [s for s in all_spans if s.get("agent_id") == agent_id]
        iterations = _extract_iterations(agent_spans)

        return _build_fsm_graph(iterations)
    except Exception as e:
        logger.warning("Failed to build FSM graph: %s", e)
        return {"states": [], "transitions": [], "loops": []}


def _build_fsm_graph(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    """Build FSM states and transitions from codegen iterations."""
    if not iterations:
        return {"states": [], "transitions": [], "loops": []}

    # Assign state IDs by fingerprint
    fingerprint_to_state: dict[str, str] = {}
    states: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []

    # Track (state_id, code_hash) for loop detection
    transition_hashes: dict[tuple[str, str], list[int]] = {}

    iteration_state_ids: list[str] = []

    for it in iterations:
        fp = _fingerprint_prompt(it["prompt"])

        if fp not in fingerprint_to_state:
            state_id = f"s{len(states)}"
            fingerprint_to_state[fp] = state_id
            mode = it["mode"]

            # Count execution history entries from prompt
            history_section = _parse_prompt_sections(it["prompt"]).get("Execution History", "")
            history_count = len([
                line for line in history_section.splitlines()
                if line.strip().startswith("[")
            ])

            states.append({
                "state_id": state_id,
                "fingerprint": fp,
                "label": f"{mode.upper()} ({history_count} history)",
                "visit_count": 0,
                "step_indices": [],
            })

        state_id = fingerprint_to_state[fp]
        # Find state dict and update
        state = next(s for s in states if s["state_id"] == state_id)
        state["visit_count"] += 1
        state["step_indices"].append(it["step_index"])
        iteration_state_ids.append(state_id)

    # Build transitions between consecutive iterations
    for i in range(len(iterations) - 1):
        from_state = iteration_state_ids[i]
        to_state = iteration_state_ids[i + 1]
        code = iterations[i]["generated_code"]
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:12]

        # Code preview: first non-comment, non-blank line
        code_lines = [
            line for line in code.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        code_preview = code_lines[0].strip() if code_lines else "(empty)"
        if len(code_preview) > 80:
            code_preview = code_preview[:77] + "..."

        transitions.append({
            "from_state": from_state,
            "to_state": to_state,
            "step_index": i,
            "code_preview": code_preview,
            "success": iterations[i].get("success", False),
        })

        key = (from_state, code_hash)
        transition_hashes.setdefault(key, []).append(i)

    # Detect loops: same (state, code) appearing more than once
    loops: list[dict[str, Any]] = []
    for (state_id, _), step_indices in transition_hashes.items():
        if len(step_indices) > 1:
            loops.append({
                "state_id": state_id,
                "count": len(step_indices),
                "step_indices": step_indices,
            })

    return {"states": states, "transitions": transitions, "loops": loops}
