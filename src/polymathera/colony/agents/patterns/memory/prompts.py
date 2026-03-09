"""Memory Architecture Guidance for ActionPolicy Prompts.

This module generates memory-aware guidance text that is included in the
ActionPolicy's system prompt. It enables the LLM to reason *about* its
memory system as a first-class cognitive resource.

The guidance describes:
- Available memory levels and their purposes
- Current memory state (entry counts, capacity)
- Dataflow between memory levels
- Available memory actions

Usage:
    ```python
    from polymathera.colony.agents.patterns.memory.prompts import (
        get_memory_architecture_guidance,
    )

    # In planning context or action policy prompt
    memory_map = await ctx_engine._build_memory_map()
    guidance = get_memory_architecture_guidance(memory_map)
    # Include `guidance` in the system prompt
    ```
"""

from __future__ import annotations

from .types import MemoryMap, MemoryScopeInfo


def get_memory_architecture_guidance(memory_map: MemoryMap) -> str:
    """Generate memory architecture guidance for ActionPolicy prompt.

    This describes the agent's memory system so the LLM can reason about it
    during planning and decision-making.

    Args:
        memory_map: Current memory map snapshot.

    Returns:
        Formatted guidance string for inclusion in LLM prompt.
    """
    sections = [
        "## Your Memory System\n",
        "You have a multi-level memory system. You can REASON ABOUT your memory "
        "(inspect it, search it, understand its structure) in addition to using it.\n",
    ]

    # Memory levels
    levels_section = _format_scope_descriptions(memory_map)
    if levels_section:
        sections.append("### Memory Levels\n")
        sections.append(levels_section)

    # Dataflow
    dataflow_section = _format_dataflow(memory_map)
    if dataflow_section:
        sections.append("### Data Flow\n")
        sections.append(dataflow_section)

    # Available actions
    sections.append("### Memory Actions\n")
    sections.append(
        "- **inspect_memory_map**: Get overview of all memory levels and dataflow\n"
        "- **inspect_scope(scope_id)**: Detailed info about a specific scope (with optional sample entries)\n"
        "- **search_memory(query)**: Semantic search across all memory levels\n"
        "- **get_memory_statistics()**: Health and capacity metrics\n"
        "- **gather_context(query)**: Retrieve memories matching a query\n"
        "- **ingest_pending()**: Process pending memory transfers between levels\n"
        "- **maintain_memories()**: Trigger maintenance (decay, prune, deduplicate)\n"
    )

    # Current state summary
    sections.append("### Current State\n")
    sections.append(_format_current_state(memory_map))

    # When to use
    sections.append("### When to Use Memory Introspection\n")
    sections.append(
        "1. **Before complex tasks**: Check what relevant knowledge you have\n"
        "2. **When stuck**: Search memory for similar past experiences\n"
        "3. **After failures**: Look for patterns in past failures\n"
        "4. **For coordination**: Check what other agents have shared\n"
        "5. **Memory pressure**: Check capacity before storing large results\n"
    )

    return "\n".join(sections)


def _format_scope_descriptions(memory_map: MemoryMap) -> str:
    """Format scope descriptions ordered by memory level."""
    # Order by cognitive level (lower to higher)
    ordered_types = [
        "sensory", "working", "stm",
        "ltm:episodic", "ltm:semantic", "ltm:procedural",
    ]

    lines: list[str] = []

    # First show ordered types
    for scope_type in ordered_types:
        info = memory_map.get_scope_by_type(scope_type)
        if info:
            lines.append(_format_single_scope(info))

    # Then show any remaining scopes (collective, shared, etc.)
    shown_ids = {
        info.scope_id
        for st in ordered_types
        if (info := memory_map.get_scope_by_type(st)) is not None
    }
    for scope_id, info in memory_map.scopes.items():
        if scope_id not in shown_ids:
            lines.append(_format_single_scope(info))

    return "\n".join(lines) if lines else ""


def _format_single_scope(info: MemoryScopeInfo) -> str:
    """Format a single scope description."""
    capacity = ""
    if info.max_entries:
        pct = info.entry_count / info.max_entries * 100
        capacity = f" ({pct:.0f}% full)"

    ttl = _format_ttl(info.ttl_seconds)

    return (
        f"- **{info.scope_type.upper()}**: {info.purpose}\n"
        f"  Entries: {info.entry_count}"
        f"{f'/{info.max_entries}' if info.max_entries else ''}{capacity}"
        f" | TTL: {ttl}"
        f"{f' | Pending: {info.pending_ingestion_count}' if info.pending_ingestion_count else ''}\n"
    )


def _format_ttl(ttl_seconds: float | None) -> str:
    """Format TTL for display."""
    if ttl_seconds is None:
        return "permanent"
    if ttl_seconds < 60:
        return f"{ttl_seconds:.0f}s"
    if ttl_seconds < 3600:
        return f"{ttl_seconds / 60:.0f}m"
    return f"{ttl_seconds / 3600:.1f}h"


def _format_dataflow(memory_map: MemoryMap) -> str:
    """Format the dataflow graph description."""
    if not memory_map.dataflow_edges:
        return ""

    lines: list[str] = ["Data flows through your memory in this direction:\n"]
    for source, target in memory_map.dataflow_edges:
        src_type = _scope_id_to_short(source, memory_map)
        tgt_type = _scope_id_to_short(target, memory_map)
        lines.append(f"  {src_type} -> {tgt_type}")

    lines.append("")
    lines.append(
        "Higher levels consolidate and distill information from lower levels. "
        "Use `ingest_pending()` to trigger transfers when you need fresh data at higher levels.\n"
    )
    return "\n".join(lines)


def _scope_id_to_short(scope_id: str, memory_map: MemoryMap) -> str:
    """Convert scope_id to short display name."""
    info = memory_map.scopes.get(scope_id)
    if info:
        return info.scope_type.upper()
    return scope_id


def _format_current_state(memory_map: MemoryMap) -> str:
    """Format current memory state summary."""
    lines = [
        f"- Total entries: {memory_map.total_entries}",
        f"- Memory levels: {len(memory_map.scopes)}",
    ]
    if memory_map.total_pending_ingestion > 0:
        lines.append(f"- Pending ingestion: {memory_map.total_pending_ingestion} entries")

    return "\n".join(lines) + "\n"
