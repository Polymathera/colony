import type { TraceSpan } from "@/api/types";
import { useMemo } from "react";

export function AgentSelector({
  spans,
  value,
  onChange,
}: {
  spans: TraceSpan[];
  value: string | null;
  onChange: (agentId: string) => void;
}) {
  const agentIds = useMemo(() => {
    const ids = new Set<string>();
    for (const s of spans) ids.add(s.agent_id);
    return Array.from(ids).sort();
  }, [spans]);

  // Auto-select first agent if none selected
  if (!value && agentIds.length > 0) {
    // Schedule to avoid setState during render
    queueMicrotask(() => onChange(agentIds[0]));
  }

  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
        Agent
      </span>
      <select
        className="rounded border border-border bg-card px-2 py-1 text-xs font-mono text-foreground focus:border-primary focus:outline-none"
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value)}
      >
        {agentIds.map((id) => (
          <option key={id} value={id}>
            {id}
          </option>
        ))}
      </select>
    </div>
  );
}
