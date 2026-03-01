import { useState } from "react";
import { useAgents, useAgentDetail } from "@/api/hooks/useAgents";
import { DataTable } from "../shared/DataTable";
import { Badge } from "../shared/Badge";
import type { AgentSummary } from "@/api/types";

const stateVariant = (state: string) => {
  if (state === "running" || state === "active") return "success";
  if (state === "suspended") return "warning";
  if (state === "error" || state === "failed") return "error";
  return "default";
};

const columns = [
  { key: "agent_id", header: "Agent ID", className: "font-mono text-xs" },
  { key: "agent_type", header: "Type" },
  {
    key: "state",
    header: "State",
    render: (row: AgentSummary) => (
      <Badge variant={stateVariant(row.state)}>
        {row.state || "unknown"}
      </Badge>
    ),
  },
  {
    key: "capabilities",
    header: "Capabilities",
    render: (row: AgentSummary) => (
      <span className="text-xs text-muted-foreground">
        {row.capabilities.length}
      </span>
    ),
  },
];

export function AgentsTab() {
  const agents = useAgents();
  const [selectedId, setSelectedId] = useState<string>("");
  const detail = useAgentDetail(selectedId);

  return (
    <div className="flex gap-4">
      {/* Agent list */}
      <div className="flex-1">
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Registered Agents
        </h2>
        <DataTable
          columns={columns}
          data={agents.data ?? []}
          onRowClick={(row) => setSelectedId(row.agent_id)}
          emptyMessage="No agents registered"
        />
      </div>

      {/* Detail panel */}
      {selectedId && (
        <div className="w-80 shrink-0 rounded-lg border bg-card p-4">
          <h3 className="mb-2 font-medium">Agent Detail</h3>
          {detail.isLoading ? (
            <p className="text-sm text-muted-foreground">Loading...</p>
          ) : detail.data ? (
            <pre className="max-h-96 overflow-auto text-xs">
              {JSON.stringify(detail.data, null, 2)}
            </pre>
          ) : (
            <p className="text-sm text-muted-foreground">
              Select an agent to view details
            </p>
          )}
        </div>
      )}
    </div>
  );
}
