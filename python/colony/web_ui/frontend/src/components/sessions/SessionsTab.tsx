import { useState } from "react";
import { useSessions, useSessionRuns } from "@/api/hooks/useSessions";
import { DataTable } from "../shared/DataTable";
import { Badge } from "../shared/Badge";
import { formatTimestamp, formatTokens } from "@/lib/utils";
import type { SessionSummary, RunSummary } from "@/api/types";

const sessionStateVariant = (state: string) => {
  if (state === "active") return "success";
  if (state === "suspended") return "warning";
  if (state === "closed" || state === "expired") return "default";
  return "info";
};

const runStatusVariant = (status: string) => {
  if (status === "completed" || status === "success") return "success";
  if (status === "running" || status === "in_progress") return "info";
  if (status === "failed" || status === "error") return "error";
  if (status === "cancelled") return "warning";
  return "default";
};

const sessionColumns = [
  {
    key: "session_id",
    header: "Session ID",
    className: "font-mono text-xs",
    render: (row: SessionSummary) => row.session_id.slice(0, 12) + "...",
  },
  { key: "tenant_id", header: "Tenant" },
  {
    key: "state",
    header: "State",
    render: (row: SessionSummary) => (
      <Badge variant={sessionStateVariant(row.state)}>
        {row.state || "unknown"}
      </Badge>
    ),
  },
  {
    key: "created_at",
    header: "Created",
    render: (row: SessionSummary) => formatTimestamp(row.created_at),
  },
  { key: "run_count", header: "Runs" },
];

const runColumns = [
  {
    key: "run_id",
    header: "Run ID",
    className: "font-mono text-xs",
    render: (row: RunSummary) => row.run_id.slice(0, 12) + "...",
  },
  { key: "agent_id", header: "Agent", className: "font-mono text-xs" },
  {
    key: "status",
    header: "Status",
    render: (row: RunSummary) => (
      <Badge variant={runStatusVariant(row.status)}>
        {row.status || "unknown"}
      </Badge>
    ),
  },
  {
    key: "input_tokens",
    header: "In Tokens",
    render: (row: RunSummary) => formatTokens(row.input_tokens),
  },
  {
    key: "output_tokens",
    header: "Out Tokens",
    render: (row: RunSummary) => formatTokens(row.output_tokens),
  },
];

export function SessionsTab() {
  const sessions = useSessions();
  const [selectedSessionId, setSelectedSessionId] = useState<string>("");
  const runs = useSessionRuns(selectedSessionId);

  return (
    <div className="space-y-6">
      {/* Sessions table */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Sessions
        </h2>
        <DataTable
          columns={sessionColumns}
          data={sessions.data ?? []}
          onRowClick={(row) => setSelectedSessionId(row.session_id)}
          emptyMessage="No sessions found"
        />
      </section>

      {/* Runs for selected session */}
      {selectedSessionId && (
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Runs — {selectedSessionId.slice(0, 12)}...
          </h2>
          <DataTable
            columns={runColumns}
            data={runs.data ?? []}
            emptyMessage={
              runs.isLoading ? "Loading runs..." : "No runs for this session"
            }
          />
        </section>
      )}
    </div>
  );
}
