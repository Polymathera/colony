import { useState } from "react";
import { useAgents, useAgentDetail, useAgentHierarchy, useAgentHistory } from "@/api/hooks/useAgents";
import { DataTable } from "../shared/DataTable";
import { Badge } from "../shared/Badge";
import { MetricCard } from "../shared/MetricCard";
import { formatTokens } from "@/lib/utils";
import type { AgentSummary, AgentHierarchyNode } from "@/api/types";

const stateVariant = (state: string) => {
  if (state === "running" || state === "active") return "success" as const;
  if (state === "suspended") return "warning" as const;
  if (state === "error" || state === "failed") return "error" as const;
  return "default" as const;
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

function buildTree(nodes: AgentHierarchyNode[]) {
  const byId = new Map(nodes.map((n) => [n.agent_id, n]));
  const children = new Map<string | null, AgentHierarchyNode[]>();
  for (const n of nodes) {
    const parent = n.parent_agent_id ?? null;
    if (!children.has(parent)) children.set(parent, []);
    children.get(parent)!.push(n);
  }
  return { byId, children };
}

function TreeNode({
  node,
  children,
  depth,
  selectedId,
  onSelect,
}: {
  node: AgentHierarchyNode;
  children: Map<string | null, AgentHierarchyNode[]>;
  depth: number;
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState(true);
  const kids = children.get(node.agent_id) ?? [];
  const hasChildren = kids.length > 0;
  const isSelected = node.agent_id === selectedId;

  return (
    <div>
      <div
        className={`flex cursor-pointer items-center gap-1.5 rounded px-2 py-1 text-xs hover:bg-accent/50 ${isSelected ? "bg-accent" : ""}`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
        onClick={() => onSelect(node.agent_id)}
      >
        {hasChildren ? (
          <button
            className="h-4 w-4 shrink-0 text-muted-foreground"
            onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }}
          >
            {expanded ? "▼" : "▶"}
          </button>
        ) : (
          <span className="h-4 w-4 shrink-0" />
        )}
        <span className="font-mono truncate">{node.agent_id.slice(0, 20)}</span>
        {node.role && (
          <Badge variant="info">{node.role}</Badge>
        )}
        <Badge variant={stateVariant(node.state)}>
          {node.state || "?"}
        </Badge>
        <span className="ml-auto text-muted-foreground">
          {node.capability_names.length} cap
        </span>
      </div>
      {expanded && kids.map((child) => (
        <TreeNode
          key={child.agent_id}
          node={child}
          children={children}
          depth={depth + 1}
          selectedId={selectedId}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}

function formatTime(ts: number | null | undefined): string {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

function AgentDetailPanel({
  agentId,
  detail,
  isLoading,
}: {
  agentId: string;
  detail: Record<string, unknown> | undefined;
  isLoading: boolean;
}) {
  const { data: history } = useAgentHistory(agentId);

  if (isLoading) {
    return (
      <div className="w-96 shrink-0 rounded-lg border bg-card p-4">
        <p className="text-sm text-muted-foreground">Loading...</p>
      </div>
    );
  }

  const state = (detail?.state as string) ?? "unknown";
  const agentType = (detail?.agent_type as string) ?? "—";
  const capabilities = (detail?.capability_names as string[]) ?? [];

  const totalInputTokens = (history?.total_input_tokens as number) ?? 0;
  const totalOutputTokens = (history?.total_output_tokens as number) ?? 0;
  const actionCount = (history?.action_count as number) ?? 0;
  const actionSuccesses = (history?.action_successes as number) ?? 0;
  const actionFailures = (history?.action_failures as number) ?? 0;
  const inferCount = (history?.infer_count as number) ?? 0;
  const firstSeen = history?.first_seen as number | null;
  const lastSeen = history?.last_seen as number | null;
  const lifecycleEvents = (history?.lifecycle_events as Array<Record<string, unknown>>) ?? [];
  const lastError = history?.last_error as Record<string, unknown> | null;

  return (
    <div className="w-96 shrink-0 rounded-lg border bg-card overflow-auto" style={{ maxHeight: "calc(100vh - 160px)" }}>
      <div className="space-y-4 p-4">
        {/* Header */}
        <div>
          <div className="flex items-center gap-2">
            <Badge variant={stateVariant(state)}>{state}</Badge>
            <span className="text-xs font-mono text-muted-foreground truncate">{agentId}</span>
          </div>
          <p className="text-xs text-muted-foreground mt-1">{agentType}</p>
        </div>

        {/* Timing */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">First seen</p>
            <p className="font-mono">{formatTime(firstSeen)}</p>
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Last seen</p>
            <p className="font-mono">{formatTime(lastSeen)}</p>
          </div>
        </div>

        {/* Resource usage */}
        <div className="grid grid-cols-2 gap-2">
          <MetricCard label="Input Tokens" value={formatTokens(totalInputTokens)} />
          <MetricCard label="Output Tokens" value={formatTokens(totalOutputTokens)} />
          <MetricCard label="LLM Calls" value={inferCount} />
          <MetricCard
            label="Actions"
            value={`${actionSuccesses}/${actionCount}`}
            subtitle={actionFailures > 0 ? `${actionFailures} failed` : undefined}
            className={actionFailures > 0 ? "border-red-800/30" : undefined}
          />
        </div>

        {/* Last error */}
        {lastError && (
          <div className="rounded-md border border-red-800 bg-red-950/30 p-2">
            <p className="text-[10px] uppercase tracking-wider text-red-400 mb-0.5">Last Error</p>
            <p className="text-xs text-red-300 font-mono break-all">
              {lastError.error as string}
            </p>
            <p className="text-[10px] text-muted-foreground mt-1">
              {formatTime(lastError.timestamp as number | null)} — {lastError.span_name as string}
            </p>
          </div>
        )}

        {/* Lifecycle timeline */}
        {lifecycleEvents.length > 0 && (
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
              Lifecycle Events
            </p>
            <div className="space-y-1">
              {lifecycleEvents.map((evt, i) => {
                const evtStatus = evt.status as string;
                const evtName = (evt.name as string)?.replace("lifecycle:", "") ?? "?";
                const evtOutput = evt.output_summary as Record<string, unknown> | undefined;
                return (
                  <div
                    key={i}
                    className="flex items-center gap-2 rounded bg-muted/30 px-2 py-1 text-xs"
                  >
                    <span className={evtStatus === "error" ? "text-red-400" : "text-emerald-400"}>
                      {evtStatus === "error" ? "✗" : "✓"}
                    </span>
                    <span className="font-medium">{evtName}</span>
                    <span className="text-muted-foreground ml-auto font-mono text-[10px]">
                      {formatTime(evt.start_wall as number | null)}
                    </span>
                    {evtOutput?.iterations != null && (
                      <span className="text-muted-foreground text-[10px]">
                        {evtOutput.iterations as number} iters
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Capabilities */}
        {capabilities.length > 0 && (
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
              Capabilities ({capabilities.length})
            </p>
            <div className="flex flex-wrap gap-1">
              {capabilities.map((cap) => (
                <Badge key={cap} variant="default">{cap}</Badge>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export function AgentsTab() {
  const agents = useAgents();
  const hierarchy = useAgentHierarchy();
  const [selectedId, setSelectedId] = useState<string>("");
  const [viewMode, setViewMode] = useState<"list" | "hierarchy">("list");
  const detail = useAgentDetail(selectedId);

  const tree = buildTree(hierarchy.data ?? []);
  const roots = tree.children.get(null) ?? [];

  return (
    <div className="flex gap-4">
      {/* Agent list / hierarchy */}
      <div className="flex-1">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Agents ({(agents.data ?? []).length})
          </h2>
          <div className="flex rounded border border-border">
            <button
              className={`px-3 py-1 text-xs ${viewMode === "list" ? "bg-accent text-accent-foreground" : ""}`}
              onClick={() => setViewMode("list")}
            >
              List
            </button>
            <button
              className={`px-3 py-1 text-xs ${viewMode === "hierarchy" ? "bg-accent text-accent-foreground" : ""}`}
              onClick={() => setViewMode("hierarchy")}
            >
              Hierarchy
            </button>
          </div>
        </div>

        {viewMode === "list" ? (
          <DataTable
            columns={columns}
            data={agents.data ?? []}
            onRowClick={(row) => setSelectedId(row.agent_id)}
            emptyMessage="No agents registered"
          />
        ) : (
          <div className="rounded-lg border bg-card p-2">
            {roots.length > 0 ? (
              roots.map((root) => (
                <TreeNode
                  key={root.agent_id}
                  node={root}
                  children={tree.children}
                  depth={0}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                />
              ))
            ) : (
              <p className="p-4 text-sm text-muted-foreground">
                {hierarchy.isLoading ? "Loading hierarchy..." : "No agents registered"}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Detail panel */}
      {selectedId && (
        <AgentDetailPanel agentId={selectedId} detail={detail.data} isLoading={detail.isLoading} />
      )}
    </div>
  );
}
