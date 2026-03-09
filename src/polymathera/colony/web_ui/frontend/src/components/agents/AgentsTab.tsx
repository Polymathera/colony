import { useState } from "react";
import { useAgents, useAgentDetail, useAgentHierarchy } from "@/api/hooks/useAgents";
import { DataTable } from "../shared/DataTable";
import { Badge } from "../shared/Badge";
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
