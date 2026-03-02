import { useState } from "react";
import {
  useBlackboardScopes,
  useBlackboardEntries,
  type BlackboardScopeSummary,
} from "@/api/hooks/useBlackboard";
import { Badge } from "../shared/Badge";
import { DataTable } from "../shared/DataTable";

const scopeColumns = [
  { key: "scope", header: "Scope" },
  { key: "scope_id", header: "Scope ID", className: "font-mono text-xs" },
  { key: "entry_count", header: "Entries" },
  {
    key: "backend_type",
    header: "Backend",
    render: (row: BlackboardScopeSummary) => (
      <Badge variant="default">{row.backend_type || "?"}</Badge>
    ),
  },
  {
    key: "newest_entry_age",
    header: "Last Write",
    render: (row: BlackboardScopeSummary) => {
      if (row.newest_entry_age == null) return "—";
      if (row.newest_entry_age < 60) return `${Math.round(row.newest_entry_age)}s ago`;
      return `${Math.round(row.newest_entry_age / 60)}m ago`;
    },
  },
  { key: "subscriber_count", header: "Subscribers" },
];

function truncateValue(val: unknown): string {
  const s = typeof val === "string" ? val : JSON.stringify(val);
  return s.length > 80 ? s.slice(0, 80) + "..." : s;
}

export function BlackboardTab() {
  const scopes = useBlackboardScopes();
  const [selected, setSelected] = useState<{ scope: string; scopeId: string; backendType: string } | null>(null);
  const entries = useBlackboardEntries(
    selected?.scope ?? "",
    selected?.scopeId ?? "",
    selected?.backendType ?? "",
  );

  return (
    <div className="space-y-6">
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Blackboard Scopes ({(scopes.data ?? []).length})
        </h2>
        <DataTable
          columns={scopeColumns}
          data={scopes.data ?? []}
          onRowClick={(row) =>
            setSelected({ scope: row.scope, scopeId: row.scope_id, backendType: row.backend_type })
          }
          emptyMessage={
            scopes.isLoading
              ? "Discovering scopes..."
              : "No active blackboard scopes"
          }
        />
      </section>

      {selected && (
        <section>
          <div className="mb-3 flex items-center gap-2">
            <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              Entries
            </h2>
            <Badge variant="info">
              {selected.scope}:{selected.scopeId}
            </Badge>
            <button
              className="ml-auto text-xs text-muted-foreground hover:text-foreground"
              onClick={() => setSelected(null)}
            >
              Close
            </button>
          </div>
          {entries.isLoading ? (
            <p className="text-sm text-muted-foreground">Loading entries...</p>
          ) : (
            <div className="overflow-auto rounded-lg border">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b bg-muted/30">
                    <th className="p-2 text-left font-medium">Key</th>
                    <th className="p-2 text-left font-medium">Value</th>
                    <th className="p-2 text-left font-medium">By</th>
                    <th className="p-2 text-left font-medium">Ver</th>
                    <th className="p-2 text-left font-medium">Tags</th>
                  </tr>
                </thead>
                <tbody>
                  {(entries.data ?? []).map((e, i) => (
                    <tr key={i} className="border-b hover:bg-accent/30">
                      <td className="p-2 font-mono">{e.key}</td>
                      <td className="max-w-xs truncate p-2 text-muted-foreground">
                        {truncateValue(e.value)}
                      </td>
                      <td className="p-2 font-mono">
                        {(e.created_by ?? "").slice(0, 16)}
                      </td>
                      <td className="p-2">{e.version}</td>
                      <td className="p-2">
                        <div className="flex flex-wrap gap-1">
                          {e.tags.slice(0, 3).map((t) => (
                            <Badge key={t} variant="default">{t}</Badge>
                          ))}
                          {e.tags.length > 3 && (
                            <span className="text-muted-foreground">
                              +{e.tags.length - 3}
                            </span>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                  {(entries.data ?? []).length === 0 && (
                    <tr>
                      <td colSpan={5} className="p-4 text-center text-muted-foreground">
                        No entries in this scope
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
