import { useState, useMemo, useCallback } from "react";
import {
  useBlackboardScopes,
  useBlackboardEntries,
  type BlackboardScopeSummary,
} from "@/api/hooks/useBlackboard";
import { Badge } from "../shared/Badge";
import { DataTable } from "../shared/DataTable";

const scopeColumns = [
  { key: "scope_id", header: "Scope ID", className: "font-mono text-xs" },
  { key: "tenant_id", header: "Tenant" },
  { key: "colony_id", header: "Colony" },
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
  const [selected, setSelected] = useState<{
    scopeId: string;
    backendType: string;
    tenantId: string | null;
    colonyId: string | null;
  } | null>(null);
  const entries = useBlackboardEntries(
    selected?.scopeId ?? "",
    selected?.backendType ?? "",
    selected?.tenantId ?? null,
    selected?.colonyId ?? null,
  );

  // Tag + text filtering
  const [searchText, setSearchText] = useState("");
  const [activeTags, setActiveTags] = useState<Set<string>>(new Set());

  const addTag = useCallback((tag: string) => {
    setActiveTags((prev) => new Set([...prev, tag]));
  }, []);

  const removeTag = useCallback((tag: string) => {
    setActiveTags((prev) => {
      const next = new Set(prev);
      next.delete(tag);
      return next;
    });
  }, []);

  const filteredEntries = useMemo(() => {
    const raw = entries.data ?? [];
    if (!searchText && activeTags.size === 0) return raw;

    const lowerSearch = searchText.toLowerCase();
    return raw.filter((e) => {
      // Tag filter: entry must have ALL active tags
      if (activeTags.size > 0) {
        for (const tag of activeTags) {
          if (!e.tags.includes(tag)) return false;
        }
      }
      // Text filter: key or value must contain search text
      if (lowerSearch) {
        const valStr = typeof e.value === "string" ? e.value : JSON.stringify(e.value);
        if (
          !e.key.toLowerCase().includes(lowerSearch) &&
          !valStr.toLowerCase().includes(lowerSearch)
        ) {
          return false;
        }
      }
      return true;
    });
  }, [entries.data, searchText, activeTags]);

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
            setSelected({
              scopeId: row.scope_id,
              backendType: row.backend_type,
              tenantId: row.tenant_id,
              colonyId: row.colony_id,
            })
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
              {selected.scopeId}
            </Badge>
            <button
              className="ml-auto text-xs text-muted-foreground hover:text-foreground"
              onClick={() => setSelected(null)}
            >
              Close
            </button>
          </div>

          {/* Search + tag filters */}
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <input
              type="text"
              placeholder="Search keys & values..."
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              className="rounded border border-border bg-background px-2 py-1 text-xs font-mono w-60"
            />
            {activeTags.size > 0 && (
              <div className="flex flex-wrap items-center gap-1">
                <span className="text-[10px] text-muted-foreground">Tags:</span>
                {[...activeTags].map((tag) => (
                  <button
                    key={tag}
                    onClick={() => removeTag(tag)}
                    className="flex items-center gap-0.5 rounded bg-primary/20 px-1.5 py-0.5 text-[10px] text-primary hover:bg-primary/30"
                  >
                    {tag} <span>&times;</span>
                  </button>
                ))}
                <button
                  onClick={() => setActiveTags(new Set())}
                  className="text-[10px] text-muted-foreground hover:text-foreground"
                >
                  Clear all
                </button>
              </div>
            )}
            {searchText || activeTags.size > 0 ? (
              <span className="text-[10px] text-muted-foreground">
                {filteredEntries.length} / {(entries.data ?? []).length}
              </span>
            ) : null}
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
                  {filteredEntries.map((e, i) => (
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
                            <button
                              key={t}
                              onClick={() => addTag(t)}
                              className="cursor-pointer"
                            >
                              <Badge variant={activeTags.has(t) ? "info" : "default"}>
                                {t}
                              </Badge>
                            </button>
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
                  {filteredEntries.length === 0 && (
                    <tr>
                      <td colSpan={5} className="p-4 text-center text-muted-foreground">
                        {(entries.data ?? []).length === 0
                          ? "No entries in this scope"
                          : "No entries match filters"}
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
