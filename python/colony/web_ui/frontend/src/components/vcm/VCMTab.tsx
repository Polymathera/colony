import { useMemo, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useVCMStats, useVCMPages, useLoadedPageEntries } from "@/api/hooks/useVCM";
import { MetricCard } from "../shared/MetricCard";
import { DataTable } from "../shared/DataTable";
import { Badge } from "../shared/Badge";
import { formatTokens } from "@/lib/utils";
import type { PageSummary, PageLoadedEntry } from "@/api/types";

const HISTOGRAM_BINS = [
  { label: "1", min: 0, max: 1 },
  { label: "2-10", min: 2, max: 10 },
  { label: "11-50", min: 11, max: 50 },
  { label: "51-100", min: 51, max: 100 },
  { label: "101-500", min: 101, max: 500 },
  { label: "501-1k", min: 501, max: 1000 },
  { label: "1k-5k", min: 1001, max: 5000 },
  { label: "5k+", min: 5001, max: Infinity },
];

type ColorMode = "loaded" | "frequency" | "recency";

const pageColumns = [
  {
    key: "page_id",
    header: "Page ID",
    className: "font-mono text-xs",
    render: (row: PageSummary) =>
      row.page_id.length > 24
        ? row.page_id.slice(0, 24) + "..."
        : row.page_id,
  },
  { key: "source", header: "Source" },
  {
    key: "tokens",
    header: "Tokens",
    render: (row: PageSummary) => formatTokens(row.tokens),
  },
  {
    key: "loaded",
    header: "Status",
    render: (row: PageSummary) => (
      <Badge variant={row.loaded ? "success" : "default"}>
        {row.loaded ? "loaded" : "unloaded"}
      </Badge>
    ),
  },
];

function getPageColor(
  page: PageSummary,
  loadedMap: Map<string, PageLoadedEntry>,
  mode: ColorMode,
  maxAccess: number,
  now: number,
): string {
  const entry = loadedMap.get(page.page_id);
  if (mode === "loaded") {
    return entry ? "#10b981" : "#374151";
  }
  if (!entry) return "#1f2937";
  if (mode === "frequency") {
    const ratio = maxAccess > 0 ? entry.total_access_count / maxAccess : 0;
    const r = Math.round(59 + ratio * 196);
    const g = Math.round(130 - ratio * 80);
    const b = Math.round(246 - ratio * 180);
    return `rgb(${r}, ${g}, ${b})`;
  }
  // recency
  const latestAccess = Math.max(...entry.locations.map((l) => l.last_access_time), 0);
  const age = now - latestAccess;
  const brightness = Math.max(0.2, 1 - Math.min(age / 300, 0.8)); // dim after 5min
  return `hsl(142, 70%, ${Math.round(brightness * 50)}%)`;
}

export function VCMTab() {
  const stats = useVCMStats();
  const pages = useVCMPages();
  const loadedEntries = useLoadedPageEntries();
  const [viewMode, setViewMode] = useState<"table" | "grid">("grid");
  const [colorMode, setColorMode] = useState<ColorMode>("loaded");
  const [hoveredPage, setHoveredPage] = useState<PageSummary | null>(null);

  const histogramData = useMemo(() => {
    const allPages = pages.data ?? [];
    if (allPages.length === 0) return [];
    return HISTOGRAM_BINS.map((bin) => ({
      name: bin.label,
      count: allPages.filter((p) => p.tokens >= bin.min && p.tokens <= bin.max).length,
    })).filter((d) => d.count > 0);
  }, [pages.data]);

  const loadedMap = useMemo(() => {
    const map = new Map<string, PageLoadedEntry>();
    for (const e of loadedEntries.data ?? []) {
      map.set(e.page_id, e);
    }
    return map;
  }, [loadedEntries.data]);

  const maxAccess = useMemo(() => {
    let max = 0;
    for (const e of loadedEntries.data ?? []) {
      if (e.total_access_count > max) max = e.total_access_count;
    }
    return max;
  }, [loadedEntries.data]);

  return (
    <div className="space-y-6">
      {/* Stats row */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          VCM Statistics
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricCard
            label="Total Pages"
            value={stats.data?.total_pages ?? 0}
          />
          <MetricCard
            label="Loaded Pages"
            value={stats.data?.loaded_pages ?? 0}
          />
          <MetricCard
            label="Page Groups"
            value={stats.data?.page_groups ?? 0}
          />
          <MetricCard
            label="Pending Faults"
            value={stats.data?.pending_faults ?? 0}
          />
        </div>
      </section>

      {/* Page size histogram */}
      {histogramData.length > 0 && (
        <section className="rounded-lg border bg-card p-4">
          <h3 className="mb-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Page Size Distribution (tokens)
          </h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={histogramData}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(216, 34%, 17%)" />
              <XAxis
                dataKey="name"
                tick={{ fill: "hsl(215, 16%, 57%)", fontSize: 11 }}
                axisLine={{ stroke: "hsl(216, 34%, 17%)" }}
              />
              <YAxis
                tick={{ fill: "hsl(215, 16%, 57%)", fontSize: 11 }}
                axisLine={{ stroke: "hsl(216, 34%, 17%)" }}
              />
              <Tooltip
                contentStyle={{
                  background: "hsl(222, 47%, 7%)",
                  border: "1px solid hsl(216, 34%, 17%)",
                  borderRadius: "0.5rem",
                  fontSize: 12,
                }}
              />
              <Bar dataKey="count" name="Pages" fill="#8b5cf6" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </section>
      )}

      {/* Page table / grid */}
      <section>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Pages ({(pages.data ?? []).length})
          </h2>
          <div className="flex items-center gap-2">
            {viewMode === "grid" && (
              <select
                className="rounded border border-border bg-background px-2 py-1 text-xs"
                value={colorMode}
                onChange={(e) => setColorMode(e.target.value as ColorMode)}
              >
                <option value="loaded">Color: Loaded Status</option>
                <option value="frequency">Color: Access Frequency</option>
                <option value="recency">Color: Last Access</option>
              </select>
            )}
            <div className="flex rounded border border-border">
              <button
                className={`px-3 py-1 text-xs ${viewMode === "table" ? "bg-accent text-accent-foreground" : ""}`}
                onClick={() => setViewMode("table")}
              >
                Table
              </button>
              <button
                className={`px-3 py-1 text-xs ${viewMode === "grid" ? "bg-accent text-accent-foreground" : ""}`}
                onClick={() => setViewMode("grid")}
              >
                Grid
              </button>
            </div>
          </div>
        </div>

        {viewMode === "table" ? (
          <DataTable
            columns={pageColumns}
            data={pages.data ?? []}
            emptyMessage={
              pages.isLoading ? "Loading pages..." : "No pages in page table"
            }
          />
        ) : (
          <div className="relative">
            <div className="grid gap-0.5" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(10px, 1fr))" }}>
              {(pages.data ?? []).map((page) => {
                const now = Date.now() / 1000;
                return (
                  <div
                    key={page.page_id}
                    className="aspect-square rounded-[2px] cursor-pointer transition-transform hover:scale-150 hover:z-10"
                    style={{ backgroundColor: getPageColor(page, loadedMap, colorMode, maxAccess, now) }}
                    onMouseEnter={() => setHoveredPage(page)}
                    onMouseLeave={() => setHoveredPage(null)}
                  />
                );
              })}
            </div>
            {/* Hover tooltip */}
            {hoveredPage && (
              <div className="pointer-events-none fixed bottom-4 left-4 z-50 max-w-sm rounded-lg border bg-card p-3 shadow-lg">
                <div className="space-y-1 text-xs">
                  <div className="font-mono font-semibold">{hoveredPage.page_id}</div>
                  <div className="text-muted-foreground">{hoveredPage.source}</div>
                  <div>Tokens: {formatTokens(hoveredPage.tokens)}</div>
                  {loadedMap.has(hoveredPage.page_id) && (
                    <div className="text-green-400">
                      Loaded — {loadedMap.get(hoveredPage.page_id)!.total_access_count} accesses
                    </div>
                  )}
                  {hoveredPage.files && hoveredPage.files.length > 0 && (
                    <div className="mt-1 border-t border-border pt-1">
                      <div className="text-muted-foreground">Files:</div>
                      {hoveredPage.files.slice(0, 5).map((f) => (
                        <div key={f} className="truncate font-mono text-[10px]">{f}</div>
                      ))}
                      {hoveredPage.files.length > 5 && (
                        <div className="text-muted-foreground">+{hoveredPage.files.length - 5} more</div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
            {/* Legend */}
            <div className="mt-2 flex gap-4 text-xs text-muted-foreground">
              {colorMode === "loaded" && (
                <>
                  <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm" style={{ background: "#10b981" }} /> Loaded</span>
                  <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm" style={{ background: "#374151" }} /> Unloaded</span>
                </>
              )}
              {colorMode === "frequency" && <span>Brighter = more accesses</span>}
              {colorMode === "recency" && <span>Brighter = more recently accessed</span>}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
