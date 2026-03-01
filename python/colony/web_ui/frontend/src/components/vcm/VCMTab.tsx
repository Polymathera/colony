import { useVCMStats, useVCMPages, useWorkingSet } from "@/api/hooks/useVCM";
import { MetricCard } from "../shared/MetricCard";
import { DataTable } from "../shared/DataTable";
import { Badge } from "../shared/Badge";
import { formatTokens } from "@/lib/utils";
import type { PageSummary } from "@/api/types";

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

export function VCMTab() {
  const stats = useVCMStats();
  const pages = useVCMPages();
  const workingSet = useWorkingSet();

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

      {/* Working set */}
      {workingSet.data && workingSet.data.pages.length > 0 && (
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Working Set ({workingSet.data.pages.length} pages)
          </h2>
          <div className="flex flex-wrap gap-1.5">
            {workingSet.data.pages.map((pageId) => (
              <Badge key={pageId} variant="success">
                {pageId.length > 16 ? pageId.slice(0, 16) + "..." : pageId}
              </Badge>
            ))}
          </div>
        </section>
      )}

      {/* Page table */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Page Table
        </h2>
        <DataTable
          columns={pageColumns}
          data={pages.data ?? []}
          emptyMessage={
            pages.isLoading ? "Loading pages..." : "No pages in page table"
          }
        />
      </section>
    </div>
  );
}
