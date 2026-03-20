import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type { PageGraphScope, PageGraphData } from "../types";

export function usePageGraphScopes() {
  return useQuery({
    queryKey: ["page-graph", "groups"],
    queryFn: () => apiFetch<PageGraphScope[]>("/vcm/page-graph/scopes"),
  });
}

export function usePageGraph(tenantId: string, colonyId: string) {
  return useQuery({
    queryKey: ["page-graph", tenantId, colonyId],
    queryFn: () =>
      apiFetch<PageGraphData>(
        `/vcm/page-graph?tenant_id=${encodeURIComponent(tenantId)}&colony_id=${encodeURIComponent(colonyId)}`,
      ),
    enabled: !!tenantId && !!colonyId,
  });
}
