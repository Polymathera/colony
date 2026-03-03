import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type { PageGraphGroup, PageGraphData } from "../types";

export function usePageGraphGroups() {
  return useQuery({
    queryKey: ["page-graph", "groups"],
    queryFn: () => apiFetch<PageGraphGroup[]>("/vcm/page-graph/groups"),
  });
}

export function usePageGraph(tenantId: string, groupId: string) {
  return useQuery({
    queryKey: ["page-graph", tenantId, groupId],
    queryFn: () =>
      apiFetch<PageGraphData>(
        `/vcm/page-graph?tenant_id=${encodeURIComponent(tenantId)}&group_id=${encodeURIComponent(groupId)}`,
      ),
    enabled: !!tenantId && !!groupId,
  });
}
