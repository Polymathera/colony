import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type {
  PageLoadedEntry,
  PageSummary,
  VCMStats,
  MapRepoRequest,
  MapRepoResponse,
} from "../types";

export function useVCMStats(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: ["vcm", "stats"],
    enabled: options?.enabled ?? true,
    queryFn: () => apiFetch<VCMStats>("/vcm/stats"),
  });
}

export function useVCMPages() {
  return useQuery({
    queryKey: ["vcm", "pages"],
    queryFn: () => apiFetch<PageSummary[]>("/vcm/pages?limit=50000"),
  });
}

export function useWorkingSet() {
  return useQuery({
    queryKey: ["vcm", "working-set"],
    queryFn: () => apiFetch<{ pages: string[] }>("/vcm/working-set"),
  });
}

export function useLoadedPageEntries() {
  return useQuery({
    queryKey: ["vcm", "loaded-pages"],
    queryFn: () => apiFetch<PageLoadedEntry[]>("/vcm/loaded-pages"),
  });
}

export function usePageDetail(pageId: string, colonyId: string, tenantId: string) {
  return useQuery({
    queryKey: ["vcm", "pages", pageId, colonyId, tenantId],
    queryFn: () =>
      apiFetch<Record<string, unknown>>(`/vcm/pages/${pageId}/${colonyId}/${tenantId}`),
    enabled: !!pageId && !!colonyId && !!tenantId,
  });
}

export interface MappingOpStatus {
  op_id: string;
  status: string;
  origin_url: string;
  started_at: number;
  completed_at: number | null;
  message: string;
  scope_id: string;
}

export function useMappingOperations() {
  return useQuery({
    queryKey: ["vcm", "map", "operations"],
    queryFn: () => apiFetch<MappingOpStatus[]>("/vcm/map/operations"),
    refetchInterval: (query) => {
      const ops = query.state.data;
      if (ops && ops.some((op) => op.status === "pending" || op.status === "running")) {
        return 2000; // Poll every 2s while active
      }
      return false;
    },
  });
}

export function useMapRepo() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: MapRepoRequest) =>
      apiFetch<MapRepoResponse>("/vcm/map", {
        method: "POST",
        body: JSON.stringify(req),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["vcm"] });
      qc.invalidateQueries({ queryKey: ["page-graph"] });
    },
  });
}
