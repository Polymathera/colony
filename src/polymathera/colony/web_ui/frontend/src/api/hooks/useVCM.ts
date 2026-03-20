import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type { PageLoadedEntry, PageSummary, VCMStats } from "../types";

export function useVCMStats() {
  return useQuery({
    queryKey: ["vcm", "stats"],
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
