import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type { PageSummary, VCMStats } from "../types";

export function useVCMStats() {
  return useQuery({
    queryKey: ["vcm", "stats"],
    queryFn: () => apiFetch<VCMStats>("/vcm/stats"),
  });
}

export function useVCMPages() {
  return useQuery({
    queryKey: ["vcm", "pages"],
    queryFn: () => apiFetch<PageSummary[]>("/vcm/pages"),
  });
}

export function useWorkingSet() {
  return useQuery({
    queryKey: ["vcm", "working-set"],
    queryFn: () => apiFetch<{ pages: string[] }>("/vcm/working-set"),
  });
}

export function usePageDetail(pageId: string) {
  return useQuery({
    queryKey: ["vcm", "pages", pageId],
    queryFn: () =>
      apiFetch<Record<string, unknown>>(`/vcm/pages/${pageId}`),
    enabled: !!pageId,
  });
}
