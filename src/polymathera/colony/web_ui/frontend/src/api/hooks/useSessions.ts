import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type { SessionSummary, RunSummary } from "../types";

export function useSessions(tenantId?: string) {
  const params = tenantId ? `?tenant_id=${tenantId}` : "";
  return useQuery({
    queryKey: ["sessions", tenantId],
    queryFn: () => apiFetch<SessionSummary[]>(`/sessions/${params}`),
  });
}

export function useSessionDetail(sessionId: string) {
  return useQuery({
    queryKey: ["sessions", sessionId],
    queryFn: () =>
      apiFetch<Record<string, unknown>>(`/sessions/${sessionId}`),
    enabled: !!sessionId,
  });
}

export function useSessionRuns(sessionId: string) {
  return useQuery({
    queryKey: ["sessions", sessionId, "runs"],
    queryFn: () =>
      apiFetch<RunSummary[]>(`/sessions/${sessionId}/runs`),
    enabled: !!sessionId,
  });
}

export function useRunDetail(runId: string) {
  return useQuery({
    queryKey: ["runs", runId],
    queryFn: () =>
      apiFetch<Record<string, unknown>>(`/sessions/runs/${runId}`),
    enabled: !!runId,
  });
}

export function useSessionStats() {
  return useQuery({
    queryKey: ["sessions", "stats"],
    queryFn: () => apiFetch<Record<string, unknown>>("/sessions/stats/overview"),
  });
}
