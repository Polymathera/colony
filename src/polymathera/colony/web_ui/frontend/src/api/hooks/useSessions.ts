import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type {
  SessionSummary,
  RunSummary,
  CreateSessionRequest,
  CreateSessionResponse,
  SessionActionResponse,
} from "../types";

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

export function useCreateSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req?: CreateSessionRequest) =>
      apiFetch<CreateSessionResponse>("/sessions/", {
        method: "POST",
        body: JSON.stringify(req ?? {}),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  });
}

export function useSuspendSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) =>
      apiFetch<SessionActionResponse>(`/sessions/${sessionId}/suspend`, {
        method: "PUT",
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  });
}

export function useResumeSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) =>
      apiFetch<SessionActionResponse>(`/sessions/${sessionId}/resume`, {
        method: "PUT",
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  });
}

export function useCloseSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) =>
      apiFetch<SessionActionResponse>(`/sessions/${sessionId}`, {
        method: "DELETE",
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["sessions"] }),
  });
}
