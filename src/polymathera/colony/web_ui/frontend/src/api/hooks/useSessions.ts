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
    onSuccess: (data) => {
      // Optimistically insert the new session into every cached
      // session list so the Sidebar's ``activeSession`` lookup matches
      // immediately after ``onSelectSession(new_id)``. Without this,
      // there is a race window between the create returning and the
      // refetch landing where ``sessions.data`` still holds the
      // pre-create list — Sidebar's stale-cleanup useEffect would
      // then snap ``activeSessionId`` back to ``null`` and the new
      // session would appear under "Recent Sessions" instead of
      // opening. The optimistic row is replaced with the real one as
      // soon as the invalidated query refetches.
      if (data.status === "created") {
        qc.setQueriesData<SessionSummary[]>(
          { queryKey: ["sessions"], exact: false },
          (prev) => {
            if (!Array.isArray(prev)) return prev;
            if (prev.some((s) => s.session_id === data.session_id)) return prev;
            const optimistic: SessionSummary = {
              session_id: data.session_id,
              tenant_id: "",
              colony_id: "",
              state: "active",
              created_at: Date.now() / 1000,
              run_count: 0,
            };
            return [optimistic, ...prev];
          },
        );
      }
      qc.invalidateQueries({ queryKey: ["sessions"] });
    },
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
