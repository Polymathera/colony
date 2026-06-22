/**
 * Hooks for the ``/api/v1/colony-status/*`` routes that power the
 * ColonyStatusPanel (P11). v1 surfaces three reads:
 *
 * - ``useColonyAlerts`` — recent bottleneck + inconsistency rows.
 * - ``useColonyRecentActivity`` — full ``interaction_log`` tail for
 *   the current colony (mix of GitHub events + mentions + alerts).
 * - ``useColonyProjectLink`` — GitHub Project deep-link derived
 *   from the colony's design monorepo URL.
 *
 * Out of v1 (deferred per p11_p12_plan.md): LLM "Next steps for
 * you", consciousness-stream excerpts, scheduled-mission tile.
 */
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";

export interface ColonyAlertRow {
  id: number;
  ts: string;
  event_kind: string; // 'bottleneck' | 'inconsistency'
  payload: Record<string, unknown>;
  refs: Array<{ kind: string; value: string }>;
  channel_ref: string | null;
}

export interface ColonyAlertsResponse {
  alerts: ColonyAlertRow[];
  count: number;
}

export function useColonyAlerts(limit = 50, options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: ["colony-status", "alerts", limit],
    queryFn: () =>
      apiFetch<ColonyAlertsResponse>(
        `/colony-status/alerts?limit=${limit}`,
      ),
    enabled: options?.enabled ?? true,
    // The alerts tile is a visible-on-load surface; refetch every
    // 30s so freshly-detected bottlenecks show without a manual
    // browser refresh. The route is a single Postgres tail query,
    // cheap to poll.
    refetchInterval: 30_000,
  });
}

export interface ColonyActivityRow {
  id: number;
  ts: string;
  event_kind: string;
  channel: string;
  channel_ref: string | null;
  user_login: string | null;
  payload: Record<string, unknown>;
  refs: Array<{ kind: string; value: string }>;
}

export interface ColonyActivityResponse {
  events: ColonyActivityRow[];
  count: number;
}

export function useColonyRecentActivity(
  limit = 50,
  options?: { enabled?: boolean },
) {
  return useQuery({
    queryKey: ["colony-status", "recent-activity", limit],
    queryFn: () =>
      apiFetch<ColonyActivityResponse>(
        `/colony-status/recent-activity?limit=${limit}`,
      ),
    enabled: options?.enabled ?? true,
    refetchInterval: 30_000,
  });
}

export interface AgentDiagnosticRow {
  id: number;
  ts: string;
  event_kind: string; // always 'agent_diagnostic'
  payload: {
    agent_id?: string;
    kind?: string;
    stop_reason?: string;
    reason?: string;
    exception_type?: string;
    exception_message?: string;
    timestamp?: number;
    [k: string]: unknown;
  };
  refs: Array<{ kind: string; value: string }>;
}

export interface AgentDiagnosticsResponse {
  diagnostics: AgentDiagnosticRow[];
  count: number;
}

export function useColonyAgentDiagnostics(
  limit = 20,
  options?: { enabled?: boolean },
) {
  return useQuery({
    queryKey: ["colony-status", "agent-diagnostics", limit],
    queryFn: () =>
      apiFetch<AgentDiagnosticsResponse>(
        `/colony-status/agent-diagnostics?limit=${limit}`,
      ),
    enabled: options?.enabled ?? true,
    refetchInterval: 30_000,
  });
}

export interface ColonyProjectLink {
  project_url: string | null;
}

export function useColonyProjectLink(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: ["colony-status", "project-link"],
    queryFn: () =>
      apiFetch<ColonyProjectLink>("/colony-status/project-link"),
    enabled: options?.enabled ?? true,
    // Rarely changes — only when the operator switches the colony's
    // design monorepo. 5-minute stale window is plenty.
    staleTime: 5 * 60_000,
  });
}
