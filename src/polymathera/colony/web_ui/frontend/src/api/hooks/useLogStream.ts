import { useState, useEffect, useRef, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type { LogSource, LogQueryResult, LogStats, LogActorSummary } from "../types";

const MAX_LOG_LINES = 5000;

/* ── Actor/source discovery ──────────────────────────────────── */

export function useLogSources(refetchInterval = 10000) {
  return useQuery({
    queryKey: ["logs", "sources"],
    queryFn: () => apiFetch<LogSource[]>("/logs/sources"),
    refetchInterval,
  });
}

/* ── SSE log stream ──────────────────────────────────────────── */

/** Connect to the SSE log stream for a specific worker (identified by node_id + pid). */
export function useLogStream(source: LogSource | null) {
  const [lines, setLines] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  // Stable key for the effect dependency
  const sourceKey = source ? `${source.node_id}:${source.pid}` : null;

  useEffect(() => {
    // Close previous connection
    esRef.current?.close();
    esRef.current = null;
    setConnected(false);

    if (!source) return;

    setLines([]);

    const params = new URLSearchParams();
    params.set("node_id", source.node_id);
    params.set("pid", String(source.pid));
    params.set("lines", "500");

    const es = new EventSource(`/api/v1/stream/logs?${params}`);
    esRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data._error) return;
        if (data.line !== undefined) {
          setLines((prev) => {
            const next = [...prev, data.line as string];
            return next.length > MAX_LOG_LINES ? next.slice(-MAX_LOG_LINES) : next;
          });
        }
      } catch {
        // ignore
      }
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourceKey]);

  const clear = useCallback(() => setLines([]), []);

  return { lines, connected, clear };
}

/* ── Persistent log queries (PostgreSQL) ──────────────────────── */

export interface PersistentLogFilters {
  session_id?: string;
  run_id?: string;
  trace_id?: string;
  actor_class?: string;
  level?: string;
  search?: string;
  since?: number;
  until?: number;
  limit?: number;
  offset?: number;
}

export function usePersistentLogs(filters: PersistentLogFilters, enabled = true) {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(filters)) {
    if (v !== undefined && v !== null && v !== "") params.set(k, String(v));
  }
  return useQuery({
    queryKey: ["logs", "persistent", params.toString()],
    queryFn: () => apiFetch<LogQueryResult>(`/logs/persistent?${params}`),
    enabled,
    refetchInterval: false,
  });
}

export function useLogStats(sessionId?: string) {
  const params = new URLSearchParams();
  if (sessionId) params.set("session_id", sessionId);
  return useQuery({
    queryKey: ["logs", "stats", sessionId ?? "all"],
    queryFn: () => apiFetch<LogStats>(`/logs/persistent/stats?${params}`),
    refetchInterval: 30000,
  });
}

export function useLogActorClasses() {
  return useQuery({
    queryKey: ["logs", "actors", "persistent"],
    queryFn: () => apiFetch<LogActorSummary[]>("/logs/persistent/actors"),
    refetchInterval: 30000,
  });
}
