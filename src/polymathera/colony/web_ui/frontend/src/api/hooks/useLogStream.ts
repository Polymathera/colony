import { useState, useEffect, useRef, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type { LogSource } from "../types";

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
