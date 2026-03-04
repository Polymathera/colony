import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch } from "../client";
import type { TraceSummary, TraceSpan } from "../types";

export function useTraces(limit = 100) {
  return useQuery({
    queryKey: ["traces", limit],
    queryFn: () => apiFetch<TraceSummary[]>(`/traces/?limit=${limit}`),
  });
}

export function useTraceSummary(traceId: string) {
  return useQuery({
    queryKey: ["traces", traceId, "summary"],
    queryFn: () =>
      apiFetch<TraceSummary>(`/traces/${traceId}/summary`),
    enabled: !!traceId,
  });
}

export function useTraceSpans(
  traceId: string,
  filters?: { runId?: string; kind?: string },
) {
  const params = new URLSearchParams();
  if (filters?.runId) params.set("run_id", filters.runId);
  if (filters?.kind) params.set("kind", filters.kind);
  const qs = params.toString() ? `?${params.toString()}` : "";

  return useQuery({
    queryKey: ["traces", traceId, "spans", filters],
    queryFn: () =>
      apiFetch<TraceSpan[]>(`/traces/${traceId}/spans${qs}`),
    enabled: !!traceId,
  });
}

/**
 * SSE hook for streaming trace spans in real-time.
 * Returns a growing map of spans and streaming status.
 */
export function useTraceStream(traceId: string | null) {
  const [spans, setSpans] = useState<Map<string, TraceSpan>>(new Map());
  const [isStreaming, setIsStreaming] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  const connect = useCallback(() => {
    if (!traceId) return;

    const url = `/api/v1/stream/traces/${traceId}`;
    const es = new EventSource(url);
    eventSourceRef.current = es;
    setIsStreaming(true);

    es.onmessage = (event) => {
      try {
        const span: TraceSpan = JSON.parse(event.data);
        setSpans((prev) => {
          const next = new Map(prev);
          next.set(span.span_id, span);
          return next;
        });
      } catch {
        // Skip malformed messages
      }
    };

    es.onerror = () => {
      setIsStreaming(false);
      es.close();
    };
  }, [traceId]);

  useEffect(() => {
    connect();
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      setIsStreaming(false);
    };
  }, [connect]);

  return { spans, isStreaming };
}
