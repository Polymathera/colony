import { useState, useEffect, useRef, useCallback } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";

/* ── Types ────────────────────────────────────────────────────── */

/** Prometheus metric type as reported by # TYPE annotations. */
export type PrometheusType = "counter" | "gauge" | "histogram" | "summary" | "untyped";

/**
 * A single scraped snapshot: metric_name → aggregated value,
 * plus _timestamp and _types metadata.
 */
export type MetricsSnapshot = Record<string, number> & {
  _timestamp: number;
  _error?: string;
  _types?: Record<string, PrometheusType>;
};

/* ── SSE live metrics stream ────────────────────────────────────── */

/**
 * Subscribe to the SSE metrics stream at /stream/metrics.
 * Returns the latest snapshot, a rolling history buffer,
 * and a merged type map accumulated across all snapshots.
 */
export function useMetricsStream(interval = 5, maxHistory = 120) {
  const [latest, setLatest] = useState<MetricsSnapshot | null>(null);
  const [history, setHistory] = useState<MetricsSnapshot[]>([]);
  const [connected, setConnected] = useState(false);
  const [types, setTypes] = useState<Record<string, PrometheusType>>({});
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const params = new URLSearchParams();
    params.set("interval", String(interval));

    const es = new EventSource(`/api/v1/stream/metrics?${params}`);
    esRef.current = es;

    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);

    es.onmessage = (ev) => {
      try {
        const data: MetricsSnapshot = JSON.parse(ev.data);
        if (!data._error) {
          // Merge type annotations (accumulate — types don't change between scrapes)
          if (data._types) {
            setTypes((prev) => ({ ...prev, ...data._types }));
          }
          setLatest(data);
          setHistory((prev) => {
            const next = [...prev, data];
            return next.length > maxHistory ? next.slice(-maxHistory) : next;
          });
        }
      } catch {
        // ignore parse errors
      }
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [interval, maxHistory]);

  const disconnect = useCallback(() => {
    esRef.current?.close();
    esRef.current = null;
    setConnected(false);
  }, []);

  return { latest, history, connected, types, disconnect };
}

/* ── REST: buffered history (initial load) ──────────────────── */

export function useMetricsHistory(last = 60, refetchInterval = 15000) {
  return useQuery({
    queryKey: ["metrics", "history", last],
    queryFn: () => apiFetch<MetricsSnapshot[]>(`/metrics/history?last=${last}`),
    refetchInterval,
  });
}

/* ── REST: single scrape ────────────────────────────────────── */

export function useScrapedMetrics(refetchInterval = 10000) {
  return useQuery({
    queryKey: ["metrics", "scraped"],
    queryFn: () => apiFetch<MetricsSnapshot>("/metrics/scraped"),
    refetchInterval,
  });
}
