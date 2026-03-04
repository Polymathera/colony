import { useState, useMemo, useCallback } from "react";
import { useTraces, useTraceSpans, useTraceStream } from "@/api/hooks/useTraces";
import { DataTable } from "../shared/DataTable";
import { Badge } from "../shared/Badge";
import { MetricCard } from "../shared/MetricCard";
import { formatTimestamp, formatTokens, cn } from "@/lib/utils";
import type { TraceSummary, TraceSpan } from "@/api/types";

/* ── Constants ───────────────────────────────────────────────── */

const KIND_COLORS: Record<string, string> = {
  run: "#8b5cf6",
  agent_step: "#3b82f6",
  plan: "#f59e0b",
  action: "#10b981",
  infer: "#ec4899",
  page_request: "#06b6d4",
  blackboard_op: "#f97316",
  child_spawn: "#a855f7",
  event_process: "#64748b",
  capability: "#14b8a6",
  custom: "#6b7280",
};

const KIND_LABELS: Record<string, string> = {
  run: "RUN",
  agent_step: "STEP",
  plan: "PLAN",
  action: "ACT",
  infer: "LLM",
  page_request: "PAGE",
  blackboard_op: "BB",
  child_spawn: "CHILD",
  event_process: "EVT",
  capability: "CAP",
  custom: "CUSTOM",
};

/* ── Helpers ─────────────────────────────────────────────────── */

const statusVariant = (status: string) => {
  if (status === "ok" || status === "completed") return "success";
  if (status === "running") return "info";
  if (status === "error") return "error";
  return "default";
};

function formatDurationMs(ms: number | null): string {
  if (ms === null || ms === undefined) return "...";
  if (ms < 1) return "<1ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

interface SpanTreeNode {
  span: TraceSpan;
  children: SpanTreeNode[];
  depth: number;
}

function buildSpanTree(spans: TraceSpan[]): SpanTreeNode[] {
  const byId = new Map<string, SpanTreeNode>();
  const roots: SpanTreeNode[] = [];

  // Sort by start_wall for stable ordering
  const sorted = [...spans].sort((a, b) => a.start_wall - b.start_wall);

  for (const span of sorted) {
    byId.set(span.span_id, { span, children: [], depth: 0 });
  }

  for (const span of sorted) {
    const node = byId.get(span.span_id)!;
    if (span.parent_span_id && byId.has(span.parent_span_id)) {
      const parent = byId.get(span.parent_span_id)!;
      node.depth = parent.depth + 1;
      parent.children.push(node);
    } else {
      roots.push(node);
    }
  }

  return roots;
}

function flattenTree(nodes: SpanTreeNode[]): SpanTreeNode[] {
  const result: SpanTreeNode[] = [];
  function walk(list: SpanTreeNode[]) {
    for (const node of list) {
      result.push(node);
      walk(node.children);
    }
  }
  walk(nodes);
  return result;
}

/* ── Trace list columns ──────────────────────────────────────── */

const traceColumns = [
  {
    key: "trace_id",
    header: "Trace ID",
    className: "font-mono text-xs",
    render: (row: TraceSummary) => row.trace_id.slice(0, 12) + "...",
  },
  {
    key: "agent_id",
    header: "Agent",
    className: "font-mono text-xs",
    render: (row: TraceSummary) => row.agent_id?.slice(0, 16) ?? "—",
  },
  {
    key: "status",
    header: "Status",
    render: (row: TraceSummary) => (
      <Badge variant={statusVariant(row.status)}>
        {row.status || "unknown"}
      </Badge>
    ),
  },
  {
    key: "span_count",
    header: "Spans",
  },
  {
    key: "run_count",
    header: "Runs",
  },
  {
    key: "total_tokens",
    header: "Tokens",
    render: (row: TraceSummary) => formatTokens(row.total_tokens),
  },
  {
    key: "start_time",
    header: "Started",
    render: (row: TraceSummary) => formatTimestamp(row.start_time),
  },
];

/* ── Sub-components ──────────────────────────────────────────── */

function SpanKindBadge({ kind }: { kind: string }) {
  const color = KIND_COLORS[kind] ?? KIND_COLORS.custom;
  const label = KIND_LABELS[kind] ?? kind.toUpperCase().slice(0, 5);
  return (
    <span
      className="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider"
      style={{ backgroundColor: `${color}20`, color }}
    >
      {label}
    </span>
  );
}

function SpanTimingBar({
  span,
  traceStart,
  traceDuration,
}: {
  span: TraceSpan;
  traceStart: number;
  traceDuration: number;
}) {
  if (traceDuration <= 0) return null;

  const offset = ((span.start_wall - traceStart) / traceDuration) * 100;
  const width =
    span.duration_ms !== null
      ? Math.max(0.5, (span.duration_ms / 1000 / traceDuration) * 100)
      : 2; // Running spans get a min bar

  const color = KIND_COLORS[span.kind] ?? KIND_COLORS.custom;

  return (
    <div className="relative h-4 w-full rounded bg-muted/30">
      <div
        className={cn(
          "absolute top-0 h-full rounded",
          span.status === "running" && "animate-pulse"
        )}
        style={{
          left: `${Math.min(offset, 98)}%`,
          width: `${Math.min(width, 100 - offset)}%`,
          backgroundColor: color,
          opacity: span.status === "error" ? 1 : 0.7,
        }}
      />
    </div>
  );
}

function WaterfallRow({
  node,
  traceStart,
  traceDuration,
  isSelected,
  onClick,
}: {
  node: SpanTreeNode;
  traceStart: number;
  traceDuration: number;
  isSelected: boolean;
  onClick: () => void;
}) {
  const { span, depth } = node;
  const indent = depth * 20;

  return (
    <div
      className={cn(
        "flex items-center gap-2 border-b px-3 py-1.5 transition-colors cursor-pointer hover:bg-muted/50",
        isSelected && "bg-primary/10 border-primary/20"
      )}
      onClick={onClick}
    >
      {/* Left: indent + kind + name */}
      <div
        className="flex items-center gap-1.5 shrink-0"
        style={{ paddingLeft: indent, minWidth: 240 }}
      >
        <SpanKindBadge kind={span.kind} />
        <span className="text-xs font-medium truncate max-w-[160px]">
          {span.name}
        </span>
        {span.status === "error" && (
          <span className="text-[10px] text-red-400">ERR</span>
        )}
      </div>

      {/* Middle: timing bar */}
      <div className="flex-1 min-w-[120px]">
        <SpanTimingBar
          span={span}
          traceStart={traceStart}
          traceDuration={traceDuration}
        />
      </div>

      {/* Right: duration + tokens */}
      <div className="flex items-center gap-3 shrink-0 text-xs text-muted-foreground">
        <span className="w-16 text-right font-mono">
          {formatDurationMs(span.duration_ms)}
        </span>
        {(span.input_tokens || span.output_tokens) && (
          <span className="w-20 text-right font-mono text-[10px]">
            {span.input_tokens ?? 0}/{span.output_tokens ?? 0}
          </span>
        )}
      </div>
    </div>
  );
}

function SpanDetail({ span }: { span: TraceSpan }) {
  return (
    <div className="space-y-4 overflow-auto p-4">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2">
          <SpanKindBadge kind={span.kind} />
          <h3 className="text-sm font-semibold">{span.name}</h3>
          <Badge variant={statusVariant(span.status)}>{span.status}</Badge>
        </div>
        <p className="mt-1 text-xs text-muted-foreground font-mono">
          {span.span_id}
        </p>
      </div>

      {/* Timing */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Duration
          </p>
          <p className="text-sm font-mono">
            {formatDurationMs(span.duration_ms)}
          </p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Start Time
          </p>
          <p className="text-sm font-mono">{formatTimestamp(span.start_wall)}</p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Agent
          </p>
          <p className="text-sm font-mono truncate">{span.agent_id}</p>
        </div>
        {span.run_id && (
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
              Run ID
            </p>
            <p className="text-sm font-mono truncate">{span.run_id}</p>
          </div>
        )}
      </div>

      {/* Error */}
      {span.error && (
        <div className="rounded-md border border-red-800 bg-red-950/30 p-3">
          <p className="text-[10px] uppercase tracking-wider text-red-400 mb-1">
            Error
          </p>
          <p className="text-xs text-red-300 font-mono break-all">
            {span.error}
          </p>
        </div>
      )}

      {/* LLM details */}
      {(span.input_tokens !== null || span.output_tokens !== null) && (
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-2">
            Token Usage
          </p>
          <div className="grid grid-cols-3 gap-2">
            <div className="rounded border p-2 text-center">
              <p className="text-lg font-bold">{formatTokens(span.input_tokens ?? 0)}</p>
              <p className="text-[10px] text-muted-foreground">Input</p>
            </div>
            <div className="rounded border p-2 text-center">
              <p className="text-lg font-bold">{formatTokens(span.output_tokens ?? 0)}</p>
              <p className="text-[10px] text-muted-foreground">Output</p>
            </div>
            {span.cache_read_tokens !== null && (
              <div className="rounded border p-2 text-center">
                <p className="text-lg font-bold">{formatTokens(span.cache_read_tokens)}</p>
                <p className="text-[10px] text-muted-foreground">Cache</p>
              </div>
            )}
          </div>
          {span.model_name && (
            <p className="mt-2 text-xs text-muted-foreground">
              Model: <span className="font-mono">{span.model_name}</span>
            </p>
          )}
        </div>
      )}

      {/* VCM Pages */}
      {span.context_page_ids && span.context_page_ids.length > 0 && (
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
            VCM Pages ({span.context_page_ids.length})
          </p>
          <div className="flex flex-wrap gap-1">
            {span.context_page_ids.map((pid) => (
              <span
                key={pid}
                className="rounded bg-cyan-500/10 px-1.5 py-0.5 text-[10px] font-mono text-cyan-400"
              >
                {pid.slice(0, 12)}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Input/Output summaries */}
      {Object.keys(span.input_summary).length > 0 && (
        <SpanDataSection label="Input" data={span.input_summary} kind={span.kind} />
      )}
      {Object.keys(span.output_summary).length > 0 && (
        <SpanDataSection label="Output" data={span.output_summary} kind={span.kind} />
      )}

      {/* Metadata */}
      {Object.keys(span.metadata).length > 0 && (
        <SpanDataSection label="Metadata" data={span.metadata} />
      )}

      {/* Tags */}
      {span.tags.length > 0 && (
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
            Tags
          </p>
          <div className="flex flex-wrap gap-1">
            {span.tags.map((tag) => (
              <Badge key={tag} variant="default">{tag}</Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function KeyValueRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-start gap-2 py-0.5">
      <span className="text-[10px] uppercase tracking-wider text-muted-foreground shrink-0 w-24">{label}</span>
      <span className={cn("text-xs break-all", mono && "font-mono")}>{value}</span>
    </div>
  );
}

function ActionResultBadge({ result }: { result: Record<string, unknown> }) {
  const success = result.success === true || result.result_type === "ActionResult";
  const resultStr = typeof result.result === "string" ? result.result : null;
  return (
    <div className={cn(
      "rounded-md border p-2",
      success ? "border-emerald-800/50 bg-emerald-950/20" : "border-red-800/50 bg-red-950/20"
    )}>
      <div className="flex items-center gap-1.5 mb-1">
        <span className={cn("text-sm", success ? "text-emerald-400" : "text-red-400")}>
          {success ? "\u2713" : "\u2717"}
        </span>
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
          {result.result_type ? String(result.result_type) : "Result"}
        </span>
      </div>
      {resultStr && (
        <pre className="text-[11px] font-mono text-muted-foreground whitespace-pre-wrap break-all max-h-40 overflow-auto">
          {tryPrettyPrint(resultStr)}
        </pre>
      )}
    </div>
  );
}

function tryPrettyPrint(s: string): string {
  try {
    const parsed = JSON.parse(s);
    return JSON.stringify(parsed, null, 2);
  } catch {
    return s;
  }
}

function SpanDataSection({
  label,
  data,
  kind,
}: {
  label: string;
  data: Record<string, unknown>;
  kind?: string;
}) {
  const [expanded, setExpanded] = useState(true);

  if (Object.keys(data).length === 0) return null;

  // For action outputs, render with success/failure badge
  if (label === "Output" && kind === "action" && ("result" in data || "result_type" in data)) {
    return (
      <div>
        <button
          className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors mb-1"
          onClick={() => setExpanded(!expanded)}
        >
          <span className="text-xs">{expanded ? "\u25BC" : "\u25B6"}</span>
          {label}
        </button>
        {expanded && <ActionResultBadge result={data} />}
      </div>
    );
  }

  // For action inputs, render key-value pairs
  if (label === "Input" && kind === "action") {
    return (
      <div>
        <button
          className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors mb-1"
          onClick={() => setExpanded(!expanded)}
        >
          <span className="text-xs">{expanded ? "\u25BC" : "\u25B6"}</span>
          {label}
        </button>
        {expanded && (
          <div className="rounded-md border bg-muted/20 p-2 space-y-0.5">
            {Object.entries(data).map(([k, v]) => (
              <KeyValueRow key={k} label={k} value={typeof v === "string" ? v : JSON.stringify(v)} mono />
            ))}
          </div>
        )}
      </div>
    );
  }

  // Default: pretty-printed JSON
  return (
    <div>
      <button
        className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors mb-1"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="text-xs">{expanded ? "\u25BC" : "\u25B6"}</span>
        {label}
      </button>
      {expanded && (
        <pre className="mt-1 rounded bg-muted/50 p-2 text-[11px] font-mono text-muted-foreground overflow-auto max-h-60 whitespace-pre-wrap break-all">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

function TraceHeader({
  spans,
  traceId,
  isStreaming,
}: {
  spans: TraceSpan[];
  traceId: string;
  isStreaming: boolean;
}) {
  const stats = useMemo(() => {
    let totalTokens = 0;
    let errors = 0;
    let minWall = Infinity;
    let maxEnd = 0;
    const agents = new Set<string>();

    for (const s of spans) {
      totalTokens += (s.input_tokens ?? 0) + (s.output_tokens ?? 0);
      if (s.status === "error") errors++;
      if (s.start_wall < minWall) minWall = s.start_wall;
      const end = s.duration_ms !== null ? s.start_wall + s.duration_ms / 1000 : s.start_wall;
      if (end > maxEnd) maxEnd = end;
      agents.add(s.agent_id);
    }

    const durationSec = spans.length > 0 ? maxEnd - minWall : 0;

    return { totalTokens, errors, durationSec, agents: agents.size, spanCount: spans.length };
  }, [spans]);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <h2 className="text-sm font-semibold">
          Trace: <span className="font-mono">{traceId.slice(0, 16)}...</span>
        </h2>
        {isStreaming && (
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <span className="inline-block h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            Live
          </span>
        )}
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <MetricCard label="Spans" value={stats.spanCount} />
        <MetricCard label="Duration" value={formatDurationMs(stats.durationSec * 1000)} />
        <MetricCard label="Tokens" value={formatTokens(stats.totalTokens)} />
        <MetricCard label="Agents" value={stats.agents} />
        <MetricCard
          label="Errors"
          value={stats.errors}
          className={stats.errors > 0 ? "border-red-800/50" : undefined}
        />
      </div>
    </div>
  );
}

/* ── Waterfall view ──────────────────────────────────────────── */

function TraceWaterfallView({
  traceId,
  onBack,
}: {
  traceId: string;
  onBack: () => void;
}) {
  const { data: restSpans } = useTraceSpans(traceId);
  const { spans: streamedSpans, isStreaming } = useTraceStream(traceId);
  const [selectedSpanId, setSelectedSpanId] = useState<string | null>(null);

  // Merge REST + streamed spans (streamed overrides for live updates)
  const allSpans = useMemo(() => {
    const merged = new Map<string, TraceSpan>();
    for (const s of restSpans ?? []) merged.set(s.span_id, s);
    for (const [id, s] of streamedSpans) merged.set(id, s);
    return Array.from(merged.values());
  }, [restSpans, streamedSpans]);

  const tree = useMemo(() => buildSpanTree(allSpans), [allSpans]);
  const flat = useMemo(() => flattenTree(tree), [tree]);

  const selectedSpan = selectedSpanId
    ? allSpans.find((s) => s.span_id === selectedSpanId) ?? null
    : null;

  // Compute trace time window
  const { traceStart, traceDuration } = useMemo(() => {
    if (allSpans.length === 0) return { traceStart: 0, traceDuration: 0 };
    let min = Infinity;
    let max = 0;
    for (const s of allSpans) {
      if (s.start_wall < min) min = s.start_wall;
      const end =
        s.duration_ms !== null ? s.start_wall + s.duration_ms / 1000 : s.start_wall;
      if (end > max) max = end;
    }
    return { traceStart: min, traceDuration: Math.max(max - min, 0.001) };
  }, [allSpans]);

  return (
    <div className="space-y-4">
      {/* Back button */}
      <button
        className="text-xs text-muted-foreground hover:text-foreground transition-colors"
        onClick={onBack}
      >
        &larr; Back to trace list
      </button>

      {/* Header stats */}
      <TraceHeader spans={allSpans} traceId={traceId} isStreaming={isStreaming} />

      {/* Kind legend */}
      <div className="flex flex-wrap gap-3">
        {Object.entries(KIND_LABELS).map(([kind, label]) => (
          <span key={kind} className="flex items-center gap-1 text-[10px] text-muted-foreground">
            <span
              className="inline-block h-2 w-2 rounded-sm"
              style={{ backgroundColor: KIND_COLORS[kind] }}
            />
            {label}
          </span>
        ))}
      </div>

      {/* Waterfall + Detail split */}
      <div className="flex gap-0 rounded-lg border overflow-hidden" style={{ height: "calc(100vh - 360px)" }}>
        {/* Left: waterfall */}
        <div className={cn("overflow-auto", selectedSpan ? "w-3/5" : "w-full")}>
          {flat.length === 0 ? (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              No spans recorded
            </div>
          ) : (
            flat.map((node) => (
              <WaterfallRow
                key={node.span.span_id}
                node={node}
                traceStart={traceStart}
                traceDuration={traceDuration}
                isSelected={node.span.span_id === selectedSpanId}
                onClick={() =>
                  setSelectedSpanId(
                    node.span.span_id === selectedSpanId ? null : node.span.span_id
                  )
                }
              />
            ))
          )}
        </div>

        {/* Right: detail panel */}
        {selectedSpan && (
          <div className="w-2/5 border-l overflow-auto bg-card">
            <SpanDetail span={selectedSpan} />
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Main component ──────────────────────────────────────────── */

export function TracesTab() {
  const { data: traces, isLoading } = useTraces();
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);

  const handleRowClick = useCallback((row: TraceSummary) => {
    setSelectedTraceId(row.trace_id);
  }, []);

  if (selectedTraceId) {
    return (
      <TraceWaterfallView
        traceId={selectedTraceId}
        onBack={() => setSelectedTraceId(null)}
      />
    );
  }

  return (
    <div className="space-y-6">
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Traces
        </h2>
        <DataTable
          columns={traceColumns}
          data={traces ?? []}
          onRowClick={handleRowClick}
          emptyMessage={isLoading ? "Loading traces..." : "No traces recorded yet"}
        />
      </section>
    </div>
  );
}
