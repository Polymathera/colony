import { useState, useMemo, useCallback, useEffect, useRef } from "react";
import { useTraces, useTraceSpans, useTraceStream } from "@/api/hooks/useTraces";
import { DataTable } from "../shared/DataTable";
import { Badge } from "../shared/Badge";
import { MetricCard } from "../shared/MetricCard";
import { formatTimestamp, formatTokens, cn } from "@/lib/utils";
import type { TraceSummary, TraceSpan } from "@/api/types";
import type { TraceViewMode } from "./traces/types";
import { TraceViewSelector } from "./traces/TraceViewSelector";
import { AgentSelector } from "./traces/AgentSelector";
import { LinearizedTimelineView } from "./traces/LinearizedTimelineView";
import { PromptDiffView } from "./traces/PromptDiffView";
import { FSMView } from "./traces/FSMView";
import { ControlFlowView } from "./traces/ControlFlowView";

/* ── Constants ───────────────────────────────────────────────── */

const KIND_COLORS: Record<string, string> = {
  run: "#8b5cf6",
  agent: "#7c3aed",
  agent_step: "#3b82f6",
  plan: "#f59e0b",
  action: "#10b981",
  infer: "#ec4899",
  page_request: "#06b6d4",
  blackboard_op: "#f97316",
  child_spawn: "#a855f7",
  event_process: "#64748b",
  capability: "#14b8a6",
  lifecycle: "#dc2626",
  custom: "#6b7280",
};

const KIND_LABELS: Record<string, string> = {
  run: "RUN",
  agent: "AGENT",
  agent_step: "STEP",
  plan: "PLAN",
  action: "ACT",
  infer: "LLM",
  page_request: "PAGE",
  blackboard_op: "BB",
  child_spawn: "CHILD",
  event_process: "EVT",
  capability: "CAP",
  lifecycle: "LIFE",
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

function flattenTree(
  nodes: SpanTreeNode[],
  collapsed: Set<string>
): SpanTreeNode[] {
  const result: SpanTreeNode[] = [];
  function walk(list: SpanTreeNode[]) {
    for (const node of list) {
      result.push(node);
      if (!collapsed.has(node.span.span_id)) {
        walk(node.children);
      }
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
      : 1.5; // Running/incomplete spans get a thin marker

  const color = KIND_COLORS[span.kind] ?? KIND_COLORS.custom;
  const isError = span.status === "error";
  const isRunning = span.status === "running";
  const isIncomplete = span.duration_ms === null && !isRunning;

  return (
    <div className="relative h-4 w-full rounded bg-muted/30">
      <div
        className={cn(
          "absolute top-0 h-full rounded",
          isRunning && "animate-pulse",
          isError && "border border-red-500",
          isIncomplete && "border border-dashed border-yellow-500/60"
        )}
        style={{
          left: `${Math.min(offset, 99)}%`,
          width: `${Math.max(0.3, Math.min(width, 100 - offset))}%`,
          backgroundColor: isError ? `${color}` : color,
          opacity: isError ? 1 : isIncomplete ? 0.4 : 0.7,
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
  isCollapsed,
  onClick,
  onToggle,
}: {
  node: SpanTreeNode;
  traceStart: number;
  traceDuration: number;
  isSelected: boolean;
  isCollapsed: boolean;
  onClick: () => void;
  onToggle: () => void;
}) {
  const { span, depth, children } = node;
  const indent = depth * 16;
  const barIndent = depth * 12; // Timing bar nesting indent
  const hasChildren = children.length > 0;

  return (
    <div
      data-span-id={span.span_id}
      className={cn(
        "flex items-center border-b px-2 py-1 transition-colors cursor-pointer hover:bg-muted/50",
        isSelected && "bg-primary/10 border-primary/20"
      )}
      onClick={onClick}
    >
      {/* Left: fixed-width name + duration column */}
      <div className="flex items-center gap-1 shrink-0 w-[280px]">
        <div
          className="flex items-center gap-1 flex-1 overflow-hidden"
          style={{ paddingLeft: indent }}
        >
          {/* Collapse/expand toggle */}
          <button
            className={cn(
              "w-4 h-4 flex items-center justify-center text-[10px] text-muted-foreground shrink-0",
              hasChildren && "hover:text-foreground"
            )}
            onClick={(e) => {
              if (hasChildren) {
                e.stopPropagation();
                onToggle();
              }
            }}
          >
            {hasChildren ? (isCollapsed ? "\u25B6" : "\u25BC") : ""}
          </button>
          <SpanKindBadge kind={span.kind} />
          {span.kind === "action" && span.status !== "running" && (
            <span className={cn(
              "text-[10px] shrink-0 font-bold",
              span.output_summary?.success === true ? "text-emerald-400" : "text-red-400"
            )}>
              {span.output_summary?.success === true ? "\u2713" : "\u2717"}
            </span>
          )}
          {span.status === "error" && span.kind !== "action" && (
            <span className="text-[10px] text-red-400 shrink-0">ERR</span>
          )}
          <span className="text-xs font-medium truncate">
            {span.name}
          </span>
        </div>
        <span className="shrink-0 w-14 text-right font-mono text-xs text-muted-foreground">
          {formatDurationMs(span.duration_ms)}
        </span>
      </div>

      {/* Right: timing bar — width driven by zoom, scrolls with parent */}
      <div className="flex-1 min-w-0" style={{ paddingLeft: barIndent }}>
        <SpanTimingBar
          span={span}
          traceStart={traceStart}
          traceDuration={traceDuration}
        />
      </div>
    </div>
  );
}

function TextModal({
  title,
  text,
  onClose,
}: {
  title: string;
  text: string;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="relative w-[80vw] max-w-4xl max-h-[80vh] rounded-lg border bg-card shadow-xl flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b px-4 py-3">
          <h3 className="text-sm font-semibold">{title}</h3>
          <button
            className="text-xs text-muted-foreground hover:text-foreground"
            onClick={onClose}
          >
            Close
          </button>
        </div>
        <pre className="flex-1 overflow-auto p-4 text-xs font-mono text-foreground/90 whitespace-pre-wrap break-words">
          {text}
        </pre>
      </div>
    </div>
  );
}

function SpanDetail({
  span,
  onNavigateToTimeline,
}: {
  span: TraceSpan;
  /** Switch to timeline view, scrolling to the iteration that contains this span. */
  onNavigateToTimeline?: (spanId: string) => void;
}) {
  const [modalContent, setModalContent] = useState<{ title: string; text: string } | null>(null);

  const inferPrompt = span.kind === "infer" ? (span.input_summary?.prompt as string | undefined) : undefined;
  const inferResponse = span.kind === "infer" ? (span.output_summary?.response as string | undefined) : undefined;

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

      {/* Cross-navigation to timeline view */}
      {onNavigateToTimeline &&
        (span.kind === "action" || span.kind === "infer" || span.kind === "agent_step") && (
        <button
          className="w-full rounded border border-blue-800/40 bg-blue-950/20 px-3 py-2 text-xs font-medium text-blue-300 hover:bg-blue-950/40 transition-colors"
          onClick={() => onNavigateToTimeline(span.span_id)}
        >
          Show in timeline view
        </button>
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

      {/* LLM Prompt/Response buttons */}
      {(inferPrompt || inferResponse) && (
        <div className="flex gap-2">
          {inferPrompt && (
            <button
              className="flex-1 rounded border border-pink-800/40 bg-pink-950/20 px-3 py-2 text-xs font-medium text-pink-300 hover:bg-pink-950/40 transition-colors"
              onClick={() => setModalContent({ title: "LLM Prompt", text: inferPrompt })}
            >
              Prompt ({(inferPrompt.length / 1000).toFixed(1)}k chars)
            </button>
          )}
          {inferResponse && (
            <button
              className="flex-1 rounded border border-pink-800/40 bg-pink-950/20 px-3 py-2 text-xs font-medium text-pink-300 hover:bg-pink-950/40 transition-colors"
              onClick={() => setModalContent({ title: "LLM Response", text: inferResponse })}
            >
              Response ({(inferResponse.length / 1000).toFixed(1)}k chars)
            </button>
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

      {/* Text modal for Prompt/Response */}
      {modalContent && (
        <TextModal
          title={modalContent.title}
          text={modalContent.text}
          onClose={() => setModalContent(null)}
        />
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

function ActionOutputPanel({ data }: { data: Record<string, unknown> }) {
  const success = data.success === true;
  const error = data.error as string | undefined;
  const output = data.output;
  const metrics = data.metrics as Record<string, unknown> | undefined;

  return (
    <div className="space-y-2">
      {/* Success/failure header */}
      <div className={cn(
        "rounded-md border px-3 py-2 flex items-center gap-2",
        success ? "border-emerald-800/50 bg-emerald-950/20" : "border-red-800/50 bg-red-950/20"
      )}>
        <span className={cn("text-sm", success ? "text-emerald-400" : "text-red-400")}>
          {success ? "\u2713" : "\u2717"}
        </span>
        <span className="text-xs font-medium">
          {success ? "Success" : "Failed"}
        </span>
        {error && (
          <span className="text-xs text-red-400 ml-2 font-mono">{error}</span>
        )}
      </div>

      {/* Output data */}
      {output != null && (
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Output</p>
          <pre className="rounded bg-muted/50 p-2 text-[11px] font-mono text-muted-foreground whitespace-pre-wrap break-all max-h-48 overflow-auto">
            {typeof output === "string" ? tryPrettyPrint(output) : JSON.stringify(output, null, 2)}
          </pre>
        </div>
      )}

      {/* Metrics */}
      {metrics && Object.keys(metrics).length > 0 && (
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">Metrics</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(metrics).map(([k, v]) => (
              <span key={k} className="rounded bg-muted/30 px-2 py-0.5 text-[10px] font-mono">
                {k}: {String(v)}
              </span>
            ))}
          </div>
        </div>
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

  // For action outputs, render structured panel with success/failure
  if (label === "Output" && kind === "action" && ("success" in data || "output" in data)) {
    return (
      <div>
        <button
          className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors mb-1"
          onClick={() => setExpanded(!expanded)}
        >
          <span className="text-xs">{expanded ? "\u25BC" : "\u25B6"}</span>
          {label}
        </button>
        {expanded && <ActionOutputPanel data={data} />}
      </div>
    );
  }

  // For action inputs, render structured fields
  if (label === "Input" && kind === "action") {
    const actionType = data.action_type as string | undefined;
    const parameters = data.parameters as Record<string, unknown> | undefined;
    const reasoning = data.reasoning as string | undefined;
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
          <div className="rounded-md border bg-muted/20 p-2 space-y-2">
            {actionType && <KeyValueRow label="Action" value={actionType} mono />}
            {reasoning && (
              <div>
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-0.5">Reasoning</p>
                <p className="text-xs text-foreground/80 italic">{reasoning}</p>
              </div>
            )}
            {parameters && Object.keys(parameters).length > 0 && (
              <div>
                <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-0.5">Parameters</p>
                <pre className="rounded bg-muted/50 p-1.5 text-[11px] font-mono text-muted-foreground whitespace-pre-wrap break-all max-h-32 overflow-auto">
                  {JSON.stringify(parameters, null, 2)}
                </pre>
              </div>
            )}
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
  const [collapsed, setCollapsed] = useState<Set<string>>(() => new Set());
  const [zoom, setZoom] = useState(1);
  const [viewMode, setViewMode] = useState<TraceViewMode>("tree");
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);

  // Pending scroll target — set by cross-navigation, consumed by useEffect after render.
  const [pendingScrollSpanId, setPendingScrollSpanId] = useState<string | null>(null);
  // Target action_span_id to scroll to in timeline view after switching.
  const [pendingTimelineSpanId, setPendingTimelineSpanId] = useState<string | null>(null);
  const waterfallRef = useRef<HTMLDivElement>(null);

  const toggleCollapsed = useCallback((spanId: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(spanId)) next.delete(spanId);
      else next.add(spanId);
      return next;
    });
  }, []);

  // Ctrl+wheel zoom on the timeline (horizontal only).
  // Must use a non-passive listener to preventDefault() the browser's
  // native Ctrl+Wheel page zoom.  React's onWheel is passive and can't
  // prevent it.
  useEffect(() => {
    const el = waterfallRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      setZoom((prev) => {
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        return Math.min(50, Math.max(0.1, +(prev * factor).toFixed(2)));
      });
    };
    el.addEventListener("wheel", handler, { passive: false });
    return () => el.removeEventListener("wheel", handler);
  }, [viewMode]); // re-attach when switching to/from tree view

  // Merge REST + streamed spans (streamed overrides for live updates)
  const allSpans = useMemo(() => {
    const merged = new Map<string, TraceSpan>();
    for (const s of restSpans ?? []) merged.set(s.span_id, s);
    for (const [id, s] of streamedSpans) merged.set(id, s);
    return Array.from(merged.values());
  }, [restSpans, streamedSpans]);

  const tree = useMemo(() => buildSpanTree(allSpans), [allSpans]);
  const flat = useMemo(() => flattenTree(tree, collapsed), [tree, collapsed]);

  const selectedSpan = selectedSpanId
    ? allSpans.find((s) => s.span_id === selectedSpanId) ?? null
    : null;

  // --- Cross-view navigation ---

  /** Timeline → Tree: switch to tree view, expand ancestors, select + scroll to the span. */
  const navigateToSpanInTree = useCallback((spanId: string) => {
    const parentOf = new Map<string, string>();
    for (const s of allSpans) {
      if (s.parent_span_id) parentOf.set(s.span_id, s.parent_span_id);
    }
    setCollapsed((prev) => {
      const next = new Set(prev);
      let cur = parentOf.get(spanId);
      while (cur) {
        next.delete(cur);
        cur = parentOf.get(cur);
      }
      return next;
    });
    setSelectedSpanId(spanId);
    setPendingScrollSpanId(spanId);
    setViewMode("tree");
  }, [allSpans]);

  /** Tree → Timeline: switch to timeline view, scroll to the iteration. */
  const navigateToTimeline = useCallback((spanId: string) => {
    setPendingTimelineSpanId(spanId);
    setViewMode("timeline");
  }, []);

  // After the tree view renders with the new selection, scroll to it.
  useEffect(() => {
    if (pendingScrollSpanId && viewMode === "tree" && waterfallRef.current) {
      const el = waterfallRef.current.querySelector(
        `[data-span-id="${pendingScrollSpanId}"]`
      );
      if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
      setPendingScrollSpanId(null);
    }
  }, [pendingScrollSpanId, viewMode, flat]); // flat changes when collapse state updates

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

      {/* View selector + agent filter */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <TraceViewSelector value={viewMode} onChange={setViewMode} />
          {viewMode !== "tree" && (
            <AgentSelector
              spans={allSpans}
              value={selectedAgentId}
              onChange={setSelectedAgentId}
            />
          )}
        </div>
        {viewMode === "tree" && (
          <span className="text-[10px] text-muted-foreground shrink-0">
            Ctrl+Scroll to zoom timeline ({zoom.toFixed(1)}x)
          </span>
        )}
      </div>

      {/* Kind legend (tree view only) */}
      {viewMode === "tree" && (
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
      )}

      {/* View content */}
      {viewMode === "tree" && (
        <div className="flex gap-0 rounded-lg border overflow-hidden" style={{ height: "calc(100vh - 360px)" }}>
          {/* Left: waterfall */}
          <div ref={waterfallRef} className={cn("h-full overflow-auto", selectedSpan ? "w-3/5" : "w-full")}>
            {flat.length === 0 ? (
              <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                No spans recorded
              </div>
            ) : (
              <div style={{ minWidth: `calc(280px + ${100 * zoom}%)` }}>
                {flat.map((node) => (
                  <WaterfallRow
                    key={node.span.span_id}
                    node={node}
                    traceStart={traceStart}
                    traceDuration={traceDuration}
                    isSelected={node.span.span_id === selectedSpanId}
                    isCollapsed={collapsed.has(node.span.span_id)}
                    onClick={() =>
                      setSelectedSpanId(
                        node.span.span_id === selectedSpanId ? null : node.span.span_id
                      )
                    }
                    onToggle={() => toggleCollapsed(node.span.span_id)}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Right: detail panel */}
          {selectedSpan && (
            <div className="w-2/5 border-l overflow-auto bg-card">
              <SpanDetail span={selectedSpan} onNavigateToTimeline={navigateToTimeline} />
            </div>
          )}
        </div>
      )}

      {viewMode === "timeline" && (
        <LinearizedTimelineView
          traceId={traceId}
          agentId={selectedAgentId}
          onNavigateToSpan={navigateToSpanInTree}
          scrollToSpanId={pendingTimelineSpanId}
          onScrollComplete={() => setPendingTimelineSpanId(null)}
        />
      )}

      {viewMode === "diff" && (
        <PromptDiffView traceId={traceId} agentId={selectedAgentId} />
      )}

      {viewMode === "fsm" && (
        <FSMView traceId={traceId} agentId={selectedAgentId} />
      )}

      {viewMode === "flow" && (
        <ControlFlowView traceId={traceId} agentId={selectedAgentId} />
      )}
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
