import { useMemo } from "react";
import { useCodegenIterations } from "@/api/hooks/useTraceAnalysis";
import { Badge } from "@/components/shared/Badge";
import { MetricCard } from "@/components/shared/MetricCard";
import { CodeBlock } from "./CodeBlock";
import type { RunCallTraceEntry } from "./types";
import { cn, formatTokens } from "@/lib/utils";
import type { CodegenIteration } from "./types";

/* ── Helpers ─────────────────────────────────────────────────── */

function formatDuration(ms: number | null): string {
  if (ms === null) return "...";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

function formatTime(wall: number | null): string {
  if (wall === null) return "—";
  return new Date(wall * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

/** Detect stuck patterns: 3+ consecutive failures or identical code. */
function detectStuckRanges(iterations: CodegenIteration[]): Set<number> {
  const stuck = new Set<number>();

  // Consecutive failures
  let failStreak = 0;
  for (let i = 0; i < iterations.length; i++) {
    if (iterations[i].success === false) {
      failStreak++;
      if (failStreak >= 3) {
        // Mark all steps in this streak
        for (let j = i - failStreak + 1; j <= i; j++) stuck.add(j);
      }
    } else {
      failStreak = 0;
    }
  }

  // Identical consecutive generated code
  for (let i = 1; i < iterations.length; i++) {
    if (
      iterations[i].generated_code === iterations[i - 1].generated_code &&
      iterations[i].generated_code.length > 0
    ) {
      stuck.add(i - 1);
      stuck.add(i);
    }
  }

  return stuck;
}

/* ── Sub-components ─────────────────────────────────────────── */

function SummaryStats({ iterations }: { iterations: CodegenIteration[] }) {
  const stats = useMemo(() => {
    let totalTokens = 0;
    let totalDurationMs = 0;
    let successes = 0;

    for (const it of iterations) {
      totalTokens += (it.input_tokens ?? 0) + (it.output_tokens ?? 0);
      totalDurationMs += it.duration_ms ?? 0;
      if (it.success === true) successes++;
    }

    const successRate =
      iterations.length > 0
        ? `${Math.round((successes / iterations.length) * 100)}%`
        : "—";

    return { totalTokens, totalDurationMs, successes, successRate };
  }, [iterations]);

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <MetricCard label="Iterations" value={iterations.length} />
      <MetricCard label="Success Rate" value={stats.successRate} />
      <MetricCard label="Tokens" value={formatTokens(stats.totalTokens)} />
      <MetricCard label="Duration" value={formatDuration(stats.totalDurationMs)} />
    </div>
  );
}

function IterationCard({
  iteration,
  isStuck,
}: {
  iteration: CodegenIteration;
  isStuck: boolean;
}) {
  const success = iteration.success === true;
  const failed = iteration.success === false;

  // Build line annotations from per-run() call trace
  const lineAnnotations = useMemo(() => {
    if (!iteration.run_call_trace) return undefined;
    const map = new Map<number, RunCallTraceEntry>();
    for (const entry of iteration.run_call_trace) {
      if (entry.line_number !== null) {
        map.set(entry.line_number, entry);
      }
    }
    return map.size > 0 ? map : undefined;
  }, [iteration.run_call_trace]);

  return (
    <div
      className={cn(
        "rounded-lg border bg-card overflow-hidden",
        isStuck && "ring-1 ring-amber-500/40",
        failed && !isStuck && "border-red-800/40",
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b bg-muted/20">
        <span className="text-xs font-bold text-muted-foreground w-8">
          #{iteration.step_index + 1}
        </span>

        <Badge
          variant={
            iteration.mode === "planning" ? "warning" : "info"
          }
        >
          {iteration.mode.toUpperCase()}
        </Badge>

        {iteration.success !== null ? (
          <Badge variant={success ? "success" : "error"}>
            {success ? "\u2713 Success" : "\u2717 Failed"}
          </Badge>
        ) : iteration.action_span_id === null ? (
          <Badge variant="warning">NOT EXECUTED</Badge>
        ) : null}

        {isStuck && (
          <Badge variant="warning">STUCK</Badge>
        )}

        <div className="flex-1" />

        <span className="text-[10px] font-mono text-muted-foreground">
          {formatTime(iteration.start_wall)}
        </span>
        <span className="text-[10px] font-mono text-muted-foreground">
          {formatDuration(iteration.duration_ms)}
        </span>
        {(iteration.input_tokens || iteration.output_tokens) && (
          <span className="text-[10px] font-mono text-muted-foreground">
            {(iteration.input_tokens ?? 0) + (iteration.output_tokens ?? 0)} tok
          </span>
        )}
      </div>

      {/* Generated code */}
      {iteration.generated_code && (
        <div className="px-3 py-2">
          <CodeBlock code={iteration.generated_code} maxHeight="240px" lineAnnotations={lineAnnotations} />
        </div>
      )}

      {/* Error details */}
      {failed && iteration.error && (
        <div className="mx-3 mb-3 rounded-md border border-red-800 bg-red-950/30 p-2">
          <p className="text-[10px] uppercase tracking-wider text-red-400 mb-0.5">
            Error
          </p>
          <p className="text-xs text-red-300 font-mono break-all">
            {iteration.error}
          </p>
        </div>
      )}
    </div>
  );
}

/* ── Main component ─────────────────────────────────────────── */

export function LinearizedTimelineView({
  traceId,
  agentId,
}: {
  traceId: string;
  agentId: string | null;
}) {
  const { data: iterations, isLoading } = useCodegenIterations(traceId, agentId);

  const stuckSteps = useMemo(
    () => detectStuckRanges(iterations ?? []),
    [iterations],
  );

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        Loading iterations...
      </div>
    );
  }

  if (!iterations || iterations.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        No codegen iterations found for this agent.
        {!agentId && " Select an agent to view iterations."}
      </div>
    );
  }

  return (
    <div className="space-y-4 overflow-auto" style={{ height: "calc(100vh - 360px)" }}>
      <SummaryStats iterations={iterations} />

      {stuckSteps.size > 0 && (
        <div className="rounded-md border border-amber-800/50 bg-amber-950/20 px-3 py-2 flex items-center gap-2">
          <span className="text-amber-400 text-sm font-bold">!</span>
          <span className="text-xs text-amber-300">
            Stuck pattern detected: {stuckSteps.size} iterations show consecutive failures
            or identical generated code.
          </span>
        </div>
      )}

      <div className="space-y-3">
        {iterations.map((it) => (
          <IterationCard
            key={it.step_index}
            iteration={it}
            isStuck={stuckSteps.has(it.step_index)}
          />
        ))}
      </div>
    </div>
  );
}
