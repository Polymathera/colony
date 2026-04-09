import { useState, useMemo } from "react";
import { useCodegenIterations } from "@/api/hooks/useTraceAnalysis";
import { useGraphLayout } from "./useGraphLayout";
import { ControlFlowGraph } from "./ControlFlowGraph";
import { CodeBlock } from "./CodeBlock";
import { Badge } from "@/components/shared/Badge";
import { cn } from "@/lib/utils";
import type { CodegenIteration } from "./types";

function DetailPanel({ iteration }: { iteration: CodegenIteration | null }) {
  if (!iteration) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-muted-foreground p-4">
        Click a node to see full code and result.
      </div>
    );
  }

  const success = iteration.success === true;
  const failed = iteration.success === false;

  return (
    <div className="space-y-3 p-4 overflow-auto h-full">
      <div className="flex items-center gap-2">
        <span className="text-xs font-bold">Step #{iteration.step_index + 1}</span>
        <Badge variant={iteration.mode === "planning" ? "warning" : "info"}>
          {iteration.mode.toUpperCase()}
        </Badge>
        {iteration.success !== null && (
          <Badge variant={success ? "success" : "error"}>
            {success ? "\u2713 Success" : "\u2717 Failed"}
          </Badge>
        )}
      </div>

      {iteration.generated_code && (
        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
            Generated Code
          </p>
          <CodeBlock code={iteration.generated_code} maxHeight="280px" />
        </div>
      )}

      {failed && iteration.error && (
        <div className="rounded-md border border-red-800 bg-red-950/30 p-2">
          <p className="text-[10px] uppercase tracking-wider text-red-400 mb-0.5">Error</p>
          <p className="text-xs text-red-300 font-mono break-all">{iteration.error}</p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-2 text-xs">
        {iteration.duration_ms !== null && (
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Duration</p>
            <p className="font-mono">{Math.round(iteration.duration_ms)}ms</p>
          </div>
        )}
        {(iteration.input_tokens || iteration.output_tokens) && (
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Tokens</p>
            <p className="font-mono">
              {(iteration.input_tokens ?? 0) + (iteration.output_tokens ?? 0)}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export function ControlFlowView({
  traceId,
  agentId,
}: {
  traceId: string;
  agentId: string | null;
}) {
  const { data: iterations, isLoading } = useCodegenIterations(traceId, agentId);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Build graph: each iteration = one node, edges = sequential flow
  const graphInput = useMemo(() => {
    if (!iterations || iterations.length === 0) return null;

    const nodes = iterations.map((it) => ({
      id: `n${it.step_index}`,
      width: 220,
      height: 70,
    }));

    const edges: Array<{ id: string; source: string; target: string }> = [];
    for (let i = 0; i < iterations.length - 1; i++) {
      edges.push({
        id: `e${i}`,
        source: `n${i}`,
        target: `n${i + 1}`,
      });
    }

    return { nodes, edges };
  }, [iterations]);

  // Edge success data for coloring
  const edgeData = useMemo(() => {
    if (!iterations) return [];
    const result: Array<{ source: string; target: string; success: boolean }> = [];
    for (let i = 0; i < iterations.length - 1; i++) {
      result.push({
        source: `n${i}`,
        target: `n${i + 1}`,
        success: iterations[i].success === true,
      });
    }
    return result;
  }, [iterations]);

  const layout = useGraphLayout(graphInput);

  const selectedIteration = selectedNodeId
    ? iterations?.find((it) => `n${it.step_index}` === selectedNodeId) ?? null
    : null;

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
      </div>
    );
  }

  const showDetail = selectedIteration !== null;

  return (
    <div className="space-y-3" style={{ height: "calc(100vh - 360px)" }}>
      <div className="flex items-center gap-3">
        <Badge variant="default">{iterations.length} step{iterations.length !== 1 ? "s" : ""}</Badge>
        <span className="text-[10px] text-muted-foreground">
          Click a node to see full code and execution result
        </span>
      </div>

      <div className="flex gap-0 rounded-lg border overflow-hidden flex-1" style={{ height: "calc(100% - 40px)" }}>
        <div className={cn("h-full bg-card", showDetail ? "w-3/5" : "w-full")}>
          {!layout.ready ? (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              Computing layout...
            </div>
          ) : (
            <ControlFlowGraph
              layoutNodes={layout.nodes}
              layoutEdges={layout.edges}
              graphWidth={layout.width}
              graphHeight={layout.height}
              iterations={iterations}
              edges={edgeData}
              selectedNodeId={selectedNodeId}
              onSelectNode={(id) => setSelectedNodeId(id === selectedNodeId ? null : id)}
            />
          )}
        </div>

        {showDetail && (
          <div className="w-2/5 border-l overflow-auto bg-card">
            <DetailPanel iteration={selectedIteration} />
          </div>
        )}
      </div>
    </div>
  );
}
