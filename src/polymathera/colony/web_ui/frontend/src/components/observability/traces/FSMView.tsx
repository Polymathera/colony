import { useState, useMemo } from "react";
import { useCodegenIterations } from "@/api/hooks/useTraceAnalysis";
import { useFSMGraph } from "@/api/hooks/useTraceAnalysis";
import { useGraphLayout } from "./useGraphLayout";
import { FSMGraph } from "./FSMGraph";
import { FSMDetailPanel } from "./FSMDetailPanel";
import { Badge } from "@/components/shared/Badge";
import { cn } from "@/lib/utils";

export function FSMView({
  traceId,
  agentId,
}: {
  traceId: string;
  agentId: string | null;
}) {
  const { data: fsmData, isLoading: fsmLoading } = useFSMGraph(traceId, agentId);
  const { data: iterations } = useCodegenIterations(traceId, agentId);
  const [selectedStateId, setSelectedStateId] = useState<string | null>(null);
  const [selectedStepIndex, setSelectedStepIndex] = useState<number | null>(null);

  // Prepare ELK input from FSM data
  const elkInput = useMemo(() => {
    if (!fsmData || fsmData.states.length === 0) return null;
    return {
      nodes: fsmData.states.map((s) => ({
        id: s.state_id,
        width: 180,
        height: s.visit_count > 1 ? 65 : 50,
      })),
      edges: fsmData.transitions.map((t, i) => ({
        id: `e${i}`,
        source: t.from_state,
        target: t.to_state,
      })),
    };
  }, [fsmData]);

  const layout = useGraphLayout(elkInput);

  const selectedState = selectedStateId
    ? fsmData?.states.find((s) => s.state_id === selectedStateId) ?? null
    : null;
  const selectedTransition = selectedStepIndex !== null
    ? fsmData?.transitions.find((t) => t.step_index === selectedStepIndex) ?? null
    : null;

  if (fsmLoading) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        Loading FSM graph...
      </div>
    );
  }

  if (!fsmData || fsmData.states.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        No codegen iterations found for this agent.
      </div>
    );
  }

  const hasLoops = fsmData.loops.length > 0;
  const showDetail = selectedState !== null || selectedTransition !== null;

  return (
    <div className="space-y-3" style={{ height: "calc(100vh - 360px)" }}>
      {/* Summary */}
      <div className="flex items-center gap-3">
        <Badge variant="default">
          {fsmData.states.length} state{fsmData.states.length !== 1 ? "s" : ""}
        </Badge>
        <Badge variant="default">
          {fsmData.transitions.length} transition{fsmData.transitions.length !== 1 ? "s" : ""}
        </Badge>
        {hasLoops && (
          <Badge variant="warning">
            {fsmData.loops.length} loop{fsmData.loops.length !== 1 ? "s" : ""} detected
          </Badge>
        )}
      </div>

      {/* Graph + Detail split */}
      <div className="flex gap-0 rounded-lg border overflow-hidden flex-1" style={{ height: "calc(100% - 40px)" }}>
        <div className={cn("h-full bg-card", showDetail ? "w-3/5" : "w-full")}>
          {!layout.ready ? (
            <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
              Computing layout...
            </div>
          ) : (
            <FSMGraph
              layoutNodes={layout.nodes}
              layoutEdges={layout.edges}
              graphWidth={layout.width}
              graphHeight={layout.height}
              states={fsmData.states}
              transitions={fsmData.transitions}
              loops={fsmData.loops}
              selectedId={selectedStateId}
              onSelectState={(id) => {
                setSelectedStateId(id === selectedStateId ? null : id);
                setSelectedStepIndex(null);
              }}
              onSelectTransition={(idx) => {
                setSelectedStepIndex(idx === selectedStepIndex ? null : idx);
                setSelectedStateId(null);
              }}
            />
          )}
        </div>

        {showDetail && (
          <div className="w-2/5 border-l overflow-auto bg-card">
            <FSMDetailPanel
              selectedState={selectedState}
              selectedTransition={selectedTransition}
              iterations={iterations ?? []}
            />
          </div>
        )}
      </div>
    </div>
  );
}
