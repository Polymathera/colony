import { Badge } from "@/components/shared/Badge";
import { CodeBlock } from "./CodeBlock";
import type { FSMState, FSMTransition, CodegenIteration } from "./types";

export function FSMDetailPanel({
  selectedState,
  selectedTransition,
  iterations,
}: {
  selectedState: FSMState | null;
  selectedTransition: FSMTransition | null;
  iterations: CodegenIteration[];
}) {
  if (!selectedState && !selectedTransition) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-muted-foreground p-4">
        Click a state or transition to see details.
      </div>
    );
  }

  if (selectedState) {
    const firstIter = iterations[selectedState.step_indices[0]];
    return (
      <div className="space-y-3 p-4 overflow-auto h-full">
        <div className="flex items-center gap-2">
          <Badge variant="info">{selectedState.state_id.toUpperCase()}</Badge>
          <span className="text-xs font-semibold">{selectedState.label}</span>
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Visits</p>
            <p className="font-mono">{selectedState.visit_count}</p>
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">Steps</p>
            <p className="font-mono">
              {selectedState.step_indices.map((i) => `#${i + 1}`).join(", ")}
            </p>
          </div>
        </div>

        <div>
          <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
            Fingerprint
          </p>
          <p className="text-[10px] font-mono text-muted-foreground break-all">
            {selectedState.fingerprint}
          </p>
        </div>

        {firstIter?.generated_code && (
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
              First Generated Code (Step #{selectedState.step_indices[0] + 1})
            </p>
            <CodeBlock code={firstIter.generated_code} maxHeight="200px" />
          </div>
        )}
      </div>
    );
  }

  if (selectedTransition) {
    const iter = iterations[selectedTransition.step_index];
    return (
      <div className="space-y-3 p-4 overflow-auto h-full">
        <div className="flex items-center gap-2">
          <Badge variant={selectedTransition.success ? "success" : "error"}>
            {selectedTransition.success ? "\u2713 Success" : "\u2717 Failed"}
          </Badge>
          <span className="text-xs text-muted-foreground">
            Step #{selectedTransition.step_index + 1}:
            {" "}{selectedTransition.from_state} &rarr; {selectedTransition.to_state}
          </span>
        </div>

        {iter?.generated_code && (
          <div>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1">
              Generated Code
            </p>
            <CodeBlock code={iter.generated_code} maxHeight="240px" />
          </div>
        )}

        {iter?.error && (
          <div className="rounded-md border border-red-800 bg-red-950/30 p-2">
            <p className="text-[10px] uppercase tracking-wider text-red-400 mb-0.5">Error</p>
            <p className="text-xs text-red-300 font-mono break-all">{iter.error}</p>
          </div>
        )}
      </div>
    );
  }

  return null;
}
