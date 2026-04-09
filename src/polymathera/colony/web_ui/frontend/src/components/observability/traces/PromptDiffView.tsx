import { useState } from "react";
import { useCodegenIterations, usePromptDiff } from "@/api/hooks/useTraceAnalysis";
import { Badge } from "@/components/shared/Badge";
import { DiffRenderer } from "./DiffRenderer";
import { cn } from "@/lib/utils";

export function PromptDiffView({
  traceId,
  agentId,
}: {
  traceId: string;
  agentId: string | null;
}) {
  const { data: iterations, isLoading: itersLoading } = useCodegenIterations(traceId, agentId);
  const [currentStep, setCurrentStep] = useState(0);

  const stepA = currentStep;
  const stepB = currentStep + 1;
  const maxStep = (iterations?.length ?? 1) - 2; // last valid step_a

  const { data: diff, isLoading: diffLoading } = usePromptDiff(
    traceId,
    agentId,
    stepA,
    stepB,
  );

  if (itersLoading) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        Loading iterations...
      </div>
    );
  }

  if (!iterations || iterations.length < 2) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        {!iterations || iterations.length === 0
          ? "No codegen iterations found for this agent."
          : "Need at least 2 iterations to compute diffs."}
      </div>
    );
  }

  const iterA = iterations[stepA];
  const iterB = iterations[stepB];

  return (
    <div className="space-y-3" style={{ height: "calc(100vh - 360px)" }}>
      {/* Navigation */}
      <div className="flex items-center gap-3">
        <button
          className={cn(
            "rounded border px-2 py-1 text-xs font-medium transition-colors",
            currentStep <= 0
              ? "border-border text-muted-foreground/40 cursor-not-allowed"
              : "border-border text-muted-foreground hover:text-foreground hover:border-primary/40",
          )}
          onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
          disabled={currentStep <= 0}
        >
          &larr; Prev
        </button>

        <span className="text-xs text-muted-foreground">
          Step{" "}
          <span className="font-mono font-bold text-foreground">{stepA + 1}</span>
          {" \u2192 "}
          <span className="font-mono font-bold text-foreground">{stepB + 1}</span>
          {" of "}
          <span className="font-mono">{iterations.length}</span>
        </span>

        <button
          className={cn(
            "rounded border px-2 py-1 text-xs font-medium transition-colors",
            currentStep >= maxStep
              ? "border-border text-muted-foreground/40 cursor-not-allowed"
              : "border-border text-muted-foreground hover:text-foreground hover:border-primary/40",
          )}
          onClick={() => setCurrentStep(Math.min(maxStep, currentStep + 1))}
          disabled={currentStep >= maxStep}
        >
          Next &rarr;
        </button>

        {/* Step badges */}
        <div className="flex items-center gap-2 ml-4">
          <Badge variant={iterA?.success ? "success" : iterA?.success === false ? "error" : "default"}>
            #{stepA + 1} {iterA?.mode?.toUpperCase()}
          </Badge>
          <span className="text-muted-foreground">&rarr;</span>
          <Badge variant={iterB?.success ? "success" : iterB?.success === false ? "error" : "default"}>
            #{stepB + 1} {iterB?.mode?.toUpperCase()}
          </Badge>
        </div>
      </div>

      {/* Identical prompt warning */}
      {diff?.is_identical && (
        <div className="rounded-md border border-amber-800/50 bg-amber-950/20 px-3 py-2 flex items-center gap-2">
          <span className="text-amber-400 text-sm font-bold">!</span>
          <span className="text-xs text-amber-300">
            Prompts are <strong>identical</strong> between steps {stepA + 1} and {stepB + 1}.
            The agent may be stuck in a loop.
          </span>
        </div>
      )}

      {/* Diff content */}
      <div className="overflow-auto flex-1" style={{ maxHeight: "calc(100vh - 480px)" }}>
        {diffLoading ? (
          <div className="flex h-32 items-center justify-center text-sm text-muted-foreground">
            Computing diff...
          </div>
        ) : diff?.error ? (
          <div className="flex h-32 items-center justify-center text-sm text-red-400">
            {diff.error}
          </div>
        ) : diff ? (
          <DiffRenderer
            sections={diff.sections}
            totalAdded={diff.total_added}
            totalRemoved={diff.total_removed}
          />
        ) : null}
      </div>
    </div>
  );
}
