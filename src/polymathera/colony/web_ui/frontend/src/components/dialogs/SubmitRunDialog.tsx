import { useState } from "react";
import { useSubmitJob } from "@/api/hooks/useJobs";
import type { AnalysisSpec } from "@/api/types";

interface SubmitRunDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmitted?: (jobId: string, sessionId: string) => void;
}

const ANALYSIS_TYPES = [
  { id: "impact", label: "Change Impact", description: "Trace ripple effects of code changes" },
  { id: "compliance", label: "Compliance", description: "License, security, and regulatory checks" },
  { id: "intent", label: "Intent Inference", description: "Understand code purpose and business logic" },
  { id: "contracts", label: "Contract Inference", description: "Derive function pre/postconditions" },
  { id: "slicing", label: "Program Slicing", description: "Extract minimal code affecting a target" },
  { id: "basic", label: "Basic Analysis", description: "General code structure analysis" },
];

export function SubmitRunDialog({ open, onClose, onSubmitted }: SubmitRunDialogProps) {
  const submitJob = useSubmitJob();
  const [originUrl, setOriginUrl] = useState("");
  const [branch, setBranch] = useState("main");
  const [selectedAnalyses, setSelectedAnalyses] = useState<Set<string>>(new Set(["basic"]));
  const [maxAgents, setMaxAgents] = useState(10);
  const [timeoutSeconds, setTimeoutSeconds] = useState(600);
  const [budgetUsd, setBudgetUsd] = useState<number | null>(null);

  if (!open) return null;

  const toggleAnalysis = (id: string) => {
    setSelectedAnalyses((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleSubmit = async () => {
    if (!originUrl.trim() || selectedAnalyses.size === 0) return;

    const analyses: AnalysisSpec[] = Array.from(selectedAnalyses).map((type) => ({
      type,
      max_agents: maxAgents,
    }));

    const result = await submitJob.mutateAsync({
      origin_url: originUrl.trim(),
      branch,
      analyses,
      timeout_seconds: timeoutSeconds,
      budget_usd: budgetUsd,
    });

    if (result.status === "submitted" && onSubmitted) {
      onSubmitted(result.job_id, result.session_id);
    }
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-lg rounded-lg border border-border bg-card shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-5 py-3">
          <h2 className="text-sm font-semibold">Start Analysis Run</h2>
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground text-lg leading-none"
          >
            &times;
          </button>
        </div>

        {/* Body */}
        <div className="space-y-4 px-5 py-4">
          {/* Repository */}
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1">
              Repository URL
            </label>
            <input
              type="text"
              value={originUrl}
              onChange={(e) => setOriginUrl(e.target.value)}
              placeholder="https://github.com/org/repo or file:///mnt/shared/codebase"
              className="w-full rounded border border-border bg-background px-3 py-1.5 text-xs font-mono focus:border-primary focus:outline-none"
            />
            <div className="grid grid-cols-2 gap-3 mt-2">
              <div>
                <label className="block text-xs font-medium text-muted-foreground mb-1">
                  Branch
                </label>
                <input
                  type="text"
                  value={branch}
                  onChange={(e) => setBranch(e.target.value)}
                  className="w-full rounded border border-border bg-background px-2 py-1.5 text-xs font-mono focus:border-primary focus:outline-none"
                />
              </div>
            </div>
          </div>

          {/* Analysis types */}
          <div>
            <label className="block text-xs font-medium text-muted-foreground mb-1.5">
              Analyses
            </label>
            <div className="grid grid-cols-2 gap-1.5">
              {ANALYSIS_TYPES.map((at) => (
                <button
                  key={at.id}
                  onClick={() => toggleAnalysis(at.id)}
                  className={`rounded border px-3 py-2 text-left transition-colors ${
                    selectedAnalyses.has(at.id)
                      ? "border-primary bg-primary/10 text-primary"
                      : "border-border text-foreground hover:border-primary/50"
                  }`}
                >
                  <div className="text-xs font-medium">{at.label}</div>
                  <div className="text-[10px] text-muted-foreground mt-0.5">{at.description}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Configuration */}
          <div className="grid grid-cols-3 gap-3">
            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1">
                Max Agents
              </label>
              <input
                type="number"
                value={maxAgents}
                onChange={(e) => setMaxAgents(Number(e.target.value))}
                min={1}
                max={50}
                className="w-full rounded border border-border bg-background px-2 py-1.5 text-xs font-mono focus:border-primary focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1">
                Timeout (s)
              </label>
              <input
                type="number"
                value={timeoutSeconds}
                onChange={(e) => setTimeoutSeconds(Number(e.target.value))}
                min={60}
                step={60}
                className="w-full rounded border border-border bg-background px-2 py-1.5 text-xs font-mono focus:border-primary focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1">
                Budget ($)
              </label>
              <input
                type="number"
                value={budgetUsd ?? ""}
                onChange={(e) =>
                  setBudgetUsd(e.target.value ? Number(e.target.value) : null)
                }
                min={0}
                step={0.5}
                placeholder="No limit"
                className="w-full rounded border border-border bg-background px-2 py-1.5 text-xs font-mono focus:border-primary focus:outline-none"
              />
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 border-t border-border px-5 py-3">
          <button
            onClick={onClose}
            className="rounded px-4 py-1.5 text-xs font-medium text-muted-foreground hover:text-foreground"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!originUrl.trim() || selectedAnalyses.size === 0 || submitJob.isPending}
            className="rounded bg-emerald-600 px-4 py-1.5 text-xs font-medium text-white hover:bg-emerald-500 disabled:opacity-50 transition-colors"
          >
            {submitJob.isPending ? "Submitting..." : "Start Run"}
          </button>
        </div>
      </div>
    </div>
  );
}
