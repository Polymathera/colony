import { cn } from "@/lib/utils";
import type { TraceViewMode } from "./types";

const VIEW_OPTIONS: { mode: TraceViewMode; label: string }[] = [
  { mode: "tree", label: "Tree" },
  { mode: "timeline", label: "Timeline" },
  { mode: "diff", label: "Diff" },
  { mode: "fsm", label: "FSM" },
  { mode: "flow", label: "Flow" },
];

export function TraceViewSelector({
  value,
  onChange,
}: {
  value: TraceViewMode;
  onChange: (mode: TraceViewMode) => void;
}) {
  return (
    <div className="inline-flex rounded-md border border-border bg-muted/30">
      {VIEW_OPTIONS.map(({ mode, label }) => (
        <button
          key={mode}
          className={cn(
            "px-3 py-1 text-xs font-medium transition-colors",
            "first:rounded-l-md last:rounded-r-md",
            value === mode
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground hover:bg-muted/60"
          )}
          onClick={() => onChange(mode)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
