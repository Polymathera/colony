import { useState } from "react";
import { cn } from "@/lib/utils";
import type { PromptDiffSection } from "./types";

function DiffSection({ section }: { section: PromptDiffSection }) {
  const [expanded, setExpanded] = useState(true);

  const added = section.changes.filter((c) => c.type === "added").length;
  const removed = section.changes.filter((c) => c.type === "removed").length;

  return (
    <div className="border-b border-border/50 last:border-b-0">
      <button
        className="flex items-center gap-2 w-full px-3 py-1.5 text-left hover:bg-muted/30 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="text-[10px] text-muted-foreground">
          {expanded ? "\u25BC" : "\u25B6"}
        </span>
        <span className="text-xs font-semibold text-foreground/80">
          {section.name}
        </span>
        <span className="text-[10px] text-muted-foreground ml-auto">
          {added > 0 && (
            <span className="text-emerald-400">+{added}</span>
          )}
          {added > 0 && removed > 0 && " / "}
          {removed > 0 && (
            <span className="text-red-400">-{removed}</span>
          )}
        </span>
      </button>

      {expanded && (
        <div className="font-mono text-[11px] leading-relaxed">
          {section.changes.map((change, i) => (
            <div
              key={i}
              className={cn(
                "px-3 py-0 whitespace-pre-wrap break-all border-l-2",
                change.type === "added" &&
                  "bg-emerald-950/20 border-l-emerald-500 text-emerald-300/90",
                change.type === "removed" &&
                  "bg-red-950/20 border-l-red-500 text-red-300/90",
                change.type === "context" &&
                  "border-l-transparent text-muted-foreground/60",
              )}
            >
              <span className="select-none text-muted-foreground/30 inline-block w-4 mr-2">
                {change.type === "added" ? "+" : change.type === "removed" ? "-" : " "}
              </span>
              {change.content}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function DiffRenderer({
  sections,
  totalAdded,
  totalRemoved,
}: {
  sections: PromptDiffSection[];
  totalAdded: number;
  totalRemoved: number;
}) {
  if (sections.length === 0 && totalAdded === 0 && totalRemoved === 0) {
    return (
      <div className="flex h-32 items-center justify-center text-sm text-muted-foreground">
        No differences found.
      </div>
    );
  }

  return (
    <div className="rounded-lg border overflow-auto bg-card">
      {/* Summary bar */}
      <div className="flex items-center gap-3 px-3 py-2 border-b bg-muted/20">
        <span className="text-xs text-muted-foreground">
          {sections.length} section{sections.length !== 1 ? "s" : ""} changed
        </span>
        <span className="text-xs text-emerald-400 font-mono">+{totalAdded}</span>
        <span className="text-xs text-red-400 font-mono">-{totalRemoved}</span>
      </div>

      {sections.map((section, i) => (
        <DiffSection key={i} section={section} />
      ))}
    </div>
  );
}
