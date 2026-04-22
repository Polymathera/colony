import { useEffect, useRef } from "react";
import type { AutocompleteSuggestion } from "./CommandParser";

interface ChatAutocompleteProps {
  suggestions: AutocompleteSuggestion[];
  selectedIndex: number;
  onSelect: (suggestion: AutocompleteSuggestion) => void;
  visible: boolean;
}

const TYPE_COLORS: Record<string, string> = {
  command: "text-blue-400",
  agent: "text-emerald-400",
  capability: "text-amber-400",
  tool: "text-purple-400",
  reference: "text-cyan-400",
};

export function ChatAutocomplete({ suggestions, selectedIndex, onSelect, visible }: ChatAutocompleteProps) {
  const listRef = useRef<HTMLDivElement>(null);

  // Scroll selected item into view
  useEffect(() => {
    if (!listRef.current) return;
    const selected = listRef.current.children[selectedIndex] as HTMLElement | undefined;
    selected?.scrollIntoView({ block: "nearest" });
  }, [selectedIndex]);

  if (!visible || suggestions.length === 0) return null;

  return (
    <div
      ref={listRef}
      className="absolute bottom-full left-0 right-0 mb-1 max-h-48 overflow-auto rounded-lg border border-border bg-card shadow-xl z-10"
    >
      {suggestions.map((s, i) => (
        <button
          key={`${s.type}-${s.value}`}
          onClick={() => onSelect(s)}
          className={`flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs transition-colors ${
            i === selectedIndex ? "bg-primary/10 text-foreground" : "text-muted-foreground hover:bg-accent/50"
          }`}
        >
          <span className={`font-mono font-medium ${TYPE_COLORS[s.type] || ""}`}>
            {s.label}
          </span>
          {s.description && (
            <span className="truncate text-[10px] text-muted-foreground">{s.description}</span>
          )}
        </button>
      ))}
    </div>
  );
}
