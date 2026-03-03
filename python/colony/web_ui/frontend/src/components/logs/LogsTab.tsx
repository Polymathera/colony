import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useLogSources, useLogStream } from "@/api/hooks/useLogStream";
import type { LogSource } from "@/api/types";

type LogLevel = "DEBUG" | "INFO" | "WARNING" | "ERROR";

const LEVEL_COLORS: Record<LogLevel, string> = {
  DEBUG: "text-zinc-500",
  INFO: "text-emerald-400",
  WARNING: "text-amber-400",
  ERROR: "text-red-400",
};

const ALL_LEVELS: LogLevel[] = ["DEBUG", "INFO", "WARNING", "ERROR"];

// Colony log format: "2025-03-03 10:30:45,123 - colony.module - LEVEL - message"
const LOG_LINE_RE = /^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+-\s+\S+\s+-\s+(\w+)\s+-/;

function parseLogLevel(line: string): LogLevel | null {
  const m = line.match(LOG_LINE_RE);
  if (!m) return null;
  const level = m[1].toUpperCase();
  if (ALL_LEVELS.includes(level as LogLevel)) return level as LogLevel;
  return null;
}

function LogLine({ line }: { line: string }) {
  const level = parseLogLevel(line);
  const colorClass = level ? LEVEL_COLORS[level] : "text-zinc-400";
  return (
    <div className={`whitespace-pre-wrap break-all leading-5 ${colorClass}`}>
      {line}
    </div>
  );
}

function SourceSelector({
  sources,
  selected,
  onSelect,
}: {
  sources: LogSource[];
  selected: LogSource | null;
  onSelect: (source: LogSource | null) => void;
}) {
  // Group sources by class_name for readability
  const grouped = useMemo(() => {
    const map = new Map<string, LogSource[]>();
    for (const s of sources) {
      const list = map.get(s.class_name) || [];
      list.push(s);
      map.set(s.class_name, list);
    }
    return map;
  }, [sources]);

  // Use actor_id as the select value, look up from sources
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const actorId = e.target.value;
      if (!actorId) {
        onSelect(null);
        return;
      }
      const found = sources.find((s) => s.actor_id === actorId);
      onSelect(found ?? null);
    },
    [sources, onSelect],
  );

  return (
    <select
      className="rounded border border-border bg-background px-2 py-1.5 text-xs font-mono"
      value={selected?.actor_id ?? ""}
      onChange={handleChange}
    >
      <option value="">Select an actor...</option>
      {[...grouped.entries()].map(([className, actors]) => (
        <optgroup key={className} label={className}>
          {actors.map((a) => (
            <option key={a.actor_id} value={a.actor_id}>
              {a.class_name} (pid={a.pid}{a.ip ? `, ${a.ip}` : ""})
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  );
}

export function LogsTab() {
  const { data: sources } = useLogSources();
  const [selectedSource, setSelectedSource] = useState<LogSource | null>(null);
  const [enabledLevels, setEnabledLevels] = useState<Set<LogLevel>>(new Set(ALL_LEVELS));
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  const { lines, connected, clear } = useLogStream(selectedSource);

  // Filter lines by log level
  const filteredLines = useMemo(() => {
    if (enabledLevels.size === ALL_LEVELS.length) return lines;
    return lines.filter((line) => {
      const level = parseLogLevel(line);
      // Keep lines that don't match the pattern (e.g. stack traces) or match enabled levels
      return level === null || enabledLevels.has(level);
    });
  }, [lines, enabledLevels]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [filteredLines, autoScroll]);

  const toggleLevel = (level: LogLevel) => {
    setEnabledLevels((prev) => {
      const next = new Set(prev);
      if (next.has(level)) {
        next.delete(level);
      } else {
        next.add(level);
      }
      return next;
    });
  };

  return (
    <div className="flex h-full flex-col gap-3">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <SourceSelector
          sources={sources ?? []}
          selected={selectedSource}
          onSelect={setSelectedSource}
        />

        {/* Level filters */}
        <div className="flex items-center gap-1">
          {ALL_LEVELS.map((level) => (
            <button
              key={level}
              className={`rounded px-2 py-1 text-[10px] font-semibold uppercase transition-colors ${
                enabledLevels.has(level)
                  ? `${LEVEL_COLORS[level]} bg-accent`
                  : "text-zinc-600 bg-transparent"
              }`}
              onClick={() => toggleLevel(level)}
            >
              {level}
            </button>
          ))}
        </div>

        {/* Auto-scroll toggle */}
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <input
            type="checkbox"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
            className="rounded"
          />
          Auto-scroll
        </label>

        {/* Clear button */}
        <button
          className="rounded px-2 py-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          onClick={clear}
        >
          Clear
        </button>

        {/* Connection indicator */}
        {selectedSource && (
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground ml-auto">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                connected ? "bg-emerald-400 animate-pulse" : "bg-red-400"
              }`}
            />
            {connected ? `Streaming (${lines.length} lines)` : "Disconnected"}
          </span>
        )}
      </div>

      {/* Log display */}
      <div
        ref={scrollRef}
        className="flex-1 min-h-0 overflow-auto rounded-lg border bg-[hsl(222,47%,5%)] p-3 font-mono text-[11px]"
      >
        {!selectedSource ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            Select an actor to view its logs
          </div>
        ) : filteredLines.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            {connected ? "Waiting for log lines..." : "Connecting..."}
          </div>
        ) : (
          filteredLines.map((line, i) => <LogLine key={i} line={line} />)
        )}
      </div>
    </div>
  );
}
