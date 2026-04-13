import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import type { RunCallTraceEntry } from "./types";

/* Lightweight Python syntax highlighting via regex.
   Not a full parser — just enough to distinguish keywords, strings,
   comments, and function calls for visual scanning. */

const PY_KEYWORDS = new Set([
  "False", "None", "True", "and", "as", "assert", "async", "await",
  "break", "class", "continue", "def", "del", "elif", "else", "except",
  "finally", "for", "from", "global", "if", "import", "in", "is",
  "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
  "while", "with", "yield",
]);

function highlightPython(code: string): React.ReactNode[] {
  // Tokenize with a regex that captures strings, comments, and words
  const TOKEN_RE = /("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|#[^\n]*|\b\w+\b|[^\s])/g;
  const nodes: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = TOKEN_RE.exec(code)) !== null) {
    // Emit any whitespace/gap between tokens as plain text
    if (match.index > lastIndex) {
      nodes.push(code.slice(lastIndex, match.index));
    }

    const token = match[0];
    const key = nodes.length;

    if (token.startsWith("#")) {
      nodes.push(<span key={key} className="text-emerald-700/70 italic">{token}</span>);
    } else if (token.startsWith('"') || token.startsWith("'")) {
      nodes.push(<span key={key} className="text-amber-400/80">{token}</span>);
    } else if (PY_KEYWORDS.has(token)) {
      nodes.push(<span key={key} className="text-blue-400 font-semibold">{token}</span>);
    } else if (token === "run" || token === "browse" || token === "signal_complete") {
      nodes.push(<span key={key} className="text-cyan-400">{token}</span>);
    } else {
      nodes.push(token);
    }
    lastIndex = match.index + token.length;
  }

  // Trailing text
  if (lastIndex < code.length) {
    nodes.push(code.slice(lastIndex));
  }

  return nodes;
}

const MAX_COLLAPSED_LINES = 20;

export function CodeBlock({
  code,
  maxHeight,
  className,
  lineAnnotations,
  onNavigateToSpan,
}: {
  code: string;
  maxHeight?: string;
  className?: string;
  lineAnnotations?: Map<number, RunCallTraceEntry>;
  /** Called when the user clicks the "show in tree" button on an annotation. */
  onNavigateToSpan?: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const [expandedAnnotations, setExpandedAnnotations] = useState<Set<number>>(() => new Set());

  const toggleAnnotation = useCallback((lineNum: number) => {
    setExpandedAnnotations((prev) => {
      const next = new Set(prev);
      if (next.has(lineNum)) next.delete(lineNum);
      else next.add(lineNum);
      return next;
    });
  }, []);

  const lines = code.split("\n");
  const needsTruncation = !expanded && lines.length > MAX_COLLAPSED_LINES;
  const displayLines = needsTruncation ? lines.slice(0, MAX_COLLAPSED_LINES) : lines;

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }, [code]);

  return (
    <div className={cn("relative group rounded-md border bg-muted/30", className)}>
      {/* Copy button */}
      <button
        onClick={handleCopy}
        className="absolute top-1.5 right-1.5 rounded px-1.5 py-0.5 text-[10px] text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity bg-muted hover:bg-muted/80"
      >
        {copied ? "Copied" : "Copy"}
      </button>

      <div
        className="overflow-auto"
        style={{ maxHeight: expanded ? undefined : maxHeight ?? "320px" }}
      >
        <pre className="p-3 text-[11px] leading-relaxed font-mono">
          <code>
            {displayLines.map((line, i) => {
              const lineNum = i + 1;
              const annotation = lineAnnotations?.get(lineNum);
              const isAnnotationExpanded = annotation && expandedAnnotations.has(lineNum);
              return (
                <div key={i}>
                  <div
                    className={cn(
                      "flex",
                      annotation && (annotation.success
                        ? "bg-emerald-950/20 border-l-2 border-l-emerald-500/40"
                        : "bg-red-950/20 border-l-2 border-l-red-500/40"),
                    )}
                  >
                    <span className="select-none text-muted-foreground/40 w-8 text-right pr-3 shrink-0">
                      {lineNum}
                    </span>
                    <span className="text-foreground/85 whitespace-pre-wrap break-all flex-1">
                      {highlightPython(line)}
                    </span>
                    {annotation && (
                      <button
                        className={cn(
                          "shrink-0 ml-2 text-[9px] font-mono px-1 rounded cursor-pointer hover:opacity-80",
                          annotation.success
                            ? "text-emerald-400 bg-emerald-500/10"
                            : "text-red-400 bg-red-500/10",
                        )}
                        onClick={() => toggleAnnotation(lineNum)}
                      >
                        {annotation.success ? "\u2713" : annotation.blocked ? "BLOCKED" : "\u2717"}
                        {" "}
                        {annotation.action_key.split(".").pop()}
                        {" "}
                        {isAnnotationExpanded ? "\u25B4" : "\u25BE"}
                      </button>
                    )}
                  </div>
                  {isAnnotationExpanded && annotation && (
                    <div className={cn(
                      "ml-11 mr-2 mb-1 rounded px-2 py-1 text-[10px] font-mono",
                      annotation.success
                        ? "bg-emerald-950/30 text-emerald-300/80"
                        : "bg-red-950/30 text-red-300/80",
                    )}>
                      {annotation.error && (
                        <div className="text-red-400 break-all">{annotation.error}</div>
                      )}
                      {annotation.output_preview && (
                        <div className="text-muted-foreground break-all mt-0.5">
                          {annotation.output_preview}
                        </div>
                      )}
                      {!annotation.error && !annotation.output_preview && (
                        <div className="text-muted-foreground/50 italic">No output</div>
                      )}
                      {onNavigateToSpan && (
                        <button
                          className="mt-1 text-[9px] text-blue-400 hover:text-blue-300 underline underline-offset-2 cursor-pointer"
                          onClick={(e) => { e.stopPropagation(); onNavigateToSpan(); }}
                        >
                          Show in tree view
                        </button>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </code>
        </pre>
      </div>

      {/* Expand/collapse button */}
      {lines.length > MAX_COLLAPSED_LINES && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full border-t py-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors bg-muted/20"
        >
          {expanded
            ? "Collapse"
            : `Show all ${lines.length} lines (+${lines.length - MAX_COLLAPSED_LINES} more)`}
        </button>
      )}
    </div>
  );
}
