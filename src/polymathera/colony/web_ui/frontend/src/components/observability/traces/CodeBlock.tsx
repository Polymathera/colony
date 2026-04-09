import { useState, useCallback } from "react";
import { cn } from "@/lib/utils";

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
}: {
  code: string;
  maxHeight?: string;
  className?: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

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
            {displayLines.map((line, i) => (
              <div key={i} className="flex">
                <span className="select-none text-muted-foreground/40 w-8 text-right pr-3 shrink-0">
                  {i + 1}
                </span>
                <span className="text-foreground/85 whitespace-pre-wrap break-all">
                  {highlightPython(line)}
                </span>
              </div>
            ))}
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
