/**
 * Typed attachments rendered alongside a chat message's markdown body.
 *
 * Each attachment carries a ``kind`` discriminator chosen by the
 * agent's ``respond_to_user`` / ``respond_to_user_with_table`` /
 * ``respond_to_user_with_diff`` action. The renderer here is dispatched on
 * that discriminator. New kinds plug in by adding a case to
 * :func:`AttachmentRenderer` without touching the action surface.
 */
import { useState } from "react";
import { cn } from "@/lib/utils";

export type Attachment =
  | CodeAttachment
  | TableAttachment
  | DiffAttachment
  | UnknownAttachment;

export interface CodeAttachment {
  kind: "code";
  content: string;
  lang?: string;
  label?: string;
}

export interface TableAttachment {
  kind: "table";
  rows: Record<string, unknown>[];
  columns?: string[];
  label?: string;
}

export interface DiffAttachment {
  kind: "diff";
  before: string;
  after: string;
  lang?: string;
  label?: string;
}

export interface UnknownAttachment {
  kind: string;
  [key: string]: unknown;
}

const COLLAPSE_LINE_THRESHOLD = 12;
const COLLAPSE_CHAR_THRESHOLD = 800;
const COLLAPSED_PREVIEW_LINES = 10;

export function AttachmentList({ attachments }: { attachments: Attachment[] }) {
  if (!attachments || attachments.length === 0) return null;
  return (
    <div className="mt-2 flex flex-col gap-2">
      {attachments.map((a, i) => (
        <AttachmentRenderer key={i} attachment={a} />
      ))}
    </div>
  );
}

function AttachmentRenderer({ attachment }: { attachment: Attachment }) {
  switch (attachment.kind) {
    case "code":
      return <CodeAttachmentView a={attachment as CodeAttachment} />;
    case "table":
      return <TableAttachmentView a={attachment as TableAttachment} />;
    case "diff":
      return <DiffAttachmentView a={attachment as DiffAttachment} />;
    default:
      // Forward-compat: unknown kinds render as a small badge so the
      // user notices the agent emitted something the UI doesn't yet
      // understand, rather than silently dropping the payload.
      return (
        <div className="rounded border border-amber-500/40 bg-amber-500/10 px-2 py-1 text-[10px] text-amber-300">
          unsupported attachment kind: <code>{attachment.kind}</code>
        </div>
      );
  }
}

// ---------------------------------------------------------------------------
// Code attachment — collapsible past the same thresholds as the inline
// markdown ``<pre>`` blocks rendered by react-markdown.
// ---------------------------------------------------------------------------

function CollapsibleBlock({
  children, label, className,
}: {
  children: React.ReactNode;
  label?: string;
  className?: string;
}) {
  return (
    <div className={cn("rounded-md border border-border bg-muted/30 overflow-hidden", className)}>
      {label && (
        <div className="px-3 py-1 border-b border-border text-[10px] font-medium text-muted-foreground bg-muted/40">
          {label}
        </div>
      )}
      {children}
    </div>
  );
}

function CodeAttachmentView({ a }: { a: CodeAttachment }) {
  const text = a.content ?? "";
  const lineCount = text ? text.split("\n").length : 0;
  const isLong =
    lineCount > COLLAPSE_LINE_THRESHOLD || text.length > COLLAPSE_CHAR_THRESHOLD;
  const [expanded, setExpanded] = useState(false);

  const hiddenLines = Math.max(0, lineCount - COLLAPSED_PREVIEW_LINES);
  const collapsedLabel =
    hiddenLines > 0
      ? `Show ${hiddenLines} more line${hiddenLines === 1 ? "" : "s"}`
      : `Show full content (${text.length.toLocaleString()} chars)`;

  const langLabel = a.lang ? a.lang : "code";
  const headerLabel = a.label ? `${langLabel} — ${a.label}` : langLabel;

  return (
    <CollapsibleBlock label={headerLabel}>
      {/* ``whitespace-pre-wrap`` + ``break-words`` so a single-line
          ``str(dict)`` payload wraps instead of horizontally
          scrolling and hiding behind the collapse button. */}
      <pre
        className={cn(
          "m-0 px-3 py-2 overflow-auto text-[10px] leading-snug whitespace-pre-wrap break-words",
          isLong && !expanded &&
            "max-h-[15rem] [mask-image:linear-gradient(to_bottom,black_70%,transparent)]",
        )}
      >
        <code className={a.lang ? `language-${a.lang}` : undefined}>{text}</code>
      </pre>
      {isLong && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="w-full border-t border-border bg-muted/40 px-3 py-1 text-[10px] text-muted-foreground hover:bg-muted/60 transition-colors text-left"
        >
          <span className="mr-1.5">{expanded ? "▼" : "▶"}</span>
          {expanded ? "Collapse" : collapsedLabel}
        </button>
      )}
    </CollapsibleBlock>
  );
}

// ---------------------------------------------------------------------------
// Table attachment — straightforward HTML table; rows that miss a
// declared column render as an empty cell.
// ---------------------------------------------------------------------------

function TableAttachmentView({ a }: { a: TableAttachment }) {
  const columns =
    a.columns && a.columns.length > 0
      ? a.columns
      : Array.from(
          new Set(a.rows.flatMap((r) => Object.keys(r ?? {}))),
        ).sort();

  const formatCell = (v: unknown): string => {
    if (v == null) return "";
    if (typeof v === "string") return v;
    if (typeof v === "number" || typeof v === "boolean") return String(v);
    try {
      return JSON.stringify(v);
    } catch {
      return String(v);
    }
  };

  return (
    <CollapsibleBlock label={a.label ?? `table — ${a.rows.length} row${a.rows.length === 1 ? "" : "s"}`}>
      <div className="overflow-auto max-h-[20rem]">
        <table className="w-full text-[10px] border-collapse">
          <thead className="sticky top-0 bg-muted/60">
            <tr>
              {columns.map((c) => (
                <th
                  key={c}
                  className="text-left px-2 py-1 border-b border-border font-medium text-muted-foreground"
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {a.rows.map((row, i) => (
              <tr key={i} className="border-b border-border/50 last:border-b-0 hover:bg-muted/20">
                {columns.map((c) => (
                  <td key={c} className="px-2 py-1 align-top font-mono break-words">
                    {formatCell(row?.[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </CollapsibleBlock>
  );
}

// ---------------------------------------------------------------------------
// Diff attachment — line-level LCS over ``before`` / ``after`` so the
// chat shows ``+`` / ``-`` markers in the ``DiffRenderer`` shape used
// by the Traces tab. Memory is O(N·M) ints; fine for chat-sized
// diffs (≤10k lines per side).
// ---------------------------------------------------------------------------

type DiffOp = { type: "context" | "added" | "removed"; content: string };

function lineDiff(beforeText: string, afterText: string): DiffOp[] {
  const a = beforeText.split("\n");
  const b = afterText.split("\n");
  const n = a.length;
  const m = b.length;
  // dp[i][j] = LCS length of a[i:] vs b[j:]
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      if (a[i] === b[j]) dp[i][j] = dp[i + 1][j + 1] + 1;
      else dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const out: DiffOp[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      out.push({ type: "context", content: a[i] });
      i++; j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      out.push({ type: "removed", content: a[i] });
      i++;
    } else {
      out.push({ type: "added", content: b[j] });
      j++;
    }
  }
  while (i < n) out.push({ type: "removed", content: a[i++] });
  while (j < m) out.push({ type: "added", content: b[j++] });
  return out;
}

function DiffAttachmentView({ a }: { a: DiffAttachment }) {
  const ops = lineDiff(a.before ?? "", a.after ?? "");
  const added = ops.filter((o) => o.type === "added").length;
  const removed = ops.filter((o) => o.type === "removed").length;

  return (
    <CollapsibleBlock
      label={
        (a.label ? `${a.label} — ` : "") +
        `+${added} / -${removed}` +
        (a.lang && a.lang !== "text" ? ` (${a.lang})` : "")
      }
    >
      <div className="font-mono text-[10px] leading-relaxed max-h-[20rem] overflow-auto">
        {ops.map((op, i) => (
          <div
            key={i}
            className={cn(
              "px-3 whitespace-pre-wrap break-words border-l-2",
              op.type === "added" &&
                "bg-emerald-950/20 border-l-emerald-500 text-emerald-300/90",
              op.type === "removed" &&
                "bg-red-950/20 border-l-red-500 text-red-300/90",
              op.type === "context" &&
                "border-l-transparent text-muted-foreground/60",
            )}
          >
            <span className="select-none text-muted-foreground/30 inline-block w-3 mr-2">
              {op.type === "added" ? "+" : op.type === "removed" ? "-" : " "}
            </span>
            {op.content}
          </div>
        ))}
      </div>
    </CollapsibleBlock>
  );
}
