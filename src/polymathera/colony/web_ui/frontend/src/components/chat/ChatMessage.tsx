import { useMemo, useState } from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import { AttachmentList, type Attachment } from "./Attachments";

// When a fenced code block exceeds either threshold, the chat message
// renders it collapsed with a "Show more" toggle — same UX shape as
// the Traces tab's section toggles. Tuned for long Python dict dumps
// the SessionAgent emits when an action returns a structured result
// (the agent is also prompted to wrap such results in ``` python `` fences).
const COLLAPSE_LINE_THRESHOLD = 12;
const COLLAPSE_CHAR_THRESHOLD = 800;
const COLLAPSED_PREVIEW_LINES = 10;

function extractText(node: unknown): string {
  // Walk the react-markdown children tree and concatenate every leaf
  // string. Used to size code blocks for the collapse heuristic without
  // forcing the caller to thread the raw source through props.
  if (typeof node === "string") return node;
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (node && typeof node === "object") {
    // ReactElement-like: { props: { children } }
    const props = (node as { props?: { children?: unknown } }).props;
    if (props && "children" in props) return extractText(props.children);
  }
  return "";
}

function CollapsiblePre({
  children, ...rest
}: React.HTMLAttributes<HTMLPreElement> & { children?: React.ReactNode }) {
  const text = useMemo(() => extractText(children), [children]);
  const lineCount = text ? text.split("\n").length : 0;
  const isLong =
    lineCount > COLLAPSE_LINE_THRESHOLD || text.length > COLLAPSE_CHAR_THRESHOLD;
  const [expanded, setExpanded] = useState(false);

  if (!isLong) {
    return <pre {...rest}>{children}</pre>;
  }

  // Button label: prefer "+N lines" when the block has multiple lines;
  // fall back to a char count when the agent dumped a single-line
  // ``str(dict)`` (no newlines but still long), so the user does not
  // see "Show -8 more lines".
  const hiddenLines = Math.max(0, lineCount - COLLAPSED_PREVIEW_LINES);
  const collapsedLabel =
    hiddenLines > 0
      ? `Show ${hiddenLines} more line${hiddenLines === 1 ? "" : "s"}`
      : `Show full content (${text.length.toLocaleString()} chars)`;

  return (
    <div className="not-prose my-2 rounded-md border border-border bg-muted/30 overflow-hidden">
      {/* ``whitespace-pre-wrap`` + ``break-words`` matter when the agent
          emits a single-line ``str(dict)`` — without them the block
          becomes one infinitely-wide horizontally-scrolling line and
          the expand button hides the very content the user wanted. */}
      <pre
        {...rest}
        className={cn(
          "m-0 px-3 py-2 overflow-auto text-[10px] leading-snug whitespace-pre-wrap break-words",
          !expanded && "max-h-[15rem] [mask-image:linear-gradient(to_bottom,black_70%,transparent)]",
          rest.className,
        )}
      >
        {children}
      </pre>
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="w-full border-t border-border bg-muted/40 px-3 py-1 text-[10px] text-muted-foreground hover:bg-muted/60 transition-colors text-left"
      >
        <span className="mr-1.5">{expanded ? "▼" : "▶"}</span>
        {expanded ? "Collapse" : collapsedLabel}
      </button>
    </div>
  );
}

const MARKDOWN_COMPONENTS: Components = {
  pre: CollapsiblePre,
};

export interface ChatMessageData {
  id: string;
  run_id: string | null;
  role: "user" | "agent" | "system";
  agent_id?: string;
  agent_type?: string;
  user_id?: string;
  username?: string;
  content: string;
  timestamp: number;
  // Agent-to-user questions
  request_id?: string;
  response_options?: string[];
  awaiting_reply?: boolean;
  // Routing hint for the click on an option button:
  //  - "human_approval" → POST to /sessions/{id}/human_approval/{request_id}/respond
  //  - undefined / other → existing WebSocket reply lane
  kind?: "human_approval" | string;
  // Run lifecycle
  run_status?: "submitted" | "spawning" | "running" | "completed" | "failed";
  // Structured attachments (code / table / diff blocks the agent
  // emits via ``respond_to_user(attachments=…)`` and the
  // ``respond_to_user_with_table`` / ``respond_to_user_with_diff`` actions). Renders
  // below the markdown content.
  attachments?: Attachment[] | null;
}

interface ChatMessageProps {
  message: ChatMessageData;
  // Receives the source message so the host can route per ``kind``
  // (human_approval vs freeform agent_question).
  onReply?: (
    message: ChatMessageData,
    content: string,
  ) => void;
}

const ROLE_COLORS: Record<string, string> = {
  user: "bg-primary/10 border-primary/20",
  agent: "bg-emerald-900/20 border-emerald-500/20",
  system: "bg-zinc-800/50 border-zinc-700/30",
};

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function ChatMessage({ message, onReply }: ChatMessageProps) {
  const { role, content, agent_id, agent_type, username, timestamp, request_id, response_options, awaiting_reply, run_status, attachments } = message;

  return (
    <div className={cn("rounded-lg border px-3 py-2 text-xs", ROLE_COLORS[role] || ROLE_COLORS.system)}>
      {/* Header: role label, agent/user info, timestamp */}
      <div className="flex items-center gap-2 mb-1">
        <span className="font-semibold capitalize">{role}</span>
        {role === "agent" && agent_type && (
          <span className="font-mono text-[10px] text-muted-foreground">
            {agent_type.split(".").pop()}
          </span>
        )}
        {role === "agent" && agent_id && (
          <span className="font-mono text-[10px] text-muted-foreground">
            {agent_id.slice(0, 12)}
          </span>
        )}
        {role === "user" && username && (
          <span className="text-[10px] text-muted-foreground">{username}</span>
        )}
        <span className="ml-auto text-[10px] text-muted-foreground">{formatTime(timestamp)}</span>
      </div>

      {/* Run status badge */}
      {run_status && (
        <div className="mb-1">
          <span className={cn(
            "inline-block rounded px-1.5 py-0.5 text-[10px] font-medium",
            run_status === "completed" ? "bg-emerald-500/20 text-emerald-400" :
            run_status === "failed" ? "bg-red-500/20 text-red-400" :
            run_status === "running" ? "bg-blue-500/20 text-blue-400" :
            "bg-zinc-500/20 text-zinc-400"
          )}>
            {run_status}
          </span>
        </div>
      )}

      {/* Content — render markdown for agent messages, plain text for user */}
      {role === "user" ? (
        <div className="whitespace-pre-wrap leading-5">{content}</div>
      ) : (
        <div className="prose prose-invert prose-xs max-w-none leading-5 [&_table]:text-[10px] [&_th]:px-2 [&_td]:px-2 [&_code]:text-[10px] [&_pre]:text-[10px]">
          <Markdown remarkPlugins={[remarkGfm]} components={MARKDOWN_COMPONENTS}>
            {content}
          </Markdown>
        </div>
      )}

      {/* Structured attachments (code / table / diff) emitted by the
          agent's ``respond_to_user`` / ``respond_to_user_with_table`` /
          ``respond_to_user_with_diff`` actions. Rendered below the markdown
          content so the human-readable summary stays at the top. */}
      {attachments && attachments.length > 0 && (
        <AttachmentList attachments={attachments} />
      )}

      {/* Agent question response options */}
      {awaiting_reply && response_options && response_options.length > 0 && request_id && agent_id && onReply && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {response_options.map((option, i) => (
            <button
              key={i}
              onClick={() => onReply(message, option)}
              className="rounded border border-primary/30 bg-primary/10 px-2.5 py-1 text-[10px] font-medium text-primary hover:bg-primary/20 transition-colors"
            >
              {option}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
