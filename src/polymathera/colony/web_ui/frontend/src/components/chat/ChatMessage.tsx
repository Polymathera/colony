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
  // Short action name (e.g. "create_decomposition"). Present iff the
  // request was a typed approval — drives the 4-choice button labels
  // (Approve once / Approve all / Reject / Abort).
  action_type?: string | null;
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
  // (human_approval vs freeform agent_question). ``explanation`` is
  // the operator's required justification on ``reject`` / ``abort``
  // (matches the Pydantic ``HumanApprovalResponse.explanation``
  // contract on the backend; the response validator rejects empty
  // explanation for those choices with 422).
  onReply?: (
    message: ChatMessageData,
    content: string,
    extra?: { explanation?: string },
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

function approvalButtonLabel(
  option: string,
  actionType: string | null,
): string {
  if (option === "reject") return "Reject";
  if (option === "abort") return "Abort";
  if (option === "approve") return "Approve";
  if (option === "approve_once") {
    return actionType ? `Approve once: ${actionType}` : "Approve once";
  }
  if (option === "approve_all") {
    return actionType
      ? `Approve all ${actionType} this session`
      : "Approve all this session";
  }
  return option;
}

// ``reject`` and ``abort`` require a non-empty operator explanation —
// the backend's ``HumanApprovalResponse`` Pydantic validator enforces
// this and the HTTP endpoint surfaces a 422 on empty. The UI mirrors
// the contract by switching the card into "compose explanation" mode
// before submission.
const CHOICES_REQUIRING_EXPLANATION = new Set(["reject", "abort"]);

export function ChatMessage({ message, onReply }: ChatMessageProps) {
  const { role, content, agent_id, agent_type, username, timestamp, request_id, response_options, awaiting_reply, run_status, attachments, action_type } = message;

  // When the operator clicks ``reject`` or ``abort``, switch the card
  // into compose mode and require a non-empty explanation before
  // submitting. ``null`` means no compose in progress (waiting for
  // the operator's initial choice).
  const [composeChoice, setComposeChoice] = useState<string | null>(null);
  const [explanationDraft, setExplanationDraft] = useState("");

  const handleChoice = (option: string) => {
    if (!onReply) return;
    if (CHOICES_REQUIRING_EXPLANATION.has(option)) {
      setComposeChoice(option);
      setExplanationDraft("");
      return;
    }
    onReply(message, option);
  };

  const submitWithExplanation = () => {
    if (!onReply || composeChoice === null) return;
    const trimmed = explanationDraft.trim();
    if (!trimmed) return;
    onReply(message, composeChoice, { explanation: trimmed });
    setComposeChoice(null);
    setExplanationDraft("");
  };

  const cancelCompose = () => {
    setComposeChoice(null);
    setExplanationDraft("");
  };

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
      {awaiting_reply && response_options && response_options.length > 0 && request_id && agent_id && onReply && composeChoice === null && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {response_options.map((option, i) => (
            <button
              key={i}
              onClick={() => handleChoice(option)}
              className={cn(
                "rounded border px-2.5 py-1 text-[10px] font-medium transition-colors",
                option === "reject"
                  ? "border-red-500/40 bg-red-500/10 text-red-300 hover:bg-red-500/20"
                  : option === "abort"
                  ? "border-red-700/50 bg-red-700/20 text-red-200 hover:bg-red-700/30"
                  : option === "approve_all"
                  ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20"
                  : "border-primary/30 bg-primary/10 text-primary hover:bg-primary/20",
              )}
            >
              {approvalButtonLabel(option, action_type ?? null)}
            </button>
          ))}
        </div>
      )}

      {/* Explanation compose mode — operator picked reject or abort; a
          non-empty justification is required before submission so the
          agent's next iteration can react to "why". */}
      {awaiting_reply && composeChoice !== null && request_id && agent_id && onReply && (
        <div className="mt-2 flex flex-col gap-1.5">
          <div className="text-[10px] text-muted-foreground">
            {composeChoice === "abort"
              ? "Aborting — explain why so the agent can record the rationale before winding down."
              : "Rejecting — explain why so the agent can adjust the next attempt."}
          </div>
          <textarea
            value={explanationDraft}
            onChange={(e) => setExplanationDraft(e.target.value)}
            placeholder="Required: operator justification"
            rows={3}
            className="w-full rounded border border-zinc-700/40 bg-zinc-900/50 px-2 py-1 text-[11px] leading-5 focus:outline-none focus:border-primary/50"
          />
          <div className="flex gap-1.5">
            <button
              onClick={submitWithExplanation}
              disabled={!explanationDraft.trim()}
              className={cn(
                "rounded border px-2.5 py-1 text-[10px] font-medium transition-colors",
                composeChoice === "abort"
                  ? "border-red-700/50 bg-red-700/20 text-red-200 hover:bg-red-700/30"
                  : "border-red-500/40 bg-red-500/10 text-red-300 hover:bg-red-500/20",
                "disabled:opacity-40 disabled:cursor-not-allowed",
              )}
            >
              Submit {composeChoice === "abort" ? "Abort" : "Reject"}
            </button>
            <button
              onClick={cancelCompose}
              className="rounded border border-zinc-700/40 px-2.5 py-1 text-[10px] font-medium text-muted-foreground hover:bg-zinc-800/50"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
