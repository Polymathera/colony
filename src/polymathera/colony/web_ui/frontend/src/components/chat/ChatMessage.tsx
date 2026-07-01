import { useMemo, useState } from "react";
import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { Highlight, themes } from "prism-react-renderer";
import { Check, X as XIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { apiFetch } from "@/api/client";
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

function extractLanguage(children: unknown): string | null {
  // Fenced markdown code blocks render as ``<pre><code className="language-X">...</code></pre>``.
  // The language is on the inner ``<code>``'s className; the
  // ``<pre>`` (which this codepath wraps) has no language hint of
  // its own. Walk the single immediate code child to find it.
  const first = Array.isArray(children) ? children[0] : children;
  const className = (first as { props?: { className?: string } } | null)
    ?.props?.className ?? "";
  const match = /(?:^|\s)language-([\w-]+)/.exec(className);
  return match ? match[1] : null;
}

function HighlightedCode({
  code,
  language,
  className,
}: {
  code: string;
  language: string;
  className?: string;
}) {
  // Prism-React-Renderer's vsDark theme matches the dashboard's dark
  // background. Lines render as flex rows so long lines word-wrap
  // alongside their highlighted spans — same behavior as the prior
  // ``whitespace-pre-wrap`` on the plain <pre>.
  return (
    <Highlight code={code.replace(/\n+$/, "")} language={language} theme={themes.vsDark}>
      {({ tokens, getLineProps, getTokenProps, style }) => (
        <pre
          className={cn(
            "m-0 px-3 py-2 overflow-auto text-[10px] leading-snug whitespace-pre-wrap break-words",
            className,
          )}
          // The Prism theme provides background + foreground; merge
          // so the dashboard's surrounding tailwind classes still win
          // on layout.
          style={{ ...style, background: "transparent" }}
        >
          {tokens.map((line, i) => {
            const lineProps = getLineProps({ line });
            return (
              <div key={i} {...lineProps}>
                {line.map((token, key) => {
                  const tokenProps = getTokenProps({ token });
                  return <span key={key} {...tokenProps} />;
                })}
              </div>
            );
          })}
        </pre>
      )}
    </Highlight>
  );
}

function CollapsiblePre({
  children, ...rest
}: React.HTMLAttributes<HTMLPreElement> & { children?: React.ReactNode }) {
  const text = useMemo(() => extractText(children), [children]);
  const language = useMemo(() => extractLanguage(children), [children]);
  const lineCount = text ? text.split("\n").length : 0;
  const isLong =
    lineCount > COLLAPSE_LINE_THRESHOLD || text.length > COLLAPSE_CHAR_THRESHOLD;
  const [expanded, setExpanded] = useState(false);

  // Body: highlighted via Prism when the markdown fence carried a
  // recognised language; plain <pre> otherwise (so non-code fenced
  // blocks like ``` <generic dump> `` keep their unstyled look).
  const body = language ? (
    <HighlightedCode
      code={text}
      language={language}
      className={cn(rest.className, !expanded && isLong && "max-h-[15rem] [mask-image:linear-gradient(to_bottom,black_70%,transparent)]")}
    />
  ) : (
    <pre
      {...rest}
      className={cn(
        "m-0 px-3 py-2 overflow-auto text-[10px] leading-snug whitespace-pre-wrap break-words",
        !expanded && isLong && "max-h-[15rem] [mask-image:linear-gradient(to_bottom,black_70%,transparent)]",
        rest.className,
      )}
    >
      {children}
    </pre>
  );

  if (!isLong) {
    // Short block: no collapse chrome. Still uses ``body`` so syntax
    // highlighting applies when language is recognised.
    if (language) {
      return (
        <div className="not-prose my-2 rounded-md border border-border bg-muted/30 overflow-hidden">
          {body}
        </div>
      );
    }
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
      {body}
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

// Thumbs on an agent message. The rating is recorded against the INFER
// span that produced the message (resolved server-side), so it lands in
// the same feedback store the Traces tab uses.
function MessageFeedback({ sessionId, messageId }: { sessionId: string; messageId: string }) {
  const [state, setState] = useState<"idle" | "up" | "down" | "error">("idle");

  const rate = async (rating: "up" | "down") => {
    try {
      await apiFetch(
        `/chat/sessions/${encodeURIComponent(sessionId)}/messages/${encodeURIComponent(messageId)}/feedback`,
        { method: "POST", body: JSON.stringify({ rating, note: null }) },
      );
      setState(rating);
    } catch {
      setState("error");
    }
  };

  return (
    <div className="mt-1 flex items-center gap-2 text-[10px] text-muted-foreground">
      <button
        type="button"
        onClick={() => rate("up")}
        className={cn("rounded px-1 hover:text-emerald-400", state === "up" && "text-emerald-400")}
        title="Helpful"
      >
        ▲
      </button>
      <button
        type="button"
        onClick={() => rate("down")}
        className={cn("rounded px-1 hover:text-red-400", state === "down" && "text-red-400")}
        title="Not helpful"
      >
        ▼
      </button>
      {(state === "up" || state === "down") && <span>recorded</span>}
      {state === "error" && <span className="text-red-400">couldn’t record</span>}
    </div>
  );
}

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
  //  - "human_approval"   → POST to /sessions/{id}/human_approval/{request_id}/respond
  //  - "human_help"       → POST to /sessions/{id}/human_help/{request_id}/respond
  //  - "guardrail_waiver" → POST to /sessions/{id}/waivers/{request_id}/{approve|reject}
  //  - "status"           → unified-timeline status entry (action lifecycle +
  //                         mission narrative). No interactive controls; rendered
  //                         compactly with a left-rail dot whose color/icon comes
  //                         from ``status_phase``.
  //  - undefined / other  → existing WebSocket reply lane
  kind?: "human_approval" | "human_help" | "guardrail_waiver" | "status" | string;
  // Status-entry phase. Only meaningful when ``kind === "status"``.
  // ``running`` → spinner; ``completed`` → check; ``failed`` → X;
  // ``undefined`` (e.g. for mission_status narratives that have no
  // lifecycle) → plain neutral dot.
  status_phase?: "running" | "completed" | "failed";
  // Extra payload travelling with the question. For ``human_help``
  // the agent stamps the ``context`` here (what it has tried + what
  // it observed) so the operator sees the situation that triggered
  // the escalation; the host renders ``extra.context`` above the
  // response surface.
  extra?: Record<string, unknown>;
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
  // (human_approval / human_help / freeform agent_question).
  // ``extra.explanation`` is the operator's required justification on
  // ``reject`` / ``abort`` for human_approval (matches the Pydantic
  // ``HumanApprovalResponse.explanation`` contract; the response
  // validator rejects empty explanation for those choices with 422).
  // ``extra.guidance`` is the operator's free-form direction for
  // human_help (matches the Pydantic
  // ``HumanHelpResponse.guidance`` contract; at least one of
  // ``chosen_option`` (passed via ``content``) and ``guidance``
  // must be non-empty — the validator surfaces 422 otherwise).
  onReply?: (
    message: ChatMessageData,
    content: string,
    extra?: { explanation?: string; guidance?: string },
  ) => void;
  // Whether to render the interactive response UI (option buttons,
  // compose textarea, guidance textarea). Default ``true`` —
  // :class:`ActiveRequestsOverlay` renders ChatMessage with the
  // default so the operator can act. :func:`ChatMessageList` passes
  // ``false`` so the timeline shows the question + (post-answer)
  // historical content without interactive controls.
  interactive?: boolean;
  // The session this message belongs to. When provided, agent messages
  // get a thumbs-up/down control. ``ChatMessageData`` itself carries no
  // session id, so the host (which knows it) threads it here.
  sessionId?: string | null;
}

// Left-rail dot color per (role, status_phase). The dot is the only
// per-entry color cue in the unified-timeline visual; cards + borders
// were removed (per 2026-06-24 chat-UI redesign — "looks better than
// those clunky cards for each message/status").
function dotClassFor(role: string, statusPhase?: string): string {
  if (statusPhase === "completed") return "bg-emerald-500";
  if (statusPhase === "failed") return "bg-red-500";
  if (statusPhase === "running") return "bg-blue-400 animate-pulse";
  if (role === "user") return "bg-primary";
  if (role === "agent") return "bg-emerald-400";
  if (role === "system") return "bg-zinc-400";
  return "bg-zinc-500";
}

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

export function ChatMessage({ message, onReply, interactive = true, sessionId }: ChatMessageProps) {
  const { role, content, agent_id, agent_type, username, timestamp, request_id, response_options, awaiting_reply, run_status, attachments, action_type, kind, extra, status_phase } = message;

  // When the operator clicks ``reject`` or ``abort``, switch the card
  // into compose mode and require a non-empty explanation before
  // submitting. ``null`` means no compose in progress (waiting for
  // the operator's initial choice).
  const [composeChoice, setComposeChoice] = useState<string | null>(null);
  const [explanationDraft, setExplanationDraft] = useState("");

  // ``human_help`` keeps a separate free-text draft. Distinct from the
  // ``human_approval`` explanation textarea (which only appears in
  // compose mode after reject/abort): for help requests the free-text
  // field is ALWAYS available because the operator may either pick an
  // ``options`` button OR write open guidance OR both. The Pydantic
  // ``HumanHelpResponse`` validator enforces "at least one
  // non-empty" — the UI submits empty guidance + an option, or
  // non-empty guidance + no option, or both; the empty/empty
  // submission is blocked by the disabled state below before it
  // reaches the server.
  const [guidanceDraft, setGuidanceDraft] = useState("");
  const isHumanHelp = kind === "human_help";
  const isStatus = kind === "status";
  const contextText = isHumanHelp && typeof extra?.context === "string"
    ? (extra.context as string)
    : "";

  const handleChoice = (option: string) => {
    if (!onReply) return;
    if (isHumanHelp) {
      // For human_help, picking an option ALSO sends whatever the
      // operator typed in the guidance textarea so both signals
      // travel together. The agent's translation step sees both.
      const guidance = guidanceDraft.trim();
      onReply(message, option, guidance ? { guidance } : undefined);
      setGuidanceDraft("");
      return;
    }
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

  const submitGuidanceOnly = () => {
    if (!onReply) return;
    const trimmed = guidanceDraft.trim();
    if (!trimmed) return;
    // ``content=""`` signals "no option picked, free-text only";
    // the host routes to the human_help endpoint which writes
    // ``chosen_option=null, guidance=<trimmed>``. The
    // ``HumanHelpResponse`` validator accepts since guidance is
    // non-empty.
    onReply(message, "", { guidance: trimmed });
    setGuidanceDraft("");
  };

  const cancelCompose = () => {
    setComposeChoice(null);
    setExplanationDraft("");
  };

  const dotClass = dotClassFor(role, status_phase);
  const phaseIcon =
    status_phase === "completed" ? <Check size={10} className="text-white" /> :
    status_phase === "failed" ? <XIcon size={10} className="text-white" /> :
    null;

  return (
    // Unified-timeline row: a left rail column (24px wide, vertical
    // line via ``before:`` pseudo-element on the parent container —
    // not here; here we provide the dot/marker for THIS entry) + a
    // right content column that holds header + body + attachments +
    // (if interactive) the response widgets.
    <div className={cn(
      "relative flex gap-2 py-1.5 pl-6",
      // Vertical rail: a 1px line at left=11 (so it visually bisects
      // the dot at left=8 w-2 = midpoint x=12). Rendered as a
      // pseudo-element so successive ChatMessage rows STACK their
      // rails and form one continuous spine without an explicit
      // parent-side container.
      "before:absolute before:left-[11px] before:top-0 before:bottom-0 before:w-px before:bg-border",
      // ``text-sm`` (14px) matches the size of the prose-sm body
      // below. Earlier this was ``text-xs`` (12px) on the wrapper +
      // a non-existent ``prose-xs`` modifier on the agent body — the
      // plugin silently fell back to default prose (16px), so agent
      // messages rendered LARGER than user. Both are now 14px;
      // status rows stay smaller via the explicit ``text-[11px]``.
      isStatus ? "text-[11px] text-muted-foreground" : "text-sm",
    )}>
      {/* Dot marker — colors discriminate role + status phase. The
          dot sits on top of the rail so it visually punctuates the
          spine. Status phases with terminal icons (completed=check,
          failed=X) render the icon inside a larger dot. */}
      <span
        className={cn(
          "absolute left-2 top-2 z-10 inline-flex h-2.5 w-2.5 items-center justify-center rounded-full ring-2 ring-card",
          dotClass,
        )}
        aria-hidden
      >
        {phaseIcon}
      </span>

      {/* Right column: header + content + (post-content) widgets. */}
      <div className="flex-1 min-w-0">
        {/* Header: role label, agent/user info, timestamp. Status
            entries collapse the header into a single muted line. */}
        {isStatus ? (
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
            {agent_id && (
              <span className="font-mono">{agent_id.slice(0, 12)}</span>
            )}
            <span className="ml-auto">{formatTime(timestamp)}</span>
          </div>
        ) : (
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
        )}

        {/* Run status badge */}
        {run_status && !isStatus && (
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

        {/* Content — render markdown for agent + status entries, plain
            text for user. User content sits in a bordered card so it
            visually pops off the rail (operator wants their own
            messages clearly distinct from the status stream). Agent +
            status are typography-only — the rail dot color is the
            distinguishing cue there. Markdown uses ``prose-sm`` (real
            14px variant) not the typo'd ``prose-xs`` (silent no-op
            that defaulted to 16px and made agent text larger than
            user). */}
        {role === "user" ? (
          <div className="rounded-md border border-primary/30 bg-primary/10 px-2.5 py-1.5 whitespace-pre-wrap leading-5">
            {content}
          </div>
        ) : isStatus ? (
          <div className="leading-5">{content}</div>
        ) : (
          <div className="prose prose-invert prose-sm max-w-none leading-5 [&_p]:my-1 [&_table]:text-[12px] [&_th]:px-2 [&_td]:px-2 [&_code]:text-[12px] [&_pre]:text-[12px]">
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

        {/* Thumbs on a finished agent response — training feedback,
            recorded against the producing INFER span. Suppressed for
            status entries and for questions still awaiting a reply. */}
        {role === "agent" && !isStatus && !awaiting_reply && sessionId && message.id && (
          <MessageFeedback sessionId={sessionId} messageId={message.id} />
        )}

        {/* Human-help context — what the agent has tried / observed
            before escalating. Renders above the response surface so
            the operator sees the situation that triggered the
            escalation before picking an option or writing guidance.
            Suppressed when ``interactive=false`` because the overlay
            owns the active-request UI; the timeline shows a clean
            historical record without the operator-action chrome. */}
        {interactive && isHumanHelp && contextText && (
          <div className="mt-2 rounded border border-zinc-700/40 bg-zinc-900/30 px-2 py-1.5">
            <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
              Agent context
            </div>
            <div className="whitespace-pre-wrap text-[11px] leading-5 text-zinc-300">
              {contextText}
            </div>
          </div>
        )}

        {/* Agent question response options. Vertical full-width stack
            so the overlay reads top-to-bottom even when the panel
            is narrow. Inline rendering in the timeline is suppressed
            via ``interactive=false`` — the operator acts in the
            overlay; the timeline records the question. */}
        {interactive && awaiting_reply && response_options && response_options.length > 0 && request_id && agent_id && onReply && composeChoice === null && (
          <div className="mt-2 flex flex-col gap-1.5">
            {response_options.map((option, i) => (
              <button
                key={i}
                onClick={() => handleChoice(option)}
                className={cn(
                  "w-full rounded border px-3 py-1.5 text-[11px] font-medium transition-colors text-left",
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

        {/* Human-help guidance textarea — ALWAYS visible alongside
            the option buttons (when awaiting_reply + kind=human_help).
            The operator may write free-form ``guidance`` instead of
            picking an option, OR write guidance AND pick an option
            (both signals reach the agent). At least one must be
            non-empty — the ``HumanHelpResponse`` Pydantic validator
            surfaces 422 if both are blank, so the Submit-guidance
            button is disabled until the textarea has trimmed content. */}
        {interactive && isHumanHelp && awaiting_reply && request_id && agent_id && onReply && (
          <div className="mt-2 flex flex-col gap-1.5">
            <div className="text-[10px] text-muted-foreground">
              {response_options && response_options.length > 0
                ? "Pick an option above, or write free-form guidance below — or both. The agent translates the reply into the missing parameter on its next iteration."
                : "Write free-form guidance for the agent. The agent translates the reply into the missing parameter on its next iteration."}
            </div>
            <textarea
              value={guidanceDraft}
              onChange={(e) => setGuidanceDraft(e.target.value)}
              placeholder="Free-form guidance (optional if you pick an option above)"
              rows={3}
              className="w-full rounded border border-zinc-700/40 bg-zinc-900/50 px-2 py-1 text-[11px] leading-5 focus:outline-none focus:border-primary/50"
            />
            <div className="flex gap-1.5">
              <button
                onClick={submitGuidanceOnly}
                disabled={!guidanceDraft.trim()}
                className={cn(
                  "w-full rounded border px-3 py-1.5 text-[11px] font-medium transition-colors",
                  "border-primary/30 bg-primary/10 text-primary hover:bg-primary/20",
                  "disabled:opacity-40 disabled:cursor-not-allowed",
                )}
              >
                Submit guidance
              </button>
            </div>
          </div>
        )}

        {/* Explanation compose mode — operator picked reject or abort; a
            non-empty justification is required before submission so the
            agent's next iteration can react to "why". */}
        {interactive && awaiting_reply && composeChoice !== null && request_id && agent_id && onReply && (
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
            <div className="flex flex-col gap-1.5">
              <button
                onClick={submitWithExplanation}
                disabled={!explanationDraft.trim()}
                className={cn(
                  "w-full rounded border px-3 py-1.5 text-[11px] font-medium transition-colors",
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
                className="w-full rounded border border-zinc-700/40 px-3 py-1.5 text-[11px] font-medium text-muted-foreground hover:bg-zinc-800/50"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
