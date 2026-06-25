import { useState, useEffect, useRef, useCallback } from "react";
import { PanelRightClose, PanelRightOpen, ShieldAlert } from "lucide-react";
import { ChatMessageList } from "./ChatMessageList";
import { ChatInput } from "./ChatInput";
import { ChatControls, type ChatControlsState } from "./ChatControls";
import { ConstraintsPanel } from "./ConstraintsPanel";
import { ActiveRequestsOverlay } from "./ActiveRequestsOverlay";
import type { ChatMessageData } from "./ChatMessage";
import { apiFetch } from "@/api/client";

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

interface ChatPanelProps {
  sessionId: string | null;
  /** Callback when a tab_activity event arrives from the WebSocket */
  onTabActivity?: (tabId: string, count: number) => void;
}

const MIN_WIDTH = 300;
const MAX_WIDTH_RATIO = 0.6;
const STORAGE_KEY_WIDTH = "colony_chat_width";
const STORAGE_KEY_COLLAPSED = "colony_chat_collapsed";

function getStoredWidth(): number {
  const stored = localStorage.getItem(STORAGE_KEY_WIDTH);
  return stored ? Math.max(MIN_WIDTH, parseInt(stored, 10)) : 380;
}

function getStoredCollapsed(): boolean {
  return localStorage.getItem(STORAGE_KEY_COLLAPSED) === "true";
}

let messageIdCounter = 0;
function nextMessageId(): string {
  return `msg_${Date.now()}_${++messageIdCounter}`;
}

export function ChatPanel({ sessionId, onTabActivity }: ChatPanelProps) {
  const [width, setWidth] = useState(getStoredWidth);
  const [collapsed, setCollapsed] = useState(getStoredCollapsed);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [chatControls, setChatControls] = useState<ChatControlsState>({});
  // PR5-B operator constraint-override panel visibility.
  const [showConstraints, setShowConstraints] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const isDragging = useRef(false);

  // Persist layout state
  useEffect(() => { localStorage.setItem(STORAGE_KEY_WIDTH, String(width)); }, [width]);
  useEffect(() => { localStorage.setItem(STORAGE_KEY_COLLAPSED, String(collapsed)); }, [collapsed]);

  const addMessage = useCallback((msg: ChatMessageData) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  // WebSocket connection lifecycle
  useEffect(() => {
    if (!sessionId) {
      setStatus("disconnected");
      return;
    }

    setStatus("connecting");
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${window.location.host}/api/v1/ws/chat/${sessionId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      addMessage({
        id: nextMessageId(),
        run_id: null,
        role: "system",
        content: `Connected to session ${sessionId.slice(0, 16)}...`,
        timestamp: Date.now(),
      });
      ws.send(JSON.stringify({ type: "list_agents" }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "agent_event" || data.type === "message") {
          const msg = data.message || data;
          const eventType = msg.event_type || data.event_type;
          const agentId = msg.agent_id || data.agent_id || "unknown";

          let content: string;
          if (eventType === "completed") {
            const value = msg.data?.value ?? msg.value;
            content = typeof value === "string" ? value : value ? JSON.stringify(value, null, 2) : "Complete.";
          } else if (eventType === "error") {
            const payload = msg.data || msg;
            content = `Error: ${payload.error || payload.message || "Unknown error"}`;
          } else if (eventType === "progress") {
            const payload = msg.data || msg;
            content = payload.message || JSON.stringify(payload);
          } else if (msg.content) {
            content = msg.content;
          } else {
            content = `[${eventType || data.type}] ${JSON.stringify(msg.data || msg)}`;
          }

          addMessage({
            id: msg.id || nextMessageId(),
            run_id: msg.run_id || null,
            role: "agent",
            agent_id: agentId,
            agent_type: msg.agent_type,
            content,
            timestamp: msg.timestamp || Date.now(),
            request_id: msg.request_id,
            response_options: msg.response_options,
            awaiting_reply: msg.awaiting_reply,
            kind: msg.kind,
            action_type: msg.action_type,
            extra: msg.extra,
            run_status: msg.run_status,
            attachments: msg.attachments,
          });
        } else if (data.type === "agent_question") {
          const msg = data.message;
          addMessage({
            id: msg.id || nextMessageId(),
            run_id: msg.run_id || null,
            role: "agent",
            agent_id: msg.agent_id,
            agent_type: msg.agent_type,
            content: msg.content,
            timestamp: msg.timestamp || Date.now(),
            request_id: msg.request_id,
            response_options: msg.response_options,
            awaiting_reply: true,
            kind: msg.kind,
            action_type: msg.action_type,
            extra: msg.extra,
            attachments: msg.attachments,
          });
        } else if (data.type === "history") {
          // History messages from server (on connect or pagination request)
          const historyMsgs: ChatMessageData[] = (data.messages || []).map((m: any) => ({
            id: m.id || nextMessageId(),
            run_id: m.run_id || null,
            role: m.role || "system",
            agent_id: m.agent_id,
            agent_type: m.agent_type,
            user_id: m.user_id,
            username: m.username,
            content: m.content || "",
            timestamp: (m.timestamp || 0) * 1000,  // server sends seconds, frontend uses ms
            request_id: m.request_id,
            response_options: m.response_options,
            awaiting_reply: m.awaiting_reply,
            kind: m.kind,
            action_type: m.action_type,
            extra: m.extra,
            run_status: m.run_status,
            attachments: m.attachments,
          }));
          if (historyMsgs.length > 0) {
            setMessages((prev) => [...historyMsgs, ...prev]);
          }
        } else if (data.type === "action_status") {
          // Action lifecycle becomes a single timeline entry per
          // ``action_id``. ``running`` appends a new entry with
          // ``status_phase=running`` (spinner dot); ``complete`` /
          // ``failed`` UPDATE that entry in place to the terminal
          // phase. Stable id ``status_action_<action_id>`` is the
          // join key — any record arriving with the same id mutates
          // the same row. Brief out-of-order arrivals self-resolve
          // because the update path is idempotent on the id.
          const id = data.action_id as string | undefined;
          if (!id) return;
          const statusId = `status_action_${id}`;
          if (data.status === "running") {
            // First-seen running record — append. If a duplicate
            // arrived (e.g. WebSocket replay), short-circuit so we
            // don't show two entries.
            setMessages((prev) => {
              if (prev.some((m) => m.id === statusId)) return prev;
              return [
                ...prev,
                {
                  id: statusId,
                  run_id: null,
                  role: "system",
                  kind: "status",
                  status_phase: "running",
                  content: _shortActionName(data.action_key || ""),
                  agent_id: data.agent_id ?? undefined,
                  timestamp: (data.started_at ? data.started_at * 1000 : Date.now()),
                },
              ];
            });
          } else {
            // Terminal phase — flip the existing entry's phase + dot
            // color. If the running record never arrived (very fast
            // action whose start was missed), synthesise a terminal
            // entry rather than silently dropping the signal.
            const phase = data.status === "complete" ? "completed" : "failed";
            setMessages((prev) => {
              if (prev.some((m) => m.id === statusId)) {
                return prev.map((m) =>
                  m.id === statusId ? { ...m, status_phase: phase } : m,
                );
              }
              return [
                ...prev,
                {
                  id: statusId,
                  run_id: null,
                  role: "system",
                  kind: "status",
                  status_phase: phase,
                  content: _shortActionName(data.action_key || ""),
                  agent_id: data.agent_id ?? undefined,
                  timestamp: Date.now(),
                },
              ];
            });
          }
        } else if (data.type === "mission_status") {
          // Mission narrative becomes one timeline entry per emitted
          // message. The prior banner shape collapsed history to
          // "latest only per mission_id"; the unified-timeline shape
          // is append-per-event so the operator sees the arc of the
          // coordinator's work. ``cleared`` payloads are no-ops at
          // the timeline layer (entries are historical; the backend
          // relay's clearing semantics applied to a singleton banner
          // that no longer exists). Each entry's id is unique per
          // (mission_id, timestamp) so duplicate replays don't
          // double-render.
          const missionId = data.mission_id as string | undefined;
          if (!missionId) return;
          if (data.cleared) return;
          const ts = typeof data.timestamp === "number"
            ? data.timestamp
            : Date.now() / 1000;
          const statusId = `status_mission_${missionId}_${ts}`;
          const text = typeof data.message === "string" ? data.message : "";
          setMessages((prev) => {
            if (prev.some((m) => m.id === statusId)) return prev;
            return [
              ...prev,
              {
                id: statusId,
                run_id: null,
                role: "system",
                kind: "status",
                // Mission narratives are informational; no
                // start/end lifecycle to drive ``status_phase``.
                // The renderer falls back to a neutral dot.
                content: text,
                agent_id: data.agent_id ?? undefined,
                timestamp: ts * 1000,
              },
            ];
          });
        } else if (data.type === "tab_activity" && onTabActivity) {
          onTabActivity(data.tab_id, data.count);
        } else if (data.type === "error") {
          addMessage({
            id: nextMessageId(),
            run_id: null,
            role: "system",
            content: `Error: ${data.message}`,
            timestamp: Date.now(),
          });
        } else if (data.type === "cancelled") {
          addMessage({
            id: nextMessageId(),
            run_id: null,
            role: "system",
            content: `Cancelled stream for agent ${data.agent_id}`,
            timestamp: Date.now(),
          });
        }
        // Ignore agents_list and other non-chat messages silently
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onerror = () => setStatus("error");

    ws.onclose = () => {
      setStatus("disconnected");
      wsRef.current = null;
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [sessionId, addMessage, onTabActivity]);

  // Clear messages when session changes. Status entries live IN the
  // ``messages`` array now (unified-timeline redesign), so a single
  // ``setMessages([])`` wipes both chat history AND inline status
  // history — no separate ``runningActions`` / ``missionStatuses``
  // state to clear.
  useEffect(() => {
    setMessages([]);
  }, [sessionId]);

  const sendMessage = useCallback((content: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    // Include controls only if any are set
    const hasControls = chatControls.vcm_context || chatControls.agent_preferences || chatControls.effort || chatControls.timeout_seconds || chatControls.budget_usd != null;
    wsRef.current.send(JSON.stringify({
      type: "message",
      content,
      ...(hasControls ? { controls: chatControls } : {}),
    }));

    addMessage({
      id: nextMessageId(),
      run_id: null,
      role: "user",
      content,
      timestamp: Date.now(),
    });
  }, [addMessage, chatControls]);

  const sendReply = useCallback((
    message: ChatMessageData,
    content: string,
    extra?: { explanation?: string; guidance?: string },
  ) => {
    const requestId = message.request_id;
    const agentId = message.agent_id;
    if (!requestId || !agentId) return;

    // Echo the user's choice into the chat history. For ``reject`` /
    // ``abort`` we include the operator's explanation in the echo so
    // the chat log records the rationale alongside the choice.
    const echoContent = extra?.explanation
      ? `${content}: ${extra.explanation}`
      : content;
    const recordEcho = () => {
      addMessage({
        id: nextMessageId(),
        run_id: null,
        role: "user",
        content: echoContent,
        timestamp: Date.now(),
      });
      setMessages((prev) => prev.map((m) =>
        m.request_id === requestId ? { ...m, awaiting_reply: false } : m
      ));
    };

    if (message.kind === "human_approval" && sessionId) {
      // Typed human-approval gate. The agent's HumanApprovalCapability
      // listens on the SESSION blackboard's
      // ``human_approval:response:*`` topic; the HTTP endpoint writes
      // there. The WebSocket reply lane is for freeform questions
      // only — using it here would route into the wrong listener.
      // ``explanation`` is the operator's justification (non-empty on
      // reject/abort, empty otherwise); the backend's
      // ``HumanApprovalResponse`` validator surfaces 422 if a reject
      // or abort arrives empty.
      apiFetch<unknown>(
        `/sessions/${encodeURIComponent(sessionId)}/human_approval/${encodeURIComponent(requestId)}/respond`,
        {
          method: "POST",
          body: JSON.stringify({
            choice: content,
            explanation: extra?.explanation ?? "",
          }),
        },
      ).then(
        () => recordEcho(),
        (err) => {
          addMessage({
            id: nextMessageId(),
            run_id: null,
            role: "system",
            content: `Failed to submit approval: ${err instanceof Error ? err.message : String(err)}`,
            timestamp: Date.now(),
          });
        },
      );
      return;
    }

    if (message.kind === "guardrail_waiver" && sessionId) {
      // Agent-initiated waiver request on a semantic guardrail rule.
      // The asking agent's ``GuardrailWaiverCapability`` listens on
      // ``guardrail_waiver:response:*``; the HTTP endpoint writes
      // there AND on approve ALSO writes the
      // ``operator_override:semantic_constraint:<cid>`` key the
      // existing ``SemanticConstraintGuardrail._read_disabled_ids``
      // reads — same plumbing as the dashboard's disable button. The
      // ``constraint_id`` round-trips via ``extra`` from the original
      // request. ``reason`` is the operator's optional rationale
      // (mapped from the same compose-mode field the human_approval
      // card surfaces on reject/abort).
      const verb = content === "approve" ? "approve" : "reject";
      // ``constraint_id`` round-trips via the ORIGINAL waiver request's
      // ``message.extra`` (the SessionOrchestrator's request-mirror
      // stamps it there). The reply ``extra`` carries the operator's
      // free-form rationale only.
      const constraintId = typeof message.extra?.constraint_id === "string"
        ? (message.extra.constraint_id as string)
        : "";
      apiFetch<unknown>(
        `/sessions/${encodeURIComponent(sessionId)}/waivers/${encodeURIComponent(requestId)}/${verb}`,
        {
          method: "POST",
          body: JSON.stringify({
            constraint_id: constraintId,
            reason: extra?.explanation ?? "",
          }),
        },
      ).then(
        () => recordEcho(),
        (err) => {
          addMessage({
            id: nextMessageId(),
            run_id: null,
            role: "system",
            content: `Failed to submit waiver decision: ${err instanceof Error ? err.message : String(err)}`,
            timestamp: Date.now(),
          });
        },
      );
      return;
    }

    if (message.kind === "human_help" && sessionId) {
      // Typed human-help (mid-flow clarification) gate. Sibling of
      // the human_approval branch above. The requesting agent's
      // ``HumanHelpCapability`` listens on the SESSION blackboard's
      // ``human_help:response:*`` topic; the HTTP endpoint writes
      // there. ``content`` carries the operator's picked option (or
      // ``""`` when they submitted free-text only); ``extra.guidance``
      // carries the free-form text. At least one must be non-empty —
      // the ``HumanHelpResponse`` Pydantic validator surfaces 422
      // otherwise.
      const chosenOption = content === "" ? null : content;
      apiFetch<unknown>(
        `/sessions/${encodeURIComponent(sessionId)}/human_help/${encodeURIComponent(requestId)}/respond`,
        {
          method: "POST",
          body: JSON.stringify({
            chosen_option: chosenOption,
            guidance: extra?.guidance ?? "",
          }),
        },
      ).then(
        () => recordEcho(),
        (err) => {
          addMessage({
            id: nextMessageId(),
            run_id: null,
            role: "system",
            content: `Failed to submit help response: ${err instanceof Error ? err.message : String(err)}`,
            timestamp: Date.now(),
          });
        },
      );
      return;
    }

    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    wsRef.current.send(JSON.stringify({ type: "reply", content, request_id: requestId, agent_id: agentId }));
    recordEcho();
  }, [addMessage, sessionId]);

  // Resize drag handler
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;

    const startX = e.clientX;
    const startWidth = width;
    const maxWidth = window.innerWidth * MAX_WIDTH_RATIO;

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      // Dragging left edge → moving left increases width
      const delta = startX - e.clientX;
      const newWidth = Math.min(maxWidth, Math.max(MIN_WIDTH, startWidth + delta));
      setWidth(newWidth);
    };

    const handleMouseUp = () => {
      isDragging.current = false;
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };

    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  }, [width]);

  const statusDot = {
    connected: "bg-emerald-400",
    connecting: "bg-amber-400 animate-pulse",
    disconnected: "bg-zinc-500",
    error: "bg-red-400",
  }[status];

  // Collapsed state: thin strip with toggle
  if (collapsed) {
    return (
      <div className="flex w-10 shrink-0 flex-col items-center border-l border-border bg-card py-3">
        <button
          onClick={() => setCollapsed(false)}
          className="text-muted-foreground hover:text-foreground"
          title="Expand chat"
        >
          <PanelRightOpen size={18} />
        </button>
        {status === "connected" && (
          <span className={`mt-2 inline-block h-2 w-2 rounded-full ${statusDot}`} />
        )}
      </div>
    );
  }

  // No session: show placeholder
  if (!sessionId) {
    return (
      <div
        className="flex shrink-0 flex-col border-l border-border bg-card"
        style={{ width }}
      >
        <div className="flex items-center justify-between border-b border-border px-3 py-2.5">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Chat</span>
          <button onClick={() => setCollapsed(true)} className="text-muted-foreground hover:text-foreground" title="Collapse chat">
            <PanelRightClose size={18} />
          </button>
        </div>
        <div className="flex flex-1 items-center justify-center text-xs text-muted-foreground p-4 text-center">
          Select or create a session to start chatting.
        </div>
      </div>
    );
  }

  return (
    <div className="flex shrink-0" style={{ width }}>
      {/* Resize handle */}
      <div
        onMouseDown={handleMouseDown}
        className="w-1 cursor-col-resize hover:bg-primary/30 active:bg-primary/50 transition-colors"
      />

      {/* Panel content */}
      <div className="flex flex-1 flex-col border-l border-border bg-card min-w-0">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-3 py-2.5">
          <div className="flex items-center gap-2">
            <span className={`inline-block h-2 w-2 rounded-full ${statusDot}`} />
            <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Chat</span>
          </div>
          <div className="flex items-center gap-2">
            {/* PR5-B: operator runtime constraint override. Opens the
                ConstraintsPanel modal which lists the SessionAgent's
                semantic constraints with toggle buttons. */}
            {sessionId && (
              <button
                onClick={() => setShowConstraints(true)}
                className="text-muted-foreground hover:text-foreground"
                title="Semantic constraints (operator override)"
              >
                <ShieldAlert size={16} />
              </button>
            )}
            <button onClick={() => setCollapsed(true)} className="text-muted-foreground hover:text-foreground" title="Collapse chat">
              <PanelRightClose size={18} />
            </button>
          </div>
        </div>
        {showConstraints && sessionId && (
          <ConstraintsPanel
            sessionId={sessionId}
            onClose={() => setShowConstraints(false)}
          />
        )}

        {/* Messages — unified timeline (status entries inline). */}
        <div className="flex-1 min-h-0">
          <ChatMessageList
            messages={messages}
            onReply={sendReply}
            emptyText={status === "connected" ? "Send a message to begin." : "Connecting..."}
          />
        </div>

        {/* Active operator requests (approval / help / waiver) pinned
            above the input. Source-of-truth is ``messages`` — the
            overlay filters to entries with ``awaiting_reply=true``
            and renders them with ``interactive=true`` so the operator
            can answer without scrolling. Auto-scroll cannot bury
            them; the timeline shows the same entries as historical
            context with ``interactive=false``. */}
        <ActiveRequestsOverlay messages={messages} onReply={sendReply} />

        {/* Controls + Input */}
        <ChatControls controls={chatControls} onChange={setChatControls} />
        <ChatInput
          onSend={sendMessage}
          disabled={status !== "connected"}
          placeholder={status === "connected" ? "Type a message, /command, or @agent..." : "Connecting..."}
        />
      </div>
    </div>
  );
}


function _shortActionName(actionKey: string): string {
  // GitHubCapability.GitHubCapability.list_issues  →  list_issues
  // VCMCapability.VCMCapability.mmap_repo          →  mmap_repo
  // signal_completion                              →  signal_completion
  if (!actionKey) return "(unknown)";
  const parts = actionKey.split(".");
  return parts[parts.length - 1] || actionKey;
}


