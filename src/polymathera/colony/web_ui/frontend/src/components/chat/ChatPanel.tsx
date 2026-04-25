import { useState, useEffect, useRef, useCallback } from "react";
import { PanelRightClose, PanelRightOpen } from "lucide-react";
import { ChatMessageList } from "./ChatMessageList";
import { ChatInput } from "./ChatInput";
import { ChatControls, type ChatControlsState } from "./ChatControls";
import type { ChatMessageData } from "./ChatMessage";

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

type RunningAction = {
  action_id: string;
  action_key: string;
  agent_id?: string | null;
  started_at?: number | null;
};

export function ChatPanel({ sessionId, onTabActivity }: ChatPanelProps) {
  const [width, setWidth] = useState(getStoredWidth);
  const [collapsed, setCollapsed] = useState(getStoredCollapsed);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [chatControls, setChatControls] = useState<ChatControlsState>({});
  // action_id → running record. Cleared on a "complete" / "failed"
  // record from the same action_id.
  const [runningActions, setRunningActions] = useState<Record<string, RunningAction>>({});
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
            run_status: msg.run_status,
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
            run_status: m.run_status,
          }));
          if (historyMsgs.length > 0) {
            setMessages((prev) => [...historyMsgs, ...prev]);
          }
        } else if (data.type === "action_status") {
          // Action lifecycle: "running" adds a banner entry; the
          // matching "complete"/"failed" record removes it. Brief
          // out-of-order arrivals (very fast actions) self-resolve
          // because the start record only adds, the end only removes.
          const id = data.action_id as string | undefined;
          if (!id) return;
          if (data.status === "running") {
            setRunningActions((prev) => ({
              ...prev,
              [id]: {
                action_id: id,
                action_key: data.action_key || "",
                agent_id: data.agent_id,
                started_at: data.started_at,
              },
            }));
          } else {
            setRunningActions((prev) => {
              if (!(id in prev)) return prev;
              const next = { ...prev };
              delete next[id];
              return next;
            });
          }
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

  // Clear messages when session changes
  useEffect(() => {
    setMessages([]);
    setRunningActions({});
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

  const sendReply = useCallback((requestId: string, agentId: string, content: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    wsRef.current.send(JSON.stringify({ type: "reply", content, request_id: requestId, agent_id: agentId }));

    addMessage({
      id: nextMessageId(),
      run_id: null,
      role: "user",
      content,
      timestamp: Date.now(),
    });

    // Mark the question as answered
    setMessages((prev) => prev.map((m) =>
      m.request_id === requestId ? { ...m, awaiting_reply: false } : m
    ));
  }, [addMessage]);

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
          <button onClick={() => setCollapsed(true)} className="text-muted-foreground hover:text-foreground" title="Collapse chat">
            <PanelRightClose size={18} />
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 min-h-0">
          <ChatMessageList
            messages={messages}
            onReply={sendReply}
            emptyText={status === "connected" ? "Send a message to begin." : "Connecting..."}
          />
        </div>

        {/* Currently-running actions banner */}
        <ActionStatusBanner
          running={runningActions}
          onAbort={() => sendMessage("/abort")}
          canAbort={status === "connected"}
        />

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


function ActionStatusBanner({
  running,
  onAbort,
  canAbort,
}: {
  running: Record<string, RunningAction>;
  onAbort: () => void;
  canAbort: boolean;
}) {
  const entries = Object.values(running);
  if (entries.length === 0) return null;
  // Most recent first so the user sees the latest action at the top.
  entries.sort((a, b) => (b.started_at || 0) - (a.started_at || 0));
  return (
    <div className="flex items-start gap-2 border-t border-border bg-accent/30 px-3 py-2 text-xs">
      <div className="flex-1 min-w-0">
        {entries.map((a) => {
          const elapsed = a.started_at
            ? Math.max(0, Math.floor(Date.now() / 1000 - a.started_at))
            : null;
          return (
            <div key={a.action_id} className="flex items-center gap-2">
              <div
                className="h-3 w-3 shrink-0 animate-spin rounded-full border-2 border-primary border-t-transparent"
                aria-hidden
              />
              <span className="font-mono text-foreground">
                {_shortActionName(a.action_key)}
              </span>
              <span className="text-muted-foreground">
                running{elapsed !== null ? ` for ${elapsed}s` : ""}
                {a.agent_id ? ` · ${a.agent_id.slice(0, 12)}` : ""}
              </span>
            </div>
          );
        })}
      </div>
      {/* One Abort button per banner — abort cancels the *current* in-flight
          action on the agent. Banner shows multiple entries only when actions
          interleave (e.g., child policies); a single /abort propagates through
          policy.abort_current() which cancels whatever is at the top of the
          dispatcher's task stack, so a single button is the correct UX. */}
      <button
        type="button"
        onClick={onAbort}
        disabled={!canAbort}
        title="Abort the running action (sends /abort on the high-priority lane)"
        className="shrink-0 rounded border border-destructive/40 bg-destructive/10 px-2 py-0.5 text-xs font-medium text-destructive hover:bg-destructive/20 disabled:cursor-not-allowed disabled:opacity-40"
      >
        Abort
      </button>
    </div>
  );
}
