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

export function ChatPanel({ sessionId, onTabActivity }: ChatPanelProps) {
  const [width, setWidth] = useState(getStoredWidth);
  const [collapsed, setCollapsed] = useState(getStoredCollapsed);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [messages, setMessages] = useState<ChatMessageData[]>([]);
  const [chatControls, setChatControls] = useState<ChatControlsState>({});
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
