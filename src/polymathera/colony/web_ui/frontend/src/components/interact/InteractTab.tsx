import { useState, useEffect, useRef, useCallback } from "react";
import { Badge } from "../shared/Badge";

/* ── Types ─────────────────────────────────────────────────────── */

type MessageRole = "system" | "user" | "agent";

interface ChatMessage {
  role: MessageRole;
  content: string;
  timestamp: string;
  agentId?: string;
}

interface AgentInfo {
  agent_id: string;
  agent_type: string;
  state: string;
}

type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

const ROLE_STYLES: Record<MessageRole, string> = {
  system: "bg-zinc-800 text-zinc-400 italic",
  user: "bg-primary/10 text-foreground",
  agent: "bg-emerald-900/20 text-emerald-100",
};

/* ── WebSocket hook ────────────────────────────────────────────── */

function useAgentChat(sessionId: string | null) {
  const wsRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [agents, setAgents] = useState<AgentInfo[]>([]);

  const addMessage = useCallback((msg: ChatMessage) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  // Connect/disconnect based on sessionId
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
        role: "system",
        content: `Connected to session ${sessionId.slice(0, 16)}...`,
        timestamp: new Date().toLocaleTimeString(),
      });
      // Request agent list on connect
      ws.send(JSON.stringify({ type: "list_agents" }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "agents_list") {
          setAgents(data.agents || []);
        } else if (data.type === "agent_event") {
          const eventType = data.event_type;
          const agentId = data.agent_id || "unknown";
          const payload = data.data || {};

          let content: string;
          if (eventType === "completed") {
            const value = payload.value;
            content = typeof value === "string"
              ? value
              : value
                ? JSON.stringify(value, null, 2)
                : "Analysis complete.";
          } else if (eventType === "error") {
            content = `Error: ${payload.error || payload.message || "Unknown error"}`;
          } else if (eventType === "progress") {
            content = payload.message || JSON.stringify(payload);
          } else {
            content = `[${eventType}] ${JSON.stringify(payload)}`;
          }

          addMessage({
            role: "agent",
            content,
            timestamp: new Date().toLocaleTimeString(),
            agentId,
          });
        } else if (data.type === "error") {
          addMessage({
            role: "system",
            content: `Error: ${data.message}`,
            timestamp: new Date().toLocaleTimeString(),
          });
        } else if (data.type === "cancelled") {
          addMessage({
            role: "system",
            content: `Cancelled stream for agent ${data.agent_id}`,
            timestamp: new Date().toLocaleTimeString(),
          });
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onerror = () => {
      setStatus("error");
    };

    ws.onclose = () => {
      setStatus("disconnected");
      wsRef.current = null;
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [sessionId, addMessage]);

  const sendMessage = useCallback(
    (agentId: string, content: string, namespace?: string) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      wsRef.current.send(
        JSON.stringify({
          type: "message",
          agent_id: agentId,
          content,
          namespace: namespace || "",
        }),
      );

      addMessage({
        role: "user",
        content,
        timestamp: new Date().toLocaleTimeString(),
        agentId,
      });
    },
    [addMessage],
  );

  const refreshAgents = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "list_agents" }));
    }
  }, []);

  return { status, messages, agents, sendMessage, refreshAgents };
}

/* ── Chat View ──────────────────────────────────────────────────── */

interface ChatViewProps {
  sessionId: string | null;
}

function ChatView({ sessionId }: ChatViewProps) {
  const { status, messages, agents, sendMessage, refreshAgents } = useAgentChat(sessionId);
  const [input, setInput] = useState("");
  const [selectedAgent, setSelectedAgent] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-select first agent
  useEffect(() => {
    if (agents.length > 0 && !selectedAgent) {
      setSelectedAgent(agents[0].agent_id);
    }
  }, [agents, selectedAgent]);

  const handleSend = () => {
    if (!input.trim() || !selectedAgent) return;
    sendMessage(selectedAgent, input.trim());
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const statusColor = {
    connected: "bg-emerald-400",
    connecting: "bg-amber-400 animate-pulse",
    disconnected: "bg-zinc-500",
    error: "bg-red-400",
  }[status];

  return (
    <div className="flex h-full flex-col">
      {/* Agent selector + status */}
      <div className="flex items-center gap-2 mb-3">
        <span className={`inline-block h-2 w-2 rounded-full ${statusColor}`} />
        <select
          className="rounded border border-border bg-background px-2 py-1.5 text-xs font-mono min-w-[200px]"
          value={selectedAgent}
          onChange={(e) => setSelectedAgent(e.target.value)}
          disabled={agents.length === 0}
        >
          {agents.length === 0 && <option value="">No agents available</option>}
          {agents.map((a) => (
            <option key={a.agent_id} value={a.agent_id}>
              {a.agent_type ? `${a.agent_type.split(".").pop()} (${a.agent_id.slice(0, 12)})` : a.agent_id.slice(0, 24)}
            </option>
          ))}
        </select>
        <button
          onClick={refreshAgents}
          className="rounded border border-border px-2 py-1.5 text-xs text-muted-foreground hover:text-foreground"
          title="Refresh agent list"
        >
          Refresh
        </button>
        <Badge variant={status === "connected" ? "success" : status === "error" ? "error" : "default"}>
          {status}
        </Badge>
      </div>

      {/* Message list */}
      <div className="flex-1 min-h-0 overflow-auto space-y-2 rounded-lg border bg-[hsl(222,47%,5%)] p-3">
        {messages.length === 0 && (
          <div className="flex h-full items-center justify-center text-xs text-muted-foreground">
            {sessionId
              ? "Select an agent and send a message to begin."
              : "Select a session from the sidebar to start chatting."}
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`rounded-lg px-3 py-2 text-xs ${ROLE_STYLES[msg.role]}`}
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="font-semibold capitalize">{msg.role}</span>
              {msg.agentId && (
                <span className="font-mono text-[10px] text-muted-foreground">
                  {msg.agentId.slice(0, 12)}
                </span>
              )}
              <span className="text-[10px] text-muted-foreground">
                {msg.timestamp}
              </span>
            </div>
            <div className="whitespace-pre-wrap leading-5">{msg.content}</div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="mt-3 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            status === "connected"
              ? "Send a message to the selected agent..."
              : "Connect to a session to chat..."
          }
          disabled={status !== "connected" || !selectedAgent}
          className="flex-1 rounded border border-border bg-background px-3 py-2 text-xs focus:border-primary focus:outline-none disabled:opacity-50"
        />
        <button
          onClick={handleSend}
          disabled={status !== "connected" || !selectedAgent || !input.trim()}
          className="rounded bg-primary px-4 py-2 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  );
}

/* ── Main Component ─────────────────────────────────────────────── */

// Access activeSessionId from AppShell context
// For now, read from URL or a simple global. In practice, this should come
// from a React context that AppShell provides.
// Workaround: InteractTab doesn't receive sessionId as prop (existing tab
// architecture renders all tabs without props). We'll read from a simple
// module-level variable that AppShell sets.

let _activeSessionId: string | null = null;

/** Called by AppShell when session changes */
export function setInteractSessionId(id: string | null) {
  _activeSessionId = id;
}

export function InteractTab() {
  // Re-render when session changes — poll the module variable
  const [sessionId, setSessionId] = useState<string | null>(_activeSessionId);

  useEffect(() => {
    const interval = setInterval(() => {
      if (_activeSessionId !== sessionId) {
        setSessionId(_activeSessionId);
      }
    }, 500);
    return () => clearInterval(interval);
  }, [sessionId]);

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center gap-2">
        <h2 className="text-sm font-semibold">Agent Chat</h2>
        {sessionId && (
          <span className="font-mono text-xs text-muted-foreground">
            Session: {sessionId.slice(0, 16)}...
          </span>
        )}
      </div>
      <div className="flex-1 min-h-0">
        <ChatView sessionId={sessionId} />
      </div>
    </div>
  );
}
