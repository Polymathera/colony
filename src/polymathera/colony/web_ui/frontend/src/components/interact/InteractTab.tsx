import { useState, useMemo } from "react";
import { Badge } from "../shared/Badge";

/* ── Placeholder data ───────────────────────────────────────── */

type MessageRole = "system" | "user" | "agent";

interface ChatMessage {
  role: MessageRole;
  content: string;
  timestamp: string;
}

const PLACEHOLDER_MESSAGES: ChatMessage[] = [
  {
    role: "system",
    content: "Connected to Colony cluster. 3 agents available.",
    timestamp: "10:30:01",
  },
  {
    role: "user",
    content: "Analyze the authentication module for security issues",
    timestamp: "10:30:15",
  },
  {
    role: "agent",
    content:
      "I'll analyze the authentication module. Loading relevant pages...\n\n" +
      "[Loaded 12 pages from scope auth-service]\n\n" +
      "Found 3 key areas to review:\n" +
      "1. Token validation in `auth/jwt.py` — expiry check is missing for refresh tokens\n" +
      "2. Password hashing in `auth/crypto.py` — uses bcrypt with cost=10 (recommend 12)\n" +
      "3. Session management in `auth/sessions.py` — no rate limiting on login attempts",
    timestamp: "10:30:42",
  },
  {
    role: "user",
    content: "Can you fix the token validation issue?",
    timestamp: "10:31:05",
  },
  {
    role: "agent",
    content:
      "I'll add expiry validation for refresh tokens. Modifying `auth/jwt.py`...\n\n" +
      "```python\n" +
      "def validate_refresh_token(token: str) -> bool:\n" +
      "    payload = decode_token(token)\n" +
      "    if payload.exp < time.time():\n" +
      "        raise TokenExpiredError('Refresh token expired')\n" +
      "    return True\n" +
      "```\n\n" +
      "Change applied. Running tests... 47/47 passed.",
    timestamp: "10:31:28",
  },
];

const PLACEHOLDER_AGENTS = [
  "Analysis Agent",
  "Research Agent",
  "Coordinator",
  "Code Review Agent",
];

const PLACEHOLDER_CAPABILITIES = [
  "code_analysis",
  "page_graph",
  "web_search",
  "file_operations",
  "blackboard_access",
  "planning",
  "tool_use",
];

const ROLE_STYLES: Record<MessageRole, string> = {
  system: "bg-zinc-800 text-zinc-400 italic",
  user: "bg-primary/10 text-foreground",
  agent: "bg-emerald-900/20 text-emerald-100",
};

/* ── Chat View ──────────────────────────────────────────────── */

function ChatView() {
  const [input, setInput] = useState("");

  return (
    <div className="flex h-full flex-col">
      {/* Agent selector */}
      <div className="flex items-center gap-2 mb-3">
        <select
          className="rounded border border-border bg-background px-2 py-1.5 text-xs font-mono"
          disabled
        >
          {PLACEHOLDER_AGENTS.map((a) => (
            <option key={a}>{a}</option>
          ))}
        </select>
        <Badge variant="warning">Coming Soon</Badge>
      </div>

      {/* Message list */}
      <div className="flex-1 min-h-0 overflow-auto space-y-2 rounded-lg border bg-[hsl(222,47%,5%)] p-3">
        {PLACEHOLDER_MESSAGES.map((msg, i) => (
          <div
            key={i}
            className={`rounded-lg px-3 py-2 text-xs ${ROLE_STYLES[msg.role]}`}
          >
            <div className="flex items-center gap-2 mb-1">
              <span className="font-semibold capitalize">{msg.role}</span>
              <span className="text-[10px] text-muted-foreground">
                {msg.timestamp}
              </span>
            </div>
            <div className="whitespace-pre-wrap leading-5">{msg.content}</div>
          </div>
        ))}
      </div>

      {/* Input */}
      <div className="mt-3 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Send a message to an agent..."
          className="flex-1 rounded border border-border bg-background px-3 py-2 text-xs"
          disabled
        />
        <button
          disabled
          className="rounded bg-primary/30 px-4 py-2 text-xs font-medium text-primary/50 cursor-not-allowed"
        >
          Send
        </button>
      </div>
    </div>
  );
}

/* ── Blueprint Builder View ─────────────────────────────────── */

function BlueprintView() {
  const [agentType, setAgentType] = useState("analysis");
  const [capabilities, setCapabilities] = useState<Set<string>>(
    new Set(["code_analysis", "page_graph"]),
  );
  const [instructions, setInstructions] = useState(
    "Analyze the codebase for security vulnerabilities and code quality issues.",
  );
  const [maxRuns, setMaxRuns] = useState(3);

  const toggleCapability = (cap: string) => {
    setCapabilities((prev) => {
      const next = new Set(prev);
      if (next.has(cap)) next.delete(cap);
      else next.add(cap);
      return next;
    });
  };

  const blueprint = useMemo(
    () =>
      JSON.stringify(
        {
          agent_type: agentType,
          capabilities: [...capabilities],
          instructions,
          max_concurrent_runs: maxRuns,
          auto_spawn: true,
        },
        null,
        2,
      ),
    [agentType, capabilities, instructions, maxRuns],
  );

  return (
    <div className="flex gap-4 h-full">
      {/* Form */}
      <div className="flex-1 space-y-4">
        <div>
          <label className="block text-xs font-medium text-muted-foreground mb-1">
            Agent Type
          </label>
          <select
            className="w-full rounded border border-border bg-background px-2 py-1.5 text-xs font-mono"
            value={agentType}
            onChange={(e) => setAgentType(e.target.value)}
          >
            <option value="analysis">Analysis Agent</option>
            <option value="research">Research Agent</option>
            <option value="coordinator">Coordinator Agent</option>
            <option value="worker">Worker Agent</option>
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium text-muted-foreground mb-1">
            Capabilities
          </label>
          <div className="flex flex-wrap gap-1.5">
            {PLACEHOLDER_CAPABILITIES.map((cap) => (
              <button
                key={cap}
                onClick={() => toggleCapability(cap)}
                className={`rounded px-2 py-1 text-[10px] font-medium transition-colors ${
                  capabilities.has(cap)
                    ? "bg-primary/20 text-primary"
                    : "bg-muted text-muted-foreground"
                }`}
              >
                {cap}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-xs font-medium text-muted-foreground mb-1">
            Instructions
          </label>
          <textarea
            className="w-full rounded border border-border bg-background px-2 py-1.5 text-xs font-mono h-24 resize-none"
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-muted-foreground mb-1">
            Max Concurrent Runs
          </label>
          <input
            type="number"
            min={1}
            max={10}
            value={maxRuns}
            onChange={(e) => setMaxRuns(Number(e.target.value))}
            className="w-24 rounded border border-border bg-background px-2 py-1.5 text-xs font-mono"
          />
        </div>

        <div className="flex items-center gap-2">
          <button
            disabled
            className="rounded bg-primary/30 px-4 py-2 text-xs font-medium text-primary/50 cursor-not-allowed"
          >
            Deploy Agent
          </button>
          <Badge variant="warning">Coming Soon</Badge>
        </div>
      </div>

      {/* JSON Preview */}
      <div className="w-80 shrink-0">
        <div className="text-xs font-medium text-muted-foreground mb-1">
          Blueprint Preview
        </div>
        <pre className="rounded-lg border bg-[hsl(222,47%,5%)] p-3 text-[11px] font-mono text-emerald-300 overflow-auto h-full">
          {blueprint}
        </pre>
      </div>
    </div>
  );
}

/* ── Main Component ─────────────────────────────────────────── */

export function InteractTab() {
  const [view, setView] = useState<"chat" | "blueprint">("chat");

  return (
    <div className="flex h-full flex-col gap-3">
      {/* View toggle */}
      <div className="flex items-center gap-2">
        <div className="flex rounded border border-border">
          <button
            className={`px-3 py-1.5 text-xs font-medium transition-colors ${
              view === "chat"
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
            onClick={() => setView("chat")}
          >
            Agent Chat
          </button>
          <button
            className={`px-3 py-1.5 text-xs font-medium transition-colors ${
              view === "blueprint"
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
            onClick={() => setView("blueprint")}
          >
            Blueprint Builder
          </button>
        </div>
        <Badge variant="warning">Preview</Badge>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0">
        {view === "chat" ? <ChatView /> : <BlueprintView />}
      </div>
    </div>
  );
}
