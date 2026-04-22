import { cn } from "@/lib/utils";

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
  // Run lifecycle
  run_status?: "submitted" | "spawning" | "running" | "completed" | "failed";
}

interface ChatMessageProps {
  message: ChatMessageData;
  onReply?: (requestId: string, agentId: string, content: string) => void;
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
  const { role, content, agent_id, agent_type, username, timestamp, request_id, response_options, awaiting_reply, run_status } = message;

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

      {/* Content */}
      <div className="whitespace-pre-wrap leading-5">{content}</div>

      {/* Agent question response options */}
      {awaiting_reply && response_options && response_options.length > 0 && request_id && agent_id && onReply && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {response_options.map((option, i) => (
            <button
              key={i}
              onClick={() => onReply(request_id, agent_id, option)}
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
