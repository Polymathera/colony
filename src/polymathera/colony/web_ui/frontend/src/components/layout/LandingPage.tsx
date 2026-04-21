import { Plus } from "lucide-react";
import { useHealthStatus } from "@/api/hooks/useInfrastructure";
import { useSessions } from "@/api/hooks/useSessions";
import { Badge } from "../shared/Badge";
import { formatTimestamp } from "@/lib/utils";

interface LandingPageProps {
  onSelectSession: (sessionId: string) => void;
  onCreateSession: () => void;
}

export function LandingPage({ onSelectSession, onCreateSession }: LandingPageProps) {
  const health = useHealthStatus();
  const sessions = useSessions();

  // Recent sessions (last 5, newest first)
  const recentSessions = (sessions.data ?? [])
    .sort((a, b) => b.created_at - a.created_at)
    .slice(0, 5);

  return (
    <div className="flex h-full flex-col items-center justify-center gap-8">
      {/* Logo */}
      <div className="text-center">
        <h1
          className="text-5xl font-bold tracking-widest text-primary"
          style={{ fontFamily: "'Orbitron', sans-serif" }}
        >
          THE COLONY
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Civilization-Building AI
        </p>
        <p className="mt-4 text-[10px] uppercase tracking-widest text-muted-foreground/60">
          By Polymathera
        </p>
      </div>

      {/* Cluster status */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              health.data?.ray_connected ? "bg-emerald-400 animate-pulse" : "bg-red-400"
            }`}
          />
          <span className="text-xs text-muted-foreground">
            {health.data?.ray_connected ? "Cluster Connected" : "Disconnected"}
          </span>
        </div>
        {health.data?.node_count != null && (
          <span className="text-xs text-muted-foreground">
            {health.data.node_count} node{health.data.node_count !== 1 ? "s" : ""}
          </span>
        )}
        <Badge variant={health.data?.ray_connected ? "success" : "error"}>
          {health.data?.ray_cluster_status ?? "unknown"}
        </Badge>
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={onCreateSession}
          className="rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          <Plus size={16} className="inline -mt-0.5" /> New Session
        </button>
      </div>

      {/* Recent sessions */}
      {recentSessions.length > 0 && (
        <div className="w-full max-w-md">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Recent Sessions
          </h3>
          <div className="rounded-lg border border-border bg-card divide-y divide-border">
            {recentSessions.map((session) => (
              <button
                key={session.session_id}
                onClick={() => onSelectSession(session.session_id)}
                className="flex w-full items-center justify-between px-4 py-2.5 text-left hover:bg-accent/50 transition-colors"
              >
                <div>
                  <span className="font-mono text-xs text-foreground">
                    {session.session_id.slice(0, 20)}...
                  </span>
                  <div className="text-[10px] text-muted-foreground">
                    {formatTimestamp(session.created_at)}
                    {session.run_count > 0 && ` \u00b7 ${session.run_count} runs`}
                  </div>
                </div>
                <Badge
                  variant={
                    session.state === "active" ? "success" :
                    session.state === "suspended" ? "warning" : "default"
                  }
                  className="text-[9px]"
                >
                  {session.state}
                </Badge>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
