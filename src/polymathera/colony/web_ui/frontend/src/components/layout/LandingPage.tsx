import { Plus } from "lucide-react";
import { useHealthStatus } from "@/api/hooks/useInfrastructure";
import { useSessions } from "@/api/hooks/useSessions";
import { Badge } from "../shared/Badge";
import { ColoniesSection } from "./ColoniesSection";
import { ColonyStatusPanel } from "./ColonyStatusPanel";
import {
  TenantGitHubInstallationSection,
  UserGitHubIdentitySection,
} from "./GitHubIdentitySection";
import { formatTimestamp } from "@/lib/utils";

interface LandingPageProps {
  onSelectSession: (sessionId: string) => void;
  onCreateSession: () => void;
  clusterReady: boolean;
  activeColonyId: string | null;
  onSelectColony: (colonyId: string) => void;
}

export function LandingPage({
  onSelectSession,
  onCreateSession,
  clusterReady,
  activeColonyId,
  onSelectColony,
}: LandingPageProps) {
  const health = useHealthStatus();
  const sessions = useSessions();

  // Recent sessions (last 5, newest first)
  const recentSessions = (sessions.data ?? [])
    .sort((a, b) => b.created_at - a.created_at)
    .slice(0, 5);

  // The session button has two distinct precondition failures we
  // were previously collapsing into a single "Waiting for cluster…"
  // label: (a) no colony is active (discovery returned zero, or the
  // operator hasn't picked one yet); (b) the cluster's deployments
  // aren't ready. Name the actual blocker so the operator knows
  // whether to wait on infra or to pick/create a colony.
  const clusterInfraReady =
    health.isSuccess && !!health.data?.deployments_ready;
  const sessionButtonLabel = !activeColonyId
    ? "Pick or create a colony first"
    : !clusterInfraReady
      ? "Waiting for cluster…"
      : "New Session";

  return (
    // ``overflow-y-auto`` because the stack (colonies, GitHub
    // installation, GitHub identity, colony status with activity,
    // actions, recent sessions) routinely exceeds viewport height.
    // ``justify-start`` + ``py-8`` replaces ``justify-center`` so the
    // top of the stack is reachable when it overflows. ``min-h-0``
    // lets the flex child shrink so the scrollbar attaches here
    // instead of escaping to the page root.
    <div className="flex h-full min-h-0 flex-col items-center gap-8 overflow-y-auto py-8">
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

      {/* Colonies — pick / create / configure before starting a session.
          Lives outside the session-tabs gate because per-colony state
          (the design-monorepo URL is the first such field) needs to be
          editable before the SessionAgent boots and reads it. */}
      <ColoniesSection
        activeColonyId={activeColonyId}
        onSelectColony={onSelectColony}
      />

      {/* Per-tenant GitHub App installation id + per-user GitHub
          OAuth identity. Both live here so they're configurable
          before a session boots — the agent metadata reads them at
          session-create time. See colony/github_identity_fix_plan.md. */}
      <TenantGitHubInstallationSection />
      <UserGitHubIdentitySection />

      {/* P11 Colony Status panel — alerts + recent activity +
          GitHub Project deep-link. Lives outside the session-tabs
          gate so the operator sees inbound GitHub events the system
          session captured without having to open a chat session.
          See colony/p11_p12_plan.md. */}
      <ColonyStatusPanel />

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={onCreateSession}
          disabled={!clusterReady}
          className="rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <Plus size={16} className="inline -mt-0.5" /> {sessionButtonLabel}
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
