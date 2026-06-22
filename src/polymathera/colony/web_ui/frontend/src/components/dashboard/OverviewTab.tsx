import { useHealthStatus, useRedisInfo } from "@/api/hooks/useInfrastructure";
import { useDeployments } from "@/api/hooks/useDeployments";
import { useAgents } from "@/api/hooks/useAgents";
import { useVCMStats } from "@/api/hooks/useVCM";
import { useSessionStats } from "@/api/hooks/useSessions";
import { useTokenUsage } from "@/api/hooks/useMetrics";
import { useColonyAgentDiagnostics } from "@/api/hooks/useColonyStatus";
import { MetricCard } from "../shared/MetricCard";
import { Badge } from "../shared/Badge";
import { formatDuration, formatTokens } from "@/lib/utils";

export function OverviewTab() {
  const health = useHealthStatus();
  const redis = useRedisInfo();
  const deployments = useDeployments();
  const agents = useAgents();
  const vcm = useVCMStats();
  const sessionStats = useSessionStats();
  const tokenUsage = useTokenUsage();
  const diagnostics = useColonyAgentDiagnostics(20);

  const h = health.data;
  const r = redis.data;
  const totals = tokenUsage.data?.totals;
  const diagRows = diagnostics.data?.diagnostics ?? [];

  return (
    <div className="space-y-6">
      {/* Cluster Health */}
      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Cluster Health
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <MetricCard
            label="Ray Cluster"
            value={h?.ray_cluster_status ?? "unknown"}
          />
          <MetricCard
            label="Nodes"
            value={h?.node_count ?? 0}
          />
          <MetricCard
            label="Redis Memory"
            value={r?.used_memory_human ?? "—"}
          />
          <MetricCard
            label="Redis Uptime"
            value={r ? formatDuration(r.uptime_in_seconds) : "—"}
          />
        </div>
      </section>

      {/* Applications */}
      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Applications
        </h2>
        {deployments.data?.length ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {deployments.data.map((app) => (
              <div
                key={app.app_name}
                className="rounded-lg border bg-card p-4 transition-colors hover:border-primary/30"
              >
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_4px_theme(colors.emerald.400)]" />
                  <h3 className="font-medium">{app.app_name}</h3>
                </div>
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {app.deployments.map((d) => (
                    <Badge key={d.deployment_name} variant="info">
                      {d.deployment_name}
                    </Badge>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            No applications found
          </p>
        )}
      </section>

      {/* Quick Stats */}
      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Activity
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <MetricCard
            label="Agents"
            value={agents.data?.length ?? 0}
          />
          <MetricCard
            label="Active Sessions"
            value={
              typeof sessionStats.data === "object" && sessionStats.data !== null
                ? (sessionStats.data as Record<string, unknown>)["active_sessions"] as number ?? 0
                : 0
            }
          />
          <MetricCard
            label="Total Pages"
            value={vcm.data?.total_pages ?? 0}
          />
          <MetricCard
            label="Loaded Pages"
            value={vcm.data?.loaded_pages ?? 0}
          />
          <MetricCard
            label="Total Tokens"
            value={formatTokens(totals?.total_tokens ?? 0)}
          />
          <MetricCard
            label="LLM Runs"
            value={totals?.run_count ?? 0}
          />
        </div>
      </section>

      {/* Agent Diagnostics — session-agent crashes, github-inbound
          quiesce, and other AgentDiagnosticProtocol events the chat
          UI can't surface (the agent is dead by the time it would
          have spoken). Hidden when steady-state empty. */}
      {diagRows.length > 0 && (
        <section>
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Agent Diagnostics
          </h2>
          <div className="rounded-lg border bg-card divide-y">
            {diagRows.map((row) => {
              const kind = row.payload.kind ?? "unknown";
              const agentId = row.payload.agent_id ?? "—";
              const detail =
                row.payload.stop_reason ??
                row.payload.exception_message ??
                row.payload.reason ??
                "";
              return (
                <div
                  key={row.id}
                  className="flex items-start gap-3 p-3 text-sm"
                >
                  <Badge
                    variant={
                      kind === "session_agent_stopped" ? "error" : "warning"
                    }
                  >
                    {kind}
                  </Badge>
                  <div className="flex-1 min-w-0">
                    <div className="font-mono text-xs text-muted-foreground truncate">
                      {agentId}
                    </div>
                    {detail && (
                      <div className="mt-0.5 text-xs text-foreground/80 line-clamp-2">
                        {String(detail)}
                      </div>
                    )}
                  </div>
                  <span className="shrink-0 text-[10px] text-muted-foreground">
                    {new Date(row.ts).toLocaleString()}
                  </span>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
