import { useHealthStatus, useRedisInfo } from "@/api/hooks/useInfrastructure";
import { useDeployments } from "@/api/hooks/useDeployments";
import { useAgents } from "@/api/hooks/useAgents";
import { useVCMStats } from "@/api/hooks/useVCM";
import { useSessionStats } from "@/api/hooks/useSessions";
import { MetricCard } from "../shared/MetricCard";
import { Badge } from "../shared/Badge";
import { formatDuration } from "@/lib/utils";

export function OverviewTab() {
  const health = useHealthStatus();
  const redis = useRedisInfo();
  const deployments = useDeployments();
  const agents = useAgents();
  const vcm = useVCMStats();
  const sessionStats = useSessionStats();

  const h = health.data;
  const r = redis.data;

  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
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

      {/* Deployments */}
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Applications
        </h2>
        {deployments.data?.length ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {deployments.data.map((app) => (
              <div
                key={app.app_name}
                className="rounded-lg border bg-card p-4 shadow-sm"
              >
                <h3 className="font-medium">{app.app_name}</h3>
                <div className="mt-2 flex flex-wrap gap-1.5">
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
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Quick Stats
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
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
        </div>
      </section>
    </div>
  );
}
