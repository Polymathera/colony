import { useHealthStatus } from "@/api/hooks/useInfrastructure";
import { useAgents } from "@/api/hooks/useAgents";
import { useVCMStats } from "@/api/hooks/useVCM";

export function StatusBar() {
  const health = useHealthStatus();
  const agents = useAgents();
  const vcm = useVCMStats();

  const rayOk = health.data?.ray_connected ?? false;
  const redisOk = health.data?.redis_connected ?? false;
  const agentCount = agents.data?.length ?? 0;
  const pageCount = vcm.data?.total_pages ?? 0;

  return (
    <footer className="flex h-7 items-center gap-4 border-t bg-muted/30 px-4 text-xs text-muted-foreground">
      <span className="flex items-center gap-1.5">
        <span
          className={`inline-block h-2 w-2 rounded-full ${
            rayOk ? "bg-emerald-500" : "bg-red-500"
          }`}
        />
        Ray: {rayOk ? "connected" : "disconnected"}
      </span>
      <span className="flex items-center gap-1.5">
        <span
          className={`inline-block h-2 w-2 rounded-full ${
            redisOk ? "bg-emerald-500" : "bg-red-500"
          }`}
        />
        Redis: {redisOk ? "connected" : "disconnected"}
      </span>
      <span>{agentCount} agents</span>
      <span>{pageCount} pages</span>
      {health.data?.node_count != null && (
        <span>{health.data.node_count} nodes</span>
      )}
    </footer>
  );
}
