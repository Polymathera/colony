import { useHealthStatus } from "@/api/hooks/useInfrastructure";
import { useAgents } from "@/api/hooks/useAgents";
import { useVCMStats } from "@/api/hooks/useVCM";

function Dot({ ok }: { ok: boolean }) {
  return (
    <span
      className={`inline-block h-1.5 w-1.5 rounded-full ${
        ok ? "bg-emerald-400 shadow-[0_0_4px_theme(colors.emerald.400)]" : "bg-red-400"
      }`}
    />
  );
}

interface StatusBarProps {
  authenticated: boolean;
}

export function StatusBar({ authenticated }: StatusBarProps) {
  const health = useHealthStatus();
  const agents = useAgents({ enabled: authenticated });
  const vcm = useVCMStats({ enabled: authenticated });

  const rayOk = health.data?.ray_connected ?? false;
  const redisOk = health.data?.redis_connected ?? false;
  const agentCount = agents.data?.length ?? 0;
  const pageCount = vcm.data?.total_pages ?? 0;

  if (!authenticated) {
    return (
      <footer className="flex h-7 shrink-0 items-center border-t bg-muted/20 px-4 text-[11px] text-muted-foreground">
        <span className="ml-auto text-muted-foreground/50">Colony v0.1</span>
      </footer>
    );
  }

  return (
    <footer className="flex h-7 shrink-0 items-center gap-4 border-t bg-muted/20 px-4 text-[11px] text-muted-foreground">
      <span className="flex items-center gap-1.5">
        <Dot ok={rayOk} />
        Ray {rayOk ? "connected" : "disconnected"}
      </span>
      <span className="flex items-center gap-1.5">
        <Dot ok={redisOk} />
        Redis {redisOk ? "connected" : "disconnected"}
      </span>
      <span className="border-l border-border pl-4">{agentCount} agents</span>
      <span>{pageCount} pages</span>
      {health.data?.node_count != null && (
        <span>{health.data.node_count} nodes</span>
      )}
      <span className="ml-auto text-muted-foreground/50">Colony v0.1</span>
    </footer>
  );
}
