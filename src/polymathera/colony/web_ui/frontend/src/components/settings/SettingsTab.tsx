import { useHealthStatus } from "@/api/hooks/useInfrastructure";
import { Badge } from "../shared/Badge";

/* ── Plugin definitions ─────────────────────────────────────── */

interface PluginEntry {
  name: string;
  description: string;
  enabled: boolean;
  builtin: boolean;
}

const PLUGINS: PluginEntry[] = [
  {
    name: "Code Analysis",
    description: "Static analysis, dependency scanning, and code quality metrics",
    enabled: true,
    builtin: true,
  },
  {
    name: "Knowledge Graph",
    description: "Page graph traversal, relationship discovery, and centrality analysis",
    enabled: true,
    builtin: true,
  },
  {
    name: "Prometheus Metrics",
    description: "Real-time metrics collection and streaming from colony deployments",
    enabled: true,
    builtin: true,
  },
  {
    name: "Custom Agent Templates",
    description: "Define and manage reusable agent blueprints with preset capabilities",
    enabled: false,
    builtin: false,
  },
  {
    name: "External LLM Providers",
    description: "Connect to OpenAI, Anthropic, or custom LLM endpoints",
    enabled: false,
    builtin: false,
  },
  {
    name: "Cost Tracking",
    description: "Track and visualize token usage costs across runs and agents",
    enabled: false,
    builtin: false,
  },
];

/* ── Main Component ─────────────────────────────────────────── */

export function SettingsTab() {
  const health = useHealthStatus();

  return (
    <div className="space-y-8 max-w-3xl">
      {/* Remote Cluster Connection */}
      <section>
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Cluster Connection
        </h2>
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <div className="flex items-center gap-3">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-sm font-medium">Local Cluster</span>
            <Badge variant="success">Connected</Badge>
          </div>

          <div className="grid grid-cols-3 gap-3 text-xs">
            <div>
              <label className="block text-muted-foreground mb-1">
                Cluster Address
              </label>
              <input
                type="text"
                value="ray://ray-head:10001"
                disabled
                className="w-full rounded border border-border bg-muted/30 px-2 py-1.5 font-mono text-muted-foreground cursor-not-allowed"
              />
            </div>
            <div>
              <label className="block text-muted-foreground mb-1">
                Auth Token
              </label>
              <input
                type="password"
                placeholder="Not required for local"
                disabled
                className="w-full rounded border border-border bg-muted/30 px-2 py-1.5 font-mono text-muted-foreground cursor-not-allowed"
              />
            </div>
            <div>
              <label className="block text-muted-foreground mb-1">
                Namespace
              </label>
              <input
                type="text"
                value="polymathera"
                disabled
                className="w-full rounded border border-border bg-muted/30 px-2 py-1.5 font-mono text-muted-foreground cursor-not-allowed"
              />
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              disabled
              className="rounded bg-primary/30 px-3 py-1.5 text-xs font-medium text-primary/50 cursor-not-allowed"
            >
              Connect to Remote Cluster
            </button>
            <Badge variant="warning">Coming Soon</Badge>
          </div>
        </div>
      </section>

      {/* Cluster Info (real data) */}
      <section>
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Cluster Info
        </h2>
        <div className="rounded-lg border bg-card p-4">
          <div className="grid grid-cols-2 gap-y-3 gap-x-6 text-xs">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Ray Connected</span>
              <Badge variant={health.data?.ray_connected ? "success" : "error"}>
                {health.data?.ray_connected ? "Yes" : "No"}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Redis Connected</span>
              <Badge variant={health.data?.redis_connected ? "success" : "error"}>
                {health.data?.redis_connected ? "Yes" : "No"}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Cluster Status</span>
              <span className="font-mono">
                {health.data?.ray_cluster_status ?? "—"}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Node Count</span>
              <span className="font-mono">
                {health.data?.node_count ?? "—"}
              </span>
            </div>
          </div>

          <div className="mt-4 pt-4 border-t grid grid-cols-3 gap-3 text-xs">
            <div>
              <label className="block text-muted-foreground mb-1">
                Max Workers
              </label>
              <input
                type="number"
                value={8}
                disabled
                className="w-full rounded border border-border bg-muted/30 px-2 py-1.5 font-mono text-muted-foreground cursor-not-allowed"
              />
            </div>
            <div>
              <label className="block text-muted-foreground mb-1">
                Auto-Scaling
              </label>
              <select
                disabled
                className="w-full rounded border border-border bg-muted/30 px-2 py-1.5 text-muted-foreground cursor-not-allowed"
              >
                <option>Disabled</option>
              </select>
            </div>
            <div>
              <label className="block text-muted-foreground mb-1">
                GPU Allocation
              </label>
              <select
                disabled
                className="w-full rounded border border-border bg-muted/30 px-2 py-1.5 text-muted-foreground cursor-not-allowed"
              >
                <option>Auto</option>
              </select>
            </div>
          </div>
        </div>
      </section>

      {/* Plugin System */}
      <section>
        <div className="mb-4 flex items-center gap-2">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
            Plugins
          </h2>
          <Badge variant="warning">Coming Soon</Badge>
        </div>
        <div className="rounded-lg border bg-card divide-y">
          {PLUGINS.map((plugin) => (
            <div
              key={plugin.name}
              className="flex items-center justify-between px-4 py-3"
            >
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium">{plugin.name}</span>
                  {plugin.builtin && (
                    <Badge variant="default">Built-in</Badge>
                  )}
                </div>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {plugin.description}
                </p>
              </div>
              <button
                disabled
                className={`relative h-6 w-11 rounded-full transition-colors cursor-not-allowed ${
                  plugin.enabled ? "bg-emerald-600/50" : "bg-zinc-700"
                }`}
              >
                <span
                  className={`absolute top-0.5 h-5 w-5 rounded-full bg-white/80 transition-transform ${
                    plugin.enabled ? "left-[22px]" : "left-0.5"
                  }`}
                />
              </button>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
