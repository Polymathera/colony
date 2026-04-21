import { useState } from "react";
import { useHealthStatus } from "@/api/hooks/useInfrastructure";
import { useColonyConfig, useTenantQuotas, useUpdateTenantQuota, type TenantQuota } from "@/api/hooks/useConfig";
import { Badge } from "../shared/Badge";

/* ── Tenant Quota Editor ──────────────────────────────────────── */

function TenantQuotaEditor({
  tenantId,
  quota,
  usage,
}: {
  tenantId: string;
  quota: Record<string, unknown>;
  usage: Record<string, unknown>;
}) {
  const updateQuota = useUpdateTenantQuota();
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState<TenantQuota>({
    max_concurrent_sessions: (quota.max_concurrent_sessions as number) ?? 10,
    max_concurrent_agents: (quota.max_concurrent_agents as number) ?? 100,
    max_total_cpu_cores: (quota.max_total_cpu_cores as number) ?? 10,
    max_total_memory_mb: (quota.max_total_memory_mb as number) ?? 51200,
    max_total_gpu_cores: (quota.max_total_gpu_cores as number) ?? 2,
    max_total_gpu_memory_mb: (quota.max_total_gpu_memory_mb as number) ?? 16384,
  });

  const handleSave = async () => {
    await updateQuota.mutateAsync({ tenantId, quota: form });
    setEditing(false);
  };

  const fields: { key: keyof TenantQuota; label: string; usageKey: string }[] = [
    { key: "max_concurrent_sessions", label: "Max Sessions", usageKey: "active_sessions" },
    { key: "max_concurrent_agents", label: "Max Agents", usageKey: "active_agents" },
    { key: "max_total_cpu_cores", label: "Max CPU Cores", usageKey: "total_cpu_cores" },
    { key: "max_total_memory_mb", label: "Max Memory (MB)", usageKey: "total_memory_mb" },
    { key: "max_total_gpu_cores", label: "Max GPU Cores", usageKey: "total_gpu_cores" },
    { key: "max_total_gpu_memory_mb", label: "Max GPU Mem (MB)", usageKey: "total_gpu_memory_mb" },
  ];

  return (
    <div className="rounded-lg border bg-card p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Tenant: </span>
          <span className="font-mono text-xs">{tenantId}</span>
        </div>
        {!editing ? (
          <button
            onClick={() => setEditing(true)}
            className="rounded bg-primary/10 px-3 py-1 text-xs font-medium text-primary hover:bg-primary/20"
          >
            Edit Quotas
          </button>
        ) : (
          <div className="flex gap-2">
            <button
              onClick={() => setEditing(false)}
              className="rounded px-3 py-1 text-xs text-muted-foreground hover:text-foreground"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={updateQuota.isPending}
              className="rounded bg-emerald-600 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
            >
              {updateQuota.isPending ? "Saving..." : "Save"}
            </button>
          </div>
        )}
      </div>
      <div className="grid grid-cols-3 gap-3">
        {fields.map(({ key, label, usageKey }) => {
          const current = (usage as Record<string, number>)[usageKey] ?? 0;
          const limit = form[key];
          const pct = limit > 0 ? (current / limit) * 100 : 0;

          return (
            <div key={key}>
              <label className="block text-[10px] text-muted-foreground mb-1">{label}</label>
              <div className="flex items-center gap-2">
                {editing ? (
                  <input
                    type="number"
                    value={form[key]}
                    onChange={(e) => setForm({ ...form, [key]: Number(e.target.value) })}
                    className="w-full rounded border border-border bg-background px-2 py-1 text-xs font-mono focus:border-primary focus:outline-none"
                  />
                ) : (
                  <div className="w-full">
                    <div className="flex justify-between text-xs font-mono">
                      <span>{current}</span>
                      <span className="text-muted-foreground">/ {limit}</span>
                    </div>
                    <div className="mt-1 h-1 w-full rounded-full bg-muted">
                      <div
                        className={`h-1 rounded-full transition-all ${
                          pct > 90 ? "bg-red-400" : pct > 70 ? "bg-amber-400" : "bg-emerald-400"
                        }`}
                        style={{ width: `${Math.min(pct, 100)}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Main Component ─────────────────────────────────────────────── */

export function SettingsTab() {
  const health = useHealthStatus();
  const colonyConfig = useColonyConfig();
  const tenantQuotas = useTenantQuotas();

  return (
    <div className="space-y-8 max-w-3xl">
      {/* Cluster Connection */}
      <section>
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Cluster Connection
        </h2>
        <div className="rounded-lg border bg-card p-4 space-y-4">
          <div className="flex items-center gap-3">
            <span
              className={`inline-block h-2.5 w-2.5 rounded-full ${
                health.data?.ray_connected ? "bg-emerald-400 animate-pulse" : "bg-red-400"
              }`}
            />
            <span className="text-sm font-medium">
              {health.data?.ray_connected ? "Connected" : "Disconnected"}
            </span>
            <Badge variant={health.data?.ray_connected ? "success" : "error"}>
              {health.data?.ray_cluster_status ?? "unknown"}
            </Badge>
          </div>

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
              <span className="text-muted-foreground">Node Count</span>
              <span className="font-mono">{health.data?.node_count ?? "—"}</span>
            </div>
          </div>
        </div>
      </section>

      {/* Colony Config (read-only overview) */}
      <section>
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Colony Configuration
        </h2>
        <div className="rounded-lg border bg-card p-4">
          {colonyConfig.isLoading && (
            <div className="text-xs text-muted-foreground">Loading...</div>
          )}
          {colonyConfig.data && (
            <pre className="text-xs font-mono text-muted-foreground overflow-auto max-h-64 whitespace-pre-wrap">
              {JSON.stringify(colonyConfig.data, null, 2)}
            </pre>
          )}
        </div>
      </section>

      {/* Tenant Quotas */}
      <section>
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
          Tenant Quotas
        </h2>
        {tenantQuotas.isLoading && (
          <div className="text-xs text-muted-foreground p-4">Loading tenants...</div>
        )}
        {tenantQuotas.data?.length === 0 && (
          <div className="rounded-lg border bg-card p-4 text-xs text-muted-foreground">
            No tenants configured. Tenants are created automatically when sessions are started.
          </div>
        )}
        <div className="space-y-3">
          {tenantQuotas.data?.map((t) => (
            <TenantQuotaEditor
              key={t.tenant_id}
              tenantId={t.tenant_id}
              quota={t.quota}
              usage={t.usage}
            />
          ))}
        </div>
      </section>

      {/* Note about config persistence */}
      <section>
        <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-4 text-xs text-amber-400">
          <strong>Note:</strong> Sessions and configuration are ephemeral in this development version.
          They are lost if the cluster restarts. Session persistence will be added in a future release.
        </div>
      </section>
    </div>
  );
}
