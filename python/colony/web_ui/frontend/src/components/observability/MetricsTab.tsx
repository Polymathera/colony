import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { useTokenUsage } from "@/api/hooks/useMetrics";
import { MetricCard } from "../shared/MetricCard";
import { formatTokens } from "@/lib/utils";

const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"];

export function MetricsTab() {
  const { data, isLoading } = useTokenUsage();

  const totals = data?.totals;
  const runs = data?.runs ?? [];
  const byAgent = data?.by_agent ?? [];

  // Pie data from server-side by-agent aggregation
  const pieData = byAgent
    .map((a) => ({ name: a.agent_id.slice(0, 16), value: a.input_tokens + a.output_tokens }))
    .filter((d) => d.value > 0)
    .sort((a, b) => b.value - a.value);

  // Bar chart data: input vs output per run
  const barData = runs
    .filter((r) => r.input_tokens > 0 || r.output_tokens > 0)
    .slice(-30)
    .map((r) => ({
      name: r.run_id.slice(0, 8),
      input: r.input_tokens,
      output: r.output_tokens,
      cache: r.cache_read_tokens,
    }));

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center text-muted-foreground">
        Loading metrics...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Totals */}
      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Token Usage
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          <MetricCard
            label="Total Tokens"
            value={formatTokens(totals?.total_tokens ?? 0)}
          />
          <MetricCard
            label="Input Tokens"
            value={formatTokens(totals?.input_tokens ?? 0)}
          />
          <MetricCard
            label="Output Tokens"
            value={formatTokens(totals?.output_tokens ?? 0)}
          />
          <MetricCard
            label="Cache Reads"
            value={formatTokens(totals?.cache_read_tokens ?? 0)}
          />
          <MetricCard
            label="Est. Cost"
            value={`$${(totals?.cost_usd ?? 0).toFixed(4)}`}
          />
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Bar chart */}
        <section className="rounded-lg border bg-card p-4">
          <h3 className="mb-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Tokens per Run
          </h3>
          {barData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(216, 34%, 17%)" />
                <XAxis
                  dataKey="name"
                  tick={{ fill: "hsl(215, 16%, 57%)", fontSize: 10 }}
                  axisLine={{ stroke: "hsl(216, 34%, 17%)" }}
                />
                <YAxis
                  tick={{ fill: "hsl(215, 16%, 57%)", fontSize: 10 }}
                  axisLine={{ stroke: "hsl(216, 34%, 17%)" }}
                />
                <Tooltip
                  contentStyle={{
                    background: "hsl(222, 47%, 7%)",
                    border: "1px solid hsl(216, 34%, 17%)",
                    borderRadius: "0.5rem",
                    fontSize: 12,
                  }}
                />
                <Bar dataKey="input" name="Input" fill="#3b82f6" radius={[2, 2, 0, 0]} />
                <Bar dataKey="output" name="Output" fill="#10b981" radius={[2, 2, 0, 0]} />
                <Bar dataKey="cache" name="Cache" fill="#f59e0b" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-[280px] items-center justify-center text-sm text-muted-foreground">
              No token data yet
            </div>
          )}
        </section>

        {/* Pie chart */}
        <section className="rounded-lg border bg-card p-4">
          <h3 className="mb-4 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Tokens by Agent
          </h3>
          {pieData.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {pieData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "hsl(222, 47%, 7%)",
                    border: "1px solid hsl(216, 34%, 17%)",
                    borderRadius: "0.5rem",
                    fontSize: 12,
                  }}
                  formatter={(value) => formatTokens(Number(value ?? 0))}
                />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex h-[280px] items-center justify-center text-sm text-muted-foreground">
              No agent data yet
            </div>
          )}
          {/* Legend */}
          <div className="mt-2 flex flex-wrap gap-3">
            {pieData.map((entry, i) => (
              <span key={entry.name} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <span
                  className="inline-block h-2 w-2 rounded-sm"
                  style={{ background: COLORS[i % COLORS.length] }}
                />
                {entry.name}
              </span>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
