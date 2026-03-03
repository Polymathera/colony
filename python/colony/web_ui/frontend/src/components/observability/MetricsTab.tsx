import { useState, useMemo } from "react";
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
  AreaChart,
  Area,
} from "recharts";
import { useTokenUsage } from "@/api/hooks/useMetrics";
import {
  useMetricsStream,
  type MetricsSnapshot,
  type PrometheusType,
} from "@/api/hooks/usePrometheus";
import { MetricCard } from "../shared/MetricCard";
import { formatTokens } from "@/lib/utils";

const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"];

const CHART_TOOLTIP_STYLE = {
  background: "hsl(222, 47%, 7%)",
  border: "1px solid hsl(216, 34%, 17%)",
  borderRadius: "0.5rem",
  fontSize: 12,
};

const AXIS_TICK = { fill: "hsl(215, 16%, 57%)", fontSize: 10 };
const AXIS_LINE = { stroke: "hsl(216, 34%, 17%)" };
const GRID_STROKE = "hsl(216, 34%, 17%)";

/* ── Helpers ─────────────────────────────────────────────────── */

/** Derive whether a metric should be rendered as a rate (/s) based on its type. */
function isRateMetric(metricKey: string, type: PrometheusType | undefined): boolean {
  if (type === "counter") return true;
  // Fallback heuristic for metrics without TYPE annotation
  return metricKey.endsWith("_total") || metricKey.endsWith("_count");
}

/** Pretty-print a metric name: strip common suffixes, replace _ with space. */
function formatMetricLabel(key: string): string {
  return key
    .replace(/_total$/, "")
    .replace(/_seconds_sum$/, " latency (s)")
    .replace(/_bytes$/, " (bytes)")
    .replace(/_/g, " ");
}

/** Pick a deterministic color from the palette for a given index. */
function pickColor(index: number): string {
  return COLORS[index % COLORS.length];
}

/* ── Sub-components ───────────────────────────────────────────── */

function TimeSeriesChart({
  label,
  metricKey,
  color,
  isDelta,
  history,
}: {
  label: string;
  metricKey: string;
  color: string;
  isDelta: boolean;
  history: MetricsSnapshot[];
}) {
  const chartData = useMemo(() => {
    if (history.length === 0) return [];

    const points: { time: string; value: number }[] = [];
    for (let i = 0; i < history.length; i++) {
      const snap = history[i];
      const raw = snap[metricKey] as number | undefined;
      if (raw === undefined) continue;

      let value: number;
      if (isDelta && i > 0) {
        const prev = (history[i - 1][metricKey] as number | undefined) ?? 0;
        const dt = snap._timestamp - history[i - 1]._timestamp;
        value = dt > 0 ? Math.max(0, (raw - prev) / dt) : 0;
      } else {
        value = raw;
      }

      points.push({
        time: new Date(snap._timestamp * 1000).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }),
        value: Math.round(value * 1000) / 1000,
      });
    }
    return points;
  }, [history, metricKey, isDelta]);

  return (
    <section className="rounded-lg border bg-card p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          {label}
        </h3>
        {isDelta && (
          <span className="text-[10px] text-muted-foreground/60">/s</span>
        )}
      </div>
      {chartData.length > 1 ? (
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id={`grad-${metricKey}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                <stop offset="95%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
            <XAxis
              dataKey="time"
              tick={AXIS_TICK}
              axisLine={AXIS_LINE}
              interval="preserveStartEnd"
              minTickGap={40}
            />
            <YAxis tick={AXIS_TICK} axisLine={AXIS_LINE} width={50} />
            <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
            <Area
              type="monotone"
              dataKey="value"
              stroke={color}
              fill={`url(#grad-${metricKey})`}
              strokeWidth={2}
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      ) : (
        <div className="flex h-[200px] items-center justify-center text-sm text-muted-foreground">
          Waiting for data...
        </div>
      )}
    </section>
  );
}

function LiveIndicator({ connected, count }: { connected: boolean; count: number }) {
  return (
    <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
      <span
        className={`inline-block h-2 w-2 rounded-full ${connected ? "bg-emerald-400 animate-pulse" : "bg-red-400"}`}
      />
      {connected ? `Live (${count} samples)` : "Disconnected"}
    </span>
  );
}

/* ── Main component ───────────────────────────────────────────── */

type MetricsView = "tokens" | "system";

export function MetricsTab() {
  const [view, setView] = useState<MetricsView>("tokens");

  return (
    <div className="space-y-4">
      {/* View toggle */}
      <div className="flex items-center gap-2">
        <button
          className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
            view === "tokens"
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-muted-foreground hover:text-foreground"
          }`}
          onClick={() => setView("tokens")}
        >
          Token Usage
        </button>
        <button
          className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
            view === "system"
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-muted-foreground hover:text-foreground"
          }`}
          onClick={() => setView("system")}
        >
          System Metrics
        </button>
      </div>

      {view === "tokens" ? <TokenUsageView /> : <SystemMetricsView />}
    </div>
  );
}

/* ── Token usage view ─────────────────────────────────────────── */

function TokenUsageView() {
  const { data, isLoading } = useTokenUsage();

  const totals = data?.totals;
  const runs = data?.runs ?? [];
  const byAgent = data?.by_agent ?? [];

  const pieData = byAgent
    .map((a) => ({ name: a.agent_id.slice(0, 16), value: a.input_tokens + a.output_tokens }))
    .filter((d) => d.value > 0)
    .sort((a, b) => b.value - a.value);

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

  const error = (data as Record<string, unknown> | undefined)?.error as string | undefined;

  return (
    <div className="space-y-6">
      {error && (
        <div className="rounded-lg border border-red-800 bg-red-950/30 px-4 py-3 text-sm text-red-400">
          Backend error: {error}
        </div>
      )}

      {/* Totals */}
      <section>
        <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Token Usage
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-6">
          <MetricCard label="Runs" value={String(totals?.run_count ?? 0)} />
          <MetricCard label="Total Tokens" value={formatTokens(totals?.total_tokens ?? 0)} />
          <MetricCard label="Input Tokens" value={formatTokens(totals?.input_tokens ?? 0)} />
          <MetricCard label="Output Tokens" value={formatTokens(totals?.output_tokens ?? 0)} />
          <MetricCard label="Cache Reads" value={formatTokens(totals?.cache_read_tokens ?? 0)} />
          <MetricCard label="Est. Cost" value={`$${(totals?.cost_usd ?? 0).toFixed(4)}`} />
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
                <CartesianGrid strokeDasharray="3 3" stroke={GRID_STROKE} />
                <XAxis dataKey="name" tick={AXIS_TICK} axisLine={AXIS_LINE} />
                <YAxis tick={AXIS_TICK} axisLine={AXIS_LINE} />
                <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
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
                  contentStyle={CHART_TOOLTIP_STYLE}
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

/* ── System metrics view (live scraping) ──────────────────────── */

/** Internal keys and noise suffixes to exclude from dynamic chart discovery. */
function isChartableMetric(key: string): boolean {
  if (key.startsWith("_")) return false;
  if (key.endsWith("_bucket")) return false;
  if (key.endsWith("_created")) return false;
  if (key.endsWith("_info")) return false;
  return true;
}

function SystemMetricsView() {
  const { history, connected, latest, types } = useMetricsStream(5, 120);

  // Dynamically discover all metric keys from the latest snapshot
  const discoveredMetrics = useMemo(() => {
    if (!latest) return [];
    return Object.keys(latest)
      .filter(isChartableMetric)
      .sort((a, b) => a.localeCompare(b))
      .map((key, i) => ({
        key,
        label: formatMetricLabel(key),
        color: pickColor(i),
        isDelta: isRateMetric(key, types[key]),
      }));
  }, [latest, types]);

  // Current value cards (top 8 for a quick glance)
  const currentValues = useMemo(() => {
    if (!latest) return [];
    return Object.entries(latest)
      .filter(([k]) => isChartableMetric(k))
      .sort(([a], [b]) => a.localeCompare(b))
      .slice(0, 8)
      .map(([k, v]) => ({ name: k.replace(/_/g, " "), value: typeof v === "number" ? v : 0 }));
  }, [latest]);

  return (
    <div className="space-y-6">
      {/* Status */}
      <LiveIndicator connected={connected} count={history.length} />

      {/* Current value cards */}
      {currentValues.length > 0 && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {currentValues.map((cv) => (
            <MetricCard
              key={cv.name}
              label={cv.name}
              value={cv.value >= 1000 ? `${(cv.value / 1000).toFixed(1)}k` : String(Math.round(cv.value * 100) / 100)}
            />
          ))}
        </div>
      )}

      {/* Time-series charts — one per discovered metric */}
      <div className="grid gap-4 lg:grid-cols-2">
        {discoveredMetrics.map((mc) => (
          <TimeSeriesChart
            key={mc.key}
            label={mc.label}
            metricKey={mc.key}
            color={mc.color}
            isDelta={mc.isDelta}
            history={history}
          />
        ))}
      </div>
    </div>
  );
}
