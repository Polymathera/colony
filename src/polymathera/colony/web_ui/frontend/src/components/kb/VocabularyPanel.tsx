/**
 * Predicate Vocabulary panel — the operator's window onto the KG's
 * predicate registry and the trigger for revision passes
 * (colony/predicate_vocabulary_plan.md §5; KB-tab surface requested
 * in the 2026-08-04 review).
 *
 * Decision-support first: the stats strip answers "is a pass worth
 * running?" BEFORE anything costs money —
 *   - singleton ratio (the join-poverty signal the pass exists to fix),
 *   - provisional count (unreviewed vocabulary growth since last pass),
 *   - estimated cluster count (≈ one judge LLM call each = the pass's
 *     approximate cost).
 *
 * A pass only PROPOSES. Each proposed operation is reviewed here
 * (type, term → target, judge confidence + rationale, generating
 * signal) and applied only with the operator's identity signing the
 * destructive ones — the backend refuses unsigned destructive ops.
 */
import { useEffect, useMemo, useState } from "react";
import { BookMarked, Play, RefreshCw } from "lucide-react";
import {
  useKBVocabApply,
  useKBVocabPropose,
  useKBVocabProposeOperations,
  useKBVocabStats,
  type VocabProposedOp,
} from "@/api/hooks/useKB";
import { useColonyDesignMonorepo } from "@/api/hooks/useRepoMap";
import { useCurrentUser } from "@/api/hooks/useAuth";
import { Badge } from "../shared/Badge";

function getActiveColonyId(): string | null {
  const id = (window as any).__colony_active_colony_id;
  return typeof id === "string" && id.length > 0 ? id : null;
}

function StatTile({
  label,
  value,
  hint,
  warn,
}: {
  label: string;
  value: string | number;
  hint?: string;
  warn?: boolean;
}) {
  return (
    <div
      className={
        "rounded-lg border px-3 py-2 " +
        (warn
          ? "border-amber-500/40 bg-amber-500/10"
          : "border-border bg-background")
      }
      title={hint}
    >
      <div className="text-xs text-muted-foreground">{label}</div>
      <div
        className={
          "text-lg font-semibold " + (warn ? "text-amber-400" : "text-foreground")
        }
      >
        {value}
      </div>
    </div>
  );
}

const AUTO_SELECT_CONFIDENCE = 0.8;

export function VocabularyPanel() {
  const colonyId = getActiveColonyId();
  const persisted = useColonyDesignMonorepo(colonyId);
  const originUrl = persisted.data?.origin_url ?? null;
  const branch = persisted.data?.branch ?? "main";
  const me = useCurrentUser();

  const statsQuery = useKBVocabStats(originUrl, branch);
  const propose = useKBVocabPropose();
  const passes = useKBVocabProposeOperations();
  const apply = useKBVocabApply();

  const latestPass = useMemo(() => {
    const ops = passes.data ?? [];
    return ops.length ? ops[ops.length - 1] : null;
  }, [passes.data]);

  const [selected, setSelected] = useState<Set<string>>(new Set());
  // When a pass completes, pre-select the judge's confident proposals;
  // the operator prunes from there rather than building from nothing.
  useEffect(() => {
    if (latestPass?.status === "completed") {
      setSelected(
        new Set(
          latestPass.proposals
            .filter((p) => p.confidence >= AUTO_SELECT_CONFIDENCE)
            .map((p) => p.op_id),
        ),
      );
    }
  }, [latestPass?.op_id, latestPass?.status]);

  const [applyMessage, setApplyMessage] = useState<string | null>(null);
  // Judge effort: batch classification is the provider's canonical
  // low-effort workload; the selector exists for when merge decisions
  // look shallow and the operator wants deeper judging.
  const [effort, setEffort] = useState<string>("low");

  if (!originUrl) {
    return null; // no design monorepo configured — nothing to curate
  }

  const stats = statsQuery.data?.stats;
  const estimatedClusters = statsQuery.data?.estimated_clusters ?? 0;
  const singletonPct = stats ? Math.round(stats.singleton_ratio * 100) : 0;
  const proposals = latestPass?.status === "completed" ? latestPass.proposals : [];
  const passRunning =
    latestPass?.status === "pending" || latestPass?.status === "running";

  const toggle = (opId: string) => {
    setSelected((curr) => {
      const next = new Set(curr);
      if (next.has(opId)) next.delete(opId);
      else next.add(opId);
      return next;
    });
  };

  const onApply = async () => {
    const chosen = proposals.filter((p) => selected.has(p.op_id));
    if (!chosen.length) return;
    setApplyMessage(null);
    const result = await apply.mutateAsync({
      origin_url: originUrl,
      branch,
      operations: chosen,
      approved_by: me.data?.vcs_login ?? "operator",
    });
    const parts = [
      `${result.applied.length} applied`,
      result.rewrite?.rewritten
        ? `${result.rewrite.rewritten} claims rewritten (${result.rewrite.deduplicated ?? 0} deduplicated)`
        : null,
      result.commit ? `commit ${result.commit} pushed` : null,
      result.failed.length ? `${result.failed.length} REFUSED` : null,
    ].filter(Boolean);
    setApplyMessage(parts.join(" · "));
    statsQuery.refetch();
  };

  return (
    <section className="rounded-xl border border-border bg-card p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BookMarked className="h-4 w-4 text-muted-foreground" />
          <h2 className="text-sm font-semibold text-foreground">
            Predicate Vocabulary
          </h2>
          <span className="text-xs text-muted-foreground">
            {branch} · {stats?.operations_applied ?? 0} operations applied
            {stats?.last_operation_at
              ? ` · last ${stats.last_operation_at.slice(0, 10)}`
              : ""}
          </span>
        </div>
        <button
          onClick={() => statsQuery.refetch()}
          className="rounded-md border border-border p-1.5 text-muted-foreground hover:bg-accent/40"
          title="Refresh stats"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="mb-3 grid grid-cols-2 gap-2 md:grid-cols-6">
        <StatTile
          label="Predicates in KG"
          value={stats?.predicates_in_kg ?? "…"}
        />
        <StatTile
          label="Singleton ratio"
          value={stats ? `${singletonPct}%` : "…"}
          hint="Predicates carrying exactly one claim. High values mean the graph has few join paths — the problem revision passes exist to fix."
          warn={!!stats && stats.singleton_ratio > 0.5}
        />
        <StatTile
          label="Provisional"
          value={stats?.provisional ?? "…"}
          hint="Minted by extraction, not yet reviewed. Growth here since the last pass is new vocabulary awaiting curation."
          warn={!!stats && stats.provisional > 100}
        />
        <StatTile label="Active" value={stats?.active ?? "…"} />
        <StatTile label="Deprecated" value={stats?.deprecated ?? "…"} />
        <StatTile
          label="Pass cost"
          value={stats ? `~${estimatedClusters} clusters` : "…"}
          hint="Candidate clusters a revision pass would judge — approximately one LLM call each."
        />
      </div>

      <div className="mb-3 flex items-center gap-3">
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
          Judge effort
          <select
            value={effort}
            onChange={(e) => setEffort(e.target.value)}
            disabled={passRunning || propose.isPending}
            className="rounded-md border border-border bg-background px-2 py-1 text-xs text-foreground"
            title="LLM effort per judged cluster. low = fastest/cheapest (recommended for synonym judging); raise if merge decisions look shallow."
          >
            <option value="low">low</option>
            <option value="medium">medium</option>
            <option value="high">high</option>
            <option value="xhigh">xhigh</option>
            <option value="max">max</option>
          </select>
        </label>
        <button
          onClick={() =>
            propose.mutate({ origin_url: originUrl, branch, effort })
          }
          disabled={passRunning || propose.isPending}
          className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Play className="h-3.5 w-3.5" />
          {passRunning ? "Revision pass running…" : "Run revision pass"}
        </button>
        {latestPass && (
          <span className="text-xs text-muted-foreground">
            {latestPass.message}
          </span>
        )}
      </div>

      {proposals.length > 0 && (
        <div className="rounded-lg border border-border">
          <div className="flex items-center justify-between border-b border-border px-3 py-2">
            <span className="text-xs text-muted-foreground">
              {proposals.length} proposed operations · {selected.size} selected
              (proposals ≥ {AUTO_SELECT_CONFIDENCE} pre-selected)
            </span>
            <button
              onClick={onApply}
              disabled={!selected.size || apply.isPending}
              className="rounded-md bg-primary px-3 py-1 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {apply.isPending
                ? "Applying…"
                : `Apply ${selected.size} as ${me.data?.vcs_login ?? "operator"}`}
            </button>
          </div>
          <div className="max-h-72 overflow-y-auto">
            <table className="w-full text-xs">
              <tbody>
                {proposals.map((p: VocabProposedOp) => (
                  <tr
                    key={p.op_id}
                    className="border-b border-border/50 last:border-0 hover:bg-accent/20"
                  >
                    <td className="w-8 px-3 py-1.5">
                      <input
                        type="checkbox"
                        checked={selected.has(p.op_id)}
                        onChange={() => toggle(p.op_id)}
                      />
                    </td>
                    <td className="w-24 py-1.5">
                      <Badge>{p.op_type}</Badge>
                    </td>
                    <td className="py-1.5 font-mono text-foreground">
                      {p.term}
                      {p.target ? ` → ${p.target}` : ""}
                    </td>
                    <td className="w-14 py-1.5 text-muted-foreground">
                      {(p.confidence * 100).toFixed(0)}%
                    </td>
                    <td className="w-40 py-1.5 text-muted-foreground">
                      {p.proposed_by.replace("revision:", "")}
                    </td>
                    <td
                      className="max-w-64 truncate py-1.5 pr-3 text-muted-foreground"
                      title={p.rationale}
                    >
                      {p.rationale}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {applyMessage && (
        <div className="mt-2 rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-400">
          {applyMessage}
        </div>
      )}
    </section>
  );
}
