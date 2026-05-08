/**
 * Knowledge Base tab — read-only window onto the corpus the agents
 * share via the process-singleton ``RetrievalDeps``.
 *
 * Three panes stacked top-to-bottom:
 *
 *   [stats strip]            total chunks / sources / tokens, plus
 *                            backend info (Qdrant URL + collection,
 *                            or "InMemory" when not wired).
 *
 *   [sources / search]       left: list of ingested sources (chunk
 *                            count + tier + format). Click to drill
 *                            into the source's chunks. Right: an
 *                            embedding-similarity search box with
 *                            ranked previews.
 *
 *   [ad-hoc ingest]          path or text-blob form for operator-
 *                            driven smoke tests of the ingestion
 *                            pipeline. The routine ingestion path is
 *                            still the SessionAgent's
 *                            ingest_repo_map_literature action.
 */
import { useMemo, useState } from "react";
import { Database, Search, FileText, Upload } from "lucide-react";
import {
  useKBChunksForSource,
  useKBIngest,
  useKBSearch,
  useKBSources,
  useKBStats,
  type KBSearchHit,
} from "@/api/hooks/useKB";
import { Badge } from "../shared/Badge";

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement>;

function PrimaryButton(props: ButtonProps) {
  const { className = "", ...rest } = props;
  return (
    <button
      {...rest}
      className={
        "rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground " +
        "hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed " +
        "focus:outline-none focus:ring-2 focus:ring-primary/40 transition-colors " +
        className
      }
    />
  );
}

function SecondaryButton(props: ButtonProps) {
  const { className = "", ...rest } = props;
  return (
    <button
      {...rest}
      className={
        "rounded-md border border-border bg-background px-3 py-1.5 text-sm font-medium text-foreground " +
        "hover:bg-accent/40 disabled:opacity-50 disabled:cursor-not-allowed " +
        "focus:outline-none focus:ring-2 focus:ring-primary/30 transition-colors " +
        className
      }
    />
  );
}

function StatTile({
  label, value, subtitle,
}: { label: string; value: string | number; subtitle?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card p-3 min-w-[10rem]">
      <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      <p className="mt-1 text-xl font-semibold tracking-tight">{value}</p>
      {subtitle && (
        <p className="mt-0.5 text-[11px] text-muted-foreground truncate">{subtitle}</p>
      )}
    </div>
  );
}

function shortSource(uri: string, max = 64): string {
  if (uri.length <= max) return uri;
  return "…" + uri.slice(uri.length - (max - 1));
}

export function KnowledgeBaseTab() {
  const stats = useKBStats();
  const sources = useKBSources();

  const [selectedSource, setSelectedSource] = useState<string | null>(null);
  const chunks = useKBChunksForSource(selectedSource);

  const search = useKBSearch();
  const [searchText, setSearchText] = useState("");
  const [searchHits, setSearchHits] = useState<KBSearchHit[]>([]);

  const ingest = useKBIngest();
  const [ingestPath, setIngestPath] = useState("");
  const [ingestTier, setIngestTier] = useState("untiered");
  const [ingestMessage, setIngestMessage] = useState<string | null>(null);

  const onSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchText.trim()) return;
    search.mutate(
      { text: searchText.trim(), max_results: 10 },
      { onSuccess: (data) => setSearchHits(data.hits) },
    );
  };

  const onIngest = (e: React.FormEvent) => {
    e.preventDefault();
    if (!ingestPath.trim()) return;
    setIngestMessage(null);
    ingest.mutate(
      { path: ingestPath.trim(), tier: ingestTier },
      {
        onSuccess: (data) => {
          if (data.error) {
            setIngestMessage(`Failed: ${data.error}`);
          } else {
            setIngestMessage(
              `Ingested ${data.source_uri} → ${data.chunks_produced} chunk(s) (${data.status})`,
            );
            setIngestPath("");
          }
        },
        onError: (err) =>
          setIngestMessage(err instanceof Error ? err.message : String(err)),
      },
    );
  };

  const backend = stats.data?.backend;
  const backendBadge = useMemo(() => {
    if (!backend) return null;
    const isQdrant = backend.vector_store === "QdrantVectorStore";
    return (
      <Badge variant={isQdrant ? "success" : "warning"}>
        {backend.vector_store}
        {isQdrant && backend.qdrant_url ? ` → ${backend.qdrant_url}` : ""}
      </Badge>
    );
  }, [backend]);

  return (
    <div className="p-4 flex flex-col gap-4 h-full">
      {/* Header */}
      <header className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Database size={18} className="text-primary" /> Knowledge Base
          </h2>
          <p className="text-xs text-muted-foreground max-w-3xl mt-1">
            Live view of the corpus shared by every agent in the colony.
            Sources are ingested by the SessionAgent's{" "}
            <code className="text-[11px] px-1 rounded bg-muted">
              ingest_repo_map_literature
            </code>{" "}
            action and via the ad-hoc form below.
          </p>
        </div>
        <div className="flex flex-col items-end gap-1">
          {backendBadge}
          {backend?.qdrant_collection && (
            <span className="text-[11px] font-mono text-muted-foreground">
              collection: {backend.qdrant_collection}
            </span>
          )}
          <span className="text-[11px] text-muted-foreground">
            embedder: {backend?.embedder_id ?? "—"} ({backend?.embedder_dimensions ?? "?"}d)
          </span>
        </div>
      </header>

      {/* Stats strip */}
      <section className="flex flex-wrap gap-3">
        <StatTile
          label="Total chunks"
          value={stats.data?.total_chunks ?? "—"}
        />
        <StatTile
          label="Sources"
          value={stats.data?.total_sources ?? "—"}
        />
        <StatTile
          label="Total tokens"
          value={stats.data?.total_tokens ?? "—"}
        />
        <StatTile
          label="By tier"
          value={
            stats.data
              ? Object.keys(stats.data.by_tier).length === 0
                ? "—"
                : Object.entries(stats.data.by_tier)
                    .map(([k, v]) => `${k}: ${v}`)
                    .join("  ·  ")
              : "—"
          }
        />
        <StatTile
          label="By type"
          value={
            stats.data
              ? Object.keys(stats.data.by_data_type).length === 0
                ? "—"
                : Object.entries(stats.data.by_data_type)
                    .map(([k, v]) => `${k}: ${v}`)
                    .join("  ·  ")
              : "—"
          }
        />
      </section>

      {/* Two-column body: sources (left) / drill-down (right) */}
      <section className="flex-1 grid grid-cols-1 lg:grid-cols-[28rem_1fr] gap-4 min-h-0">
        {/* Sources list */}
        <div className="rounded-lg border border-border bg-card flex flex-col overflow-hidden min-h-0">
          <div className="px-3 py-2 border-b border-border flex items-center justify-between">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <FileText size={14} /> Sources
            </h3>
            <span className="text-[11px] text-muted-foreground">
              {sources.data?.sources.length ?? 0} source(s)
            </span>
          </div>
          <div className="flex-1 overflow-auto">
            {sources.isLoading ? (
              <div className="text-xs text-muted-foreground p-3">Loading…</div>
            ) : (sources.data?.sources.length ?? 0) === 0 ? (
              <div className="text-xs text-muted-foreground p-3">
                Empty corpus. Use{" "}
                <code className="text-[11px] px-1 rounded bg-muted">
                  ingest_repo_map_literature
                </code>{" "}
                from the SessionAgent or the form below to populate it.
              </div>
            ) : (
              <ul className="divide-y divide-border">
                {sources.data!.sources.map((s) => (
                  <li
                    key={s.source}
                    onClick={() => setSelectedSource(s.source)}
                    className={
                      "px-3 py-2 cursor-pointer hover:bg-accent/40 transition-colors " +
                      (selectedSource === s.source ? "bg-accent/40" : "")
                    }
                  >
                    <div
                      className="text-xs font-mono truncate"
                      title={s.source}
                    >
                      {shortSource(s.source)}
                    </div>
                    <div className="mt-1 flex items-center gap-1.5 flex-wrap">
                      <Badge variant="info">{s.chunk_count} chunks</Badge>
                      <span className="text-[11px] text-muted-foreground">
                        {s.total_tokens} tok
                      </span>
                      {s.data_types.map((dt) => (
                        <Badge key={dt}>{dt}</Badge>
                      ))}
                      {s.tiers.map((t) => (
                        <Badge key={t} variant="default">
                          {t}
                        </Badge>
                      ))}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Right pane: search + drilldown + ingest */}
        <div className="flex flex-col gap-4 min-h-0 overflow-hidden">
          {/* Search */}
          <div className="rounded-lg border border-border bg-card overflow-hidden">
            <div className="px-3 py-2 border-b border-border">
              <h3 className="text-sm font-medium flex items-center gap-2">
                <Search size={14} /> Embedding search
              </h3>
            </div>
            <form onSubmit={onSearch} className="p-3 flex items-center gap-2">
              <input
                type="text"
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                placeholder="Free-text query…"
                className="flex-1 px-3 py-1.5 rounded-md border border-border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
              />
              <PrimaryButton type="submit" disabled={search.isPending}>
                {search.isPending ? "Searching…" : "Search"}
              </PrimaryButton>
            </form>
            {searchHits.length > 0 && (
              <ul className="divide-y divide-border max-h-64 overflow-auto">
                {searchHits.map((h) => (
                  <li key={h.chunk_id} className="px-3 py-2">
                    <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                      <Badge variant="info">#{h.rank + 1}</Badge>
                      <span className="font-mono">
                        score {h.score.toFixed(3)}
                      </span>
                      <span className="font-mono truncate" title={h.source}>
                        {shortSource(h.source, 48)}
                      </span>
                    </div>
                    <div className="mt-1 text-xs whitespace-pre-wrap">
                      {h.text_preview}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Drill-down */}
          <div className="flex-1 rounded-lg border border-border bg-card overflow-hidden flex flex-col min-h-0">
            <div className="px-3 py-2 border-b border-border flex items-center justify-between">
              <h3 className="text-sm font-medium">
                {selectedSource ? "Chunks" : "Pick a source"}
              </h3>
              {selectedSource && (
                <span
                  className="text-[11px] text-muted-foreground font-mono truncate max-w-[36rem]"
                  title={selectedSource}
                >
                  {shortSource(selectedSource, 60)}
                </span>
              )}
            </div>
            <div className="flex-1 overflow-auto">
              {!selectedSource ? (
                <div className="text-xs text-muted-foreground p-3">
                  Click a source on the left to inspect its chunks.
                </div>
              ) : chunks.isLoading ? (
                <div className="text-xs text-muted-foreground p-3">Loading…</div>
              ) : (chunks.data?.chunks.length ?? 0) === 0 ? (
                <div className="text-xs text-muted-foreground p-3">
                  No chunks for this source.
                </div>
              ) : (
                <ul className="divide-y divide-border">
                  {chunks.data!.chunks.map((c) => (
                    <li key={c.chunk_id} className="px-3 py-2">
                      <div className="flex items-center gap-2 text-[11px] text-muted-foreground flex-wrap">
                        <span className="font-mono">{c.chunk_id}</span>
                        {c.section_path && (
                          <span title={c.section_path}>{c.section_path}</span>
                        )}
                        <Badge>{c.data_type}</Badge>
                        <Badge variant="default">{c.tier}</Badge>
                        <span>{c.token_count} tok</span>
                        {c.page_number != null && <span>p.{c.page_number}</span>}
                      </div>
                      <div className="mt-1 text-xs whitespace-pre-wrap">
                        {c.text_preview}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>

          {/* Ad-hoc ingest */}
          <div className="rounded-lg border border-border bg-card overflow-hidden">
            <div className="px-3 py-2 border-b border-border">
              <h3 className="text-sm font-medium flex items-center gap-2">
                <Upload size={14} /> Ad-hoc ingest
              </h3>
            </div>
            <form onSubmit={onIngest} className="p-3 flex items-center gap-2 flex-wrap">
              <input
                type="text"
                value={ingestPath}
                onChange={(e) => setIngestPath(e.target.value)}
                placeholder="/mnt/shared/path/to/file.pdf"
                className="flex-1 min-w-[20rem] px-3 py-1.5 rounded-md border border-border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-primary/30"
              />
              <select
                value={ingestTier}
                onChange={(e) => setIngestTier(e.target.value)}
                className="px-2 py-1.5 rounded-md border border-border bg-background text-sm"
              >
                <option value="untiered">untiered</option>
                <option value="tier_1_foundations">tier 1 (foundations)</option>
                <option value="tier_2_standards">tier 2 (standards)</option>
                <option value="tier_3_research">tier 3 (research)</option>
                <option value="tier_4_software_docs">tier 4 (software docs)</option>
              </select>
              <SecondaryButton type="submit" disabled={ingest.isPending}>
                {ingest.isPending ? "Ingesting…" : "Ingest"}
              </SecondaryButton>
            </form>
            {ingestMessage && (
              <div
                className={
                  "mx-3 mb-3 text-xs rounded border px-2.5 py-1.5 " +
                  (ingestMessage.startsWith("Failed")
                    ? "border-red-500/40 bg-red-500/10 text-red-400"
                    : "border-emerald-500/40 bg-emerald-500/10 text-emerald-400")
                }
              >
                {ingestMessage}
              </div>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
