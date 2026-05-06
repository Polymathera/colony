/**
 * Design Monorepo tab — read-only view of a repo's `.colony/repo_map.yaml`,
 * its directory tree, a dry-run preview of the mmap_application_scope
 * calls the materialiser would issue, AND the entry point for actually
 * mapping the repo into VCM (per-source checkboxes + Map to VCM button +
 * confirmation modal). PUT-and-commit on the YAML is intentionally
 * deferred; the YAML viewer is a `<pre>` block, not Monaco.
 *
 * The colony's design-monorepo URL is configured on the LandingPage
 * (Colonies panel → pencil → Save). The tab reads the persisted URL
 * to pre-populate the inspector but does not write back here.
 */
import { useEffect, useMemo, useState } from "react";
import {
  useColonyDesignMonorepo,
  useRepoMap,
  useRepoMapPreview,
  useRepoTree,
  type RepoMapSource,
  type RepoTreeNode,
} from "@/api/hooks/useRepoMap";
import { useMapRepo } from "@/api/hooks/useVCM";

const DEFAULT_BRANCH = "main";

function getActiveColonyId(): string | null {
  const id = (window as any).__colony_active_colony_id;
  return typeof id === "string" && id.length > 0 ? id : null;
}

export function RepoMapTab() {
  const colonyId = getActiveColonyId();
  const persisted = useColonyDesignMonorepo(colonyId);

  const [originInput, setOriginInput] = useState("");
  const [branchInput, setBranchInput] = useState(DEFAULT_BRANCH);
  const [submitted, setSubmitted] = useState<{
    originUrl: string;
    branch: string;
  } | null>(null);

  // Pre-populate the inspector from the persisted colony config and
  // auto-load it once the URL is known. Paste-once UX.
  useEffect(() => {
    const url = persisted.data?.origin_url;
    if (!url) return;
    setOriginInput((curr) => curr || url);
    setBranchInput((curr) =>
      curr === DEFAULT_BRANCH ? persisted.data!.branch : curr,
    );
    setSubmitted((curr) =>
      curr ?? { originUrl: url, branch: persisted.data!.branch },
    );
  }, [persisted.data]);

  const repoMapQuery = useRepoMap(submitted);
  const treeQuery = useRepoTree(submitted, { maxDepth: 6, maxNodes: 5000 });
  const preview = useRepoMapPreview();
  const mapRepo = useMapRepo();

  // Per-source enabled-set. Reset to "all enabled" whenever the loaded
  // sources change — operator can then untick what they don't want.
  const sources = repoMapQuery.data?.sources ?? [];
  const sourcesKey = sources.map((s) => s.name).join("|");
  const [enabledNames, setEnabledNames] = useState<Set<string>>(new Set());
  useEffect(() => {
    setEnabledNames(new Set(sources.map((s) => s.name)));
    // sourcesKey collapses the array identity into a stable string so
    // we re-init only when the actual list of names changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sourcesKey]);

  const [confirmOpen, setConfirmOpen] = useState(false);

  const onLoad = (e: React.FormEvent) => {
    e.preventDefault();
    if (!originInput.trim()) return;
    setSubmitted({ originUrl: originInput.trim(), branch: branchInput.trim() || DEFAULT_BRANCH });
  };

  const onPreview = () => {
    if (!submitted) return;
    preview.mutate({
      origin_url: submitted.originUrl,
      branch: submitted.branch,
    });
  };

  const onConfirmMap = () => {
    if (!submitted) return;
    // ``enabled_sources`` is omitted when every source is ticked —
    // matches the backend's "None ⇒ map every row" contract and keeps
    // the request body small for the common case.
    const allEnabled = enabledNames.size === sources.length;
    mapRepo.mutate({
      origin_url: submitted.originUrl,
      branch: submitted.branch,
      ...(allEnabled ? {} : { enabled_sources: Array.from(enabledNames) }),
    });
    setConfirmOpen(false);
  };

  const toggleSource = (name: string) => {
    setEnabledNames((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const canMap = !!submitted && sources.length > 0 && enabledNames.size > 0;

  return (
    <div className="p-4 flex flex-col gap-4 h-full">
      <header className="flex flex-col gap-2">
        <h2 className="text-lg font-semibold">Design Monorepo</h2>
        <p className="text-xs text-muted-foreground max-w-3xl">
          Inspect the <code>.colony/repo_map.yaml</code> of a design
          monorepo, browse its directory tree, dry-run the materialiser
          to see exactly which VCM mappings the cluster would create,
          and trigger the actual mapping with the per-source checkboxes
          below. The colony's design-monorepo URL is configured on the
          landing page (Colonies panel → pencil → Save).
        </p>
        <form onSubmit={onLoad} className="flex flex-wrap items-center gap-2">
          <input
            value={originInput}
            onChange={(e) => setOriginInput(e.target.value)}
            placeholder="https://github.com/example/design-monorepo.git"
            className="px-2 py-1 rounded border bg-background text-sm w-[28rem]"
          />
          <input
            value={branchInput}
            onChange={(e) => setBranchInput(e.target.value)}
            placeholder="branch"
            className="px-2 py-1 rounded border bg-background text-sm w-32"
          />
          <button
            type="submit"
            className="px-3 py-1 rounded bg-primary text-primary-foreground text-sm"
          >
            Load
          </button>
          <button
            type="button"
            onClick={onPreview}
            disabled={!submitted}
            className="px-3 py-1 rounded border text-sm disabled:opacity-50"
          >
            Preview mmap calls
          </button>
          <button
            type="button"
            onClick={() => setConfirmOpen(true)}
            disabled={!canMap || mapRepo.isPending}
            className="px-3 py-1 rounded bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 text-sm disabled:opacity-50"
          >
            {mapRepo.isPending ? "Mapping…" : "Map to VCM"}
          </button>
        </form>
        {!colonyId ? (
          <div className="text-xs text-muted-foreground">
            No active colony. Pick one in the header dropdown or on the
            landing page.
          </div>
        ) : persisted.data?.origin_url ? (
          <div className="text-xs text-muted-foreground">
            Saved on this colony:{" "}
            <code>{persisted.data.origin_url}</code>{" "}
            (branch <code>{persisted.data.branch}</code>) — edit on the
            landing page.
          </div>
        ) : (
          <div className="text-xs text-muted-foreground">
            No design monorepo configured for the active colony{" "}
            (<code>{colonyId.slice(0, 16)}…</code>). Set one on the
            landing page (Colonies panel → pencil → Save), then refresh
            this tab. Until then, you can paste any URL above to
            inspect it ad-hoc.
          </div>
        )}
        {repoMapQuery.isError && (
          <div className="text-xs text-red-500">
            {(repoMapQuery.error as Error)?.message ?? "Failed to load repo map."}
          </div>
        )}
        {mapRepo.isSuccess && (
          <div className="text-xs text-emerald-400">
            Mapping started (op {mapRepo.data?.op_id ?? "?"}). Watch
            progress on the VCM tab.
          </div>
        )}
        {mapRepo.isError && (
          <div className="text-xs text-red-500">
            {(mapRepo.error as Error)?.message ?? "Map to VCM failed."}
          </div>
        )}
      </header>

      <div className="flex flex-1 gap-4 overflow-hidden">
        <section className="w-1/3 border rounded p-2 overflow-auto">
          <h3 className="text-sm font-medium mb-2">Tree</h3>
          {treeQuery.isLoading && <div className="text-xs">Loading…</div>}
          {treeQuery.data ? (
            <RepoTree node={treeQuery.data.root} depth={0} />
          ) : null}
        </section>

        <section className="flex-1 flex flex-col gap-3 overflow-hidden">
          <SourcesPanel
            hasFile={repoMapQuery.data?.has_repo_map_file ?? false}
            yaml={repoMapQuery.data?.raw_yaml ?? null}
            sources={sources}
            enabledNames={enabledNames}
            toggleSource={toggleSource}
            loading={repoMapQuery.isLoading}
          />
          {preview.data && (
            <PreviewPanel
              baseScopeId={preview.data.base_scope_id}
              sources={preview.data.sources}
            />
          )}
        </section>
      </div>

      {confirmOpen && submitted && (
        <ConfirmMapDialog
          originUrl={submitted.originUrl}
          branch={submitted.branch}
          enabledNames={Array.from(enabledNames)}
          totalSources={sources.length}
          onCancel={() => setConfirmOpen(false)}
          onConfirm={onConfirmMap}
        />
      )}
    </div>
  );
}

/* -------------------------------------------------------------------- */

function RepoTree({ node, depth }: { node: RepoTreeNode; depth: number }) {
  const [open, setOpen] = useState(depth < 1);
  const indent = { paddingLeft: `${depth * 12}px` };
  const label = node.path === "." ? "/" : node.path.split("/").pop();
  if (!node.is_dir) {
    return (
      <div style={indent} className="text-xs">
        {label}
      </div>
    );
  }
  return (
    <div>
      <div
        style={indent}
        className="text-xs cursor-pointer select-none"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? "▾" : "▸"} {label}/
      </div>
      {open &&
        node.children.map((c) => (
          <RepoTree key={c.path} node={c} depth={depth + 1} />
        ))}
    </div>
  );
}

function SourcesPanel({
  hasFile,
  yaml,
  sources,
  enabledNames,
  toggleSource,
  loading,
}: {
  hasFile: boolean;
  yaml: string | null;
  sources: RepoMapSource[];
  enabledNames: Set<string>;
  toggleSource: (name: string) => void;
  loading: boolean;
}) {
  return (
    <div className="border rounded flex flex-col overflow-hidden">
      <div className="px-3 py-2 border-b flex items-center justify-between">
        <h3 className="text-sm font-medium">repo_map.yaml</h3>
        <span className="text-xs text-muted-foreground">
          {loading
            ? "loading…"
            : hasFile
            ? "from .colony/repo_map.yaml"
            : "default fallback (no file present)"}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-0 overflow-hidden">
        {/* Left: per-source toggles. The user ticks rows they want
            mapped; the Map to VCM button passes the resulting set as
            ``enabled_sources``. */}
        <div className="border-r overflow-auto p-2">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1.5">
            Sources to map
          </div>
          {sources.length === 0 && (
            <div className="text-xs text-muted-foreground italic">
              no sources
            </div>
          )}
          {sources.map((s) => (
            <SourceCheckbox
              key={s.name}
              source={s}
              enabled={enabledNames.has(s.name)}
              onToggle={() => toggleSource(s.name)}
            />
          ))}
        </div>
        {/* Right: raw YAML for reference. */}
        <pre className="text-xs p-3 overflow-auto bg-muted/40 whitespace-pre-wrap font-mono">
          {yaml ?? renderDefaultSummary(sources)}
        </pre>
      </div>
    </div>
  );
}

function SourceCheckbox({
  source, enabled, onToggle,
}: {
  source: RepoMapSource;
  enabled: boolean;
  onToggle: () => void;
}) {
  // Surface the per-source paging knobs (and chunk knobs for
  // literature) so the operator sees what config each row uses
  // without having to read the YAML.
  const meta = useMemo(() => {
    const bits: string[] = [source.type];
    if (source.start_dir) bits.push(`@ ${source.start_dir}`);
    if (source.flush_threshold != null) bits.push(`flush=${source.flush_threshold}`);
    if (source.flush_token_budget != null) bits.push(`tok=${source.flush_token_budget}`);
    if (source.pinned) bits.push("pinned");
    if (source.chunk_target_tokens != null) bits.push(`chunk=${source.chunk_target_tokens}`);
    return bits.join(" · ");
  }, [source]);

  return (
    <label className="flex items-start gap-2 py-1 cursor-pointer hover:bg-accent/30 px-1 rounded">
      <input
        type="checkbox"
        checked={enabled}
        onChange={onToggle}
        className="mt-0.5"
      />
      <div className="flex flex-col">
        <span className="text-xs font-medium">{source.name}</span>
        <span className="text-[10px] text-muted-foreground">{meta}</span>
      </div>
    </label>
  );
}

function renderDefaultSummary(sources: RepoMapSource[]): string {
  if (sources.length === 0) return "(no sources)";
  return [
    "# No .colony/repo_map.yaml found — using the default fallback:",
    "schema_version: 1",
    "sources:",
    ...sources.map((s) => `  - { name: ${s.name}, type: ${s.type} }`),
  ].join("\n");
}

function PreviewPanel({
  baseScopeId,
  sources,
}: {
  baseScopeId: string;
  sources: { name: string; scope_id: string; mmap_kwargs: Record<string, unknown> }[];
}) {
  return (
    <div className="border rounded flex flex-col overflow-hidden">
      <div className="px-3 py-2 border-b">
        <h3 className="text-sm font-medium">
          Preview — mmap calls (base scope:{" "}
          <code className="text-xs">{baseScopeId}</code>)
        </h3>
      </div>
      <div className="overflow-auto p-2 flex flex-col gap-2">
        {sources.map((s) => (
          <div key={s.scope_id} className="border rounded p-2 text-xs">
            <div className="font-medium">
              {s.name}{" "}
              <span className="text-muted-foreground">→ {s.scope_id}</span>
            </div>
            <pre className="mt-1 text-[11px] whitespace-pre-wrap font-mono">
              {JSON.stringify(s.mmap_kwargs, null, 2)}
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}

function ConfirmMapDialog({
  originUrl, branch, enabledNames, totalSources, onCancel, onConfirm,
}: {
  originUrl: string;
  branch: string;
  enabledNames: string[];
  totalSources: number;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const allEnabled = enabledNames.length === totalSources;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-md rounded-lg border border-border bg-card shadow-xl">
        <div className="border-b border-border px-5 py-3">
          <h2 className="text-sm font-semibold">Map to VCM?</h2>
        </div>
        <div className="space-y-3 px-5 py-4 text-xs">
          <div>
            <div className="text-muted-foreground">Repository</div>
            <code className="break-all">{originUrl}</code>{" "}
            (branch <code>{branch}</code>)
          </div>
          <div>
            <div className="text-muted-foreground">Sources</div>
            <div>
              {allEnabled
                ? `All ${totalSources} source${totalSources === 1 ? "" : "s"} from repo_map.yaml`
                : `${enabledNames.length} of ${totalSources}: ${enabledNames.join(", ") || "(none)"}`}
            </div>
          </div>
          <div className="rounded border border-border bg-muted/30 px-3 py-2 text-muted-foreground">
            Mapping runs in the background on the cluster. Progress and
            results appear on the <strong>VCM</strong> tab.
          </div>
        </div>
        <div className="flex items-center justify-end gap-2 border-t border-border px-5 py-3">
          <button
            onClick={onCancel}
            className="rounded px-4 py-1.5 text-xs font-medium text-muted-foreground hover:text-foreground"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={enabledNames.length === 0}
            className="rounded bg-primary px-4 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          >
            Map to VCM
          </button>
        </div>
      </div>
    </div>
  );
}
