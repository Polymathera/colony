/**
 * Design Monorepo tab — read-only view of a repo's `.colony/repo_map.yaml`,
 * its directory tree, and a dry-run preview of the mmap_application_scope
 * calls the materialiser would issue. PUT-and-commit on the YAML is
 * intentionally deferred; the YAML viewer is a `<pre>` block, not Monaco.
 *
 * The colony's design-monorepo URL is configured on the LandingPage
 * (Colonies panel → pencil → Save). The tab reads the persisted URL
 * to pre-populate the inspector but does not write back here.
 */
import { useEffect, useState } from "react";
import {
  useColonyDesignMonorepo,
  useRepoMap,
  useRepoMapPreview,
  useRepoTree,
  type RepoMapSource,
  type RepoTreeNode,
} from "@/api/hooks/useRepoMap";

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

  return (
    <div className="p-4 flex flex-col gap-4 h-full">
      <header className="flex flex-col gap-2">
        <h2 className="text-lg font-semibold">Design Monorepo</h2>
        <p className="text-xs text-muted-foreground max-w-3xl">
          Inspect the <code>.colony/repo_map.yaml</code> of a design
          monorepo, browse its directory tree, and dry-run the
          materialiser to see exactly which VCM mappings the cluster
          would create. The colony's design-monorepo URL is configured
          on the landing page (Colonies panel → pencil → Save).
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
        </form>
        {persisted.data?.origin_url && (
          <div className="text-xs text-muted-foreground">
            Saved on this colony:{" "}
            <code>{persisted.data.origin_url}</code>{" "}
            (branch <code>{persisted.data.branch}</code>) — edit on the
            landing page.
          </div>
        )}
        {repoMapQuery.isError && (
          <div className="text-xs text-red-500">
            {(repoMapQuery.error as Error)?.message ?? "Failed to load repo map."}
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
          <YamlPanel
            hasFile={repoMapQuery.data?.has_repo_map_file ?? false}
            yaml={repoMapQuery.data?.raw_yaml ?? null}
            sources={repoMapQuery.data?.sources ?? []}
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

function YamlPanel({
  hasFile,
  yaml,
  sources,
  loading,
}: {
  hasFile: boolean;
  yaml: string | null;
  sources: RepoMapSource[];
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
      <pre className="text-xs p-3 overflow-auto flex-1 bg-muted/40 whitespace-pre-wrap font-mono">
        {yaml ?? renderDefaultSummary(sources)}
      </pre>
    </div>
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
