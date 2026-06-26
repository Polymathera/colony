/**
 * Design Monorepo tab — read-only view of a repo's `.colony/repo_map.yaml`,
 * its directory tree, a dry-run preview of the mmap_application_scope
 * calls the materialiser would issue, AND the entry points for the two
 * orthogonal materialisation operations:
 *
 *   - "Map to VCM"        → ``vcm_sources:``        → ``/vcm/map``
 *   - "Ingest Knowledge"  → ``knowledge_sources:``  → ``/kb/ingest-repo-map``
 *
 * Each section gets its own checkbox list — ticks on one side never
 * imply ticks on the other. The colony's design-monorepo URL is
 * configured on the LandingPage (Colonies panel → pencil → Save).
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Tree, type NodeRendererProps } from "react-arborist";
import { Highlight, themes } from "prism-react-renderer";
import {
  ChevronDown,
  ChevronRight,
  File as FileIcon,
  FileCode,
  FileText,
  Folder,
  FolderOpen,
} from "lucide-react";
import {
  useColonyDesignMonorepo,
  useColonyEnabledKnowledgeSources,
  useColonyEnabledVcmSources,
  useRepoMap,
  useRepoMapPreview,
  useRepoTree,
  useSetColonyEnabledKnowledgeSources,
  useSetColonyEnabledVcmSources,
  type KnowledgeSourceRow,
  type RepoTreeNode,
  type VcmSourceRow,
} from "@/api/hooks/useRepoMap";
import { useMapRepo } from "@/api/hooks/useVCM";
import { useKBIngestRepoMap, useKBRehydrate } from "@/api/hooks/useKB";

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
  const ingestKb = useKBIngestRepoMap();
  const rehydrateKg = useKBRehydrate();

  // Two independent persisted selections — one per panel. The
  // dashboard hydrates the local state from the colony's persisted
  // source selection so the same checkboxes the operator left on
  // last visit (and the same selection the chat-driven action sees)
  // are reflected here. On every toggle we PUT the new set; the
  // backend / agent reads the persisted value at action time —
  // there is no request-body filter on the buttons.
  const vcmSources = repoMapQuery.data?.vcm_sources ?? [];
  const kbSources = repoMapQuery.data?.knowledge_sources ?? [];
  const vcmSourcesKey = vcmSources.map((s) => s.name).join("|");
  const kbSourcesKey = kbSources.map((s) => s.name).join("|");

  const persistedVcm = useColonyEnabledVcmSources(colonyId);
  const persistedKb = useColonyEnabledKnowledgeSources(colonyId);
  const setPersistedVcm = useSetColonyEnabledVcmSources(colonyId);
  const setPersistedKb = useSetColonyEnabledKnowledgeSources(colonyId);

  const [enabledVcm, setEnabledVcm] = useState<Set<string>>(new Set());
  const [enabledKb, setEnabledKb] = useState<Set<string>>(new Set());

  // Hydrate from persisted state when the loaded rows change.
  // ``persisted.enabled === null`` means "all enabled" — the default
  // before the operator has ever toggled. ``enabled`` ⊆ rows; rows
  // outside the persisted set are unchecked.
  useEffect(() => {
    const persisted = persistedVcm.data?.enabled;
    if (persisted === null || persisted === undefined) {
      setEnabledVcm(new Set(vcmSources.map((s) => s.name)));
    } else {
      const rowNames = new Set(vcmSources.map((s) => s.name));
      setEnabledVcm(new Set(persisted.filter((n) => rowNames.has(n))));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vcmSourcesKey, persistedVcm.data]);
  useEffect(() => {
    const persisted = persistedKb.data?.enabled;
    if (persisted === null || persisted === undefined) {
      setEnabledKb(new Set(kbSources.map((s) => s.name)));
    } else {
      const rowNames = new Set(kbSources.map((s) => s.name));
      setEnabledKb(new Set(persisted.filter((n) => rowNames.has(n))));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kbSourcesKey, persistedKb.data]);

  const [confirmMapOpen, setConfirmMapOpen] = useState(false);
  const [confirmIngestOpen, setConfirmIngestOpen] = useState(false);

  const onLoad = (e: React.FormEvent) => {
    e.preventDefault();
    if (!originInput.trim()) return;
    setSubmitted({
      originUrl: originInput.trim(),
      branch: branchInput.trim() || DEFAULT_BRANCH,
    });
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
    // Selection lives in the colony's persisted source-selection
    // state; the backend reads it at action time. No body filter.
    mapRepo.mutate({
      origin_url: submitted.originUrl,
      branch: submitted.branch,
    });
    setConfirmMapOpen(false);
  };

  const onConfirmIngest = () => {
    if (!submitted) return;
    ingestKb.mutate({
      origin_url: submitted.originUrl,
      branch: submitted.branch,
    });
    setConfirmIngestOpen(false);
  };

  const onRehydrate = () => {
    if (!submitted) return;
    rehydrateKg.mutate({
      origin_url: submitted.originUrl,
      branch: submitted.branch,
    });
  };

  // Persist on every checkbox toggle so the chat-driven
  // ``ingest_repo_map_literature`` action sees the operator's
  // current selection without a parameter. ``null`` means "all
  // enabled" — the convention shared by the materialiser.
  const persistVcm = (next: Set<string>) => {
    if (!colonyId) return;
    const all = next.size === vcmSources.length;
    setPersistedVcm.mutate(all ? null : Array.from(next));
  };
  const persistKb = (next: Set<string>) => {
    if (!colonyId) return;
    const all = next.size === kbSources.length;
    setPersistedKb.mutate(all ? null : Array.from(next));
  };

  const toggleVcm = (name: string) => {
    setEnabledVcm((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      persistVcm(next);
      return next;
    });
  };
  const toggleKb = (name: string) => {
    setEnabledKb((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      persistKb(next);
      return next;
    });
  };

  const canMap = !!submitted && vcmSources.length > 0 && enabledVcm.size > 0;
  const canIngest = !!submitted && kbSources.length > 0 && enabledKb.size > 0;

  return (
    <div className="p-4 flex flex-col gap-4 h-full">
      <header className="flex flex-col gap-3">
        <div>
          <h2 className="text-lg font-semibold">Design Monorepo</h2>
          <p className="text-xs text-muted-foreground max-w-3xl mt-1">
            Inspect <code>.colony/repo_map.yaml</code>, browse the
            directory tree, dry-run the materialiser, and trigger the
            actual mapping into VCM. The colony's design-monorepo URL is
            configured on the landing page (Colonies → pencil → Save).
          </p>
        </div>

        {/* Row 1: source inputs. Row 2: action cluster, right-aligned.
            Two rows beats one wrapping row — keeps the inputs readable
            at any width and keeps actions visually grouped. */}
        <form onSubmit={onLoad} className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <input
              value={originInput}
              onChange={(e) => setOriginInput(e.target.value)}
              placeholder="https://github.com/example/design-monorepo.git"
              className="px-3 py-1.5 rounded-md border border-border bg-background text-sm flex-1 min-w-[20rem] focus:outline-none focus:ring-2 focus:ring-primary/30"
            />
            <input
              value={branchInput}
              onChange={(e) => setBranchInput(e.target.value)}
              placeholder="branch"
              className="px-3 py-1.5 rounded-md border border-border bg-background text-sm w-32 focus:outline-none focus:ring-2 focus:ring-primary/30"
            />
          </div>
          <div className="flex items-center justify-end gap-2">
            <SecondaryButton type="submit">Load</SecondaryButton>
            <SecondaryButton
              type="button"
              onClick={onPreview}
              disabled={!submitted}
            >
              Preview mmap calls
            </SecondaryButton>
            <PrimaryButton
              type="button"
              onClick={() => setConfirmMapOpen(true)}
              disabled={!canMap || mapRepo.isPending}
            >
              {mapRepo.isPending ? "Mapping…" : "Map to VCM"}
            </PrimaryButton>
            <PrimaryButton
              type="button"
              onClick={() => setConfirmIngestOpen(true)}
              disabled={!canIngest || ingestKb.isPending}
            >
              {ingestKb.isPending ? "Ingesting…" : "Ingest Knowledge"}
            </PrimaryButton>
            <SecondaryButton
              type="button"
              onClick={onRehydrate}
              disabled={!submitted || rehydrateKg.isPending}
              title="Load the KG snapshot from origin/<branch> into the shared Kùzu store."
            >
              {rehydrateKg.isPending ? "Rehydrating…" : "Rehydrate KG"}
            </SecondaryButton>
          </div>
        </form>

        <Banner
          colony={colonyId}
          persistedUrl={persisted.data?.origin_url ?? null}
          persistedBranch={persisted.data?.branch ?? null}
          loadError={
            repoMapQuery.isError
              ? (repoMapQuery.error as Error)?.message ?? "Failed to load repo map."
              : null
          }
          mapSuccess={
            mapRepo.isSuccess
              ? `Mapping started (op ${mapRepo.data?.op_id ?? "?"}). Watch progress on the VCM tab.`
              : null
          }
          mapError={
            mapRepo.isError
              ? (mapRepo.error as Error)?.message ?? "Map to VCM failed."
              : null
          }
          ingestSuccess={
            ingestKb.isSuccess
              ? `KB ingest started (op ${ingestKb.data?.op_id ?? "?"}). Watch progress on the Knowledge Base tab.`
              : null
          }
          ingestError={
            ingestKb.isError
              ? (ingestKb.error as Error)?.message ?? "KB ingest failed."
              : null
          }
        />
      </header>

      <div className="flex flex-1 gap-4 overflow-hidden min-h-0">
        <section className="w-72 shrink-0 rounded-lg border border-border bg-card flex flex-col overflow-hidden">
          <div className="px-3 py-2 border-b border-border">
            <h3 className="text-sm font-medium">Tree</h3>
          </div>
          <div className="flex-1 overflow-hidden">
            {treeQuery.isLoading && (
              <div className="text-xs text-muted-foreground p-3">Loading…</div>
            )}
            {treeQuery.data && <RepoTreeView root={treeQuery.data.root} />}
          </div>
        </section>

        <section className="flex-1 flex flex-col gap-3 overflow-hidden min-w-0">
          <SourcesPanel
            hasFile={repoMapQuery.data?.has_repo_map_file ?? false}
            yaml={repoMapQuery.data?.raw_yaml ?? null}
            vcmSources={vcmSources}
            kbSources={kbSources}
            enabledVcm={enabledVcm}
            enabledKb={enabledKb}
            toggleVcm={toggleVcm}
            toggleKb={toggleKb}
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

      {confirmMapOpen && submitted && (
        <ConfirmDialog
          title="Map to VCM?"
          actionLabel="Map to VCM"
          originUrl={submitted.originUrl}
          branch={submitted.branch}
          enabledNames={Array.from(enabledVcm)}
          totalSources={vcmSources.length}
          tailMessage="Mapping runs in the background on the cluster. Progress and results appear on the VCM tab."
          onCancel={() => setConfirmMapOpen(false)}
          onConfirm={onConfirmMap}
        />
      )}
      {confirmIngestOpen && submitted && (
        <ConfirmDialog
          title="Ingest into Knowledge Base?"
          actionLabel="Ingest"
          originUrl={submitted.originUrl}
          branch={submitted.branch}
          enabledNames={Array.from(enabledKb)}
          totalSources={kbSources.length}
          tailMessage="Ingestion runs in the background on the cluster. Progress and results appear on the Knowledge Base tab."
          onCancel={() => setConfirmIngestOpen(false)}
          onConfirm={onConfirmIngest}
        />
      )}
    </div>
  );
}

/* -------------------------------------------------------------------- */
/* Buttons — two styles only: PrimaryButton (the CTA), SecondaryButton  */
/* (everything else). Local to this tab to avoid premature shared-      */
/* primitive churn across the dashboard.                                */
/* -------------------------------------------------------------------- */

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

/* -------------------------------------------------------------------- */
/* Status banner — colony-context + load/map states, all in one strip.  */
/* Replaces the old vertical pile of mismatched status divs.            */
/* -------------------------------------------------------------------- */

function Banner({
  colony, persistedUrl, persistedBranch, loadError,
  mapSuccess, mapError, ingestSuccess, ingestError,
}: {
  colony: string | null;
  persistedUrl: string | null;
  persistedBranch: string | null;
  loadError: string | null;
  mapSuccess: string | null;
  mapError: string | null;
  ingestSuccess: string | null;
  ingestError: string | null;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      {!colony ? (
        <div className="text-xs text-muted-foreground">
          No active colony. Pick one in the header dropdown or on the
          landing page.
        </div>
      ) : persistedUrl ? (
        <div className="text-xs text-muted-foreground">
          Saved on this colony: <code>{persistedUrl}</code> (branch{" "}
          <code>{persistedBranch}</code>) — edit on the landing page.
        </div>
      ) : (
        <div className="text-xs text-muted-foreground">
          No design monorepo configured for the active colony{" "}
          (<code>{colony.slice(0, 16)}…</code>). Set one on the landing
          page (Colonies → pencil → Save), then refresh this tab.
        </div>
      )}
      {loadError && (
        <div className="text-xs rounded border border-red-500/40 bg-red-500/10 px-2.5 py-1.5 text-red-400">
          {loadError}
        </div>
      )}
      {mapSuccess && (
        <div className="text-xs rounded border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-1.5 text-emerald-400">
          {mapSuccess}
        </div>
      )}
      {mapError && (
        <div className="text-xs rounded border border-red-500/40 bg-red-500/10 px-2.5 py-1.5 text-red-400">
          {mapError}
        </div>
      )}
      {ingestSuccess && (
        <div className="text-xs rounded border border-emerald-500/40 bg-emerald-500/10 px-2.5 py-1.5 text-emerald-400">
          {ingestSuccess}
        </div>
      )}
      {ingestError && (
        <div className="text-xs rounded border border-red-500/40 bg-red-500/10 px-2.5 py-1.5 text-red-400">
          {ingestError}
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------------------- */
/* Tree — react-arborist + lucide icons.                                */
/* -------------------------------------------------------------------- */

interface ArboristNode {
  id: string;
  name: string;
  isDir: boolean;
  children?: ArboristNode[];
}

function toArborist(node: RepoTreeNode): ArboristNode {
  return {
    id: node.path,
    name: node.path === "." ? "/" : node.path.split("/").pop() ?? node.path,
    isDir: node.is_dir,
    children: node.is_dir ? node.children.map(toArborist) : undefined,
  };
}

function RepoTreeView({ root }: { root: RepoTreeNode }) {
  // arborist requires an array at the top level; expose root's
  // children directly when root is the synthetic ".", otherwise wrap.
  const data = useMemo<ArboristNode[]>(() => {
    const rootNode = toArborist(root);
    return root.path === "." ? rootNode.children ?? [] : [rootNode];
  }, [root]);
  const { ref, height } = useElementSize<HTMLDivElement>();

  return (
    <div ref={ref} className="h-full overflow-hidden">
      <Tree<ArboristNode>
        data={data}
        openByDefault={false}
        width="100%"
        height={height || 1}
        indent={16}
        rowHeight={24}
        paddingTop={4}
        paddingBottom={4}
        disableDrag
        disableDrop
        disableEdit
        disableMultiSelection
        className="text-xs"
      >
        {TreeNode}
      </Tree>
    </div>
  );
}

/**
 * Track the rendered height of an element so virtualised children
 * (react-arborist's ``<Tree>``) get a real number rather than the
 * fill-parent CSS the rest of the layout relies on. ResizeObserver
 * is on every browser the dashboard targets.
 */
function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [height, setHeight] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const h = entries[0]?.contentRect.height ?? 0;
      setHeight(Math.floor(h));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return { ref, height };
}

function iconForFile(name: string) {
  const ext = name.includes(".") ? name.slice(name.lastIndexOf(".") + 1).toLowerCase() : "";
  const code = new Set([
    "py", "ts", "tsx", "js", "jsx", "go", "rs", "java", "c", "cpp", "h",
    "hpp", "rb", "sh", "yaml", "yml", "toml", "json", "ipynb",
  ]);
  const text = new Set(["md", "txt", "rst", "csv", "tsv", "log"]);
  if (code.has(ext)) return FileCode;
  if (text.has(ext)) return FileText;
  return FileIcon;
}

function TreeNode({ node, style, dragHandle }: NodeRendererProps<ArboristNode>) {
  const isDir = node.data.isDir;
  const Chevron = node.isOpen ? ChevronDown : ChevronRight;
  const FolderGlyph = node.isOpen ? FolderOpen : Folder;
  const FileGlyph = iconForFile(node.data.name);

  return (
    <div
      ref={dragHandle}
      style={style}
      className={
        "flex items-center gap-1.5 px-2 cursor-pointer select-none rounded-sm " +
        "hover:bg-accent/30 " +
        (node.isSelected ? "bg-accent/40 " : "")
      }
      onClick={() => isDir && node.toggle()}
    >
      {isDir ? (
        <Chevron size={12} className="shrink-0 text-muted-foreground" />
      ) : (
        <span className="w-3 shrink-0" />
      )}
      {isDir ? (
        <FolderGlyph size={14} className="shrink-0 text-sky-500/80" />
      ) : (
        <FileGlyph size={14} className="shrink-0 text-muted-foreground" />
      )}
      <span className="truncate">{node.data.name}</span>
    </div>
  );
}

/* -------------------------------------------------------------------- */
/* Sources / YAML panel.                                                */
/* -------------------------------------------------------------------- */

function SourcesPanel({
  hasFile, yaml, vcmSources, kbSources,
  enabledVcm, enabledKb, toggleVcm, toggleKb, loading,
}: {
  hasFile: boolean;
  yaml: string | null;
  vcmSources: VcmSourceRow[];
  kbSources: KnowledgeSourceRow[];
  enabledVcm: Set<string>;
  enabledKb: Set<string>;
  toggleVcm: (name: string) => void;
  toggleKb: (name: string) => void;
  loading: boolean;
}) {
  return (
    <div className="rounded-lg border border-border bg-card flex flex-col overflow-hidden min-h-0">
      <div className="px-3 py-2 border-b border-border flex items-center justify-between">
        <h3 className="text-sm font-medium">repo_map.yaml</h3>
        <span className="text-[11px] text-muted-foreground">
          {loading
            ? "loading…"
            : hasFile
            ? "from .colony/repo_map.yaml"
            : "default fallback (no file present)"}
        </span>
      </div>
      {/* Two stacked checkbox columns (VCM + KB), then the YAML view.
          The two operations are orthogonal — ticks on one side never
          imply ticks on the other. */}
      <div
        className="grid overflow-hidden min-h-0"
        style={{ gridTemplateColumns: "18rem 1fr" }}
      >
        <div className="border-r border-border overflow-auto p-3 flex flex-col gap-4">
          <CheckboxList
            label="VCM sources"
            empty="no vcm_sources"
            rows={vcmSources.map((s) => ({
              name: s.name,
              meta: vcmRowMeta(s),
              enabled: enabledVcm.has(s.name),
              onToggle: () => toggleVcm(s.name),
            }))}
          />
          <CheckboxList
            label="Knowledge sources"
            empty="no knowledge_sources"
            rows={kbSources.map((s) => ({
              name: s.name,
              meta: kbRowMeta(s),
              enabled: enabledKb.has(s.name),
              onToggle: () => toggleKb(s.name),
            }))}
          />
        </div>
        <YamlView text={yaml ?? renderDefaultSummary(vcmSources)} />
      </div>
    </div>
  );
}

function vcmRowMeta(source: VcmSourceRow): string {
  const bits: string[] = [source.type];
  if (source.start_dir) bits.push(`@ ${source.start_dir}`);
  if (source.flush_threshold != null) bits.push(`flush=${source.flush_threshold}`);
  if (source.flush_token_budget != null) bits.push(`tok=${source.flush_token_budget}`);
  if (source.pinned) bits.push("pinned");
  if (source.chunk_target_tokens != null) bits.push(`chunk=${source.chunk_target_tokens}`);
  return bits.join(" · ");
}

function kbRowMeta(source: KnowledgeSourceRow): string {
  const bits: string[] = [];
  if (source.profile) bits.push(source.profile);
  if (source.paths.length > 0) bits.push(source.paths.join(", "));
  return bits.join(" · ");
}

function YamlView({ text }: { text: string }) {
  return (
    <Highlight code={text} language="yaml" theme={themes.vsDark}>
      {({ className, style, tokens, getLineProps, getTokenProps }) => (
        <pre
          className={
            (className ?? "") +
            " text-[11px] leading-5 font-mono p-3 overflow-auto m-0 bg-muted/40"
          }
          style={{ ...style, background: undefined }}
        >
          {tokens.map((line, i) => {
            const lineProps = getLineProps({ line, key: i });
            return (
              <div
                key={i}
                className={lineProps.className}
                style={lineProps.style}
              >
                <span className="inline-block w-7 pr-2 text-right text-muted-foreground/60 select-none">
                  {i + 1}
                </span>
                {line.map((token, j) => {
                  const tp = getTokenProps({ token, key: j });
                  return (
                    <span key={j} className={tp.className} style={tp.style}>
                      {tp.children}
                    </span>
                  );
                })}
              </div>
            );
          })}
        </pre>
      )}
    </Highlight>
  );
}

interface CheckboxRow {
  name: string;
  meta: string;
  enabled: boolean;
  onToggle: () => void;
}

function CheckboxList({
  label, empty, rows,
}: {
  label: string;
  empty: string;
  rows: CheckboxRow[];
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-2">
        {label}
      </div>
      {rows.length === 0 ? (
        <div className="text-xs text-muted-foreground italic">{empty}</div>
      ) : (
        <div className="flex flex-col gap-0.5">
          {rows.map((row) => (
            <label
              key={row.name}
              className="flex items-start gap-2 py-1 px-1 cursor-pointer hover:bg-accent/30 rounded"
            >
              <input
                type="checkbox"
                checked={row.enabled}
                onChange={row.onToggle}
                className="mt-0.5 accent-primary"
              />
              <div className="flex flex-col min-w-0">
                <span className="text-xs font-medium truncate">{row.name}</span>
                <span className="text-[10px] text-muted-foreground truncate">
                  {row.meta}
                </span>
              </div>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

function renderDefaultSummary(sources: VcmSourceRow[]): string {
  if (sources.length === 0) return "(no sources)";
  return [
    "# No .colony/repo_map.yaml found — using the default fallback:",
    "schema_version: 1",
    "vcm_sources:",
    ...sources.map((s) => `  - { name: ${s.name}, type: ${s.type} }`),
  ].join("\n");
}

/* -------------------------------------------------------------------- */
/* Preview panel + Confirm dialog — same shape as before, restyled to   */
/* match the unified card / button language.                            */
/* -------------------------------------------------------------------- */

function PreviewPanel({
  baseScopeId, sources,
}: {
  baseScopeId: string;
  sources: { name: string; scope_id: string; mmap_kwargs: Record<string, unknown> }[];
}) {
  return (
    <div className="rounded-lg border border-border bg-card flex flex-col overflow-hidden">
      <div className="px-3 py-2 border-b border-border">
        <h3 className="text-sm font-medium">
          Preview — mmap calls
          <span className="ml-2 text-xs text-muted-foreground font-normal">
            base scope <code>{baseScopeId}</code>
          </span>
        </h3>
      </div>
      <div className="overflow-auto p-3 flex flex-col gap-2">
        {sources.map((s) => (
          <div
            key={s.scope_id}
            className="rounded-md border border-border bg-background/60 p-2 text-xs"
          >
            <div className="font-medium">
              {s.name}{" "}
              <span className="text-muted-foreground font-normal">
                → {s.scope_id}
              </span>
            </div>
            <pre className="mt-1 text-[11px] whitespace-pre-wrap font-mono text-muted-foreground">
              {JSON.stringify(s.mmap_kwargs, null, 2)}
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}

function ConfirmDialog({
  title, actionLabel, originUrl, branch,
  enabledNames, totalSources, tailMessage, onCancel, onConfirm,
}: {
  title: string;
  actionLabel: string;
  originUrl: string;
  branch: string;
  enabledNames: string[];
  totalSources: number;
  tailMessage: string;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const allEnabled = enabledNames.length === totalSources;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="w-full max-w-md rounded-lg border border-border bg-card shadow-xl">
        <div className="border-b border-border px-5 py-3">
          <h2 className="text-sm font-semibold">{title}</h2>
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
                ? `All ${totalSources} row${totalSources === 1 ? "" : "s"} from repo_map.yaml`
                : `${enabledNames.length} of ${totalSources}: ${enabledNames.join(", ") || "(none)"}`}
            </div>
          </div>
          <div className="rounded-md border border-border bg-muted/30 px-3 py-2 text-muted-foreground">
            {tailMessage}
          </div>
        </div>
        <div className="flex items-center justify-end gap-2 border-t border-border px-5 py-3">
          <SecondaryButton onClick={onCancel}>Cancel</SecondaryButton>
          <PrimaryButton onClick={onConfirm} disabled={enabledNames.length === 0}>
            {actionLabel}
          </PrimaryButton>
        </div>
      </div>
    </div>
  );
}
