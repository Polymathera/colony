/**
 * LandingPage's "Colonies" panel: switch the active colony, create a
 * new one, and edit per-colony state inline (currently the design-
 * monorepo URL; two more slots are reserved for future per-colony
 * fields).
 *
 * The Design Monorepo dashboard tab inspects content; this panel
 * configures the colony itself, so it lives outside the session-tabs
 * gate. Aligns with the workflow:
 *
 *   1. Log in.
 *   2. Pick or create a colony here.
 *   3. Set the design-monorepo URL inline.
 *   4. Click "New Session" — the SessionAgent reads the URL from
 *      ``AgentMetadata.parameters[design_monorepo_url]``.
 */
import { useEffect, useState } from "react";
import { Pencil, Plus } from "lucide-react";
import {
  useColonies,
  useCreateColony,
  type ColonyInfo,
} from "@/api/hooks/useAuth";
import {
  useColonyDesignMonorepo,
  useColonyGitAttribution,
  useSetColonyDesignMonorepo,
  useSetColonyGitAttribution,
  type ColonyGitAttributionConfig,
} from "@/api/hooks/useRepoMap";


interface ColoniesSectionProps {
  activeColonyId: string | null;
  onSelectColony: (colonyId: string) => void;
}


export function ColoniesSection({
  activeColonyId, onSelectColony,
}: ColoniesSectionProps) {
  const colonies = useColonies();
  const [creating, setCreating] = useState(false);

  return (
    <div className="w-full max-w-md">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Colonies
        </h3>
        <button
          onClick={() => setCreating((v) => !v)}
          className="text-xs text-primary hover:underline flex items-center gap-1"
        >
          <Plus size={12} /> New colony
        </button>
      </div>
      {creating && (
        <NewColonyForm
          onClose={() => setCreating(false)}
          onCreated={(colonyId) => {
            // A new colony only matters if it becomes the active one —
            // otherwise the next session will still boot under whatever
            // colony was active before, and the Design Monorepo tab
            // ends up reading a different colony's row than the one
            // just edited.
            onSelectColony(colonyId);
          }}
        />
      )}
      <div className="rounded-lg border border-border bg-card divide-y divide-border">
        {(colonies.data ?? []).map((c) => (
          <ColonyRow
            key={c.colony_id}
            colony={c}
            isActive={c.colony_id === activeColonyId}
            onSelect={() => onSelectColony(c.colony_id)}
            onSelectColony={onSelectColony}
          />
        ))}
        {colonies.data && colonies.data.length === 0 && (
          <div className="px-4 py-3 text-xs text-muted-foreground">
            No colonies yet — click "New colony" to create one.
          </div>
        )}
      </div>
    </div>
  );
}


function NewColonyForm({
  onClose, onCreated,
}: {
  onClose: () => void;
  onCreated: (colonyId: string) => void;
}) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const create = useCreateColony();

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    try {
      const created = await create.mutateAsync({
        name: name.trim(),
        description: description.trim(),
      });
      onCreated(created.colony_id);
      onClose();
    } catch {
      // useCreateColony exposes the error; rendering below.
    }
  };

  return (
    <form
      onSubmit={submit}
      className="mb-2 rounded-lg border border-border bg-card p-3 flex flex-col gap-2"
    >
      <input
        autoFocus
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Colony name"
        className="px-2 py-1 rounded border bg-background text-sm"
      />
      <input
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="Description (optional)"
        className="px-2 py-1 rounded border bg-background text-sm"
      />
      <div className="flex gap-2">
        <button
          type="submit"
          disabled={!name.trim() || create.isPending}
          className="px-3 py-1 rounded bg-primary text-primary-foreground text-xs disabled:opacity-50"
        >
          {create.isPending ? "Creating…" : "Create"}
        </button>
        <button
          type="button"
          onClick={onClose}
          className="px-3 py-1 rounded border text-xs"
        >
          Cancel
        </button>
      </div>
      {create.error && (
        <div className="text-xs text-red-500">
          {(create.error as Error).message}
        </div>
      )}
    </form>
  );
}


function ColonyRow({
  colony, isActive, onSelect, onSelectColony,
}: {
  colony: ColonyInfo;
  isActive: boolean;
  onSelect: () => void;
  onSelectColony: (colonyId: string) => void;
}) {
  return (
    <div
      className={`px-4 py-3 ${
        isActive ? "bg-accent/30" : "hover:bg-accent/20"
      }`}
    >
      <button
        type="button"
        onClick={onSelect}
        className="flex w-full items-center justify-between text-left"
      >
        <div className="flex items-center gap-2">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              isActive ? "bg-emerald-400" : "bg-muted-foreground/40"
            }`}
          />
          <span className="text-sm font-medium text-foreground">
            {colony.name}
          </span>
          {colony.is_default && (
            <span className="text-[9px] uppercase tracking-wider text-muted-foreground">
              default
            </span>
          )}
        </div>
        <span className="text-[10px] text-muted-foreground font-mono">
          {colony.colony_id.slice(0, 16)}
        </span>
      </button>
      {/* Per-colony field slots. Each slot is a single line:
          ``label: <value | "Not configured"> [Edit]``.
          Slot 1 — design-monorepo URL.
          Slot 2 — git-commit attribution (principal + co-author + name/email).
          Slot 3 — reserved for future per-colony state. Add new rows
          here following the ``DesignMonorepoField`` pattern; keep the
          order stable so users see the same layout per colony. */}
      <div className="mt-2 flex flex-col gap-1">
        <DesignMonorepoField
          colonyId={colony.colony_id}
          onSelectColony={onSelectColony}
        />
        <GitAttributionField
          colonyId={colony.colony_id}
          onSelectColony={onSelectColony}
        />
        {/* TODO(per-colony-field-3): future slot. */}
      </div>
    </div>
  );
}


function DesignMonorepoField({
  colonyId, onSelectColony,
}: {
  colonyId: string;
  onSelectColony: (colonyId: string) => void;
}) {
  const cfg = useColonyDesignMonorepo(colonyId);
  const set = useSetColonyDesignMonorepo(colonyId);
  const [editing, setEditing] = useState(false);
  const [draftUrl, setDraftUrl] = useState("");
  const [draftBranch, setDraftBranch] = useState("main");
  // Transient confirmation flag — the save lands in Postgres
  // immediately (independent of cluster-ready state), but the form
  // collapses back to the read view on success which makes the save
  // easy to miss. A short-lived "Saved" indicator answers "did this
  // actually persist?" without needing a refresh.
  const [savedAt, setSavedAt] = useState<number | null>(null);

  const startEdit = () => {
    // Make this colony the active one before editing — otherwise the
    // user can save a URL onto colony A while colony B is active, and
    // the next session (which boots under the active colony) reads B's
    // empty row in the Design Monorepo tab.
    onSelectColony(colonyId);
    setDraftUrl(cfg.data?.origin_url ?? "");
    setDraftBranch(cfg.data?.branch ?? "main");
    setEditing(true);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!draftUrl.trim()) return;
    try {
      await set.mutateAsync({
        origin_url: draftUrl.trim(),
        branch: draftBranch.trim() || "main",
      });
      cfg.refetch();
      setEditing(false);
      setSavedAt(Date.now());
    } catch {
      // useSetColonyDesignMonorepo exposes the error; rendered below.
    }
  };

  useEffect(() => {
    if (savedAt === null) return;
    const t = setTimeout(() => setSavedAt(null), 2500);
    return () => clearTimeout(t);
  }, [savedAt]);

  if (editing) {
    return (
      <form onSubmit={submit} className="flex flex-col gap-1">
        <div className="flex flex-wrap gap-1">
          <span className="text-[10px] text-muted-foreground self-center">
            Design monorepo:
          </span>
          <input
            autoFocus
            value={draftUrl}
            onChange={(e) => setDraftUrl(e.target.value)}
            placeholder="https://github.com/me/repo.git"
            className="px-1.5 py-0.5 rounded border bg-background text-xs flex-1 min-w-[14rem]"
          />
          <input
            value={draftBranch}
            onChange={(e) => setDraftBranch(e.target.value)}
            placeholder="branch"
            className="px-1.5 py-0.5 rounded border bg-background text-xs w-20"
          />
          <button
            type="submit"
            disabled={!draftUrl.trim() || set.isPending}
            className="px-2 py-0.5 rounded bg-primary text-primary-foreground text-xs disabled:opacity-50"
          >
            {set.isPending ? "…" : "Save"}
          </button>
          <button
            type="button"
            onClick={() => setEditing(false)}
            className="px-2 py-0.5 rounded border text-xs"
          >
            Cancel
          </button>
        </div>
        {set.error && (
          <span className="text-[10px] text-red-500">
            {(set.error as Error).message}
          </span>
        )}
      </form>
    );
  }

  const url = cfg.data?.origin_url;
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="text-muted-foreground">Design monorepo:</span>
      {url ? (
        <code className="text-foreground truncate max-w-[18rem]" title={url}>
          {url}
        </code>
      ) : (
        <span className="text-muted-foreground italic">Not configured</span>
      )}
      <button
        type="button"
        onClick={startEdit}
        className="text-muted-foreground hover:text-foreground"
        title="Edit"
      >
        <Pencil size={11} />
      </button>
      {savedAt !== null && (
        <span className="text-[10px] text-emerald-400">Saved.</span>
      )}
    </div>
  );
}


function GitAttributionField({
  colonyId, onSelectColony,
}: {
  colonyId: string;
  onSelectColony: (colonyId: string) => void;
}) {
  // Per-commit attribution preference shown as a single read-only
  // line, with an inline edit form that toggles open. Mirrors
  // ``DesignMonorepoField`` shape so the Colonies panel feels
  // consistent.
  //
  // Per-user identity (``git_user_name`` / ``git_user_email``) lives
  // on the user profile now, OAuth-verified from GitHub. See
  // colony/github_identity_fix_plan.md.
  const cfg = useColonyGitAttribution(colonyId);
  const set = useSetColonyGitAttribution(colonyId);
  const [editing, setEditing] = useState(false);
  const [draftPrincipal, setDraftPrincipal] = useState("colony");
  const [draftCoAuthor, setDraftCoAuthor] = useState<string>("user");
  const [savedAt, setSavedAt] = useState<number | null>(null);

  const startEdit = () => {
    onSelectColony(colonyId);
    setDraftPrincipal(cfg.data?.commit_principal ?? "colony");
    setDraftCoAuthor(cfg.data?.commit_co_author ?? "");
    setEditing(true);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const payload: ColonyGitAttributionConfig = {
      commit_principal: draftPrincipal.trim() || "colony",
      // Empty input → no co-author trailer.
      commit_co_author: draftCoAuthor.trim() || null,
    };
    try {
      await set.mutateAsync(payload);
      cfg.refetch();
      setEditing(false);
      setSavedAt(Date.now());
    } catch {
      // useSetColonyGitAttribution surfaces the error below.
    }
  };

  useEffect(() => {
    if (savedAt === null) return;
    const t = setTimeout(() => setSavedAt(null), 2500);
    return () => clearTimeout(t);
  }, [savedAt]);

  if (editing) {
    return (
      <form onSubmit={submit} className="flex flex-col gap-1">
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-[10px] text-muted-foreground self-center">
            Commit attribution:
          </span>
          <input
            autoFocus
            value={draftPrincipal}
            onChange={(e) => setDraftPrincipal(e.target.value)}
            placeholder="principal (colony / user / agent / …)"
            title="Free-form. Well-known: user, colony, agent. Anything else is treated as an agent-type label."
            className="px-1.5 py-0.5 rounded border bg-background text-xs w-44"
          />
          <span className="text-[10px] text-muted-foreground">+</span>
          <input
            value={draftCoAuthor}
            onChange={(e) => setDraftCoAuthor(e.target.value)}
            placeholder="co-author (blank = none)"
            title="Same value space as principal. Blank disables the Co-Authored-By: trailer."
            className="px-1.5 py-0.5 rounded border bg-background text-xs w-40"
          />
        </div>
        <div className="flex flex-wrap items-center gap-1">
          <button
            type="submit"
            disabled={set.isPending}
            className="px-2 py-0.5 rounded bg-primary text-primary-foreground text-xs disabled:opacity-50"
          >
            {set.isPending ? "…" : "Save"}
          </button>
          <button
            type="button"
            onClick={() => setEditing(false)}
            className="px-2 py-0.5 rounded border text-xs"
          >
            Cancel
          </button>
        </div>
        {set.error && (
          <span className="text-[10px] text-red-500">
            {(set.error as Error).message}
          </span>
        )}
      </form>
    );
  }

  const principal = cfg.data?.commit_principal ?? "colony";
  const coAuthor = cfg.data?.commit_co_author;
  return (
    <div className="flex items-center gap-2 text-[11px]">
      <span className="text-muted-foreground">Commit attribution:</span>
      <code className="text-foreground">
        {principal}
        {coAuthor ? ` + ${coAuthor}` : ""}
      </code>
      <button
        type="button"
        onClick={startEdit}
        className="text-muted-foreground hover:text-foreground"
        title="Edit"
      >
        <Pencil size={11} />
      </button>
      {savedAt !== null && (
        <span className="text-[10px] text-emerald-400">Saved.</span>
      )}
    </div>
  );
}
