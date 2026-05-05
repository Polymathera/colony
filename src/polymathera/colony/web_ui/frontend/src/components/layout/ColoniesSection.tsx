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
import { useState } from "react";
import { Pencil, Plus } from "lucide-react";
import {
  useColonies,
  useCreateColony,
  type ColonyInfo,
} from "@/api/hooks/useAuth";
import {
  useColonyDesignMonorepo,
  useSetColonyDesignMonorepo,
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
        <NewColonyForm onClose={() => setCreating(false)} />
      )}
      <div className="rounded-lg border border-border bg-card divide-y divide-border">
        {(colonies.data ?? []).map((c) => (
          <ColonyRow
            key={c.colony_id}
            colony={c}
            isActive={c.colony_id === activeColonyId}
            onSelect={() => onSelectColony(c.colony_id)}
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


function NewColonyForm({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const create = useCreateColony();

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    try {
      await create.mutateAsync({
        name: name.trim(),
        description: description.trim(),
      });
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
  colony, isActive, onSelect,
}: {
  colony: ColonyInfo;
  isActive: boolean;
  onSelect: () => void;
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
          Slots 2 & 3 — reserved for future per-colony state. Add new
          rows here following the ``DesignMonorepoField`` pattern;
          keep the order stable so users see the same layout per
          colony. */}
      <div className="mt-2 flex flex-col gap-1">
        <DesignMonorepoField colonyId={colony.colony_id} />
        {/* TODO(per-colony-field-2): future slot. */}
        {/* TODO(per-colony-field-3): future slot. */}
      </div>
    </div>
  );
}


function DesignMonorepoField({ colonyId }: { colonyId: string }) {
  const cfg = useColonyDesignMonorepo(colonyId);
  const set = useSetColonyDesignMonorepo(colonyId);
  const [editing, setEditing] = useState(false);
  const [draftUrl, setDraftUrl] = useState("");
  const [draftBranch, setDraftBranch] = useState("main");

  const startEdit = () => {
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
    } catch {
      // useSetColonyDesignMonorepo exposes the error; rendered below.
    }
  };

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
    </div>
  );
}
