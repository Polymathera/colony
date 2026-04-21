import { useState } from "react";
import { Plus, PanelLeftClose, PanelLeftOpen, Pause, Play, X, Rocket } from "lucide-react";
import { useSessions, useCreateSession, useSuspendSession, useResumeSession, useCloseSession } from "@/api/hooks/useSessions";
import { Badge } from "../shared/Badge";
import { formatTimestamp } from "@/lib/utils";

const stateVariant = (state: string) => {
  if (state === "active") return "success";
  if (state === "suspended") return "warning";
  if (state === "closed" || state === "archived") return "default";
  return "info";
};

interface SidebarProps {
  activeSessionId: string | null;
  onSelectSession: (sessionId: string | null) => void;
  onStartRun: () => void;
  colonyReady: boolean;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}

export function Sidebar({
  activeSessionId,
  onSelectSession,
  onStartRun,
  colonyReady,
  collapsed,
  onToggleCollapsed,
}: SidebarProps) {
  const sessions = useSessions();
  const createSession = useCreateSession();
  const suspendSession = useSuspendSession();
  const resumeSession = useResumeSession();
  const closeSession = useCloseSession();
  const [showNameInput, setShowNameInput] = useState(false);
  const [newSessionName, setNewSessionName] = useState("");

  const activeSession = sessions.data?.find(
    (s) => s.session_id === activeSessionId,
  );

  const handleCreateSession = async () => {
    const name = newSessionName.trim() || undefined;
    const result = await createSession.mutateAsync({ name });
    if (result.status === "created") {
      onSelectSession(result.session_id);
      setShowNameInput(false);
      setNewSessionName("");
    } else {
      // API returned 200 but with an error status — show it
      const { showErrorToast } = await import("../shared/ErrorToast");
      showErrorToast(result.message || "Failed to create session");
    }
  };

  if (collapsed) {
    return (
      <div className="flex w-10 shrink-0 flex-col items-center border-r border-border bg-card py-3 gap-3">
        <button
          onClick={onToggleCollapsed}
          className="text-muted-foreground hover:text-foreground"
          title="Expand sidebar"
        >
          <PanelLeftOpen size={18} />
        </button>
        <button
          onClick={handleCreateSession}
          disabled={createSession.isPending}
          className="text-primary hover:text-primary/80"
          title="New Session"
        >
          <Plus size={20} />
        </button>
      </div>
    );
  }

  return (
    <div className="flex w-56 shrink-0 flex-col border-r border-border bg-card">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-3 py-2.5">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Sessions
        </span>
        <button
          onClick={onToggleCollapsed}
          className="text-muted-foreground hover:text-foreground"
          title="Collapse sidebar"
        >
          <PanelLeftClose size={18} />
        </button>
      </div>

      {/* New Session */}
      <div className="border-b border-border p-2 space-y-1.5">
        {showNameInput ? (
          <>
            <input
              type="text"
              value={newSessionName}
              onChange={(e) => setNewSessionName(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleCreateSession(); if (e.key === "Escape") setShowNameInput(false); }}
              placeholder="Session name (optional)"
              autoFocus
              className="w-full rounded border border-border bg-background px-2 py-1 text-xs focus:border-primary focus:outline-none"
            />
            <div className="flex gap-1">
              <button
                onClick={handleCreateSession}
                disabled={createSession.isPending || !colonyReady}
                className="flex-1 rounded bg-primary/10 px-2 py-1 text-xs font-medium text-primary hover:bg-primary/20 disabled:opacity-50"
              >
                {createSession.isPending ? "Creating..." : "Create"}
              </button>
              <button
                onClick={() => { setShowNameInput(false); setNewSessionName(""); }}
                className="rounded px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
              >
                Cancel
              </button>
            </div>
          </>
        ) : (
          <button
            onClick={() => setShowNameInput(true)}
            disabled={!colonyReady}
            className="w-full rounded bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary hover:bg-primary/20 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Plus size={14} className="inline -mt-0.5" /> New Session
          </button>
        )}
      </div>

      {/* Session List */}
      <div className="flex-1 overflow-auto">
        {sessions.isLoading && (
          <div className="p-3 text-xs text-muted-foreground">Loading...</div>
        )}
        {sessions.data?.length === 0 && (
          <div className="p-3 text-xs text-muted-foreground">No sessions yet</div>
        )}
        {sessions.data?.map((session) => (
          <button
            key={session.session_id}
            onClick={() => onSelectSession(session.session_id)}
            className={`w-full border-b border-border px-3 py-2 text-left transition-colors ${
              session.session_id === activeSessionId
                ? "bg-primary/10"
                : "hover:bg-accent/50"
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] text-foreground truncate">
                {session.session_id.slice(0, 16)}
              </span>
              <Badge variant={stateVariant(session.state)} className="text-[9px] px-1 py-0">
                {session.state}
              </Badge>
            </div>
            <div className="mt-0.5 flex items-center gap-2 text-[10px] text-muted-foreground">
              <span>{formatTimestamp(session.created_at)}</span>
              {session.run_count > 0 && (
                <span>{session.run_count} run{session.run_count > 1 ? "s" : ""}</span>
              )}
            </div>
          </button>
        ))}
      </div>

      {/* Active session controls */}
      {activeSession && (
        <div className="border-t border-border p-2 space-y-1.5">
          <button
            onClick={onStartRun}
            disabled={!colonyReady}
            className="w-full rounded bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-400 hover:bg-emerald-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Rocket size={14} className="inline -mt-0.5" /> Start Run
          </button>
          <div className="flex gap-1">
            {activeSession.state === "active" && (
              <button
                onClick={() => suspendSession.mutate(activeSession.session_id)}
                disabled={suspendSession.isPending}
                className="flex-1 rounded bg-amber-500/10 px-2 py-1 text-[10px] font-medium text-amber-400 hover:bg-amber-500/20 disabled:opacity-50"
              >
                <Pause size={12} className="inline" /> Suspend
              </button>
            )}
            {activeSession.state === "suspended" && (
              <button
                onClick={() => resumeSession.mutate(activeSession.session_id)}
                disabled={resumeSession.isPending}
                className="flex-1 rounded bg-emerald-500/10 px-2 py-1 text-[10px] font-medium text-emerald-400 hover:bg-emerald-500/20 disabled:opacity-50"
              >
                <Play size={12} className="inline" /> Resume
              </button>
            )}
            <button
              onClick={() => {
                closeSession.mutate(activeSession.session_id);
                onSelectSession(null);
              }}
              disabled={closeSession.isPending}
              className="flex-1 rounded bg-red-500/10 px-2 py-1 text-[10px] font-medium text-red-400 hover:bg-red-500/20 disabled:opacity-50"
            >
              <X size={12} className="inline" /> Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
