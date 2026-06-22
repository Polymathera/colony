/**
 * Operator runtime override panel for semantic constraints (PR5-B UI).
 *
 * Modal opened from the chat header. Lists the SessionAgent's active
 * semantic constraints with toggle buttons that POST to
 * /sessions/{id}/constraints/{cid}/{disable|enable}.
 *
 * The matching backend lives in routers/sessions.py; the
 * SessionOrchestratorCapability subscribes to
 * OperatorOverrideProtocol on the high-priority lane so the toggle
 * takes effect without restarting the agent.
 */

import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { apiFetch } from "@/api/client";

interface Constraint {
  id: string;
  rule_nl: string;
  scope: string;
  failure_mode: string;
  disabled: boolean;
}

interface ConstraintsPanelProps {
  sessionId: string;
  onClose: () => void;
}

export function ConstraintsPanel({ sessionId, onClose }: ConstraintsPanelProps) {
  const [constraints, setConstraints] = useState<Constraint[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);

  const load = async () => {
    try {
      const data = await apiFetch<Constraint[]>(
        `/sessions/${sessionId}/constraints`,
      );
      setConstraints(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const toggle = async (c: Constraint) => {
    const verb = c.disabled ? "enable" : "disable";
    setPending(c.id);
    try {
      await apiFetch(
        `/sessions/${sessionId}/constraints/${c.id}/${verb}`,
        { method: "POST" },
      );
      // Optimistic update; a refresh would re-fetch.
      setConstraints((prev) =>
        prev ? prev.map((row) =>
          row.id === c.id ? { ...row, disabled: !row.disabled } : row,
        ) : prev,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(null);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onClose}
    >
      <div
        className="w-[min(560px,calc(100vw-32px))] max-h-[80vh] overflow-auto rounded-lg border border-border bg-card p-4 shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold uppercase tracking-wider">
            Semantic Constraints
          </h2>
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground"
            title="Close"
          >
            <X size={18} />
          </button>
        </div>

        <p className="text-xs text-muted-foreground mb-3">
          Runtime toggles for the SessionAgent's runtime guardrails.
          Disabled constraints are skipped (no precondition, no
          verifier, no advisory) for the rest of this session.
        </p>

        {error && (
          <div className="mb-3 rounded border border-destructive bg-destructive/10 p-2 text-xs text-destructive">
            {error}
          </div>
        )}

        {constraints === null && !error && (
          <div className="text-xs text-muted-foreground">Loading…</div>
        )}

        {constraints !== null && constraints.length === 0 && (
          <div className="text-xs text-muted-foreground">
            No semantic constraints active on this session.
          </div>
        )}

        {constraints !== null && constraints.length > 0 && (
          <ul className="flex flex-col gap-2">
            {constraints.map((c) => (
              <li
                key={c.id}
                className="rounded border border-border p-3 flex flex-col gap-1"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <code className="text-xs font-mono break-all">{c.id}</code>
                    <div className="text-[10px] text-muted-foreground mt-0.5">
                      scope: <span className="font-mono">{c.scope}</span> · on
                      failure: <span className="font-mono">{c.failure_mode}</span>
                    </div>
                  </div>
                  <button
                    onClick={() => toggle(c)}
                    disabled={pending === c.id}
                    className={`text-xs px-2 py-1 rounded border ${
                      c.disabled
                        ? "border-emerald-600 text-emerald-700 hover:bg-emerald-50"
                        : "border-amber-600 text-amber-700 hover:bg-amber-50"
                    } disabled:opacity-50 whitespace-nowrap`}
                    title={c.disabled ? "Re-enable this constraint" : "Disable for the rest of this session"}
                  >
                    {pending === c.id
                      ? "…"
                      : c.disabled
                      ? "Enable"
                      : "Disable"}
                  </button>
                </div>
                <p className="text-xs text-muted-foreground line-clamp-3">
                  {c.rule_nl}
                </p>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
