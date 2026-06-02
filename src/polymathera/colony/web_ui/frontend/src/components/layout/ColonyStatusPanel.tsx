/**
 * ColonyStatusPanel — the visible payoff for the §8-§14 backend stack.
 *
 * v1 surfaces three tiles per design doc §15 (other tiles deferred —
 * see ../../../../../../../p11_p12_plan.md):
 *
 *   - Alerts: recent bottlenecks + inconsistencies from the
 *     interaction_log (P11 extension to InteractionLogCapability).
 *   - Recent activity: the full interaction_log tail mixed across
 *     GitHub events + mentions + alerts.
 *   - Project link: deep-link to the configured design monorepo's
 *     GitHub Project board (the design doc §15 non-duplication
 *     principle — Project board is the source of truth for
 *     roadmap/kanban; this panel shows what Projects can't).
 */
import { ExternalLink, AlertTriangle, Activity } from "lucide-react";
import {
  useColonyAlerts,
  useColonyProjectLink,
  useColonyRecentActivity,
} from "@/api/hooks/useColonyStatus";
import type {
  ColonyActivityRow,
  ColonyAlertRow,
} from "@/api/hooks/useColonyStatus";


function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}


function AlertItem({ row }: { row: ColonyAlertRow }) {
  // Bottleneck payload: ``{kind, severity, repo, issue_number,
  // title, url, stale_days, summary, suggested_remedies}`` (from
  // DesignProcessCapability.identify_bottlenecks).
  // Inconsistency payload: shape varies — render best-effort.
  const payload = row.payload ?? {};
  const summary =
    (payload.summary as string | undefined) ||
    (payload.title as string | undefined) ||
    (payload.message as string | undefined) ||
    `(${row.event_kind})`;
  const url = row.channel_ref;
  return (
    <li className="flex items-start gap-2 py-2 text-sm">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
      <div className="flex-1 min-w-0">
        <div className="text-foreground">{summary}</div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="uppercase">{row.event_kind}</span>
          <span>·</span>
          <span>{formatTimestamp(row.ts)}</span>
          {url && (
            <>
              <span>·</span>
              <a
                href={url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-primary hover:underline"
              >
                Open <ExternalLink className="h-3 w-3" />
              </a>
            </>
          )}
        </div>
      </div>
    </li>
  );
}


function ActivityItem({ row }: { row: ColonyActivityRow }) {
  // Concise one-liner — the panel is a tail, not a detail view.
  // event_kind covers github_issue_event / github_comment_event /
  // github_pr_event / mention_event / bottleneck / inconsistency.
  // Pull the repo + issue number out of refs when present.
  const issueRef = row.refs?.find((r) => r.kind === "issue")?.value;
  const mentionRef = row.refs?.find((r) => r.kind === "mention")?.value;
  const payload = row.payload ?? {};
  const title =
    (payload.title as string | undefined) ||
    (payload.summary as string | undefined) ||
    issueRef ||
    `(${row.event_kind})`;
  return (
    <li className="flex items-start gap-2 py-1.5 text-xs">
      <Activity className="mt-0.5 h-3 w-3 shrink-0 text-muted-foreground" />
      <div className="flex-1 min-w-0">
        <div className="truncate text-foreground">
          {mentionRef && (
            <span className="mr-1 rounded bg-primary/10 px-1 text-primary">
              @{mentionRef}
            </span>
          )}
          {title}
        </div>
        <div className="flex items-center gap-2 text-muted-foreground">
          <span className="uppercase">{row.event_kind}</span>
          {row.user_login && (
            <>
              <span>·</span>
              <span>{row.user_login}</span>
            </>
          )}
          <span>·</span>
          <span>{formatTimestamp(row.ts)}</span>
          {row.channel_ref && (
            <>
              <span>·</span>
              <a
                href={row.channel_ref}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-primary hover:underline"
              >
                Open <ExternalLink className="h-3 w-3" />
              </a>
            </>
          )}
        </div>
      </div>
    </li>
  );
}


export function ColonyStatusPanel() {
  const alerts = useColonyAlerts(20);
  const activity = useColonyRecentActivity(30);
  const projectLink = useColonyProjectLink();

  return (
    <div className="w-full max-w-md">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Colony Status
        </h3>
        {projectLink.data?.project_url && (
          <a
            href={projectLink.data.project_url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
          >
            Open Project board <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </div>
      <div className="rounded-lg border border-border bg-card p-4">

        {/* Alerts tile */}
        <div className="mb-4">
          <div className="mb-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Alerts
          </div>
          {alerts.isLoading ? (
            <div className="text-xs text-muted-foreground">Loading…</div>
          ) : alerts.data?.alerts.length ? (
            <ul className="divide-y divide-border">
              {alerts.data.alerts.map((row) => (
                <AlertItem key={row.id} row={row} />
              ))}
            </ul>
          ) : (
            <div className="text-xs text-muted-foreground">
              No alerts. Bottleneck detections + design-context
              inconsistencies land here as agents discover them.
            </div>
          )}
        </div>

        {/* Recent activity tile */}
        <div>
          <div className="mb-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Recent Activity
          </div>
          {activity.isLoading ? (
            <div className="text-xs text-muted-foreground">Loading…</div>
          ) : activity.data?.events.length ? (
            <ul className="divide-y divide-border">
              {activity.data.events.map((row) => (
                <ActivityItem key={row.id} row={row} />
              ))}
            </ul>
          ) : (
            <div className="text-xs text-muted-foreground">
              No recent events. GitHub issues / comments / PRs +
              ``@colony`` mentions flow in via the webhook receiver
              or the poll-mode capability.
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
