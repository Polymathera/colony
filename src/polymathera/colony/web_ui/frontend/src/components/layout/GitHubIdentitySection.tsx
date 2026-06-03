/**
 * LandingPage panels for the VCS identity surface.
 *
 *   - :func:`UserVcsIdentitySection` — read-only display of the
 *     user's OAuth-verified VCS identity (provider, login, name).
 *     Populated by the sign-in walker; there's no "Connect/Disconnect"
 *     button because sign-in IS the connect now (PR 3).
 *
 *   - :func:`TenantGitHubInstallationSection` — text input for the
 *     numeric installation id the tenant gets when they install the
 *     Polymathera Colony GitHub App into their org. Mostly auto-
 *     populated by the sign-in walker (``GET /user/installations``);
 *     this UI is the manual-override / debug surface.
 */
import { useEffect, useState } from "react";
import { Pencil } from "lucide-react";
import { useCurrentUser } from "@/api/hooks/useAuth";
import {
  useSetTenantGitHubInstallation,
  useTenantGitHubInstallation,
} from "@/api/hooks/useGitHubIdentity";


export function UserVcsIdentitySection() {
  const me = useCurrentUser();

  if (me.isLoading) {
    return (
      <div className="w-full max-w-md text-xs text-muted-foreground">
        Loading identity…
      </div>
    );
  }

  const user = me.data;
  if (!user || !user.vcs_login) {
    // Should not happen in normal flow — sign-in always populates
    // these fields — but defensive when the cookies somehow get out
    // of sync with the DB row.
    return null;
  }

  const providerLabel =
    user.vcs_provider === "github" ? "GitHub" : (user.vcs_provider || "VCS");

  return (
    <div className="w-full max-w-md">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Your VCS Identity
        </h3>
      </div>
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted-foreground">Provider:</span>
            <span className="text-foreground">{providerLabel}</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted-foreground">Login:</span>
            <code className="text-foreground">{user.vcs_login}</code>
          </div>
          <div className="text-[10px] text-muted-foreground">
            Re-verify by signing out and signing back in.
          </div>
        </div>
      </div>
    </div>
  );
}


// Back-compat alias so existing imports keep working until callers
// migrate to the new name. Drop in a follow-up cleanup.
export const UserGitHubIdentitySection = UserVcsIdentitySection;


export function TenantGitHubInstallationSection() {
  const installation = useTenantGitHubInstallation();
  const set = useSetTenantGitHubInstallation();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  useEffect(() => {
    if (!editing) {
      setDraft(installation.data?.installation_id ?? "");
    }
  }, [editing, installation.data]);

  if (installation.isLoading) {
    return null;
  }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const cleaned = draft.trim();
    await set.mutateAsync(cleaned || null);
    setEditing(false);
  };

  return (
    <div className="w-full max-w-md">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Tenant GitHub Installation
        </h3>
      </div>
      <div className="rounded-lg border border-border bg-card p-4">
        <p className="mb-2 text-[10px] text-muted-foreground">
          The numeric id from the Polymathera Colony App's installation
          page (e.g. ``https://github.com/organizations/&lt;org&gt;/settings/installations/&lt;id&gt;``).
          Colony uses it to mint REST tokens scoped to this tenant's
          repos.
        </p>
        {editing ? (
          <form onSubmit={submit} className="flex items-center gap-2">
            <input
              autoFocus
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="installation id (e.g. 12345678)"
              className="flex-1 px-2 py-1 rounded border bg-background text-xs"
            />
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
          </form>
        ) : (
          <div className="flex items-center gap-2 text-sm">
            <span className="text-muted-foreground">Installation id:</span>
            <code className="text-foreground">
              {installation.data?.installation_id ?? "(not set)"}
            </code>
            <button
              type="button"
              onClick={() => setEditing(true)}
              className="text-muted-foreground hover:text-foreground"
              title="Edit"
            >
              <Pencil size={12} />
            </button>
          </div>
        )}
        {set.error && (
          <div className="mt-2 text-[10px] text-red-500">
            {(set.error as Error).message}
          </div>
        )}
      </div>
    </div>
  );
}
