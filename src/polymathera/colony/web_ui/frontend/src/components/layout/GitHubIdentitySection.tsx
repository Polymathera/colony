/**
 * LandingPage panels for the GitHub identity surface.
 *
 * Two sections, both per-user/per-tenant (not per-colony):
 *
 *   - :func:`UserGitHubIdentitySection` — the "Connect GitHub" button,
 *     and once connected, the verified ``(login, email)`` pair the
 *     user is identified as on GitHub. All fields are read-only: the
 *     OAuth callback is the only writer (typing the email would let a
 *     user attribute commits to anyone — see
 *     ``colony/github_identity_fix_plan.md`` §3).
 *
 *   - :func:`TenantGitHubInstallationSection` — text input for the
 *     numeric installation id the tenant gets when they install the
 *     Polymathera Colony GitHub App into their org. Used by the
 *     capability layer to mint REST tokens scoped to that tenant.
 *
 * Both live on the LandingPage alongside ``ColoniesSection`` so the
 * user can set them up before starting a session.
 */
import { useEffect, useState } from "react";
import { Pencil } from "lucide-react";
import {
  startGitHubConnect,
  useDisconnectGitHub,
  useSetTenantGitHubInstallation,
  useTenantGitHubInstallation,
  useUserGitHubIdentity,
} from "@/api/hooks/useGitHubIdentity";


function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}


export function UserGitHubIdentitySection() {
  const identity = useUserGitHubIdentity();
  const disconnect = useDisconnectGitHub();

  if (identity.isLoading) {
    return (
      <div className="w-full max-w-md text-xs text-muted-foreground">
        Loading GitHub identity…
      </div>
    );
  }

  return (
    <div className="w-full max-w-md">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Your GitHub Identity
        </h3>
      </div>
      <div className="rounded-lg border border-border bg-card p-4">
        {identity.data?.connected ? (
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground">Login:</span>
              <code className="text-foreground">
                {identity.data.github_login}
              </code>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-muted-foreground">Email:</span>
              <span className="text-foreground">
                {identity.data.github_email}
              </span>
            </div>
            {identity.data.git_user_name && (
              <div className="flex items-center gap-2 text-sm">
                <span className="text-muted-foreground">Name:</span>
                <span className="text-foreground">
                  {identity.data.git_user_name}
                </span>
              </div>
            )}
            <div className="text-[10px] text-muted-foreground">
              Verified {formatTimestamp(identity.data.github_last_verified_at)}
            </div>
            <div className="flex gap-2 pt-1">
              <button
                type="button"
                onClick={() => startGitHubConnect()}
                className="px-2 py-0.5 rounded border text-xs"
                title="Re-verify by running the OAuth flow again"
              >
                Re-verify
              </button>
              <button
                type="button"
                onClick={() => disconnect.mutate()}
                disabled={disconnect.isPending}
                className="px-2 py-0.5 rounded border text-xs text-red-500 disabled:opacity-50"
              >
                {disconnect.isPending ? "…" : "Disconnect"}
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            <p className="text-xs text-muted-foreground">
              Connect your GitHub account so Colony can attribute
              commits to you and assign issues to your real GitHub
              profile. Colony never acts AS you — your verified
              identity is fetched once and the token is discarded.
            </p>
            <button
              type="button"
              onClick={() => startGitHubConnect()}
              className="self-start px-3 py-1 rounded bg-primary text-primary-foreground text-xs"
            >
              Connect GitHub
            </button>
          </div>
        )}
        {disconnect.error && (
          <div className="mt-2 text-[10px] text-red-500">
            {(disconnect.error as Error).message}
          </div>
        )}
      </div>
    </div>
  );
}


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
