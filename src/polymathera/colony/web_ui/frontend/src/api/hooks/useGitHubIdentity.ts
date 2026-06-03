/**
 * Per-tenant GitHub App installation id hook.
 *
 * The per-user GitHub identity that used to live in this file
 * (``useUserGitHubIdentity`` / ``useDisconnectGitHub`` /
 * ``startGitHubConnect``) is gone — sign-in IS the connect now, so
 * the user's verified identity is already on the
 * :func:`useCurrentUser` response (``vcs_login`` / ``vcs_provider``
 * fields).
 *
 * The tenant-installation surface stays because it's a separate
 * concern: the tenant admin pastes the App installation id their
 * org received when they installed Colony's GitHub App. Today's
 * sign-in walker auto-populates ``github_installation_id`` from
 * ``GET /user/installations`` (see services/user_tenant_sync.py)
 * so this UI is now mostly a debug/override surface.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../client";


export interface TenantGitHubInstallation {
  installation_id: string | null;
}

export function useTenantGitHubInstallation() {
  return useQuery({
    queryKey: ["tenant", "github-installation"],
    queryFn: () =>
      apiFetch<TenantGitHubInstallation>("/tenants/me/github-installation"),
    retry: false,
    staleTime: 60 * 1000,
  });
}

export function useSetTenantGitHubInstallation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (installation_id: string | null) =>
      apiFetch<TenantGitHubInstallation>(
        "/tenants/me/github-installation",
        {
          method: "PUT",
          body: JSON.stringify({ installation_id }),
        },
      ),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["tenant", "github-installation"] });
    },
  });
}


// ----------------------------------------------------------------------
// Discoverable repos — cached by the sign-in walker for the dropdown UI
// ----------------------------------------------------------------------

export interface DiscoverableRepo {
  vcs_repo_id: string;
  vcs_repo_full_name: string;
  default_branch: string;
  user_permission: string;     // "read" | "write" | "admin"
  has_colony_marker: boolean;  // TRUE for repos with .colony/
  clone_url: string | null;    // Pre-rendered by the tenant's provider
}

/**
 * Repos the sign-in walker (services/colony_discovery) discovered for
 * the caller's active tenant. Powers the "+ New Colony" form's repo
 * dropdown + the per-colony "Design monorepo" picker.
 *
 * Refreshes whenever the user signs in (new walker pass). v1 doesn't
 * auto-refresh from inside the dashboard — operator clicks
 * "Sign out and back in" to trigger a fresh discovery walk.
 */
export function useDiscoverableRepos() {
  return useQuery({
    queryKey: ["tenant", "discoverable-repos"],
    queryFn: () =>
      apiFetch<DiscoverableRepo[]>("/tenants/me/discoverable-repos"),
    retry: false,
    staleTime: 60 * 1000,
  });
}

