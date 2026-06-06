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


// ----------------------------------------------------------------------
// Per-colony GitHub Project (v2) attachment
// ----------------------------------------------------------------------

export interface DiscoverableProject {
  node_id: string;
  title: string;
  number: number | null;
  url: string | null;
}

export interface DiscoverableProjectsResponse {
  repo: string;
  projects: DiscoverableProject[];
  error: string | null;
}

/**
 * Open Projects v2 boards on the colony's design monorepo.
 *
 * Used by the colony settings UI to populate the project picker.
 * Returns the full response — when ``error`` is non-null, the UI
 * should render the error message inline (App permission missing,
 * monorepo URL not set, repo not on github.com).
 *
 * Caller passes ``enabled=false`` to hold the request until the
 * picker is actually open (avoids a wasted GraphQL round-trip on
 * every render of the colony list).
 */
export function useColonyDiscoverableProjects(
  colony_id: string | null,
  enabled: boolean = true,
) {
  return useQuery({
    queryKey: ["colony", colony_id, "discoverable-projects"],
    queryFn: () =>
      apiFetch<DiscoverableProjectsResponse>(
        `/colonies/${colony_id}/discoverable-projects`,
      ),
    enabled: enabled && !!colony_id,
    retry: false,
    staleTime: 30 * 1000,
  });
}


/**
 * Open Projects v2 boards on an arbitrary repo (``owner/name``).
 *
 * Used by the "+ New Colony" form to populate the project picker
 * after the operator picks a repo but before the colony exists.
 * The colony-level variant above can't be used pre-colony.
 *
 * Pass ``repo=null`` to disable the query (e.g. before any repo is
 * picked).
 */
export function useTenantDiscoverableProjects(repo: string | null) {
  return useQuery({
    queryKey: ["tenant", "discoverable-projects", repo],
    queryFn: () =>
      apiFetch<DiscoverableProjectsResponse>(
        `/tenants/me/discoverable-projects?repo=${encodeURIComponent(repo!)}`,
      ),
    enabled: !!repo && repo.includes("/"),
    retry: false,
    staleTime: 30 * 1000,
  });
}


export interface ColonyGitHubProjectConfig {
  node_id: string | null;
  title: string | null;
}

export function useColonyGitHubProject(colony_id: string | null) {
  return useQuery({
    queryKey: ["colony", colony_id, "github-project"],
    queryFn: () =>
      apiFetch<ColonyGitHubProjectConfig>(
        `/colonies/${colony_id}/github-project`,
      ),
    enabled: !!colony_id,
    retry: false,
    staleTime: 60 * 1000,
  });
}


export function useSetColonyGitHubProject(colony_id: string | null) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (payload: { node_id: string; title: string | null }) =>
      apiFetch<ColonyGitHubProjectConfig>(
        `/colonies/${colony_id}/github-project`,
        {
          method: "PUT",
          body: JSON.stringify(payload),
        },
      ),
    onSuccess: () => {
      qc.invalidateQueries({
        queryKey: ["colony", colony_id, "github-project"],
      });
    },
  });
}

