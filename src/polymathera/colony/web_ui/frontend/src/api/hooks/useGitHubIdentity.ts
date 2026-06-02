/**
 * Hooks for the per-user GitHub identity + per-tenant App installation
 * surfaces (backend lives in
 * ``colony/src/polymathera/colony/web_ui/backend/routers/{github_oauth,tenants}.py``).
 *
 * The "Connect GitHub" flow is a server-side redirect, not a fetch:
 * the browser navigates to ``/api/v1/auth/github/connect``, the
 * backend redirects to GitHub, the user approves, GitHub redirects
 * back to ``/api/v1/auth/github/callback`` which sets the verified
 * identity on the user row. The UI just calls
 * :func:`startGitHubConnect` (window.location assignment) and
 * refetches :func:`useUserGitHubIdentity` when the user lands back.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../client";

// ----------------------------------------------------------------------
// Per-user GitHub identity (OAuth-verified)
// ----------------------------------------------------------------------

export interface UserGitHubIdentity {
  connected: boolean;
  github_login?: string;
  github_user_id?: number;
  github_email?: string;
  git_user_name?: string | null;
  github_connected_at?: string | null;
  github_last_verified_at?: string | null;
}

export function useUserGitHubIdentity() {
  return useQuery({
    queryKey: ["user", "github-identity"],
    queryFn: () => apiFetch<UserGitHubIdentity>("/users/me/github"),
    retry: false,
    staleTime: 60 * 1000,
  });
}

export function useDisconnectGitHub() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<{ cleared: boolean }>("/users/me/github", {
        method: "DELETE",
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["user", "github-identity"] });
    },
  });
}

/**
 * Kick off the OAuth flow by navigating the browser to the connect
 * endpoint. The backend redirects to GitHub; on success the user
 * returns to the dashboard at the callback URL and the next
 * :func:`useUserGitHubIdentity` poll sees the populated identity.
 */
export function startGitHubConnect(): void {
  window.location.assign("/api/v1/auth/github/connect");
}

// ----------------------------------------------------------------------
// Per-tenant GitHub App installation id
// ----------------------------------------------------------------------

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
