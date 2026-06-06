/**
 * Auth hooks — VCS-OAuth-only.
 *
 * Sign-in is a full-page navigation to ``/api/v1/auth/{provider_id}/sign-in``
 * (the backend 302s to the provider's authorize URL). There is no
 * password sign-in, so no ``useSignup`` / ``useLogin`` mutation — the
 * UI just calls :func:`startVcsSignIn` from a "Sign in with GitHub"
 * button.
 *
 * Shape matches ``web_ui/backend/routers/auth.py``.
 */
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../client";

export interface UserInfo {
  user_id: string;
  vcs_login: string | null;
  vcs_provider: string | null;
  active_colony_id: string | null;
  created_at: string | null;
  tenants: TenantMembership[];
  colonies: ColonyInfo[];
}

export interface TenantMembership {
  tenant_id: string;
  name: string;
  vcs_provider: string;
  vcs_org_login: string | null;
  role: string;
}

export interface ColonyInfo {
  colony_id: string;
  name: string;
  tenant_id: string;
  description: string;
  // PR 4: a colony binds to a VCS repo (many-to-one — multiple
  // colonies can point at the same repo). NULL during the transient
  // window between row create and repo selection.
  vcs_repo_id: string | null;
  vcs_repo_full_name: string | null;
  default_branch: string | null;
  created_at: string | null;
}

export interface AuthResponse {
  user_id: string;
  vcs_login: string;
  active_colony_id: string | null;
  message: string;
}


export function useCurrentUser() {
  return useQuery({
    queryKey: ["auth", "me"],
    queryFn: () => apiFetch<UserInfo>("/auth/me"),
    retry: false,
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchInterval: false,
  });
}


/**
 * Kick off the VCS OAuth flow by navigating to the provider's
 * sign-in endpoint. Backend 302s to the provider; on callback the
 * user lands back at ``/`` with auth cookies set.
 */
export function startVcsSignIn(providerId: string): void {
  window.location.assign(`/api/v1/auth/${providerId}/sign-in`);
}


export function useLogout() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () =>
      apiFetch<{ message: string }>("/auth/logout", { method: "POST" }),
    onSuccess: () => {
      qc.clear(); // Clear all cached data on logout
    },
  });
}


/**
 * Switch the user's active colony. Persists ``users.active_colony_id``
 * AND re-issues the JWT with the new ``(tenant_id, colony_id)`` pair
 * so subsequent requests are scoped correctly without a page reload.
 */
export function useSwitchActiveColony() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (colony_id: string) =>
      apiFetch<AuthResponse>("/auth/me/active-colony", {
        method: "PATCH",
        body: JSON.stringify({ colony_id }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["auth", "me"] });
      // Many other queries' scoping depends on the JWT's tenant/colony
      // claims — invalidate everything so they re-run under the new
      // context.
      qc.invalidateQueries({ queryKey: ["colonies"] });
      qc.invalidateQueries({ queryKey: ["sessions"] });
    },
  });
}


export function useColonies(options?: { enabled?: boolean }) {
  return useQuery({
    queryKey: ["colonies"],
    queryFn: () => apiFetch<ColonyInfo[]>("/colonies/"),
    enabled: options?.enabled ?? true,
    retry: false,
  });
}


export interface CreateColonyRequest {
  name: string;
  description?: string;
  // Optional repo binding — populated by the "New Colony" form when
  // the user picks from the discoverable-repos dropdown. Backend
  // derives ``design_monorepo_url`` from the tenant's provider when
  // all three are supplied.
  vcs_repo_id?: string | null;
  vcs_repo_full_name?: string | null;
  default_branch?: string | null;
  // Optional commit-attribution preferences. Backend defaults to
  // ``commit_principal='colony'``, ``commit_co_author='user'`` when
  // both are null.
  commit_principal?: string | null;
  commit_co_author?: string | null;
  // GitHub Project (v2) attachment. Required when ``vcs_repo_full_name``
  // is set — backend rejects the create call otherwise. ``null`` is
  // only valid for bare colonies (no repo bound).
  github_project_node_id?: string | null;
  github_project_title?: string | null;
}

export function useCreateColony() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: CreateColonyRequest) =>
      apiFetch<ColonyInfo>("/colonies/", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["colonies"] });
      qc.invalidateQueries({ queryKey: ["auth", "me"] });
    },
  });
}

