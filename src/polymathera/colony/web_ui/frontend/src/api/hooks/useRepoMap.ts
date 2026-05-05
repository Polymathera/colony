import { useMutation, useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";

export interface RepoMapSource {
  name: string;
  type: "git_repo" | "literature" | string;
  origin_url?: string | null;
  submodule?: string | null;
  branch?: string;
  commit?: string;
  start_dir?: string | null;
  include_globs?: string[] | null;
  exclude_globs?: string[] | null;
  binary_policy?: "skip" | "include" | null;
  static?: boolean | null;
  chunk_target_tokens?: number | null;
  chunk_overlap_tokens?: number | null;
}

export interface RepoMapResponse {
  origin_url: string;
  branch: string;
  commit: string;
  has_repo_map_file: boolean;
  raw_yaml: string | null;
  sources: RepoMapSource[];
}

export interface RepoTreeNode {
  path: string;
  is_dir: boolean;
  children: RepoTreeNode[];
}

export interface RepoTreeResponse {
  origin_url: string;
  branch: string;
  commit: string;
  root: RepoTreeNode;
}

export interface PreviewedSource {
  name: string;
  scope_id: string;
  mmap_kwargs: Record<string, unknown>;
}

export interface RepoMapPreviewResponse {
  origin_url: string;
  base_scope_id: string;
  sources: PreviewedSource[];
}

export interface RepoMapQueryArgs {
  originUrl: string;
  branch?: string;
  commit?: string;
}

function buildQs(args: RepoMapQueryArgs, extra: Record<string, string> = {}) {
  const params = new URLSearchParams({
    origin_url: args.originUrl,
    branch: args.branch ?? "main",
    commit: args.commit ?? "HEAD",
    ...extra,
  });
  return params.toString();
}

export function useRepoMap(args: RepoMapQueryArgs | null) {
  return useQuery({
    queryKey: ["repo-map", args],
    queryFn: () => apiFetch<RepoMapResponse>(`/repo-map?${buildQs(args!)}`),
    enabled: !!args && !!args.originUrl,
  });
}

export function useRepoTree(
  args: RepoMapQueryArgs | null,
  opts?: { maxDepth?: number; maxNodes?: number },
) {
  const extra: Record<string, string> = {};
  if (opts?.maxDepth) extra.max_depth = String(opts.maxDepth);
  if (opts?.maxNodes) extra.max_nodes = String(opts.maxNodes);
  return useQuery({
    queryKey: ["repo-map", "tree", args, opts],
    queryFn: () =>
      apiFetch<RepoTreeResponse>(`/repo-map/tree?${buildQs(args!, extra)}`),
    enabled: !!args && !!args.originUrl,
  });
}

/**
 * Dry-run the materialiser. Returns the list of mmap_application_scope
 * kwargs the dashboard's "Map Repo" flow would issue, with no side
 * effects.
 */
export function useRepoMapPreview() {
  return useMutation({
    mutationFn: (request: {
      origin_url: string;
      branch?: string;
      commit?: string;
      base_scope_id?: string;
    }) =>
      apiFetch<RepoMapPreviewResponse>("/repo-map/preview", {
        method: "POST",
        body: JSON.stringify({
          branch: "main",
          commit: "HEAD",
          ...request,
        }),
      }),
  });
}

// ----------------------------------------------------------------------
// Per-colony design-monorepo URL — persisted on the ``colonies`` row.
// ----------------------------------------------------------------------

export interface ColonyDesignMonorepoConfig {
  origin_url: string | null;
  branch: string;
  commit: string;
}

export function useColonyDesignMonorepo(colonyId: string | null) {
  return useQuery({
    queryKey: ["colonies", colonyId, "design-monorepo"],
    queryFn: () =>
      apiFetch<ColonyDesignMonorepoConfig>(
        `/colonies/${colonyId}/design-monorepo`,
      ),
    enabled: !!colonyId,
  });
}

export function useSetColonyDesignMonorepo(colonyId: string | null) {
  return useMutation({
    mutationFn: (request: {
      origin_url: string;
      branch?: string;
      commit?: string;
    }) =>
      apiFetch<ColonyDesignMonorepoConfig>(
        `/colonies/${colonyId}/design-monorepo`,
        {
          method: "PUT",
          body: JSON.stringify({
            branch: "main",
            commit: "HEAD",
            ...request,
          }),
        },
      ),
  });
}
