import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";

export interface BlackboardScopeSummary {
  scope: string;
  scope_id: string;
  entry_count: number;
  oldest_entry_age: number | null;
  newest_entry_age: number | null;
  backend_type: string;
  subscriber_count: number;
  error?: string;
}

export interface BlackboardEntryInfo {
  key: string;
  value: unknown;
  version: number;
  created_by: string | null;
  updated_by: string | null;
  created_at: number;
  updated_at: number;
  tags: string[];
}

export function useBlackboardScopes() {
  return useQuery({
    queryKey: ["blackboard", "scopes"],
    queryFn: () => apiFetch<BlackboardScopeSummary[]>("/blackboard/scopes"),
    refetchInterval: 10000,
  });
}

export function useBlackboardEntries(scope: string, scopeId: string, backendType: string = "") {
  return useQuery({
    queryKey: ["blackboard", "entries", scope, scopeId],
    queryFn: () => {
      const params = backendType ? `?backend_type=${encodeURIComponent(backendType)}` : "";
      return apiFetch<BlackboardEntryInfo[]>(
        `/blackboard/scopes/${scope}/${scopeId}/entries${params}`
      );
    },
    enabled: !!scope && !!scopeId,
    refetchInterval: 5000,
  });
}
