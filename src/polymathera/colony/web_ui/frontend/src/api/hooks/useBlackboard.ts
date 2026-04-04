import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";

export interface BlackboardScopeSummary {
  scope: string;
  scope_id: string;
  tenant_id: string | null;
  colony_id: string | null;
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

export function useBlackboardEntries(
  scopeId: string,
  backendType: string = "",
  tenantId: string | null = null,
  colonyId: string | null = null,
) {
  return useQuery({
    queryKey: ["blackboard", "entries", scopeId],
    queryFn: () => {
      const params = new URLSearchParams();
      if (backendType) params.set("backend_type", backendType);
      if (tenantId) params.set("tenant_id", tenantId);
      if (colonyId) params.set("colony_id", colonyId);
      const qs = params.toString();
      return apiFetch<BlackboardEntryInfo[]>(
        `/blackboard/scopes/${scopeId}/entries${qs ? `?${qs}` : ""}`
      );
    },
    enabled: !!scopeId,
    refetchInterval: 5000,
  });
}
