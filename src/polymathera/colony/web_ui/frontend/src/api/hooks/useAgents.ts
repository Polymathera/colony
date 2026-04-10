import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type { AgentHierarchyNode, AgentSummary } from "../types";

export function useAgents() {
  return useQuery({
    queryKey: ["agents"],
    queryFn: () => apiFetch<AgentSummary[]>("/agents/"),
  });
}

export function useAgentDetail(agentId: string) {
  return useQuery({
    queryKey: ["agents", agentId],
    queryFn: () => apiFetch<Record<string, unknown>>(`/agents/${agentId}`),
    enabled: !!agentId,
  });
}

export function useAgentHierarchy() {
  return useQuery({
    queryKey: ["agents", "hierarchy"],
    queryFn: () => apiFetch<AgentHierarchyNode[]>("/agents/hierarchy"),
  });
}

export function useAgentHistory(agentId: string | null) {
  return useQuery({
    queryKey: ["agents", agentId, "history"],
    queryFn: () => apiFetch<Record<string, unknown>>(`/agents/${agentId}/history`),
    enabled: !!agentId,
  });
}

export function useSystemStats() {
  return useQuery({
    queryKey: ["agents", "stats"],
    queryFn: () => apiFetch<Record<string, unknown>>("/agents/stats/system"),
  });
}
