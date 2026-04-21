import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../client";

export interface TenantQuota {
  max_concurrent_sessions: number;
  max_concurrent_agents: number;
  max_total_cpu_cores: number;
  max_total_memory_mb: number;
  max_total_gpu_cores: number;
  max_total_gpu_memory_mb: number;
}

export interface TenantQuotaResponse {
  tenant_id: string;
  quota: Record<string, unknown>;
  usage: Record<string, unknown>;
}

export function useColonyConfig() {
  return useQuery({
    queryKey: ["config", "colony"],
    queryFn: () => apiFetch<Record<string, unknown>>("/config/colony"),
    refetchInterval: 30000,
  });
}

export function useTenantQuotas() {
  return useQuery({
    queryKey: ["config", "tenants"],
    queryFn: () => apiFetch<TenantQuotaResponse[]>("/config/tenants/"),
    refetchInterval: 15000,
  });
}

export function useTenantQuota(tenantId: string) {
  return useQuery({
    queryKey: ["config", "tenants", tenantId],
    queryFn: () => apiFetch<TenantQuotaResponse>(`/config/tenants/${tenantId}/quota`),
    enabled: !!tenantId,
  });
}

export function useUpdateTenantQuota() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ tenantId, quota }: { tenantId: string; quota: TenantQuota }) =>
      apiFetch<Record<string, unknown>>(`/config/tenants/${tenantId}/quota`, {
        method: "PUT",
        body: JSON.stringify(quota),
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["config", "tenants"] }),
  });
}
