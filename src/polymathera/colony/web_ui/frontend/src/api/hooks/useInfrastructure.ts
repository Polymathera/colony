import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type { HealthStatus, RedisInfo } from "../types";

export function useHealthStatus() {
  return useQuery({
    queryKey: ["infra", "status"],
    queryFn: () => apiFetch<HealthStatus>("/infra/status"),
    refetchInterval: (query) => {
      // Poll every 5s until deployments are ready, then slow down to 30s
      return query.state.data?.deployments_ready ? 30000 : 5000;
    },
  });
}

export function useRedisInfo() {
  return useQuery({
    queryKey: ["infra", "redis"],
    queryFn: () => apiFetch<RedisInfo>("/infra/redis"),
  });
}
