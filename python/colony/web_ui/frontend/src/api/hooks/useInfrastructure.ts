import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type { HealthStatus, RedisInfo } from "../types";

export function useHealthStatus() {
  return useQuery({
    queryKey: ["infra", "status"],
    queryFn: () => apiFetch<HealthStatus>("/infra/status"),
  });
}

export function useRedisInfo() {
  return useQuery({
    queryKey: ["infra", "redis"],
    queryFn: () => apiFetch<RedisInfo>("/infra/redis"),
  });
}
