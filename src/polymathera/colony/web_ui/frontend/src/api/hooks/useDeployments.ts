import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type { ApplicationSummary } from "../types";

export function useDeployments() {
  return useQuery({
    queryKey: ["deployments"],
    queryFn: () => apiFetch<ApplicationSummary[]>("/deployments/"),
  });
}

export function useDeploymentHealth(appName: string, deploymentName: string) {
  return useQuery({
    queryKey: ["deployments", appName, deploymentName, "health"],
    queryFn: () =>
      apiFetch<Record<string, unknown>>(
        `/deployments/${appName}/${deploymentName}/health`
      ),
    enabled: !!appName && !!deploymentName,
  });
}
