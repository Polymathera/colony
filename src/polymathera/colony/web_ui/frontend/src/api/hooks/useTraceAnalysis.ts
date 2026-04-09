/** React Query hooks for trace analysis endpoints. */

import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";
import type {
  CodegenIteration,
  PromptDiffResult,
  FSMGraph,
} from "@/components/observability/traces/types";

export function useCodegenIterations(traceId: string | null, agentId: string | null) {
  return useQuery({
    queryKey: ["trace-analysis", "iterations", traceId, agentId],
    queryFn: () =>
      apiFetch<CodegenIteration[]>(
        `/traces/${traceId}/codegen-iterations?agent_id=${encodeURIComponent(agentId!)}`
      ),
    enabled: !!traceId && !!agentId,
  });
}

export function usePromptDiff(
  traceId: string | null,
  agentId: string | null,
  stepA: number,
  stepB: number,
) {
  return useQuery({
    queryKey: ["trace-analysis", "diff", traceId, agentId, stepA, stepB],
    queryFn: () =>
      apiFetch<PromptDiffResult>(
        `/traces/${traceId}/prompt-diff?agent_id=${encodeURIComponent(agentId!)}&step_a=${stepA}&step_b=${stepB}`
      ),
    enabled: !!traceId && !!agentId && stepA >= 0 && stepB >= 0 && stepA !== stepB,
  });
}

export function useFSMGraph(traceId: string | null, agentId: string | null) {
  return useQuery({
    queryKey: ["trace-analysis", "fsm", traceId, agentId],
    queryFn: () =>
      apiFetch<FSMGraph>(
        `/traces/${traceId}/fsm?agent_id=${encodeURIComponent(agentId!)}`
      ),
    enabled: !!traceId && !!agentId,
  });
}
