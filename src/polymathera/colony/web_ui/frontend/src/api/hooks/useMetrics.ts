import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "../client";

interface TokenRun {
  run_id: string;
  agent_id: string;
  status: string;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  llm_calls: number;
  cost_usd: number;
  started_at: number | null;
}

interface TokenTotals {
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  total_tokens: number;
  cost_usd: number;
  run_count: number;
}

interface AgentTokenSummary {
  agent_id: string;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  llm_calls: number;
  cost_usd: number;
  run_count: number;
}

export interface TokenUsageResponse {
  runs: TokenRun[];
  totals: TokenTotals;
  by_agent?: AgentTokenSummary[];
  error?: string;
}

export function useTokenUsage(sessionId?: string) {
  const params = sessionId ? `?session_id=${sessionId}` : "";
  return useQuery({
    queryKey: ["metrics", "tokens", sessionId],
    queryFn: () => apiFetch<TokenUsageResponse>(`/metrics/tokens${params}`),
    refetchInterval: 10000,
  });
}
