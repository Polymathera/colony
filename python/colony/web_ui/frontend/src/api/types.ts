/* Response types matching backend api_models.py */

export interface HealthStatus {
  ray_connected: boolean;
  redis_connected: boolean;
  postgres_connected: boolean;
  ray_cluster_status: string;
  node_count: number;
}

export interface RedisInfo {
  connected_clients: number;
  used_memory_human: string;
  total_commands_processed: number;
  keyspace_hits: number;
  keyspace_misses: number;
  uptime_in_seconds: number;
}

export interface DeploymentSummary {
  app_name: string;
  deployment_name: string;
  proxy_actor_name: string;
}

export interface ApplicationSummary {
  app_name: string;
  created_at: number;
  deployments: DeploymentSummary[];
}

export interface AgentSummary {
  agent_id: string;
  agent_type: string;
  state: string;
  capabilities: string[];
}

export interface SessionSummary {
  session_id: string;
  tenant_id: string;
  state: string;
  created_at: number;
  run_count: number;
}

export interface RunSummary {
  run_id: string;
  session_id: string;
  agent_id: string;
  status: string;
  started_at: number | null;
  completed_at: number | null;
  input_tokens: number;
  output_tokens: number;
}

export interface PageSummary {
  page_id: string;
  source: string;
  tokens: number;
  loaded: boolean;
}

export interface VCMStats {
  total_pages: number;
  loaded_pages: number;
  page_groups: number;
  pending_faults: number;
}
