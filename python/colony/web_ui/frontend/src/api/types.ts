/* Response types matching backend api_models.py */

export interface HealthStatus {
  ray_connected: boolean;
  redis_connected: boolean;
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
  files?: string[];
}

export interface PageLocationSummary {
  deployment_name: string;
  client_id: string;
  access_count: number;
  last_access_time: number;
  load_time: number;
}

export interface PageLoadedEntry {
  page_id: string;
  size: number;
  tenant_id: string;
  total_access_count: number;
  locations: PageLocationSummary[];
}

export interface AgentHierarchyNode {
  agent_id: string;
  agent_type: string;
  state: string;
  role: string | null;
  parent_agent_id: string | null;
  capability_names: string[];
  bound_pages: string[];
  tenant_id: string;
}

export interface VCMStats {
  total_pages: number;
  loaded_pages: number;
  page_groups: number;
  pending_faults: number;
}

export interface LogSource {
  actor_id: string;
  class_name: string;
  node_id: string;
  pid: number;
  ip: string;
  repr_name: string;
}
