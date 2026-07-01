/* Response types matching backend api_models.py */

export interface HealthStatus {
  ray_connected: boolean;
  redis_connected: boolean;
  deployments_ready: boolean;
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
  colony_id: string;
  state: string;
  created_at: number;
  run_count: number;
  // P8-0: "user" for chat sessions, "system" for the per-colony
  // colony-singleton host (always-on, no user attached).
  session_kind?: string;
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
  colony_id: string;
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
  colony_id: string;
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

export interface PersistentLogEntry {
  log_id: string;
  timestamp: number;
  level: string;
  logger_name: string;
  message: string;
  module: string;
  func_name: string;
  line_no: number;
  pid: number;
  thread_name: string;
  actor_class: string;
  node_id: string;
  tenant_id: string | null;
  colony_id: string | null;
  session_id: string | null;
  run_id: string | null;
  trace_id: string | null;
  exc_info: string | null;
}

export interface LogQueryResult {
  logs: PersistentLogEntry[];
  count: number;
  error?: string;
}

export interface LogStats {
  total: number;
  errors: number;
  warnings: number;
  sessions: number;
  actors: number;
  earliest: number | null;
  latest: number | null;
}

export interface LogActorSummary {
  actor_class: string;
  log_count: number;
  latest: number;
}

export interface PageGraphScope {
  tenant_id: string;
  colony_id: string;
  scope_id: string;
}

export interface PageGraphNode {
  id: string;
  x: number;
  y: number;
  z: number;
}

export interface PageGraphEdge {
  source: string;
  target: string;
  weight: number;
  confidence: number;
  relationship_types: string[];
}

export interface PageGraphData {
  nodes: PageGraphNode[];
  edges: PageGraphEdge[];
  node_count: number;
  edge_count: number;
  tenant_id?: string;
  colony_id?: string;
  error?: string;
}

/* Observability / Tracing */

export type SpanRating = "up" | "down";

export interface SpanFeedback {
  author: string;
  rating: SpanRating;
  note: string | null;
  updated_wall: number | null;
}

export interface TraceSpan {
  span_id: string;
  trace_id: string;
  parent_span_id: string | null;
  run_id: string | null;
  agent_id: string;
  name: string;
  kind: string;
  start_wall: number;
  duration_ms: number | null;
  status: "running" | "ok" | "error";
  error: string | null;
  input_summary: Record<string, unknown>;
  output_summary: Record<string, unknown>;
  input_tokens: number | null;
  output_tokens: number | null;
  cache_read_tokens: number | null;
  model_name: string | null;
  context_page_ids: string[] | null;
  ring: string | null;
  service_name: string | null;
  tags: string[];
  metadata: Record<string, unknown>;
  /** Present on the REST view; absent on synthetic/streamed spans. */
  feedback?: SpanFeedback[];
}

export interface TraceSummary {
  trace_id: string;
  agent_id: string;
  status: string;
  start_time: number;
  span_count: number;
  run_count: number;
  total_tokens: number;
}

/* Session mutations */

export interface CreateSessionRequest {
  name?: string | null;
  ttl_seconds?: number | null;
  fork_from_session_id?: string | null;
}

export interface CreateSessionResponse {
  session_id: string;
  status: string;
  message: string;
}

export interface SessionActionResponse {
  session_id: string;
  success: boolean;
  message: string;
}

/* VCM repo mapping */

export interface MapRepoRequest {
  origin_url: string;
  branch?: string;
  commit?: string;
  repo_id?: string | null;
  // Per-row selection lives in the colony's persisted source-selection
  // state (PUT /colonies/{id}/enabled-vcm-sources). The Design Monorepo
  // tab writes it on every checkbox toggle; the materialiser reads it.
  // Single source of truth — no body field here.
}


/* Job submission */

export interface MissionSpec {
  type: string;
  coordinator_version?: string;
  max_agents?: number;
  quality_threshold?: number;
  max_iterations?: number;
  batching_policy?: string;
  extra_capabilities?: string[];
  parameters?: Record<string, unknown>;
}

export interface JobSubmitRequest {
  session_id: string;
  missions: MissionSpec[];
  timeout_seconds?: number;
  budget_usd?: number | null;
}

export interface JobSubmitResponse {
  job_id: string;
  session_id: string;
  status: string;
  missions: string[];
  message: string;
}

export interface JobStatusResponse {
  job_id: string;
  session_id: string;
  status: string;
  missions_completed: number;
  missions_total: number;
  message: string;
}

/* Chat controls — sent alongside messages to configure the session agent */

export interface ChatControls {
  vcm_context?: {
    repo_ids?: string[];
    file_patterns?: string[];
    dir_paths?: string[];
    page_ids?: string[];
    languages?: string[];
    exclude_patterns?: string[];
  };
  agent_preferences?: {
    mission_types?: string[];
    max_agents?: number;
    capabilities?: string[];
    tools?: string[];
    tool_config?: Record<string, Record<string, unknown>>;
  };
  effort?: "low" | "medium" | "high";
  timeout_seconds?: number;
  budget_usd?: number | null;
}
