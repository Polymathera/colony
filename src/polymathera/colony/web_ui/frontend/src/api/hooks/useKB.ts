import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../client";

export interface KBBackendInfo {
  vector_store: string;
  embedder_id: string;
  embedder_dimensions: number;
  qdrant_url: string | null;
  qdrant_collection: string | null;
}

export interface KBStatsResponse {
  total_chunks: number;
  total_sources: number;
  total_tokens: number;
  by_tier: Record<string, number>;
  by_data_type: Record<string, number>;
  backend: KBBackendInfo;
}

export interface KBSourceRow {
  source: string;
  chunk_count: number;
  total_tokens: number;
  data_types: string[];
  tiers: string[];
}

export interface KBSourcesResponse {
  sources: KBSourceRow[];
}

export interface KBChunkRow {
  chunk_id: string;
  section_path: string;
  data_type: string;
  tier: string;
  token_count: number;
  page_number: number | null;
  text_preview: string;
  /**
   * IDs of figures the chunk references. Populated by the chunker
   * when the section's markdown contained ``colony-image://`` URIs;
   * the chat tab renders a "📷 N" badge on chunks that have them.
   */
  figure_ids: string[];
  /**
   * Which extractor produced this chunk (``mistral_ocr`` /
   * ``anthropic`` / ``marker`` / …). ``null`` for legacy chunks
   * ingested before the multimodal pipeline landed.
   */
  metadata_origin: string | null;
}

export interface KBChunksResponse {
  source: string;
  chunks: KBChunkRow[];
}

export interface KBSearchHit {
  chunk_id: string;
  score: number;
  rank: number;
  source: string;
  section_path: string;
  data_type: string;
  tier: string;
  text_preview: string;
}

export interface KBSearchResponse {
  hits: KBSearchHit[];
}

export interface KBIngestRequest {
  path?: string | null;
  text?: string | null;
  source_uri?: string | null;
  tier?: string;
}

export interface KBIngestResponse {
  record_id: string;
  source_uri: string;
  status: string;
  chunks_produced: number;
  error: string | null;
}

export function useKBStats() {
  return useQuery({
    queryKey: ["kb", "stats"],
    queryFn: () => apiFetch<KBStatsResponse>("/kb/stats"),
  });
}

export function useKBSources() {
  return useQuery({
    queryKey: ["kb", "sources"],
    queryFn: () => apiFetch<KBSourcesResponse>("/kb/sources"),
  });
}

export function useKBChunksForSource(sourceUri: string | null, limit = 200) {
  return useQuery({
    queryKey: ["kb", "sources", sourceUri, "chunks", limit],
    queryFn: () => {
      const qs = new URLSearchParams({
        source_uri: sourceUri ?? "",
        limit: String(limit),
      });
      return apiFetch<KBChunksResponse>(`/kb/sources/chunks?${qs.toString()}`);
    },
    enabled: !!sourceUri,
  });
}

export function useKBSearch() {
  return useMutation({
    mutationFn: (request: {
      text: string;
      max_results?: number;
      source_prefix?: string | null;
      data_types?: string[];
    }) =>
      apiFetch<KBSearchResponse>("/kb/search", {
        method: "POST",
        body: JSON.stringify({ max_results: 10, ...request }),
      }),
  });
}

export function useKBIngest() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (request: KBIngestRequest) =>
      apiFetch<KBIngestResponse>("/kb/ingest", {
        method: "POST",
        body: JSON.stringify(request),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["kb", "stats"] });
      qc.invalidateQueries({ queryKey: ["kb", "sources"] });
    },
  });
}


// ----------------------------------------------------------------------
// /kb/ingest-repo-map — bulk ingest from ``knowledge_sources:`` rows.
// Mirrors useMapRepo's shape (background op + polled status). The
// per-row selection lives in the colony's persisted source-selection
// state (PUT /colonies/{id}/enabled-knowledge-sources) — no body
// filter here, single source of truth shared with the chat-driven
// ``ingest_repo_map_literature`` action.
// ----------------------------------------------------------------------

export interface IngestRepoMapRequest {
  origin_url: string;
  branch?: string;
  commit?: string;
}

export interface IngestRepoMapOpStatus {
  op_id: string;
  status: "pending" | "running" | "completed" | "error";
  origin_url: string;
  started_at: number;
  completed_at: number | null;
  message: string;
  ingested: number;
  failed: number;
}

export function useKBIngestRepoMap() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (req: IngestRepoMapRequest) =>
      apiFetch<IngestRepoMapOpStatus>("/kb/ingest-repo-map", {
        method: "POST",
        body: JSON.stringify({ branch: "main", commit: "HEAD", ...req }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["kb"] });
    },
  });
}

export function useKBIngestRepoMapOperations() {
  return useQuery({
    queryKey: ["kb", "ingest-repo-map", "operations"],
    queryFn: () =>
      apiFetch<IngestRepoMapOpStatus[]>("/kb/ingest-repo-map/operations"),
    refetchInterval: (query) => {
      const ops = query.state.data;
      if (ops && ops.some((op) => op.status === "pending" || op.status === "running")) {
        return 2000;
      }
      return false;
    },
  });
}
