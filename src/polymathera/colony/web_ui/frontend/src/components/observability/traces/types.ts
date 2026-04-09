/** Shared types for trace analysis views. */

export type TraceViewMode = "tree" | "timeline" | "diff" | "fsm" | "flow";

export interface CodegenIteration {
  step_index: number;
  agent_step_span_id: string;
  infer_span_id: string | null;
  action_span_id: string | null;
  prompt: string;
  generated_code: string;
  success: boolean | null;
  error: string | null;
  duration_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  mode: string;
  start_wall: number | null;
}

export interface PromptDiffSection {
  name: string;
  changes: Array<{ type: "added" | "removed" | "context"; content: string }>;
}

export interface PromptDiffResult {
  step_a: number;
  step_b: number;
  sections: PromptDiffSection[];
  total_added: number;
  total_removed: number;
  is_identical: boolean;
  error?: string;
}

export interface FSMState {
  state_id: string;
  fingerprint: string;
  label: string;
  visit_count: number;
  step_indices: number[];
}

export interface FSMTransition {
  from_state: string;
  to_state: string;
  step_index: number;
  code_preview: string;
  success: boolean;
}

export interface FSMLoop {
  state_id: string;
  count: number;
  step_indices: number[];
}

export interface FSMGraph {
  states: FSMState[];
  transitions: FSMTransition[];
  loops: FSMLoop[];
}
