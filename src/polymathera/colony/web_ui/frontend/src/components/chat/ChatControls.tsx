import { useState } from "react";
import { ChevronDown, ChevronUp, Database, Bot, Wrench, Gauge } from "lucide-react";

export interface ChatControlsState {
  vcm_context?: {
    repo_ids?: string[];
    file_patterns?: string[];
    dir_paths?: string[];
    page_ids?: string[];
    languages?: string[];
    exclude_patterns?: string[];
  };
  agent_preferences?: {
    analysis_types?: string[];
    max_agents?: number;
    capabilities?: string[];
    tools?: string[];
  };
  effort?: "low" | "medium" | "high";
  timeout_seconds?: number;
  budget_usd?: number | null;
}

interface ChatControlsProps {
  controls: ChatControlsState;
  onChange: (controls: ChatControlsState) => void;
}

const ANALYSIS_TYPES = ["impact", "compliance", "intent", "contracts", "slicing", "basic"];
const EFFORT_LEVELS = ["low", "medium", "high"] as const;

export function ChatControls({ controls, onChange }: ChatControlsProps) {
  const [expanded, setExpanded] = useState(false);
  const [activeSection, setActiveSection] = useState<string | null>(null);

  const toggleSection = (section: string) => {
    setActiveSection((prev) => (prev === section ? null : section));
    if (!expanded) setExpanded(true);
  };

  const analysisTypes = controls.agent_preferences?.analysis_types || [];
  const maxAgents = controls.agent_preferences?.max_agents || 10;
  const effort = controls.effort || "medium";
  const timeout = controls.timeout_seconds || 600;

  const toggleAnalysisType = (type: string) => {
    const current = [...analysisTypes];
    const idx = current.indexOf(type);
    if (idx >= 0) current.splice(idx, 1);
    else current.push(type);
    onChange({
      ...controls,
      agent_preferences: { ...controls.agent_preferences, analysis_types: current },
    });
  };

  // Summary of active controls (shown when collapsed)
  const summaryParts: string[] = [];
  if (analysisTypes.length > 0) summaryParts.push(`${analysisTypes.length} analyses`);
  if (controls.vcm_context?.file_patterns?.length) summaryParts.push(`${controls.vcm_context.file_patterns.length} filters`);
  if (effort !== "medium") summaryParts.push(`effort: ${effort}`);

  return (
    <div className="border-t border-border">
      {/* Toggle bar */}
      <div className="flex items-center gap-1 px-3 py-1.5">
        <button onClick={() => toggleSection("context")} className={sectionBtnClass(activeSection === "context")} title="VCM Context">
          <Database size={12} />
        </button>
        <button onClick={() => toggleSection("agents")} className={sectionBtnClass(activeSection === "agents")} title="Agent Preferences">
          <Bot size={12} />
        </button>
        <button onClick={() => toggleSection("tools")} className={sectionBtnClass(activeSection === "tools")} title="Tools">
          <Wrench size={12} />
        </button>
        <button onClick={() => toggleSection("effort")} className={sectionBtnClass(activeSection === "effort")} title="Effort & Limits">
          <Gauge size={12} />
        </button>

        {summaryParts.length > 0 && (
          <span className="ml-2 text-[10px] text-muted-foreground">{summaryParts.join(" | ")}</span>
        )}

        <button onClick={() => setExpanded((v) => !v)} className="ml-auto text-muted-foreground hover:text-foreground">
          {expanded ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
        </button>
      </div>

      {/* Expanded panels */}
      {expanded && activeSection && (
        <div className="border-t border-border px-3 py-2 space-y-2">
          {activeSection === "context" && (
            <VCMContextSection
              context={controls.vcm_context || {}}
              onChange={(vcm_context) => onChange({ ...controls, vcm_context })}
            />
          )}
          {activeSection === "agents" && (
            <AgentPreferencesSection
              analysisTypes={analysisTypes}
              maxAgents={maxAgents}
              onToggleType={toggleAnalysisType}
              onMaxAgentsChange={(n) => onChange({
                ...controls,
                agent_preferences: { ...controls.agent_preferences, max_agents: n },
              })}
            />
          )}
          {activeSection === "tools" && (
            <ToolsSection
              tools={controls.agent_preferences?.tools || []}
              onChange={(tools) => onChange({
                ...controls,
                agent_preferences: { ...controls.agent_preferences, tools },
              })}
            />
          )}
          {activeSection === "effort" && (
            <EffortSection
              effort={effort}
              timeout={timeout}
              budget={controls.budget_usd ?? null}
              onEffortChange={(e) => onChange({ ...controls, effort: e })}
              onTimeoutChange={(t) => onChange({ ...controls, timeout_seconds: t })}
              onBudgetChange={(b) => onChange({ ...controls, budget_usd: b })}
            />
          )}
        </div>
      )}
    </div>
  );
}

function sectionBtnClass(active: boolean): string {
  return `rounded p-1.5 text-xs transition-colors ${
    active
      ? "bg-primary/10 text-primary"
      : "text-muted-foreground hover:text-foreground hover:bg-accent/50"
  }`;
}

// --- Sub-sections ---

function VCMContextSection({
  context,
  onChange,
}: {
  context: NonNullable<ChatControlsState["vcm_context"]>;
  onChange: (ctx: NonNullable<ChatControlsState["vcm_context"]>) => void;
}) {
  const [newPattern, setNewPattern] = useState("");

  const addPattern = () => {
    const trimmed = newPattern.trim();
    if (!trimmed) return;
    const patterns = [...(context.file_patterns || []), trimmed];
    onChange({ ...context, file_patterns: patterns });
    setNewPattern("");
  };

  const removePattern = (idx: number) => {
    const patterns = [...(context.file_patterns || [])];
    patterns.splice(idx, 1);
    onChange({ ...context, file_patterns: patterns });
  };

  return (
    <div className="space-y-1.5">
      <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">VCM Context</label>
      <div className="flex flex-wrap gap-1">
        {(context.file_patterns || []).map((p, i) => (
          <span key={i} className="inline-flex items-center gap-1 rounded bg-cyan-500/10 px-2 py-0.5 text-[10px] text-cyan-400">
            {p}
            <button onClick={() => removePattern(i)} className="hover:text-foreground">&times;</button>
          </span>
        ))}
      </div>
      <div className="flex gap-1">
        <input
          type="text"
          value={newPattern}
          onChange={(e) => setNewPattern(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); addPattern(); } }}
          placeholder="src/**/*.py, *.ts, etc."
          className="flex-1 rounded border border-border bg-background px-2 py-1 text-[10px] focus:border-primary focus:outline-none"
        />
        <button onClick={addPattern} className="rounded bg-primary/10 px-2 py-1 text-[10px] text-primary hover:bg-primary/20">Add</button>
      </div>
    </div>
  );
}

function AgentPreferencesSection({
  analysisTypes,
  maxAgents,
  onToggleType,
  onMaxAgentsChange,
}: {
  analysisTypes: string[];
  maxAgents: number;
  onToggleType: (type: string) => void;
  onMaxAgentsChange: (n: number) => void;
}) {
  return (
    <div className="space-y-1.5">
      <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Analysis Types</label>
      <div className="flex flex-wrap gap-1">
        {ANALYSIS_TYPES.map((type) => (
          <button
            key={type}
            onClick={() => onToggleType(type)}
            className={`rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
              analysisTypes.includes(type)
                ? "bg-primary/10 text-primary border border-primary/30"
                : "bg-accent/30 text-muted-foreground border border-transparent hover:border-border"
            }`}
          >
            {type}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-2">
        <label className="text-[10px] text-muted-foreground">Max agents:</label>
        <input
          type="number"
          value={maxAgents}
          onChange={(e) => onMaxAgentsChange(Number(e.target.value))}
          min={1}
          max={50}
          className="w-16 rounded border border-border bg-background px-2 py-0.5 text-[10px] font-mono focus:border-primary focus:outline-none"
        />
      </div>
    </div>
  );
}

function ToolsSection({
  tools,
  onChange,
}: {
  tools: string[];
  onChange: (tools: string[]) => void;
}) {
  const availableTools = [
    { id: "web-search", label: "Web Search" },
    { id: "repl", label: "Code REPL" },
  ];

  const toggleTool = (id: string) => {
    const current = [...tools];
    const idx = current.indexOf(id);
    if (idx >= 0) current.splice(idx, 1);
    else current.push(id);
    onChange(current);
  };

  return (
    <div className="space-y-1.5">
      <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Tools</label>
      <div className="flex flex-wrap gap-1">
        {availableTools.map((tool) => (
          <button
            key={tool.id}
            onClick={() => toggleTool(tool.id)}
            className={`rounded px-2 py-0.5 text-[10px] font-medium transition-colors ${
              tools.includes(tool.id)
                ? "bg-purple-500/10 text-purple-400 border border-purple-500/30"
                : "bg-accent/30 text-muted-foreground border border-transparent hover:border-border"
            }`}
          >
            {tool.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function EffortSection({
  effort,
  timeout,
  budget,
  onEffortChange,
  onTimeoutChange,
  onBudgetChange,
}: {
  effort: string;
  timeout: number;
  budget: number | null;
  onEffortChange: (e: "low" | "medium" | "high") => void;
  onTimeoutChange: (t: number) => void;
  onBudgetChange: (b: number | null) => void;
}) {
  return (
    <div className="space-y-1.5">
      <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Effort & Limits</label>
      <div className="flex items-center gap-3">
        {EFFORT_LEVELS.map((level) => (
          <label key={level} className="flex items-center gap-1 text-[10px] text-muted-foreground cursor-pointer">
            <input
              type="radio"
              name="effort"
              checked={effort === level}
              onChange={() => onEffortChange(level)}
              className="accent-primary"
            />
            <span className={effort === level ? "text-foreground font-medium" : ""}>{level}</span>
          </label>
        ))}
      </div>
      <div className="flex gap-3">
        <div className="flex items-center gap-1">
          <label className="text-[10px] text-muted-foreground">Timeout:</label>
          <input
            type="number"
            value={timeout}
            onChange={(e) => onTimeoutChange(Number(e.target.value))}
            min={60}
            step={60}
            className="w-20 rounded border border-border bg-background px-2 py-0.5 text-[10px] font-mono focus:border-primary focus:outline-none"
          />
          <span className="text-[10px] text-muted-foreground">s</span>
        </div>
        <div className="flex items-center gap-1">
          <label className="text-[10px] text-muted-foreground">Budget:</label>
          <input
            type="number"
            value={budget ?? ""}
            onChange={(e) => onBudgetChange(e.target.value ? Number(e.target.value) : null)}
            min={0}
            step={0.5}
            placeholder="none"
            className="w-20 rounded border border-border bg-background px-2 py-0.5 text-[10px] font-mono focus:border-primary focus:outline-none"
          />
          <span className="text-[10px] text-muted-foreground">$</span>
        </div>
      </div>
    </div>
  );
}
