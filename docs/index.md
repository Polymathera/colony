# Colony

**Polymathera's no-RAG, cache-aware multi-agent framework for extremely long, dense contexts (1B+ tokens).**

Colony is a framework for building *tightly-coupled, self-evolving, self-improving, self-aware multi-agent systems* (**agent colonies**) that reason over extremely long context without retrieval-augmented generation (RAG). Instead of fragmenting context into chunks and retrieving snippets, Colony keeps the entire context "live" over a cluster of *one or more* LLMs through a cluster-level virtual memory system that manages LLM KV caches in the same way an operating system manages (almost unlimited) virtual memory over finite physical memory.

!!! tip "Colony's Vision"
    Colony's goal is to be the most efficient *country of geniuses in a datacenter* — the ideal substrate for **civilization-building AI**.


!!! tip "Pre-Alpha Early Access"

    Colony is still in pre-alpha early access. The API is not stable and the framework is under active development. We welcome feedback and contributions, but be aware that breaking changes may occur.


!!! tip "Who should use Colony?"

    Colony is designed for **engineers building complex multi-agent systems** that require reasoning over extremely long contexts. It is not a general-purpose agent framework or a consumer product. If you are looking for a simple agent orchestration tool or a way to add tool use to an LLM, Colony may not be the right fit. It runs over a Ray cluster (local or in the cloud) and it can be resource-intensive and expensive.

## Key Ideas

- **NoRAG**: Colony keeps the full context live and accessible, not filtered through retrieval. Colony manages all kinds of context (code, text, data) through distributed KV cache paging, not vector search.

- **Cache-Aware Agents**: Agents are aware of what's in GPU memory (at the cluster level) and consciously plan their work to maximize cache reuse.

- **Agents All the Way Down**: General intelligence emerges from the right composition of *agent capabilities* and *multi-agent patterns*. Every cognitive process -- attention, memory, planning, confidence tracking -- is a pluggable policy with a default implementation.

- **Distributed Reasoning Patterns**: Multi-agent game protocols (hypothesis games, contract nets, negotiation) combat specific LLM failure modes like hallucination, laziness, and goal drift.

## Getting Started

```bash
pip install polymathera-colony
```

See the [Installation](getting-started/installation.md) guide and [Quick Start](getting-started/quickstart.md) tutorial.

## Architecture at a Glance

<style>
/* ── Architecture Diagram ── */
.arch-svg { width: 100%; max-width: 940px; margin: 0 auto; display: block; }

/* Groups — hover effects */
.arch-svg .group       { transition: filter .2s; cursor: pointer; }
.arch-svg .group:hover { filter: brightness(1.04) drop-shadow(0 4px 12px rgba(0,0,0,.10)); }

/* Boxes */
.arch-svg .box  { rx: 8; ry: 8; stroke-width: 1.4; transition: stroke-width .15s; }
.arch-svg .group:hover .box { stroke-width: 2.2; }

/* Text — uses system font stack */
.arch-svg text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.arch-svg .title  { font-size: 13.5px; font-weight: 600; fill: #1e1b4b; }
.arch-svg .body   { font-size: 11px;   fill: #374151; }
.arch-svg .detail { font-size: 10px;   fill: #6b7280; }

/* Arrow lines & labels */
.arch-svg .arrow       { stroke-width: 1.6; fill: none; marker-end: url(#ah); }
.arch-svg .arrow-label { font-size: 13px; fill: #6b7280; font-weight: 500; }


/* Animated flow dash */
@keyframes flowDash { to { stroke-dashoffset: -14; } }
.arch-svg .flow { stroke-dasharray: 7 4; animation: flowDash 1s linear infinite; }

/* ── Dark mode (Material slate) ── */
[data-md-color-scheme="slate"] .arch-svg .title  { fill: #e0e7ff; }
[data-md-color-scheme="slate"] .arch-svg .body   { fill: #cbd5e1; }
[data-md-color-scheme="slate"] .arch-svg .detail { fill: #94a3b8; }
[data-md-color-scheme="slate"] .arch-svg .arrow-label { fill: #94a3b8; font-weight: 500; }
[data-md-color-scheme="slate"] .arch-svg .group:hover { filter: brightness(1.12) drop-shadow(0 4px 12px rgba(255,255,255,.06)); }
/* Dark-mode box fills */
[data-md-color-scheme="slate"] .arch-svg .r-sys  { fill: #1e1b4b; stroke: #6d28d9; }
[data-md-color-scheme="slate"] .arch-svg .r-agt  { fill: #2e1065; stroke: #7c3aed; }
[data-md-color-scheme="slate"] .arch-svg .r-bb   { fill: #172554; stroke: #2563eb; }
[data-md-color-scheme="slate"] .arch-svg .r-vcm  { fill: #052e16; stroke: #059669; }
[data-md-color-scheme="slate"] .arch-svg .r-llm  { fill: #422006; stroke: #d97706; }
[data-md-color-scheme="slate"] .arch-svg .r-node { fill: #451a03; stroke: #ea580c; }
[data-md-color-scheme="slate"] .arch-svg .r-ctx  { fill: #1c1917; stroke: #78716c; }
[data-md-color-scheme="slate"] .arch-svg .r-src  { fill: #292524; stroke: #78716c; }
[data-md-color-scheme="slate"] .arch-svg .r-ext  { fill: #18181b; stroke: #52525b; }
[data-md-color-scheme="slate"] .arch-svg .r-inf  { fill: #1e293b; stroke: #475569; }
</style>
<div>
<svg class="arch-svg" viewBox="0 0 940 544" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="ah" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <path d="M0,0 L10,3.5 L0,7 Z" fill="#7c3aed"/>
    </marker>
    <marker id="ah-blue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <path d="M0,0 L10,3.5 L0,7 Z" fill="#3b82f6"/>
    </marker>
    <marker id="ah-gray" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <path d="M0,0 L10,3.5 L0,7 Z" fill="#78716c"/>
    </marker>
  </defs>

  <!-- ══════════ AGENT SYSTEM ══════════ -->
  <a href="architecture/agent-system/">
  <g class="group" id="g-agents">
    <rect class="box r-sys" x="20" y="10" width="900" height="140" fill="#f5f3ff" stroke="#8b5cf6"/>
    <text class="title" x="470" y="32" text-anchor="middle">Agent System</text>
    <!-- Agent 1 -->
    <rect class="box r-agt" x="36" y="44" width="175" height="94" fill="#ede9fe" stroke="#a78bfa"/>
    <text class="title" x="124" y="66" text-anchor="middle">Agent 1</text>
    <text class="body"  x="124" y="82" text-anchor="middle">Capabilities (Memory,</text>
    <text class="body"  x="124" y="95" text-anchor="middle">Games, Confidence, ...)</text>
    <text class="body"  x="124" y="112" text-anchor="middle">ActionPolicy (MPC)</text>
    <text class="detail" x="124" y="128" text-anchor="middle">Hook System (AOP)</text>
    <!-- Agent 2 -->
    <rect class="box r-agt" x="224" y="44" width="175" height="94" fill="#ede9fe" stroke="#a78bfa"/>
    <text class="title" x="312" y="66" text-anchor="middle">Agent 2</text>
    <text class="body"  x="312" y="82" text-anchor="middle">Capabilities (Memory,</text>
    <text class="body"  x="312" y="95" text-anchor="middle">Games, Confidence, ...)</text>
    <text class="body"  x="312" y="112" text-anchor="middle">ActionPolicy (MPC)</text>
    <text class="detail" x="312" y="128" text-anchor="middle">Hook System (AOP)</text>
    <!-- Agent N -->
    <rect class="box r-agt" x="412" y="44" width="175" height="94" fill="#ede9fe" stroke="#a78bfa"/>
    <text class="title" x="500" y="66" text-anchor="middle">Agent N</text>
    <text class="body"  x="500" y="82" text-anchor="middle">Capabilities (Memory,</text>
    <text class="body"  x="500" y="95" text-anchor="middle">Games, Confidence, ...)</text>
    <text class="body"  x="500" y="112" text-anchor="middle">ActionPolicy (MPC)</text>
    <text class="detail" x="500" y="128" text-anchor="middle">Hook System (AOP)</text>
    <!-- ··· separator + more agents hint -->
    <text class="detail" x="604" y="94" text-anchor="middle">···</text>
    <!-- Capability composition note -->
    <text class="body"   x="648" y="62"  text-anchor="start" font-style="italic">Each agent is composed of capabilities</text>
    <text class="body"   x="648" y="78"  text-anchor="start" font-style="italic">wired together through its ActionPolicy:</text>
    <text class="detail" x="648" y="96"  text-anchor="start">Memory · Attention · Grounding</text>
    <text class="detail" x="648" y="110" text-anchor="start">Confidence · Reflection · Planning</text>
    <text class="detail" x="648" y="124" text-anchor="start">Games (Hypothesis, Contract Net,</text>
    <text class="detail" x="648" y="138" text-anchor="start">Negotiation, Consensus)</text>
  </g>
  </a>

  <!-- ══════════ ARROWS: Agents → Blackboard / VCM ══════════ -->
  <line class="arrow flow" x1="150" y1="150" x2="150" y2="196" stroke="#8b5cf6"/>
  <text class="arrow-label" x="164" y="178">read / write / query / mmap</text>
  <line class="arrow flow" x1="630" y1="150" x2="630" y2="196" stroke="#8b5cf6"/>
  <text class="arrow-label" x="644" y="178">infer_with_suffix / page_graph_ops</text>

  <!-- ══════════ BLACKBOARD ══════════ -->
  <a href="architecture/blackboard/">
  <g class="group" id="g-bb">
    <rect class="box r-bb" x="20" y="198" width="260" height="148" fill="#eff6ff" stroke="#3b82f6"/>
    <text class="title" x="150" y="222" text-anchor="middle">Blackboard (Redis)</text>
    <text class="body"  x="36" y="244">Shared state &amp; event pub/sub</text>
    <text class="body"  x="36" y="260">Optimistic concurrency (OCC)</text>
    <text class="body"  x="36" y="276">Agent coordination</text>
    <text class="body"  x="36" y="298">Memory scopes:</text>
    <text class="detail" x="36" y="314">Working · STM · LTM</text>
    <text class="detail" x="36" y="328">Episodic · Semantic · Procedural</text>
  </g>
  </a>

  <!-- ══════════ EXTERNAL SOURCES ══════════ -->
  <g class="group" id="g-ext">
    <rect class="box r-ext" x="20" y="362" width="260" height="58" fill="#f9fafb" stroke="#9ca3af"/>
    <text class="title" x="150" y="386" text-anchor="middle">External Sources</text>
    <text class="body"  x="150" y="404" text-anchor="middle">Git repos · documents · KBs · APIs</text>
  </g>

  <!-- ══════════ VCM ══════════ -->
  <a href="architecture/virtual-context-memory/">
  <g class="group" id="g-vcm">
    <rect class="box r-vcm" x="340" y="198" width="580" height="280" fill="#ecfdf5" stroke="#10b981"/>
    <text class="title" x="630" y="222" text-anchor="middle">Virtual Context Memory (VCM)</text>
    <text class="body"  x="630" y="242" text-anchor="middle">Page Table · Page Attention Graph · Cache Scheduling · Page Faults</text>
    <!-- LLM Cluster -->
    <rect class="box r-llm" x="356" y="254" width="548" height="80" fill="#fffbeb" stroke="#f59e0b"/>
    <text class="title" x="630" y="274" text-anchor="middle">LLM Cluster (GPU Nodes)</text>
    <rect class="box r-node" x="372" y="284" width="155" height="38" fill="#fef3c7" stroke="#fbbf24"/>
    <text class="body"  x="450" y="300" text-anchor="middle">LLM Node 1</text>
    <text class="detail" x="450" y="314" text-anchor="middle">KV Cache</text>
    <rect class="box r-node" x="540" y="284" width="155" height="38" fill="#fef3c7" stroke="#fbbf24"/>
    <text class="body"  x="618" y="300" text-anchor="middle">LLM Node 2</text>
    <text class="detail" x="618" y="314" text-anchor="middle">KV Cache</text>
    <rect class="box r-node" x="708" y="284" width="155" height="38" fill="#fef3c7" stroke="#fbbf24"/>
    <text class="body"  x="786" y="300" text-anchor="middle">LLM Node N</text>
    <text class="detail" x="786" y="314" text-anchor="middle">KV Cache</text>
    <!-- Context Sources -->
    <rect class="box r-ctx" x="356" y="348" width="548" height="68" fill="#f5f5f4" stroke="#a8a29e"/>
    <text class="title" x="630" y="368" text-anchor="middle">Context Sources (mapped as pages)</text>
    <rect class="box r-src" x="372" y="378" width="122" height="28" fill="#e7e5e4" stroke="#a8a29e"/>
    <text class="body" x="433" y="396" text-anchor="middle">Git Repos</text>
    <rect class="box r-src" x="504" y="378" width="122" height="28" fill="#e7e5e4" stroke="#a8a29e"/>
    <text class="body" x="565" y="396" text-anchor="middle">Knowledge Bases</text>
    <rect class="box r-src" x="636" y="378" width="122" height="28" fill="#e7e5e4" stroke="#a8a29e"/>
    <text class="body" x="697" y="396" text-anchor="middle">Blackboard Data</text>
    <rect class="box r-src" x="768" y="378" width="120" height="28" fill="#e7e5e4" stroke="#a8a29e"/>
    <text class="body" x="828" y="396" text-anchor="middle">Custom</text>
    <!-- VCM features -->
    <text class="detail" x="630" y="438" text-anchor="middle">Nonuniform pages · Soft/hard affinity · Advisory/mandatory groups · Prefetching</text>
    <text class="detail" x="630" y="454" text-anchor="middle">Amortized cost: O(N²) → O(N log N) as page graph stabilizes</text>
  </g>
  </a>

  <!-- ══════════ ARROWS: BB / External → VCM context sources (mmap) ══════════ -->
  <line class="arrow flow" x1="280" y1="340" x2="340" y2="370" stroke="#3b82f6" style="marker-end:url(#ah-blue)"/>
  <text class="arrow-label" x="282" y="350" text-anchor="start">mmap / munmap</text>
  <line class="arrow flow" x1="280" y1="391" x2="340" y2="391" stroke="#78716c" style="marker-end:url(#ah-gray)"/>
  <text class="arrow-label" x="290" y="385" text-anchor="middle">mmap / munmap</text>

  <!-- ══════════ INFRASTRUCTURE BAR ══════════ -->
  <rect class="box r-inf" x="20" y="492" width="900" height="50" fill="#f1f5f9" stroke="#cbd5e1"/>
  <text class="title" x="470" y="522" text-anchor="middle">Infrastructure: Ray Cluster (actors, autoscaling) · Redis (state, pub/sub)</text>

</svg>
</div>


## Documentation

| Section | Description |
|---------|-------------|
| [Philosophy](philosophy/index.md) | Why Colony exists and what makes it different |
| [Architecture](architecture/index.md) | Technical architecture of each subsystem |
| [Design Insights](design-insights/index.md) | Deep dives into novel design decisions |
| [Guides](guides/colony-env.md) | Practical how-to guides |
| [Contributing](contributing.md) | How to contribute to Colony |
