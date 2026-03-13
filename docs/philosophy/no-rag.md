# The NoRAG Paradigm

Colony rejects retrieval-augmented generation (RAG) as the foundation for deep reasoning over extremely long context.

<!--

## Explicit Context Is Better Than Implicit Context

LLMs learn vast amounts of knowledge during training, but this knowledge is *implicit* -- encoded in weights, distributed across layers, accessible only through the model's finite-depth forward pass. When a task requires deep, systemic reasoning (understanding a complex codebase, conducting multi-step scientific research, synthesizing a legal argument across thousands of documents), the model must first *explicate* the relevant implicit knowledge into explicit, live context before it can reason over it.

<mark style="background-color: #d0d0d0">This is why chain-of-thought (CoT) prompting works: it forces the model to externalize intermediate reasoning steps as explicit text. But CoT has a ceiling. Reproducing *all* the implicit context necessary for a successful inference through CoT alone is not possible. The benefits plateau because the model cannot reconstruct, through sequential token generation, the full web of implicit associations that the task demands.</mark>

!!! tip "The Design Principle"
    If CoT plateaus because it cannot externalize enough implicit context, the solution is not better prompting -- it is providing more explicit context to reason over. Colony emphasizes reasoning over extremely long context (potentially billions of tokens) precisely because of this principle.

-->


## Why Not RAG?

!!! abstract "The RAG Problem"
    RAG activates only sparse subsets of a corpus at a time. For tasks that require **local (sparse) reasoning** -- adding type annotations to a codebase, answering factual questions from a knowledge base -- this is fine. But Colony targets a different class of problems: tasks requiring **global (systemic, dense) reasoning** that synthesize insights from many disparate parts of the context across many iterative passes.


!!! abstract "The NoRAG Advantage"
    In dense reasoning tasks, breakthroughs are unlocked by new insights synthesized from *unpredictable* combinations of previously known facts. A retrieval system, by definition, must predict which facts are relevant before the reasoning happens. This creates a chicken-and-egg problem: the most valuable connections are precisely the ones a retrieval model would not predict, because they span distant and seemingly unrelated parts of the context.

> The major difference between RAG and NoRAG is that RAG optimizes for recall of known-relevant information, while NoRAG optimizes for the ability to synthesize new insights from all available information. RAG hides context that "seems" irrelevant -- precisely the context where breakthroughs come from.

> A major difference between RAG and NoRAG is the type of queries they excel at: RAG is suited for queries with known-relevant information, while NoRAG is designed for **continuous research queries** that require synthesizing new insights from all available information.


<style>
/* ── NoRAG Diagrams ── */
.norag-svg text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }

/* Box fills/strokes */
.norag-svg .r-panel-rag   { fill: #f9fafb; stroke: #9ca3af; }
.norag-svg .r-panel-norag { fill: #f5f3ff; stroke: #8b5cf6; }
.norag-svg .r-hdr-rag     { fill: #9ca3af; }
.norag-svg .r-hdr-norag   { fill: #7c3aed; }
.norag-svg .r-query       { fill: white;   stroke: #d1d5db; }
.norag-svg .r-query-norag { fill: white;   stroke: #a78bfa; }
.norag-svg .r-retrieve    { fill: #f5f5f4; stroke: #78716c; }
.norag-svg .r-warning     { fill: #fef3c7; stroke: #f59e0b; }
.norag-svg .r-vcm         { fill: #ecfdf5; stroke: #10b981; }
.norag-svg .r-agt         { fill: #ede9fe; stroke: #a78bfa; }
.norag-svg .r-bb          { fill: #eff6ff; stroke: #3b82f6; }
.norag-svg .r-vcs         { fill: #f5f5f4; stroke: #78716c; }
.norag-svg .r-tile        { fill: #e7e5e4; stroke: #78716c; }
.norag-svg .r-llm         { fill: #fffbeb; stroke: #f59e0b; }
.norag-svg .r-gpu         { fill: #fef3c7; stroke: #fbbf24; }
.norag-svg .r-gpu-free    { fill: #fde68a; stroke: #fbbf24; }
.norag-svg .r-page-loaded { fill: #10b981; }

/* Text fills */
.norag-svg .t-white    { fill: white; }
.norag-svg .t-title    { fill: #1e1b4b; }
.norag-svg .t-body     { fill: #374151; }
.norag-svg .t-muted    { fill: #6b7280; }
.norag-svg .t-hint     { fill: #9ca3af; }
.norag-svg .t-stone-dk { fill: #292524; }
.norag-svg .t-stone    { fill: #78716c; }
.norag-svg .t-stone-md { fill: #57534e; }
.norag-svg .t-stone-lt { fill: #a8a29e; }
.norag-svg .t-vcm-dk   { fill: #064e3b; }
.norag-svg .t-vcm      { fill: #065f46; }
.norag-svg .t-vcm-md   { fill: #047857; }
.norag-svg .t-vcm-fb   { fill: #10b981; }
.norag-svg .t-vcm-xdk  { fill: #14532d; }
.norag-svg .t-vcm-dk2  { fill: #166534; }
.norag-svg .t-bb-dk    { fill: #1e40af; }
.norag-svg .t-bb       { fill: #3b82f6; }
.norag-svg .t-bb-md    { fill: #2563eb; }
.norag-svg .t-llm-dk   { fill: #78350f; }
.norag-svg .t-llm      { fill: #92400e; }
.norag-svg .t-llm-md   { fill: #b45309; }
.norag-svg .t-agt-dk   { fill: #4c1d95; }
.norag-svg .t-warn-dk  { fill: #92400e; }
.norag-svg .t-warn     { fill: #b45309; }

/* ── Dark mode (Material slate) ── */
[data-md-color-scheme="slate"] .norag-svg .r-panel-rag   { fill: #1c1917; stroke: #57534e; }
[data-md-color-scheme="slate"] .norag-svg .r-panel-norag { fill: #1e1b4b; stroke: #6d28d9; }
[data-md-color-scheme="slate"] .norag-svg .r-hdr-rag     { fill: #44403c; }
[data-md-color-scheme="slate"] .norag-svg .r-hdr-norag   { fill: #4c1d95; }
[data-md-color-scheme="slate"] .norag-svg .r-query       { fill: #1f2937; stroke: #374151; }
[data-md-color-scheme="slate"] .norag-svg .r-query-norag { fill: #1f2937; stroke: #7c3aed; }
[data-md-color-scheme="slate"] .norag-svg .r-retrieve    { fill: #292524; stroke: #57534e; }
[data-md-color-scheme="slate"] .norag-svg .r-warning     { fill: #422006; stroke: #d97706; }
[data-md-color-scheme="slate"] .norag-svg .r-vcm         { fill: #052e16; stroke: #059669; }
[data-md-color-scheme="slate"] .norag-svg .r-agt         { fill: #2e1065; stroke: #7c3aed; }
[data-md-color-scheme="slate"] .norag-svg .r-bb          { fill: #172554; stroke: #2563eb; }
[data-md-color-scheme="slate"] .norag-svg .r-vcs         { fill: #1c1917; stroke: #57534e; }
[data-md-color-scheme="slate"] .norag-svg .r-tile        { fill: #292524; stroke: #57534e; }
[data-md-color-scheme="slate"] .norag-svg .r-llm         { fill: #422006; stroke: #d97706; }
[data-md-color-scheme="slate"] .norag-svg .r-gpu         { fill: #451a03; stroke: #ea580c; }
[data-md-color-scheme="slate"] .norag-svg .r-gpu-free    { fill: #78350f; stroke: #d97706; }
[data-md-color-scheme="slate"] .norag-svg .r-page-loaded { fill: #059669; }
[data-md-color-scheme="slate"] .norag-svg .t-white    { fill: white; }
[data-md-color-scheme="slate"] .norag-svg .t-title    { fill: #e0e7ff; }
[data-md-color-scheme="slate"] .norag-svg .t-body     { fill: #cbd5e1; }
[data-md-color-scheme="slate"] .norag-svg .t-muted    { fill: #94a3b8; }
[data-md-color-scheme="slate"] .norag-svg .t-hint     { fill: #6b7280; }
[data-md-color-scheme="slate"] .norag-svg .t-stone-dk { fill: #e7e5e4; }
[data-md-color-scheme="slate"] .norag-svg .t-stone    { fill: #a8a29e; }
[data-md-color-scheme="slate"] .norag-svg .t-stone-md { fill: #d6d3d1; }
[data-md-color-scheme="slate"] .norag-svg .t-stone-lt { fill: #78716c; }
[data-md-color-scheme="slate"] .norag-svg .t-vcm-dk   { fill: #6ee7b7; }
[data-md-color-scheme="slate"] .norag-svg .t-vcm      { fill: #34d399; }
[data-md-color-scheme="slate"] .norag-svg .t-vcm-md   { fill: #34d399; }
[data-md-color-scheme="slate"] .norag-svg .t-vcm-fb   { fill: #34d399; }
[data-md-color-scheme="slate"] .norag-svg .t-vcm-xdk  { fill: #a7f3d0; }
[data-md-color-scheme="slate"] .norag-svg .t-vcm-dk2  { fill: #6ee7b7; }
[data-md-color-scheme="slate"] .norag-svg .t-bb-dk    { fill: #bfdbfe; }
[data-md-color-scheme="slate"] .norag-svg .t-bb       { fill: #93c5fd; }
[data-md-color-scheme="slate"] .norag-svg .t-bb-md    { fill: #93c5fd; }
[data-md-color-scheme="slate"] .norag-svg .t-llm-dk   { fill: #fed7aa; }
[data-md-color-scheme="slate"] .norag-svg .t-llm      { fill: #fdba74; }
[data-md-color-scheme="slate"] .norag-svg .t-llm-md   { fill: #fbbf24; }
[data-md-color-scheme="slate"] .norag-svg .t-agt-dk   { fill: #ddd6fe; }
[data-md-color-scheme="slate"] .norag-svg .t-warn-dk  { fill: #fef3c7; }
[data-md-color-scheme="slate"] .norag-svg .t-warn     { fill: #fde68a; }
</style>

<div style="margin:1.5rem 0;">
<svg class="norag-svg" viewBox="0 0 900 358" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:900px;display:block;margin:0 auto;">
  <defs>
    <marker id="ah-r" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#78716c"/></marker>
    <marker id="ah-g" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#7c3aed"/></marker>
    <marker id="ah-g2" markerWidth="8" markerHeight="6" refX="1" refY="3" orient="auto"><path d="M8,0 L0,3 L8,6 Z" fill="#10b981"/></marker>
  </defs>
  <!-- ── RAG panel ── -->
  <rect x="8" y="8" width="418" height="342" rx="6" class="r-panel-rag" stroke-width="1.5"/>
  <rect x="8" y="8" width="418" height="28" rx="6" class="r-hdr-rag"/>
  <rect x="8" y="26" width="418" height="10" class="r-hdr-rag"/>
  <text x="217" y="27" text-anchor="middle" font-size="13" font-weight="600" class="t-white">RAG</text>
  <!-- Query -->
  <rect x="122" y="50" width="190" height="28" rx="4" class="r-query" stroke-width="1.2"/>
  <text x="217" y="69" text-anchor="middle" font-size="11" font-weight="600" class="t-title">Query / Task</text>
  <!-- arrow -->
  <line x1="217" y1="78" x2="217" y2="96" stroke="#78716c" stroke-width="1.4" marker-end="url(#ah-r)"/>
  <text x="225" y="91" font-size="9.5" class="t-hint">vector similarity search</text>
  <!-- Retrieve Top-K -->
  <rect x="50" y="100" width="334" height="38" rx="4" class="r-retrieve" stroke-width="1.2"/>
  <text x="217" y="117" text-anchor="middle" font-size="11" font-weight="600" class="t-body">Retrieve Top-K chunks</text>
  <text x="217" y="131" text-anchor="middle" font-size="9.5" class="t-muted">predicts relevance before reasoning happens</text>
  <!-- arrow -->
  <line x1="217" y1="138" x2="217" y2="156" stroke="#78716c" stroke-width="1.4" marker-end="url(#ah-r)"/>
  <text x="225" y="151" font-size="9.5" class="t-hint">sparse context window</text>
  <!-- LLM subset -->
  <rect x="50" y="160" width="334" height="28" rx="4" class="r-query" stroke-width="1.2"/>
  <text x="217" y="179" text-anchor="middle" font-size="11" class="t-body">LLM reasons over sparse subset</text>
  <!-- arrow -->
  <line x1="217" y1="188" x2="217" y2="206" stroke="#78716c" stroke-width="1.4" marker-end="url(#ah-r)"/>
  <!-- Miss connections -->
  <rect x="50" y="210" width="334" height="38" rx="4" class="r-warning" stroke-width="1.5"/>
  <text x="217" y="227" text-anchor="middle" font-size="11" font-weight="600" class="t-warn-dk">Cross-cutting connections missed</text>
  <text x="217" y="241" text-anchor="middle" font-size="9.5" class="t-warn">most valuable links span seemingly unrelated pages</text>
  <!-- bottom note -->
  <text x="217" y="272" text-anchor="middle" font-size="10" font-style="italic" class="t-muted">Optimizes for recall of known-relevant info.</text>
  <text x="217" y="286" text-anchor="middle" font-size="10" font-style="italic" class="t-muted">Hides context that "seems" irrelevant — precisely the</text>
  <text x="217" y="300" text-anchor="middle" font-size="10" font-style="italic" class="t-muted">context where breakthroughs come from.</text>

  <!-- ── VS divider ── -->
  <line x1="450" y1="20" x2="450" y2="348" stroke="#e5e7eb" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="450" y="192" text-anchor="middle" font-size="12" font-weight="700" class="t-hint">vs</text>

  <!-- ── NoRAG panel ── -->
  <rect x="474" y="8" width="418" height="342" rx="6" class="r-panel-norag" stroke-width="1.5"/>
  <rect x="474" y="8" width="418" height="28" rx="6" class="r-hdr-norag"/>
  <rect x="474" y="26" width="418" height="10" class="r-hdr-norag"/>
  <text x="683" y="27" text-anchor="middle" font-size="13" font-weight="600" class="t-white">NoRAG (Colony)</text>
  <!-- Task -->
  <rect x="588" y="50" width="190" height="28" rx="4" class="r-query-norag" stroke-width="1.2"/>
  <text x="683" y="69" text-anchor="middle" font-size="11" font-weight="600" class="t-title">Query / Task</text>
  <!-- arrow -->
  <line x1="683" y1="78" x2="683" y2="96" stroke="#7c3aed" stroke-width="1.4" marker-end="url(#ah-g)"/>
  <!-- WorkingSet box -->
  <rect x="484" y="100" width="370" height="76" rx="4" class="r-vcm" stroke-width="1.2"/>
  <text x="669" y="117" text-anchor="middle" font-size="10.5" font-weight="700" class="t-vcm-dk">WorkingSetCapability: cluster-wide KV cache coordination</text>
  <text x="669" y="131" text-anchor="middle" font-size="9.5" class="t-vcm">virtual pages ≫ KV cache capacity  ·  page graph guides selection</text>
  <text x="669" y="144" text-anchor="middle" font-size="9" class="t-vcm">centrality → BFS traversal → cache-aware scoring → eviction candidates</text>
  <text x="669" y="157" text-anchor="middle" font-size="9" class="t-vcm-md">state: vcm:working_set:{tenant_id} on Blackboard — shared across all agents</text>
  <text x="669" y="170" text-anchor="middle" font-size="9" class="t-vcm-md">request_pages() · release_pages() · score_pages() · identify_eviction_candidates()</text>
  <!-- arrow -->
  <line x1="683" y1="176" x2="683" y2="194" stroke="#7c3aed" stroke-width="1.4" marker-end="url(#ah-g)"/>
  <text x="691" y="189" font-size="9.5" class="t-hint">page fault → VCM loads / evicts</text>
  <!-- infer_with_suffix -->
  <rect x="484" y="198" width="370" height="28" rx="4" class="r-agt" stroke-width="1.2"/>
  <text x="669" y="217" text-anchor="middle" font-size="11" class="t-body">infer_with_suffix over loaded pages</text>
  <!-- arrow -->
  <line x1="683" y1="226" x2="683" y2="244" stroke="#7c3aed" stroke-width="1.4" marker-end="url(#ah-g)"/>
  <!-- Results box -->
  <rect x="484" y="248" width="370" height="50" rx="4" class="r-bb" stroke-width="1.2"/>
  <text x="669" y="265" text-anchor="middle" font-size="10.5" font-weight="700" class="t-bb-dk">Scope-aware results → Blackboard</text>
  <text x="669" y="279" text-anchor="middle" font-size="9.5" class="t-bb-md">tagged: source_agent · source_pages · scope</text>
  <text x="669" y="292" text-anchor="middle" font-size="9" class="t-bb-md">merge · detect_contradictions · synthesize across agents</text>
  <!-- feedback arrow -->
  <path d="M 854 248 C 876 220 876 158 854 176" stroke="#10b981" stroke-width="1.2" stroke-dasharray="4 3" fill="none" marker-end="url(#ah-g2)"/>
  <text x="878" y="208" font-size="9" class="t-vcm-fb">record_query</text>
  <text x="878" y="219" font-size="9" class="t-vcm-fb">_resolution()</text>
  <text x="878" y="230" font-size="9" class="t-vcm-fb">→ strengthen</text>
  <text x="878" y="241" font-size="9" class="t-vcm-fb">graph edges</text>
  <!-- bottom note -->
  <text x="669" y="320" text-anchor="middle" font-size="10" font-style="italic" class="t-muted">Discovers unknown-relevant connections.</text>
  <text x="669" y="334" text-anchor="middle" font-size="10" font-style="italic" class="t-muted">Graph stabilizes across rounds → O(N²) → O(N log N) amortized.</text>
</svg>
</div>

## Deep Research as a Game

Colony views deep research as a **game** where the moves available to agents are combinations of facts that offer the smallest leap to new insights. This framing has concrete architectural consequences:

- The **game state** is the full set of live context pages plus accumulated findings
- A **move** is a synthesis step that connects facts from different pages into a new insight
- The **strategy** is the order and combination in which pages are visited and cross-referenced
- **Winning** means reaching the deepest insights that the context can support

For this game to work, the entire context must remain live. *You cannot play chess if most of the board is hidden behind a retrieval layer that only shows you the squares it thinks are relevant*.

!!! warning "The Retrieval Trap"
    Retrieval systems optimize for *recall of known-relevant information*. Deep reasoning requires *discovery of unknown-relevant connections*. These are fundamentally different objectives, and optimizing for the first actively harms the second by hiding context that "seems" irrelevant.

## Why Not RNNs or State Space Models?

Recurrent neural networks (RNNs) and state space models (SSMs) like Mamba offer an alternative to transformers for processing long sequences: they compress context into a fixed-size hidden state. This sounds efficient, but it has a fatal flaw for deep reasoning.

Once an RNN or SSM decides to forget some context, **it cannot recover it**. The compression is irreversible. Information that seemed unimportant in early layers may turn out to be critical ten reasoning steps later, and there is no mechanism to retrieve it.

LLMs with external memory (Colony's architecture) can always retrieve forgotten context from external storage -- blackboard state, VCM pages, agent findings -- when the reasoning process discovers it is needed. This is the same advantage that random-access memory has over streaming tape: you can go back.

!!! note "Irreversible Forgetting"
    This is not a limitation that better training will fix. It is a structural property of recurrent architectures. The hidden state has finite capacity, and any compression scheme must discard information. Deep reasoning over extremely long context requires that *nothing* be permanently discarded until the task is complete.

## Virtual Memory for LLMs

If you cannot retrieve-and-forget, you need a system that can manage context at the scale of billions of tokens. Colony's answer is to treat KV cache management like an operating system treats virtual memory.

| OS Virtual Memory | Colony VCM |
|---|---|
| Virtual address space | Virtual context pages |
| Physical RAM | GPU KV cache capacity |
| Page tables | Global page table |
| Page faults | Page faults |
| Working set | Active pages in KV caches of all LLM instances |
| Page replacement (LRU, etc.) | Page replacement (LRU, etc.) |
| Prefetching | Speculative page loading from page graph |

Context is partitioned into **pages** and managed through a **Virtual Context Manager (VCM)** that operates at the cluster level -- across GPU nodes, not just within a single device. Pages are loaded into and evicted from KV caches based on access patterns, with a dynamically-updated **page attention graph** that captures which pages answer queries from which other pages.

<div style="margin:1.5rem 0;">
<svg class="norag-svg" viewBox="0 0 900 310" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:900px;display:block;margin:0 auto;">
  <defs>
    <marker id="vcm-ah" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#10b981"/></marker>
    <marker id="vcm-ah-b" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#a78bfa"/></marker>
    <marker id="vcm-ah-o" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#f59e0b"/></marker>
  </defs>

  <!-- ── Virtual Context Space (left, large) ── -->
  <rect x="8" y="8" width="290" height="260" rx="6" class="r-vcs" stroke-width="1.4" stroke-dasharray="5 3"/>
  <text x="153" y="28" text-anchor="middle" font-size="11.5" font-weight="700" class="t-stone-dk">Virtual Context Space</text>
  <text x="153" y="42" text-anchor="middle" font-size="9.5" class="t-stone">(total size ≫ KV cache capacity)</text>
  <!-- page grid - many small tiles -->
  <!-- row 1 -->
  <rect x="20" y="52" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="39" y="66" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">repo/1</text>
  <rect x="64" y="52" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="83" y="66" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">repo/2</text>
  <rect x="108" y="52" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="127" y="66" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">repo/3</text>
  <rect x="152" y="52" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="171" y="66" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">doc/1</text>
  <rect x="196" y="52" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="215" y="66" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">doc/2</text>
  <rect x="240" y="52" width="50" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="265" y="66" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">kb/1</text>
  <!-- row 2 -->
  <rect x="20" y="78" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="39" y="92" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">repo/4</text>
  <rect x="64" y="78" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="83" y="92" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">repo/5</text>
  <rect x="108" y="78" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="127" y="92" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">repo/6</text>
  <rect x="152" y="78" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="171" y="92" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">doc/3</text>
  <rect x="196" y="78" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="215" y="92" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">doc/4</text>
  <rect x="240" y="78" width="50" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="265" y="92" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">kb/2</text>
  <!-- row 3 -->
  <rect x="20" y="104" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="39" y="118" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">repo/7</text>
  <rect x="64" y="104" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="83" y="118" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">repo/8</text>
  <rect x="108" y="104" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="127" y="118" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">repo/9</text>
  <rect x="152" y="104" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="171" y="118" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">doc/5</text>
  <rect x="196" y="104" width="38" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="215" y="118" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">doc/6</text>
  <rect x="240" y="104" width="50" height="20" rx="2" class="r-tile" stroke-width="0.8"/><text x="265" y="118" text-anchor="middle" font-size="8" font-family="monospace" class="t-stone-md">kb/3</text>
  <!-- ellipsis rows -->
  <text x="153" y="138" text-anchor="middle" font-size="14" class="t-stone-lt">⋯ ⋯ ⋯ ⋯ ⋯</text>
  <text x="153" y="152" text-anchor="middle" font-size="10" class="t-stone-lt">N pages total  (N ≫ KV capacity)</text>
  <!-- Page Graph annotation -->
  <rect x="20" y="165" width="268" height="42" rx="4" class="r-vcs" stroke-width="1"/>
  <text x="154" y="181" text-anchor="middle" font-size="9.5" font-weight="600" class="t-stone-dk">Page Attention Graph (NetworkX DiGraph)</text>
  <text x="154" y="195" text-anchor="middle" font-size="9" class="t-stone-md">edges: centrality · BFS · discovered_dependency</text>
  <text x="154" y="207" text-anchor="middle" font-size="9" class="t-stone-md">weights updated by record_query_resolution()</text>
  <!-- Blackboard annotation -->
  <rect x="20" y="216" width="268" height="40" rx="4" class="r-bb" stroke-width="1"/>
  <text x="154" y="232" text-anchor="middle" font-size="9.5" font-weight="600" class="t-bb-dk">Blackboard: cluster-wide working set state</text>
  <text x="154" y="246" text-anchor="middle" font-size="9" class="t-bb">vcm:working_set:{tenant_id}  ·  results:partial:{tenant_id}:*</text>
  <text x="154" y="257" text-anchor="middle" font-size="9" class="t-muted">(all agents coordinate here)</text>

  <!-- ── VCM middle ── -->
  <rect x="330" y="100" width="130" height="76" rx="6" class="r-vcm" stroke-width="1.5"/>
  <text x="395" y="120" text-anchor="middle" font-size="11" font-weight="700" class="t-vcm-dk">VCM</text>
  <text x="395" y="134" text-anchor="middle" font-size="9" class="t-vcm">Page Table</text>
  <text x="395" y="146" text-anchor="middle" font-size="9" class="t-vcm">Cache Scheduling</text>
  <text x="395" y="158" text-anchor="middle" font-size="9" class="t-vcm">Page Fault Handler</text>
  <text x="395" y="170" text-anchor="middle" font-size="9" class="t-vcm">Eviction / Prefetch</text>

  <!-- arrows: virtual space ↔ VCM -->
  <line x1="298" y1="138" x2="330" y2="138" stroke="#10b981" stroke-width="1.4" marker-end="url(#vcm-ah)"/>
  <text x="314" y="133" text-anchor="middle" font-size="8" class="t-hint">load</text>
  <line x1="330" y1="148" x2="298" y2="148" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="3 2" marker-end="url(#vcm-ah-o)"/>
  <text x="314" y="162" text-anchor="middle" font-size="8" class="t-hint">evict</text>

  <!-- ── KV Cache Cluster (right) ── -->
  <rect x="492" y="8" width="400" height="292" rx="6" class="r-llm" stroke-width="1.5"/>
  <text x="692" y="28" text-anchor="middle" font-size="11.5" font-weight="700" class="t-llm-dk">KV Cache Cluster  (physical capacity)</text>
  <text x="692" y="42" text-anchor="middle" font-size="9.5" class="t-llm-md">holds only the current working set — a subset of all virtual pages</text>

  <!-- GPU node 1 -->
  <rect x="504" y="52" width="175" height="112" rx="4" class="r-gpu" stroke-width="1.2"/>
  <text x="591" y="68" text-anchor="middle" font-size="10" font-weight="600" class="t-llm">GPU Node 1  · KV Cache</text>
  <!-- loaded pages -->
  <rect x="514" y="76" width="70" height="20" rx="2" class="r-page-loaded"/><text x="549" y="90" text-anchor="middle" font-size="8" font-family="monospace" class="t-white">repo/1 ✓</text>
  <rect x="590" y="76" width="70" height="20" rx="2" class="r-page-loaded"/><text x="625" y="90" text-anchor="middle" font-size="8" font-family="monospace" class="t-white">repo/2 ✓</text>
  <rect x="514" y="102" width="70" height="20" rx="2" class="r-page-loaded"/><text x="549" y="116" text-anchor="middle" font-size="8" font-family="monospace" class="t-white">doc/3 ✓</text>
  <rect x="590" y="102" width="70" height="20" rx="2" class="r-page-loaded"/><text x="625" y="116" text-anchor="middle" font-size="8" font-family="monospace" class="t-white">kb/1 ✓</text>
  <rect x="514" y="128" width="146" height="20" rx="2" class="r-gpu-free" stroke-width="0.8"/>
  <text x="587" y="142" text-anchor="middle" font-size="8" class="t-llm-dk">… remaining slots free</text>

  <!-- GPU node 2 -->
  <rect x="697" y="52" width="187" height="112" rx="4" class="r-gpu" stroke-width="1.2"/>
  <text x="790" y="68" text-anchor="middle" font-size="10" font-weight="600" class="t-llm">GPU Node 2  · KV Cache</text>
  <rect x="707" y="76" width="76" height="20" rx="2" class="r-page-loaded"/><text x="745" y="90" text-anchor="middle" font-size="8" font-family="monospace" class="t-white">repo/5 ✓</text>
  <rect x="791" y="76" width="76" height="20" rx="2" class="r-page-loaded"/><text x="829" y="90" text-anchor="middle" font-size="8" font-family="monospace" class="t-white">doc/1 ✓</text>
  <rect x="707" y="102" width="76" height="20" rx="2" class="r-page-loaded"/><text x="745" y="116" text-anchor="middle" font-size="8" font-family="monospace" class="t-white">doc/5 ✓</text>
  <rect x="791" y="102" width="76" height="20" rx="2" class="r-page-loaded"/><text x="829" y="116" text-anchor="middle" font-size="8" font-family="monospace" class="t-white">kb/3 ✓</text>
  <rect x="707" y="128" width="160" height="20" rx="2" class="r-gpu-free" stroke-width="0.8"/>
  <text x="787" y="142" text-anchor="middle" font-size="8" class="t-llm-dk">… remaining slots free</text>

  <!-- GPU node N -->
  <rect x="504" y="180" width="380" height="28" rx="4" class="r-gpu" stroke-width="1" stroke-dasharray="4 2"/>
  <text x="694" y="199" text-anchor="middle" font-size="10" class="t-llm-md">GPU Node N  ·  KV Cache  ·  …</text>

  <!-- Agents -->
  <rect x="504" y="222" width="380" height="28" rx="4" class="r-agt" stroke-width="1.2"/>
  <text x="694" y="241" text-anchor="middle" font-size="10" font-weight="600" class="t-agt-dk">Agents: infer_with_suffix against loaded pages (cached tokens reused)</text>

  <!-- Working set note -->
  <rect x="504" y="260" width="380" height="34" rx="4" class="r-vcm" stroke-width="1"/>
  <text x="694" y="276" text-anchor="middle" font-size="9.5" font-weight="600" class="t-vcm-xdk">WorkingSetCapability: coordinates which pages are hot</text>
  <text x="694" y="289" text-anchor="middle" font-size="9" class="t-vcm-dk2">request_pages() triggers VCM load  ·  hard/soft affinity keeps related pages co-located</text>

  <!-- VCM → cluster arrow -->
  <line x1="460" y1="138" x2="492" y2="108" stroke="#10b981" stroke-width="1.4" marker-end="url(#vcm-ah)"/>
  <line x1="460" y1="140" x2="492" y2="170" stroke="#10b981" stroke-width="1.4" marker-end="url(#vcm-ah)"/>
</svg>
</div>

This is not a metaphor. Colony implements actual page fault semantics, working set tracking, and cache-aware scheduling -- the same fundamental mechanisms that made virtual memory one of the most successful abstractions in computing history. The difference is that "physical memory" is GPU KV cache capacity distributed across a cluster, and "addresses" are semantic page identifiers rather than integers.

!!! note "Inference with Suffix"
    Since LLMs are causal models, KV caches stay valid (and, hence, useful) only if a cached sequence is a prefix of the current input. Colony's `infer_with_suffix()` API allows agents to specify a suffix to append to the cached page content, enabling flexible reuse of cached tokens across different reasoning steps. Moreover, the page contents themselves are prefixed with a system message that provides context about the page's origin and relevance, and that also explains the reasoning process over sharded/paged context.



## The Payoff: Amortized Efficiency

The initial cost of reasoning over all pages is high: $O(N^2)$ for routing queries among $N$ pages. But as the page attention graph stabilizes over successive reasoning rounds, the amortized cost per round drops to $O(N \log N)$. Deep reasoning tasks inherently require many rounds, so the graph has time to stabilize and the amortized cost dominates. This is especially true for research tasks, where:
- The context corpus (git repos, docs, KBs) changes slowly (e.g., cumulative knowledge).
- The number of reasoning rounds can be very large.

!!! tip "Amortized Efficiency"
    This is the same insight behind persistent data structures and amortized analysis in algorithms: pay a high upfront cost to build structure that makes all subsequent operations cheaper. Colony applies this principle to multi-agent reasoning over extremely long context.

