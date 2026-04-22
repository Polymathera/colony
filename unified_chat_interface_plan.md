# Unified Chat Interface Plan

## Problem Statement

The dashboard currently has two divergent interaction modes:

1. **"Start Run" button** (`SubmitRunDialog`) — selects analysis types and parameters, spawns coordinators in the background, user monitors via polling
2. **"Interact" tab** — WebSocket chat with a specific agent, buried as one of 11 tabs

These need to merge into a single chat interface that is the **primary portal** for all user interaction with the Colony. All other tabs become observability lenses, not interaction surfaces.

## Conceptual Model

```
Colony (workspace/project)
  └─ Session (like a Slack channel — long-lived conversation thread)
       └─ Run (individual execution triggered by a user message)
```

- **Colony**: groups assets (repos, docs, tools) and agents. Multiple concurrent sessions.
- **Session**: sequence of runs and interactions evolving colony state. Focused on a long-running task or goal.
- **Run**: single execution in response to a user message. Can spawn coordinators, workers, use tools.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      AppShell                                │
│  ┌───────┐  ┌──────────────────────────────────────────────┐ │
│  │Sidebar│  │  ┌──────────────────────┐  ┌──────────────┐  │ │
│  │       │  │  │   Tab Content Area   │  │  Chat Panel  │  │ │
│  │Session│  │  │  (Overview, Agents,  │  │  (resizable) │  │ │
│  │ List  │  │  │   VCM, Traces, etc.) │  │              │  │ │
│  │       │  │  │                      │  │  messages    │  │ │
│  │       │  │  │                      │  │  input bar   │  │ │
│  │       │  │  │                      │  │  controls    │  │ │
│  │       │  │  └──────────────────────┘  └──────────────┘  │ │
│  └───────┘  └──────────────────────────────────────────────┘ │
│  [StatusBar]                                                 │
└──────────────────────────────────────────────────────────────┘
```

The chat panel is **always visible** alongside tabs (not a tab itself), resizable via drag handle, and collapsible.

## Detailed Design

### 1. Chat Panel (Frontend)

**New component**: `components/chat/ChatPanel.tsx`

The chat panel is a persistent side panel (right side of the main content area), not a tab. It is:
- Always visible when a session is active
- Resizable via a vertical drag handle (min 300px, max 60% of viewport)
- Collapsible to a thin strip with a toggle button
- State (width, collapsed) persisted to `localStorage`

**Sub-components**:

| Component | Responsibility |
|-----------|---------------|
| `ChatPanel.tsx` | Container: resize handle, collapse toggle, renders header + messages + input |
| `ChatMessageList.tsx` | Scrollable message history with auto-scroll, renders message bubbles |
| `ChatMessage.tsx` | Single message: user, agent, or system. Supports markdown, code blocks, expandable sections |
| `ChatInput.tsx` | Text input with send button, @-mention autocomplete, #-reference autocomplete, attachment controls |
| `ChatControls.tsx` | Expandable controls panel above the input: VCM context selector, agent/capability selector, effort/quota sliders, "Map Content" button |
| `RunStatusBanner.tsx` | Inline banner in message list showing active run status (spawning, running N agents, completed) |

**Message format** (frontend model):

```typescript
interface ChatMessage {
  id: string;
  run_id: string | null;        // which run this message belongs to
  role: "user" | "agent" | "system";
  agent_id?: string;            // which agent sent this (for agent messages)
  agent_type?: string;          // human-readable agent type
  content: string;              // markdown content
  timestamp: number;
  // Agent-to-user questions
  request_id?: string;          // for routing user's reply back to the requesting agent
  response_options?: string[];  // multiple-choice options (rendered as inline buttons)
  awaiting_reply?: boolean;     // true if agent is blocked waiting for user input
  // Rich content (future)
  attachments?: Attachment[];
  // Run lifecycle messages
  run_status?: "submitted" | "spawning" | "running" | "completed" | "failed";
  run_summary?: RunSummary;     // token counts, duration, agents used
}
```

### 2. Agent-to-User Communication

The chat is **bidirectional** — agents can proactively post messages to the user, not just respond to user messages. This is a core interaction pattern:

- **Agent questions**: A coordinator analyzing code discovers an ambiguous pattern and asks the user: "The `auth` module has two competing session strategies. Which should I treat as canonical?" The user responds in the chat, and the coordinator uses the answer to continue.
- **Agent progress updates**: "Spawned 5 workers. 3/5 clusters analyzed so far. Found 2 high-severity issues."
- **Agent decision requests**: "I found a circular dependency between modules A and B. Should I (1) flag it and continue, or (2) deep-dive into resolution strategies?"

**Implementation**: Agents post to the user via the blackboard using `SessionChatProtocol`. The WebSocket bridge subscribes to the session's chat scope and relays agent messages to the frontend in real-time. The user's reply is routed back to the requesting agent (tracked by `request_id` in the protocol).

**Frontend rendering**: Agent questions render as distinct message types with optional response affordances — inline reply buttons for multiple-choice decisions, or the user types a free-form response. The message tracks which agent asked and the `request_id` so the response routes back correctly.

### 3. Chat Syntax

The chat input supports Slack-like syntax parsed client-side for autocomplete and server-side for routing:

**@ — Addressing agents, capabilities, and tools**:

| Syntax | Purpose | Example |
|--------|---------|---------|
| `@agent_type` | Address a specific agent type | `@impact-coordinator analyze the auth module` |
| `@agent_id` | Address a specific agent instance | `@agent_a1b2c3 what's your progress?` |
| `@capability` | Request a specific capability be used | `@consciousness reflect on the last run` |
| `@tool:name` | Request a specific tool | `@tool:web-search find the latest CVE for openssl` |

**# — Referencing VCM assets**:

| Syntax | Purpose | Example |
|--------|---------|---------|
| `#repo:name` | Reference a VCM scope/repo | `#repo:crewAI focus on the agents/ directory` |
| `#page:id` | Reference a specific VCM page | `#page:pg_abc123` |
| `#file:path` | Reference a file path within a repo | `#file:src/auth/middleware.py` |
| `#dir:path` | Reference a directory | `#dir:src/agents/` |
| `#lang:name` | Filter by programming language | `#lang:python only Python files` |

**/ — Commands (structured actions)**:

| Syntax | Purpose | Example |
|--------|---------|---------|
| `/analyze` | Start a structured analysis run | `/analyze impact --max-agents 5 --quality 0.8` |
| `/map` | Map content to VCM | `/map https://github.com/org/repo --branch main` |
| `/abort` | Abort the current run or a specific run | `/abort` or `/abort run_abc123` |
| `/status` | Show current run/session status | `/status` |
| `/agents` | List active agents in this session | `/agents` |
| `/set` | Set a session parameter | `/set max-agents=5` or `/set effort=high` |
| `/help` | Show available commands and syntax | `/help` or `/help analyze` |
| `/context` | Show or modify active VCM context | `/context` or `/context add #repo:crewAI` |

**@ and # triggers** show autocomplete dropdowns populated from live data (agent registry, VCM page index). `/` shows a command palette with descriptions and argument hints.

### 4. Session Agent (Backend)

**New component**: `agents/sessions/session_agent.py`

Each session gets a **Session Agent** — a coordinator that:
- Receives all user messages for the session
- Decides how to handle each message (route to existing agent, spawn new coordinators, use tools, respond directly)
- Manages the lifecycle of runs within the session
- Maintains conversational context across runs

```python
class SessionAgent(Agent):
    """Per-session orchestrator agent.
    
    Spawned via AgentHandle.from_blueprint() when a session is created.
    Receives user messages via blackboard, decides execution strategy,
    and orchestrates runs by spawning child coordinator agents.
    Communicates results back via blackboard events.
    """
    
    # Actions this agent needs to perform:
    #
    # 1. Spawn coordinator agents for analysis runs
    #    → Uses AgentPoolCapability (already exists) to manage child agents
    #
    # 2. Monitor running coordinators and relay progress to the user
    #    → Subscribes to child agent events via blackboard streaming
    #
    # 3. Query VCM for page listings, search results
    #    → Calls VCM deployment handle (list_stored_pages, etc.)
    #    → Does NOT need VCMAnalysisCapability (that's for page-level code analysis)
    #
    # 4. Trigger VCM mapping of new repos/content
    #    → Calls VCM deployment handle (mmap_application_scope)
    #
    # 5. Query session/run state (active runs, token usage, etc.)
    #    → Calls SessionManager deployment handle
    #
    # 6. Cancel running agents on user request
    #    → Blackboard interrupt via InterruptionProtocol
    #
    # 7. Web search to answer user questions with external context
    #    → Needs a tool/capability for web search (does not exist yet)
    #    → Used when the user asks questions that require external knowledge
    #      (e.g., "what's the latest CVE for openssl?", "how does X library work?")
    #
    # Capabilities to bind (existing):
    # - AgentPoolCapability — spawn and manage child coordinator agents
    # - ConsciousnessCapability — self-monitoring, introspection
    # - WorkingMemoryCapability — maintain conversational context across runs
    #
    # New capabilities needed:
    # - SessionOrchestratorCapability — wraps actions 2-6 above (VCM queries,
    #   mapping, session state, cancellation). Thin wrapper over deployment handles.
    # - WebSearchCapability — wraps a web search API for action 7. Used by the
    #   session agent to answer user questions requiring external knowledge.
    #
    # Communication:
    # - Receives user messages via SessionChatProtocol on the blackboard
    # - Sends responses back via the same protocol
    # - Spawns child coordinators via AgentPoolCapability
    # - Monitors children via blackboard event streaming
```

**Decision flow for user messages**:

```
User message arrives
  ├─ Is this a reply to an agent question? (has request_id context)
  │   └─ Route reply back to the requesting agent via blackboard
  │
  ├─ Starts with /command? → Parse and execute command
  │   ├─ /analyze → Spawn coordinator agents (like current SubmitRunDialog flow)
  │   ├─ /map → Trigger VCM mapping (like current MapContentDialog flow)
  │   ├─ /abort → Cancel current run via InterruptionProtocol
  │   ├─ /status → Query session/run state, respond
  │   ├─ /set → Update session parameters (max-agents, effort, etc.)
  │   ├─ /agents → List active agents, respond
  │   ├─ /context → Show/modify active VCM context
  │   └─ /help → Show help text, respond
  │
  ├─ Contains @agent reference? → Route to specific agent
  │   ├─ @agent_type → Find agent by type, forward via blackboard
  │   ├─ @capability → Request session agent use specific capability
  │   └─ @tool:name → Request session agent use specific tool
  │
  └─ Plain message → Session agent handles directly
      ├─ If agents are running → interpret as guidance/interrupt
      ├─ If question about results → query blackboard/VCM, respond
      └─ If new task → decide whether to spawn agents or respond
```

**Agent-to-user flow** (reverse direction):

```
Agent posts to SessionChatProtocol on blackboard
  ├─ WebSocket bridge picks up the event
  ├─ Sends { type: "message", message: ChatMessage } to client
  │   message.role = "agent"
  │   message.agent_id = requesting agent
  │   message.request_id = for routing user's reply back
  │
  ├─ If message has response_options (multiple choice):
  │   └─ Frontend renders inline buttons for each option
  │
  └─ User replies → routed back to requesting agent (see above)
```

### 5. Backend Chat Protocol (WebSocket)

**Endpoint**: `ws://host/api/v1/ws/session/{session_id}/chat`

Replaces the current `/ws/chat/{session_id}`. The key difference: messages go to the **session agent**, not directly to individual agents.

**Client → Server messages**:

```typescript
// User sends a chat message (new interaction → creates a run)
{ type: "message", content: string, controls?: ChatControls }

// User replies to an agent question (routes back to the requesting agent)
{ type: "reply", content: string, request_id: string, agent_id: string }

// User requests to cancel the current run
{ type: "cancel_run", run_id?: string }

// User requests message history
{ type: "history", before?: string, limit?: number }
```

Where `ChatControls` captures the UI control state:

```typescript
interface ChatControls {
  // VCM context selection — query expression that narrows the page set
  vcm_context?: {
    repo_ids?: string[];          // specific repos by scope ID
    file_patterns?: string[];     // glob patterns: "src/**/*.py", "*.ts"
    dir_paths?: string[];         // specific directories: "src/agents/"
    page_ids?: string[];          // specific page IDs
    languages?: string[];         // filter by language: "python", "typescript"
    exclude_patterns?: string[];  // exclude globs: "test/**", "*.md"
  };
  // Agent/capability/tool preferences
  agent_preferences?: {
    analysis_types?: string[];    // impact, compliance, intent, contracts, slicing, basic
    max_agents?: number;
    capabilities?: string[];      // specific capabilities to bind to spawned agents
    tools?: string[];             // specific tools to make available: "web-search", "repl"
    tool_config?: Record<string, Record<string, unknown>>;  // per-tool configuration
  };
  // Resource limits
  effort?: "low" | "medium" | "high";
  timeout_seconds?: number;
  budget_usd?: number | null;
}
```

**Server → Client messages**:

```typescript
// Agent/system message in the chat (includes agent questions via request_id/response_options)
{ type: "message", message: ChatMessage }

// Run lifecycle events (spawning, progress, completion)
{ type: "run_event", run_id: string, event_type: string, data: any }

// Agent question requiring user input (highlighted in UI, optional response buttons)
{ type: "agent_question", message: ChatMessage }  // message.awaiting_reply = true

// Tab activity notification (new data in a tab)
{ type: "tab_activity", tab_id: string, count: number }

// Message history response
{ type: "history", messages: ChatMessage[], has_more: boolean }

// Error
{ type: "error", message: string }
```

### 6. Tab Activity Indicators

When a run produces new data visible in other tabs, the backend sends `tab_activity` events:

| Tab | Activity trigger |
|-----|-----------------|
| Agents | New agent spawned or agent state changed |
| VCM | New pages mapped |
| Page Graph | Graph updated |
| Blackboard | New blackboard entries |
| Logs | New log entries for this session |
| Traces | New spans for this session |

**Frontend**: Each tab in the `TabBar` shows a small indicator dot when `tab_activity` count > 0 for that tab. Clicking the tab clears its indicator.

### 7. Layout Changes

**Remove**:
- `InteractTab` (replaced by `ChatPanel`)
- `SessionsTab` (sessions are in the sidebar)
- `SubmitRunDialog` (replaced by chat `/analyze` command and `ChatControls`)
- "Start Run" button from Sidebar (replaced by chat)
- "interact" from TABS array
- "sessions" from TABS array

**Modify**:
- `AppShell.tsx`: Main content area becomes a horizontal split: `TabContent | ChatPanel`. The chat panel is rendered outside the tab system.
- `Sidebar.tsx`: Remove "Start Run" button. Keep session list, create/suspend/resume/close controls.
- `LandingPage.tsx`: "New Session" button stays. When a session is created, the chat panel opens automatically.
- `TabBar.tsx`: Add activity indicator dot support via a `notifications` prop.

**New TABS array** (9 tabs, down from 11):

```typescript
const TABS: Tab[] = [
  { id: "overview",   label: "Overview",    icon: <LayoutDashboard /> },
  { id: "agents",     label: "Agents",      icon: <Bot /> },
  { id: "vcm",        label: "VCM",         icon: <Database /> },
  { id: "graph",      label: "Page Graph",  icon: <GitFork /> },
  { id: "blackboard", label: "Blackboard",  icon: <ClipboardList /> },
  { id: "logs",       label: "Logs",        icon: <ScrollText /> },
  { id: "traces",     label: "Traces",      icon: <Activity /> },
  { id: "metrics",    label: "Metrics",     icon: <Gauge /> },
  { id: "settings",   label: "Settings",    icon: <Settings /> },
];
```

### 8. Message Persistence

Chat messages need to survive page refreshes. Two options:

**Option A (chosen)**: Store messages in the session's run history via `SessionManagerDeployment`. Each user message creates a run. Agent responses are run events. The chat endpoint loads history from runs on connect.

This reuses existing infrastructure:
- `create_run()` → user message
- `add_run_event()` → agent responses, progress
- `get_session_runs()` → message history

**Additional field needed**: Add `content` (the user's message text) to the run's `input_data`, and surface agent responses from run events.

**Option B**: Dedicated chat message storage (PostgreSQL table). More flexible but adds a new persistence layer. Defer to later if Option A proves insufficient.

### 9. ChatControls UI

The input area has an expandable controls section (collapsed by default, toggle via button):

```
┌───────────────────────────────────────────────────────┐
│  [Context ▾]  [Agents ▾]  [Tools ▾]  [Effort ▾] [Map] │  ← collapsed: buttons
├───────────────────────────────────────────────────────┤
│  VCM Context:                                         │  ← expanded
│    Repos: [crewAI ×] [colony ×] [+ add]               │
│    Files: [src/**/*.py] [× remove] [+ add pattern]    │
│    Language: [Python ×] [TypeScript ×]                │
│    Exclude: [test/** ×] [+ add]                       │
│  Agents:                                              │
│    Analysis: [Impact ✓] [Compliance ✓] [Intent]      │
│    Max agents: [  10]                                 │
│    Capabilities: [Consciousness ✓] [Critique]         │
│  Tools:                                               │
│    [Web Search ✓] [REPL] [+ configure]                │
│  Effort: ○ Low  ● Medium  ○ High                      │
│    Timeout: [600]s   Budget: [$___]                   │
├───────────────────────────────────────────────────────┤
│  [Type a message...                            Send]  │
└───────────────────────────────────────────────────────┘
```

These controls are **optional** — sending a plain text message works without touching them. The session agent uses defaults or infers from context. When controls are set, they're sent as `ChatControls` alongside the message. The `/set` command provides a text-based alternative to these UI controls (e.g., `/set max-agents=5 effort=high`).

## Implementation Phases

### Phase 1: Chat Panel Shell (Frontend only)

**Goal**: Replace the Interact tab with a persistent side panel. No backend changes.

**Files to create**:
- `components/chat/ChatPanel.tsx` — resizable panel container
- `components/chat/ChatMessageList.tsx` — message display
- `components/chat/ChatMessage.tsx` — individual message rendering
- `components/chat/ChatInput.tsx` — text input with send button

**Files to modify**:
- `AppShell.tsx` — render ChatPanel alongside tab content, remove "interact" and "sessions" from TABS, remove SubmitRunDialog, remove "Start Run" from Sidebar props
- `Sidebar.tsx` — remove "Start Run" button and `onStartRun` prop
- `TabBar.tsx` — add notification dot support

**Files to delete**:
- `components/interact/InteractTab.tsx` (replaced by `ChatPanel`)
- `components/dialogs/SubmitRunDialog.tsx` (replaced by chat controls)

**Behavior**: Uses existing WebSocket `/ws/chat/{session_id}` endpoint. Messages go directly to agents (same as current InteractTab). This phase just moves the chat from a tab to a persistent panel.

### Phase 2: Tab Activity Indicators

**Goal**: Tabs show activity dots when new data arrives during a run.

**Files to modify**:
- `TabBar.tsx` — accept `notifications: Record<string, number>` prop, render dots
- `AppShell.tsx` — track notification counts, clear on tab visit
- `api/hooks/useInfrastructure.ts` or new `useTabActivity.ts` — listen for WebSocket `tab_activity` events

**Backend**:
- `routers/chat.py` — emit `tab_activity` messages when runs produce observable state changes

### Phase 3: Session Agent (Backend)

**Goal**: User messages go to a session agent instead of directly to individual agents.

**Files to create**:
- `agents/sessions/session_agent.py` — `SessionAgent(Agent)` subclass with message routing, command parsing, and agent orchestration capabilities
- `agents/sessions/chat_protocol.py` — `SessionChatProtocol(BlackboardProtocol)` defining request/response/event keys for user↔session agent communication

**Files to modify**:
- `agents/sessions/manager.py` — `create_session` spawns a `SessionAgent` via `AgentHandle.from_blueprint()` and stores the handle in the session state
- `routers/chat.py` — rewrite to send messages to the session's `SessionAgent` handle via `run_streamed()` instead of directly to individual agents
- `routers/jobs.py` — job submission can be triggered by the session agent internally (not just the REST endpoint)

**Key design**: The `SessionAgent` is a regular `Agent` subclass, spawned per-session via `AgentHandle.from_blueprint()`. It communicates via the blackboard using `SessionChatProtocol`. For simple queries it responds directly using its own capabilities. For analysis requests, it spawns child coordinator agents (same agents that `_run_job` currently spawns) and monitors them via blackboard events.

### Phase 4: Chat Commands and Syntax

**Goal**: `/analyze`, `/map`, `/abort`, `/status` commands. `@agent` and `#ref` syntax with autocomplete.

**Files to create**:
- `components/chat/ChatAutocomplete.tsx` — dropdown for @-mentions and #-references
- `components/chat/CommandParser.ts` — client-side command parsing

**Files to modify**:
- `ChatInput.tsx` — add autocomplete trigger on `@`, `#`, `/`
- `agents/sessions/session_agent.py` — handle parsed commands
- `routers/chat.py` — forward parsed commands to session agent

### Phase 5: Chat Controls

**Goal**: VCM context selector, agent preferences, effort/quota controls in the chat input area.

**Files to create**:
- `components/chat/ChatControls.tsx` — expandable controls panel
- `components/chat/VCMContextSelector.tsx` — repo/file/page selector
- `components/chat/AgentPreferences.tsx` — analysis type chips, max agents
- `components/chat/EffortSelector.tsx` — effort level radio buttons

**Files to modify**:
- `ChatInput.tsx` — render ChatControls above input, include in message payload
- `api/types.ts` — add `ChatControls` type
- `routers/chat.py` — pass controls to session agent
- `agents/sessions/session_agent.py` — use controls to configure runs

### Phase 6: Message Persistence

**Goal**: Chat history survives page refresh.

**Files to modify**:
- `routers/chat.py` — on WebSocket connect, load history from session runs and send to client
- `agents/sessions/manager.py` — ensure run input_data and events contain enough info to reconstruct chat messages
- `ChatPanel.tsx` — request history on mount, prepend to message list

## Migration Notes

- The `SubmitRunDialog` functionality moves to the chat's `/analyze` command + `ChatControls`. The dialog component is deleted.
- The `InteractTab` is deleted entirely. Its WebSocket logic moves to `ChatPanel`.
- The "Start Run" button in the Sidebar is removed. Users type in the chat or use `/analyze`.
- The "Sessions" tab is removed. The sidebar already shows sessions.
- The `MapContentDialog` in VCMTab stays — it's a shortcut for mapping without the chat. The chat's `/map` command is an alternative path.
- The jobs REST API (`POST /jobs/submit`) stays as a programmatic API. The chat is the UI layer on top.

## Open Questions

1. **Session agent lifecycle**: One `SessionAgent` instance per session, spawned via `AgentHandle.from_blueprint()`. If it crashes, can be re-spawned from the blueprint — but how does it recover conversational context? Reload from run history? Only take note of this in the implementation for now. We will address this later for all agents not just `SessionAgent`.
2. **Multi-user sessions**: Can multiple users chat in the same session? If so, need user attribution on messages. Yes. A session is more like a Slack channel than a 1:1 conversation. Messages should include `user_id` and render the user's name/avatar in the UI.
3. **Streaming responses**: The session agent should stream its responses token-by-token (like an LLM chat) or buffer and send complete messages? Token-by-token feels more responsive but requires LLM integration in the session agent. For MVP, buffer complete messages and send when ready. We can add streaming later if needed.
4. **Chat history storage**: Option A (reuse run events) may be too coarse for conversational back-and-forth. If a user sends 10 quick questions, that's 10 runs with minimal events. Consider a lightweight `chat_messages` table in PostgreSQL. OK. Use a PostgreSQL table for chat messages, linked to sessions.
5. **Offline agents**: When the session agent spawns coordinators, how does it relay their progress to the chat? Via blackboard events → WebSocket bridge? Need to define the event flow clearly. Current thinking is that agents self-report progress by posting messages to the `SessionChatProtocol` on the blackboard. The WebSocket bridge listens for these and sends them to the UI to be rendered as collapsible "inner monologue" messages.
