# Game Engine

Colony's game engine provides structured multi-agent deliberation through formal game-theoretic protocols. Games serve as correctness mechanisms -- they combat hallucination, laziness, goal drift, and miscommunication by forcing agents into adversarial or cooperative interactions with defined rules and payoffs.

## Why Games?

LLM agents suffer from predictable failure modes. The game engine maps each failure mode to a game-theoretic countermeasure:

| Failure Mode | Game Countermeasure |
|-------------|---------------------|
| **Hallucination** | Every claim is a `ScopeAwareResult` with confidence, evidence, and missing context. No single agent's output is final. |
| **Laziness** | Contract Net task allocation with reputation-based bid selection. Agents compete for tasks. |
| **Goal drift** | `ObjectiveGuardAgent` checks each new draft against the original goal. Plans and intentions are explicit (BDI). |
| **Miscommunication** | Meta-agent normalizes vocabulary and detects mismatched assumptions across agents. |

## Four Game Types

### Hypothesis Game
One agent proposes a hypothesis; others refute or refine it. Continues until the hypothesis is accepted (sufficient evidence), rejected (contradicting evidence), or refined into a new hypothesis.

### Bidding / Contract Game
Multiple agents bid to take a subtask (Contract Net Protocol). The supervisor evaluates bids based on capability match, reputation, current load, and cache affinity. The winning bidder commits to delivery.

### Negotiation Game
Agents with conflicting constraints exchange offers until they reach agreement or escalate to an arbiter. Used for resource allocation, page assignment, and conflicting analysis results.

### Consensus Game
Agents vote or provide evidence; a meta-agent aggregates using configurable voting rules. Used for final decisions where multiple independent analyses must be reconciled.

## Roles

Games define fixed roles with specific permissions:

| Role | Responsibility |
|------|---------------|
| **Proposer** | Initiates hypotheses, claims, or offers |
| **Skeptic** | Challenges claims, demands evidence, identifies weaknesses |
| **Grounder** | Connects claims to specific evidence in context pages |
| **Arbiter** | Makes final decisions when agents cannot agree |
| **Planner** | Decomposes tasks and coordinates agent assignments |

Roles map to move permissions -- a Skeptic can `challenge` and `request_evidence` but cannot `propose`. This prevents agents from stepping outside their designated function.

## GameProtocolCapability

`polymathera.colony.agents.patterns.games.state.GameProtocolCapability` is the base class for game implementations. It extends `AgentCapability` with:

- **State machine**: Game phases (setup, active, terminal) with validated transitions
- **Move validation**: Moves checked against role permissions before application
- **Shared scope**: All participants share a blackboard namespace via `scope_id=game_id`
- **Memory integration**: Terminal game states are captured via `MemoryProducerConfig` for episodic memory

```python
class MyGameProtocol(GameProtocolCapability):
    role_permissions = {
        "proposer": {"propose", "revise"},
        "skeptic": {"challenge", "request_evidence"},
        "arbiter": {"accept", "reject", "request_revision"},
    }

    async def apply_move(self, state, move):
        # Implement state transitions
        ...
```

## Agent Communication Language

Messages between agents are not plain strings. Each message has structure:

- **Illocutionary force**: The intent -- `inform`, `request`, `propose`, `promise`, `challenge`, `accept`, `reject`
- **Content**: The payload (claim, evidence, offer, etc.)
- **Preconditions**: What must be true for this message to be valid
- **Expected effects**: What the sender expects to change

This follows the FIPA Agent Communication Language model, adapted for LLM agents where the "content" is often natural language with structured metadata.

## Hybrid Architecture

The game engine uses a hybrid deliberative-reactive architecture:

- **Deliberative core**: The LLM handles planning, explanation, and reasoning about game state
- **Reactive rules**: Automated triggers that surround the LLM core:
    - Auto-invoke validators on every move
    - Auto-escalate on low confidence
    - Auto-trigger conflict resolution when contradictions are detected

Each agent's mental state is represented partly in prompts (natural language reasoning) and partly in structured state on the blackboard:

- **Beliefs**: References to blackboard entries the agent considers true
- **Goals**: Explicit `Goal` objects with success criteria
- **Intentions**: Current plans and sub-tasks

## Advanced Mechanisms

### No-Regret Learning

Algorithms like Exp3/EXP4 adjust the mixture over agents and strategies based on quality metrics across games. Over time, the system learns which agents perform well at which roles and which strategies succeed in which contexts.

### Targeted Learning

`TargetedLearningManager` clusters past tasks by similarity and learns per-cluster tuning parameters. Different types of analysis tasks may require different game configurations, and the system adapts automatically.

### VCG-Style Incentives

Vickrey-Clarke-Groves mechanism design rewards agents for their marginal contribution to global performance. This discourages free-riding and encourages honest reporting of capabilities and confidence levels.

### Social Choice Theory

Voting rules aggregate evaluator rankings for consensus games. Arrow's impossibility theorem informs rule selection -- no voting rule is perfect, so the choice depends on which properties matter most for the specific decision.

### Coalition Games

Agents can form stable, high-value coalitions for complex tasks. Coalition value is approximated empirically over time. Stable coalitions persist across tasks; unstable ones are dissolved and reformed.

### Epistemic Logic

Agent mental states are structured and inspectable:

- Beliefs and knowledge are classified as "agent's belief" vs. "common knowledge"
- Only `common=True` propositions appear in final confirmed reports
- This prevents individual agent biases from propagating to system output

## Integration with Action Policies

Games are invoked by action policies as plugin policies. The `CacheAwareActionPolicy` can delegate to a game protocol when it detects a situation requiring multi-agent deliberation:

- Conflicting analysis results from child agents trigger a hypothesis game
- Resource contention triggers a negotiation game
- Task decomposition triggers a bidding game
- Final synthesis triggers a consensus game

The game events flow into the planning context, and the LLM planner decides how to respond -- games inform the LLM's reasoning rather than bypassing it.
