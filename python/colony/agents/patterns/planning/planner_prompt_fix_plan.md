# Plan: Fix Action Duplication in Planner Prompt + Prevent Incomplete Action Keys

## Problem 1: Duplicate Action Descriptions

Action keys like `ConsciousnessCapability.b79b5858.consciousness_update_self_concept` appear **twice** in the LLM planner prompt. This wastes input tokens and degrades LLM performance.

### Root Cause

The duplication happens because capabilities are passed to the action dispatcher from TWO sources:

1. **`base.py:2053`**: `action_providers=list(self._capabilities.values())` — passes all capabilities to the policy constructor, stored as `self._action_providers`
2. **`base.py:2077`**: `self.action_policy.use_agent_capabilities([bp.key for bp in self.capability_blueprints])` — registers the **same** capabilities in `_used_agent_capabilities`
3. **`policies.py:1662`**: `action_providers=capability_providers + self._action_providers` — combines BOTH lists

The `ActionDispatcher` iterates over all providers and calls `_create_object_action_group()` for each, producing duplicate action groups with the same keys.

### Fix

**File**: `colony/python/colony/agents/patterns/actions/policies.py` (~line 1650-1665)

The simplest correct fix: `_create_action_dispatcher()` should NOT concatenate both sources. `capability_providers` (from `_used_agent_capabilities`) is the authoritative source — it's the list the user/agent explicitly opted into. `self._action_providers` contains the same capabilities passed at construction time.

Option A (preferred): In `_create_action_dispatcher()`, use only `capability_providers` and deduplicate against `self._action_providers`:
```python
async def _create_action_dispatcher(self) -> None:
    if self._action_dispatcher:
        return
    capability_providers = self.get_used_capabilities()
    # self._action_providers may overlap with capability_providers
    # (capabilities are passed both at construction and via use_agent_capabilities).
    # Deduplicate by identity to avoid duplicate action groups in the prompt.
    seen = set(id(p) for p in capability_providers)
    extra_providers = [p for p in self._action_providers if id(p) not in seen]
    self._action_dispatcher = ActionDispatcher(
        agent=self.agent,
        action_policy=self,
        action_map=self._action_map,
        action_providers=capability_providers + extra_providers,
    )
    await self._action_dispatcher.initialize()
```

Option B: Don't pass `action_providers` in the constructor at all — let `use_agent_capabilities()` be the only registration path. This requires changing `base.py` to not pass `action_providers=list(self._capabilities.values())`.

Option A is safer because it preserves the ability to pass non-capability action providers at construction time.

---

## Problem 2: LLM Selects Incomplete Action Keys

The LLM outputs `consciousness_get_self_concept` instead of `ConsciousnessCapability.b79b5858.consciousness_get_self_concept`. This causes "Unknown action type" errors.

### Current Prompt Format

```
## Available Actions

### Planning & Execution Control — manages the agent's plan lifecycle...
- ConsciousnessCapability.b79b5858.consciousness_update_self_concept: Update the agent's self-concept...
- ConsciousnessCapability.b79b5858.consciousness_get_self_concept: Get the agent's self-concept...
...
```

And the output format instruction:
```
"action_type": "<one of the available action types above>",
```

### Why the LLM Truncates Keys

1. The keys are long compound strings (`ClassName.hash.method_name`) with no visual delimiter making the full key obvious
2. The prompt says "action types" which the LLM interprets as the short method name
3. The flat markdown list format makes it easy to skip the prefix
4. No structural reinforcement that the FULL string (including dots) is the key

### Proposed Fix: XML-Structured Action Descriptions

**File**: `colony/python/colony/agents/patterns/planning/strategies.py` (~line 186-194)

Replace the flat markdown format with XML tags that unambiguously delimit action keys:

```python
@staticmethod
def _format_action_descriptions(planning_context: PlanningContext) -> str:
    """Format action descriptions from planning context into prompt text."""
    sections = []
    for group in planning_context.action_descriptions:
        sections.append(f'<action-group key="{group.group_key}">')
        if group.group_description:
            sections.append(f"  <description>{group.group_description}</description>")
        for action_key, action_desc in group.action_descriptions.items():
            sections.append(f'  <action key="{action_key}">{action_desc}</action>')
        sections.append("</action-group>")
    return "\n".join(sections)
```

This produces:
```xml
<action-group key="ConsciousnessCapability.b79b5858">
  <description>Agent self-awareness and metacognition</description>
  <action key="ConsciousnessCapability.b79b5858.consciousness_update_self_concept">Update the agent's self-concept...</action>
  <action key="ConsciousnessCapability.b79b5858.consciousness_get_self_concept">Get the agent's self-concept...</action>
</action-group>
```

And update the output format instruction to reference the XML attribute:
```
"action_type": "<the exact key attribute from an <action> element above>",
```

### Alternative Strategy: Prompt Formatting as a Parameter

The user asked whether the prompt formatting strategy should be parameterizable. This makes sense for experimenting with different formats:

**File**: `colony/python/colony/agents/patterns/planning/strategies.py`

Add a `PromptFormattingStrategy` protocol:

```python
class PromptFormattingStrategy(ABC):
    """Strategy for formatting action descriptions in the planner prompt.

    Different formatting strategies may affect LLM accuracy on action key
    selection. This abstraction allows experimentation and evaluation.
    """

    @abstractmethod
    def format_action_descriptions(self, action_descriptions: list[ActionGroupDescription]) -> str:
        """Format action group descriptions into prompt text."""
        ...

    @abstractmethod
    def format_action_type_instruction(self) -> str:
        """Return the instruction text for the action_type field in the output format."""
        ...


class MarkdownPromptFormatting(PromptFormattingStrategy):
    """Original markdown list formatting (legacy)."""

    def format_action_descriptions(self, action_descriptions):
        sections = []
        for group in action_descriptions:
            if group.group_description:
                sections.append(f"\n### {group.group_description}")
            for action_key, action_desc in group.action_descriptions.items():
                sections.append(f"- {action_key}: {action_desc}")
        return "\n".join(sections)

    def format_action_type_instruction(self):
        return '"action_type": "<one of the available action types above>"'


class XMLPromptFormatting(PromptFormattingStrategy):
    """XML-structured formatting for unambiguous action key delimitation."""

    def format_action_descriptions(self, action_descriptions):
        sections = []
        for group in action_descriptions:
            sections.append(f'<action-group key="{group.group_key}">')
            if group.group_description:
                sections.append(f"  <description>{group.group_description}</description>")
            for action_key, action_desc in group.action_descriptions.items():
                sections.append(f'  <action key="{action_key}">{action_desc}</action>')
            sections.append("</action-group>")
        return "\n".join(sections)

    def format_action_type_instruction(self):
        return '"action_type": "<the exact key= attribute from an <action> element above>"'
```

Then `PlanningStrategyPolicy` takes an optional `prompt_formatting: PromptFormattingStrategy` parameter, defaulting to `XMLPromptFormatting()`.

---

## Files Summary

| File | Change |
|------|--------|
| `colony/python/colony/agents/patterns/actions/policies.py` | Deduplicate action providers in `_create_action_dispatcher()` |
| `colony/python/colony/agents/patterns/planning/strategies.py` | Add `PromptFormattingStrategy` protocol + `XMLPromptFormatting` default; update `_format_action_descriptions` and output format instructions |

---

## Open Questions

1. Should we also add **fuzzy matching** as a fallback in `ActionDispatcher.dispatch()` when the exact key isn't found? (e.g., if the LLM outputs `consciousness_get_self_concept`, try matching the suffix against known keys). This would be a safety net, not a replacement for fixing the prompt.

2. Should the XML format use short aliases alongside full keys to reduce token cost? e.g.:
   ```xml
   <action key="ConsciousnessCapability.b79b5858.consciousness_get_self_concept" alias="consciousness_get_self_concept">...</action>
   ```
   Then accept either the full key or the alias in the plan output.


The answer for both questions is No for now — let's first see if the XML formatting alone resolves the issue before adding complexity.
