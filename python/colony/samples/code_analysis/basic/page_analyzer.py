"""PageAnalyzer: Analyzes a SINGLE page and produces compact summary.

This agent is bound to exactly ONE page and produces a compact summary
that can be used as a "key" for attention mechanism.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from overrides import override

from colony.agents.base import Agent, AgentCapability
from colony.agents.models import AgentSuspensionState
from colony.cluster.models import InferenceResponse
from colony.agents.patterns.scope import ScopeAwareResult
from colony.agents.patterns.actions.policies import action_executor


logger = logging.getLogger(__name__)


class PageAnalyzerCapability(AgentCapability):
    """Analyzes a SINGLE page and produces compact summary.

    Bound to exactly ONE page. Produces summary that's 1-2KB,
    not multi-MB page content. This summary serves as the "key"
    in the key-query-value attention mechanism.

    Design:
    - One PageAnalyzer per page
    - High parallelism, no page contention
    - Produces compact structured summary
    - Writes results to blackboard for ClusterAnalyzer

    Rationale:
    - PageAnalyzer is a leaf agent with a single, well-defined atomic task
    - Workflow is linear and deterministic: load page → analyze → produce summary → stop
    - No complex decision-making or multi-step planning required
    - Everything happens in one LLM call
    - Adding a reasoning loop would be over-engineering for this simple use case

    Comparison:
    - PageAnalyzer: Single atomic task → Simple FSM (this agent)
    - ClusterAnalyzerV2: Multi-page iterative refinement → Reasoning loop
    - Coordinator: Multi-agent orchestration → FSM (could use reasoning loop for advanced scenarios)

    Simple workflow:
    1. Analyze page → produce summary
    2. Write to blackboard
    3. Stop
    """

    async def initialize(self) -> None:
        """Initialize page analyzer."""
        await super().initialize()

        # This agent is bound to exactly ONE page
        if len(self.agent.bound_pages) != 1:
            raise ValueError(f"PageAnalyzer must be bound to exactly ONE page, got {len(self.agent.bound_pages)}")

        self.page_id = self.agent.bound_pages[0]
        self.summary: dict | None = None
        self.analysis_complete = False

        logger.info(f"PageAnalyzer initialized for page {self.page_id}")

    @action_executor()
    async def analyze_page(self) -> ScopeAwareResult[dict]:
        """Analyze the single bound page and produce compact summary."""
        # Load configuration from metadata
        from .config import PageAnalyzerConfig
        config_data = self.agent.metadata.parameters.get("config", {})
        config = PageAnalyzerConfig(**config_data)

        # Request page load
        await self.agent.request_page(self.page_id, priority=config.request_priority)

        # Wait for page to be loaded
        loaded = await self._wait_for_page(self.page_id, timeout_s=config.wait_timeout_seconds)
        if not loaded:
            raise RuntimeError(f"Page {self.page_id} failed to load after {config.wait_timeout_seconds}s")

        # Analyze with LLM - produces COMPACT summary
        response: InferenceResponse = await self.agent.infer(
            context_page_ids=[self.page_id],
            prompt=self._build_analysis_prompt(),
            max_tokens=config.max_tokens_summary
        )

        # Parse summary
        self.summary = self._parse_summary(response.generated_text)

        # Validate summary is compact
        summary_size = len(json.dumps(self.summary))
        if summary_size > config.summary_size_limit:
            logger.warning(
                f"Summary for {self.page_id} is {summary_size} bytes, exceeds 4KB limit. "
                "Truncating..."
            )
            self.summary = self._truncate_summary(self.summary)

        # Write to blackboard for ClusterAnalyzer
        await self._write_summary_to_blackboard()

        logger.info(f"Page analysis complete for {self.page_id}")

        # Return result (decorator will wrap in ActionResult)
        return {"page_id": self.page_id, "summary_size": len(json.dumps(self.summary))}

    async def _wait_for_page(self, page_id: str, timeout_s: float) -> bool:
        """Wait for page to be loaded."""
        start = time.time()
        while time.time() - start < timeout_s:
            if await self.agent.is_page_loaded(page_id):
                return True
            await asyncio.sleep(0.5)
        return False

    def _build_analysis_prompt(self) -> str:
        """Build prompt for page analysis.

        CRITICAL: Prompt must emphasize producing COMPACT summary.
        """
        return """Analyze this code shard and produce a COMPACT SUMMARY (not full content).

Extract:
1. **Definitions**: Classes, functions, constants defined HERE
2. **Dependencies**: External references (imports, calls to external code)
3. **Exports**: What this shard provides to other code
4. **Issues**: Any obvious bugs, vulnerabilities, or code smells
5. **Complexity**: Rough complexity assessment (simple/medium/complex)

**CRITICAL**: Your response must be COMPACT (under 2KB).
Do NOT reproduce code. ONLY provide summary information.

**Output Format** (JSON):
{
    "page_id": "...",
    "definitions": {
        "classes": [{"name": "Foo", "methods": ["bar", "baz"], "purpose": "Brief description"}],
        "functions": [{"name": "compute", "signature": "compute(x, y)", "purpose": "Brief description"}],
        "constants": ["MAX_SIZE", "DEFAULT_TIMEOUT"]
    },
    "dependencies": {
        "imports": ["module.Foo", "package.util"],
        "external_calls": ["some_function", "SomeClass.method"]
    },
    "exports": ["Foo", "compute", "MAX_SIZE"],
    "issues": [
        {"type": "security", "severity": "high", "description": "SQL injection risk"},
        {"type": "performance", "severity": "medium", "description": "O(n^2) algorithm"}
    ],
    "complexity": "medium",
    "summary": "One-sentence summary of what this code does"
}

Be concise. Focus on facts, not speculation."""

    def _parse_summary(self, text: str) -> dict:
        """Parse LLM response into summary dict."""
        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)

        try:
            summary = json.loads(text)
            # Ensure page_id is set
            summary["page_id"] = self.page_id
            return summary
        except json.JSONDecodeError as e:
            # Fallback: return minimal summary
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return {
                "page_id": self.page_id,
                "error": f"Failed to parse LLM response: {e}",
                "raw_text": text[:1000],  # First 1KB
                "summary": "Parse error - manual review needed"
            }

    def _truncate_summary(self, summary: dict) -> dict:
        """Truncate summary to fit 4KB limit."""
        # Remove raw_text if present
        if "raw_text" in summary:
            del summary["raw_text"]

        # Truncate arrays
        if "definitions" in summary:
            if "classes" in summary["definitions"]:
                summary["definitions"]["classes"] = summary["definitions"]["classes"][:5]
            if "functions" in summary["definitions"]:
                summary["definitions"]["functions"] = summary["definitions"]["functions"][:10]

        if "issues" in summary:
            summary["issues"] = summary["issues"][:5]

        return summary

    async def _write_summary_to_blackboard(self) -> None:
        """Write summary to blackboard for cluster analyzer.

        Uses manager delegation to access blackboard.
        """
        parent_id = self.agent.metadata.parent_agent_id
        if not parent_id:
            logger.warning("No parent_id in metadata, cannot write to blackboard")
            return

        # Get shared blackboard with parent
        blackboard = await self.agent.get_blackboard(scope="shared", scope_id=parent_id)

        # Write page summary to blackboard
        await blackboard.write(f"page_summary:{self.page_id}", self.summary)

        logger.info(f"Wrote summary for {self.page_id} to blackboard (parent: {parent_id})")

    @override
    async def serialize_suspension_state(self, state: AgentSuspensionState) -> AgentSuspensionState:
        """Serialize PageAnalyzer-specific state.

        Overrides base implementation to add PageAnalyzer-specific state.
        Calls super() first to get base state, then adds subclass state.

        Returns:
            AgentSuspensionState with all agent state serialized
        """
        # Get base state from parent
        state = await super().serialize_suspension_state(state)

        # Store PageAnalyzer-specific state in custom_data with special key
        state.custom_data["_page_analyzer_state"] = {
            "page_id": self.page_id,
            "summary": self.summary,
            "analysis_complete": self.analysis_complete,
        }

        return state

    @override
    async def deserialize_suspension_state(
        self,
        state: AgentSuspensionState
    ) -> None:
        """Restore PageAnalyzer-specific state from suspension.

        Overrides base implementation to restore PageAnalyzer-specific state.
        Calls super() first to restore base state, then restores subclass state.

        Args:
            state: AgentSuspensionState to restore from
        """
        # Restore base state first
        await super().deserialize_suspension_state(state)

        # Restore PageAnalyzer-specific state
        custom_state = state.custom_data.get("_page_analyzer_state", {})
        if custom_state:
            self.page_id = custom_state.get("page_id", self.agent.bound_pages[0] if self.agent.bound_pages else None)
            self.summary = custom_state.get("summary")
            self.analysis_complete = custom_state.get("analysis_complete", False)

            logger.info(
                f"Restored PageAnalyzer state: page_id={self.page_id}, "
                f"analysis_complete={self.analysis_complete}"
            )



class PageAnalyzer(Agent):
    """Analyzes a SINGLE page and produces compact summary.

    Bound to exactly ONE page. Produces summary that's 1-2KB,
    not multi-MB page content. This summary serves as the "key"
    in the key-query-value attention mechanism.

    Design:
    - One PageAnalyzer per page
    - High parallelism, no page contention
    - Produces compact structured summary
    - Writes results to blackboard for ClusterAnalyzer
    """

    async def initialize(self) -> None:
        """Initialize page analyzer."""
        self.add_capability_classes([
            PageAnalyzerCapability
        ])
        await super().initialize()


