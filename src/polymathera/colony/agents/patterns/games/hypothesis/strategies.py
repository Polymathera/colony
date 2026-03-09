"""Hypothesis formation, evidence gathering, and evaluation strategies.

This module defines the pluggable strategy interfaces and implementations:
- HypothesisFormationStrategy: Generate hypotheses from context
- EvidenceGatheringStrategy: Find evidence for/against hypotheses
- HypothesisEvaluationStrategy: Evaluate hypotheses against evidence

Strategies can be swapped dynamically during game execution.
"""

from __future__ import annotations

import json
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from overrides import override

from ...models import Hypothesis, QueryAnswer
from .types import (
    HypothesisContext,
    HypothesisFormationTrigger,
    Evidence,
    EvidenceType,
    EvaluationResult,
    EvaluationDecision,
    TriggerType,
)
from ...attention.attention import PageQuery
from ...capabilities.grounding import GroundingRequest

if TYPE_CHECKING:
    from ....base import Agent
    from .capabilities import ChallengeRecord
    from ...capabilities.grounding import GroundingCapability
    from ...attention.incremental import IncrementalQueryCapability


logger = logging.getLogger(__name__)


# ============================================================================
# Hypothesis Formation Strategy
# ============================================================================


class HypothesisFormationStrategy(ABC):
    """Strategy for forming hypotheses from context.

    Implementations:
    - LLMHypothesisFormation: Use LLM to generate hypotheses
    - RuleBasedHypothesisFormation: Apply domain rules
    - TemplateHypothesisFormation: Fill in templates
    - CompositeHypothesisFormation: Combine strategies
    """

    def __init__(self, agent: Agent):
        self.agent = agent

    @abstractmethod
    async def form_hypotheses(
        self,
        context: HypothesisContext,
        max_hypotheses: int = 5,
    ) -> list[Hypothesis]:
        """Form hypotheses from context.

        Args:
            context: Formation context with observations and constraints
            max_hypotheses: Maximum number to generate

        Returns:
            List of hypotheses ordered by priority/confidence
        """
        ...

    @abstractmethod
    def should_form_hypothesis(
        self,
        trigger: HypothesisFormationTrigger,
    ) -> bool:
        """Determine if trigger warrants new hypothesis formation.

        Args:
            trigger: The triggering event

        Returns:
            True if should form new hypotheses
        """
        ...

    @abstractmethod
    async def rank_hypotheses(
        self,
        hypotheses: list[Hypothesis],
        context: HypothesisContext,
    ) -> list[Hypothesis]:
        """Rank hypotheses by priority for validation.

        Args:
            hypotheses: Hypotheses to rank
            context: Context for ranking decisions

        Returns:
            Hypotheses ordered by priority (highest first)
        """
        ...


class LLMHypothesisFormation(HypothesisFormationStrategy):
    """Form hypotheses using LLM inference.

    Transferred from HypothesisDrivenExplorer.form_hypothesis().
    """

    def __init__(
        self,
        agent: Agent,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ):
        """Initialize LLM-based formation.

        Args:
            agent: Agent for LLM inference
            temperature: Sampling temperature
            max_tokens: Max tokens per response
        """
        super().__init__(agent)
        self.temperature = temperature
        self.max_tokens = max_tokens

    @override
    async def form_hypotheses(
        self,
        context: HypothesisContext,
        max_hypotheses: int = 5,
    ) -> list[Hypothesis]:
        """Form hypotheses using LLM."""

        prompt = self._build_formation_prompt(context, max_hypotheses)

        # Get structured output from LLM
        response = await self.agent.infer(
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Parse response into hypotheses
        hypotheses = self._parse_hypotheses(response.generated_text, context)

        return hypotheses[:max_hypotheses]

    @override
    def should_form_hypothesis(
        self,
        trigger: HypothesisFormationTrigger,
    ) -> bool:
        """Determine if trigger warrants formation."""
        # Always form on explicit request
        if trigger.trigger_type == TriggerType.EXPLICIT:
            return True

        # Form on new observations with sufficient confidence
        if trigger.trigger_type == TriggerType.OBSERVATION:
            confidence = trigger.data.get("confidence", 0.5)
            return confidence > 0.4

        # Form when analysis completes
        if trigger.trigger_type == TriggerType.ANALYSIS_COMPLETE:
            return True

        # Form on contradiction detection
        if trigger.trigger_type == TriggerType.CONTRADICTION:
            return True

        # Form on game outcome if rejected and retriable
        if trigger.trigger_type == TriggerType.GAME_OUTCOME:
            outcome = trigger.data.get("outcome", {})
            return not outcome.get("success", True) and outcome.get("retriable", False)

        return False

    @override
    async def rank_hypotheses(
        self,
        hypotheses: list[Hypothesis],
        context: HypothesisContext,
    ) -> list[Hypothesis]:
        """Rank by confidence and relevance to investigation goal."""
        # Simple ranking by confidence
        # Could use LLM for more sophisticated ranking
        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

    def _build_formation_prompt(
        self,
        context: HypothesisContext,
        max_hypotheses: int,
    ) -> str:
        """Build prompt for hypothesis formation."""
        observations_text = "\n".join(
            f"- [{obs.observation_type.value}] {obs.description} (confidence: {obs.confidence:.2f})"
            for obs in context.observations
        )

        prior_text = ""
        if context.prior_hypotheses:
            prior_text = "\nPrior hypotheses (avoid duplicates):\n" + "\n".join(
                f"- {h.statement}" for h in context.prior_hypotheses
            )

        constraints_text = ""
        if context.constraints:
            constraints_text = "\nConstraints:\n" + "\n".join(
                f"- {c}" for c in context.constraints
            )

        known_facts_text = ""
        if context.known_facts:
            known_facts_text = "\nKnown facts:\n" + "\n".join(
                f"- {f}" for f in context.known_facts
            )

        goal_text = ""
        if context.investigation_goal:
            goal_text = f"\nInvestigation goal: {context.investigation_goal}"

        return f"""Form testable hypotheses based on the following observations.

Domain: {context.domain.value}
Subject: {context.subject.subject_type.value} - {context.subject.subject_id}
{goal_text}

Observations:
{observations_text}
{prior_text}
{known_facts_text}
{constraints_text}

Generate up to {max_hypotheses} hypotheses that are:
1. Testable (can be verified through evidence)
2. Specific (not too vague)
3. Actionable (guides further analysis)
4. Non-redundant (different from prior hypotheses)

For each hypothesis, provide:
- statement: Clear hypothesis statement
- test_queries: List of queries to test the hypothesis
- supporting_evidence: Initial evidence (from observations)
- confidence: Initial confidence (0.0-1.0)

Format as JSON list of hypothesis objects."""

    def _parse_hypotheses(
        self,
        response_text: str,
        context: HypothesisContext,
    ) -> list[Hypothesis]:
        """Parse LLM response into Hypothesis objects."""

        hypotheses = []

        try:
            # Try to extract JSON from response
            # Handle markdown code blocks
            text = response_text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())

            if isinstance(data, list):
                for item in data:
                    hyp = Hypothesis(
                        hypothesis_id=f"hyp_{uuid.uuid4().hex[:8]}",
                        statement=item.get("statement", ""),
                        test_queries=item.get("test_queries", []),
                        supporting_evidence=item.get("supporting_evidence", []),
                        confidence=item.get("confidence", 0.5),
                        status="pending",
                    )
                    hypotheses.append(hyp)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse hypothesis response: {e}")
            # Create a single hypothesis from the response
            hypotheses.append(Hypothesis(
                hypothesis_id=f"hyp_{uuid.uuid4().hex[:8]}",
                statement=response_text[:500],
                test_queries=[],
                supporting_evidence=[],
                confidence=0.3,
                status="pending",
            ))

        return hypotheses


class RuleBasedHypothesisFormation(HypothesisFormationStrategy):
    """Form hypotheses using domain-specific rules.

    Useful for well-understood domains where patterns are known.
    """

    def __init__(
        self,
        agent: Agent,
        rules: dict[str, list[dict[str, Any]]] | None = None,
    ):
        """Initialize with domain rules.

        Args:
            rules: Dict of domain -> list of rule dicts
                   Each rule: {observation_type, hypothesis_template, confidence}
        """
        super().__init__(agent)
        self.rules = rules or {}

    @override
    async def form_hypotheses(
        self,
        context: HypothesisContext,
        max_hypotheses: int = 5,
    ) -> list[Hypothesis]:
        """Form hypotheses by applying rules to observations."""

        domain_rules = self.rules.get(context.domain.value, [])
        hypotheses = []

        for obs in context.observations:
            for rule in domain_rules:
                if rule.get("observation_type") == obs.observation_type.value:
                    template = rule.get("hypothesis_template", "")
                    statement = template.format(
                        description=obs.description,
                        source=obs.source or "unknown",
                        subject=context.subject.subject_id,
                    )

                    hyp = Hypothesis(
                        hypothesis_id=f"hyp_{uuid.uuid4().hex[:8]}",
                        statement=statement,
                        test_queries=rule.get("test_queries", []),
                        supporting_evidence=[str(e) for e in obs.evidence],
                        confidence=rule.get("confidence", 0.5) * obs.confidence,
                        status="pending",
                    )
                    hypotheses.append(hyp)

                    if len(hypotheses) >= max_hypotheses:
                        return hypotheses

        return hypotheses

    @override
    def should_form_hypothesis(
        self,
        trigger: HypothesisFormationTrigger,
    ) -> bool:
        """Form only on observation or explicit triggers."""
        return trigger.trigger_type in (
            TriggerType.OBSERVATION,
            TriggerType.EXPLICIT,
            TriggerType.ANALYSIS_COMPLETE,
        )

    @override
    async def rank_hypotheses(
        self,
        hypotheses: list[Hypothesis],
        context: HypothesisContext,
    ) -> list[Hypothesis]:
        """Rank by confidence."""
        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)


# ============================================================================
# Evidence Gathering Strategy
# ============================================================================


class EvidenceGatheringStrategy(ABC):
    """Strategy for gathering evidence to support/refute hypotheses.

    This is a mechanism-agnostic interface. Implementations may gather evidence
    through any means: page-graph queries, code execution, LLM reasoning,
    external API calls, experiment execution, mathematical proofs, etc.

    Each implementation:
    - Looks up needed capabilities via ``self.agent.get_capability_by_type()``
    - Declares what it can handle via ``can_gather()``
    - Interprets ``Hypothesis.test_queries`` according to its mechanism
    - Returns classified Evidence items

    Implementations:
    - QueryBasedEvidence: Page-graph queries via IncrementalQueryCapability
    - LLMReasoningEvidence: LLM analysis of existing agent context
    - CompositeEvidenceStrategy: Combines multiple strategies
    """

    def __init__(self, agent: Agent):
        self.agent = agent

    @abstractmethod
    async def gather_evidence(
        self,
        hypothesis: Hypothesis,
        context: HypothesisContext,
        challenges: list[ChallengeRecord] | None = None,
    ) -> list[Evidence]:
        """Gather evidence for hypothesis.

        Implementations should:
        1. Look up required capabilities via ``self.agent.get_capability_by_type()``
        2. Interpret ``hypothesis.test_queries`` according to their mechanism
        3. Classify results as supporting/contradicting/neutral Evidence

        Args:
            hypothesis: Hypothesis with test_queries to process
            context: Context with subject and scope
            challenges: Specific challenges to address

        Returns:
            List of evidence items classified as supporting/contradicting/neutral
        """
        ...

    def can_gather(
        self,
        hypothesis: Hypothesis,
        context: HypothesisContext,
    ) -> bool:
        """Check if this strategy can gather evidence for the given hypothesis.

        Default returns True (strategy can always attempt). Override to restrict
        based on available capabilities, hypothesis properties, or domain.

        Examples:
        - QueryBasedEvidence returns True only if IncrementalQueryCapability is on agent
        - Code execution strategies return True only if execution tool is available
        - LLMReasoningEvidence always returns True (LLM is always available)

        Args:
            hypothesis: The hypothesis to gather evidence for
            context: The hypothesis context

        Returns:
            True if this strategy can attempt evidence gathering
        """
        return True


class QueryBasedEvidence(EvidenceGatheringStrategy):
    """Gather evidence by querying the page graph via IncrementalQueryCapability.

    Interprets ``Hypothesis.test_queries`` as page-graph search queries, processes
    them through ``IncrementalQueryCapability``, and optionally uses
    ``GroundingCapability`` to generate additional targeted queries for open
    challenges.

    Required capability on agent: ``IncrementalQueryCapability``
    Optional capability on agent: ``GroundingCapability``

    Integrates with existing capabilities:
    - GroundingCapability for claim validation
    - IncrementalQueryCapability for iterative search
    
    Data flow:
        Hypothesis.test_queries: list[str]
            │
            ├── Convert each query string → PageQuery
            │       │
            │       └── IncrementalQueryCapability.get_answer(query, pages) → QueryAnswer
            │               │
            │               └── Classify QueryAnswer → Evidence (supporting/contradicting)
            │
            └── (For challenges) GroundingCapability.generate_grounding_query(claim) → PageQuery
                    │
                    └── IncrementalQueryCapability.get_answer(query, pages) → QueryAnswer
                            │
                            └── Classify QueryAnswer → Evidence

    The GroundingAgent (meta_agents/grounding.py) illustrates the correct pattern:
    it has BOTH GroundingCapability and IncrementalQueryCapability. The action
    policy orchestrates: grounding generates queries → incremental query processes them.

    Flow:
        1. Hypothesis.test_queries → PageQuery objects
        2. Each PageQuery → IncrementalQueryCapability.get_answer() → QueryAnswer
        3. QueryAnswer → classified as Evidence (supporting/contradicting based on content)
        4. For challenges: GroundingCapability generates additional targeted queries
        5. Those queries also go through IncrementalQueryCapability
    """

    def __init__(self, agent: Agent):
        """Initialize with agent only; capabilities are looked up dynamically.

        Args:
            agent: Agent that must have IncrementalQueryCapability attached
        """
        super().__init__(agent)

    def _get_query_capability(self) -> IncrementalQueryCapability:
        """Look up IncrementalQueryCapability on the agent.

        Raises:
            RuntimeError: If IncrementalQueryCapability is not attached to agent
        """
        from ...attention.incremental import IncrementalQueryCapability

        cap = self.agent.get_capability_by_type(IncrementalQueryCapability)
        if cap is None:
            raise RuntimeError(
                f"QueryBasedEvidence requires IncrementalQueryCapability on agent "
                f"{self.agent.agent_id}, but none was found. Attach the capability "
                f"before using this strategy."
            )
        return cap

    def _get_grounding_capability(self) -> GroundingCapability | None:
        """Look up optional GroundingCapability on the agent."""
        from ...capabilities.grounding import GroundingCapability

        return self.agent.get_capability_by_type(GroundingCapability)

    @override
    def can_gather(
        self,
        hypothesis: Hypothesis,
        context: HypothesisContext,
    ) -> bool:
        """Returns True only if IncrementalQueryCapability is attached to agent."""
        from ...attention.incremental import IncrementalQueryCapability

        return self.agent.get_capability_by_type(IncrementalQueryCapability) is not None

    @override
    async def gather_evidence(
        self,
        hypothesis: Hypothesis,
        context: HypothesisContext,
        challenges: list[ChallengeRecord] | None = None,
    ) -> list[Evidence]:
        """Process hypothesis test_queries through IncrementalQueryCapability."""
        evidence: list[Evidence] = []
        initial_pages = context.subject.related_subjects

        # Step 1: Process hypothesis.test_queries
        # These are the queries the formation strategy specifically designed to test this hypothesis.
        for query_text in hypothesis.test_queries:
            page_query = PageQuery(
                query_text=query_text,
                metadata={"hypothesis_id": hypothesis.hypothesis_id},
            )

            query_answer = await self._get_query_capability().get_answer(
                query=page_query,
                pages=initial_pages,
            )

            # Classify the answer as evidence
            classified = await self._classify_answer(
                query_answer=query_answer,
                hypothesis=hypothesis,
                query_text=query_text,
            )
            evidence.extend(classified)

        # Step 2: For open challenges, generate additional targeted queries
        if challenges:
            for challenge in challenges:
                if challenge.status == "open":
                    challenge_evidence = await self._gather_for_challenge(
                        hypothesis, challenge, context
                    )
                    evidence.extend(challenge_evidence)

        return evidence

    async def _gather_for_challenge(
        self,
        hypothesis: Hypothesis,
        challenge: ChallengeRecord,
        context: HypothesisContext,
    ) -> list[Evidence]:
        """Gather evidence specifically for a challenge."""
        initial_pages = context.subject.related_subjects

        challenge_queries = await self._generate_challenge_queries(
            hypothesis, challenge, context
        )
        evidence: list[Evidence] = []
        for page_query in challenge_queries:
            query_answer = await self._get_query_capability().get_answer(
                query=page_query,
                pages=initial_pages,
            )
            classified = await self._classify_answer(
                query_answer=query_answer,
                hypothesis=hypothesis,
                query_text=page_query.query_text,
                challenge=challenge,
            )
            evidence.extend(classified)
        return evidence

    async def _classify_answer(
        self,
        query_answer: QueryAnswer,
        hypothesis: Hypothesis,
        query_text: str,
        challenge: ChallengeRecord | None = None,
    ) -> list[Evidence]:
        """Classify a QueryAnswer as supporting, contradicting, or neutral evidence.

        Uses the agent's LLM to classify whether the answer supports or
        contradicts the hypothesis.

        Args:
            query_answer: Answer from IncrementalQueryCapability.get_answer()
            hypothesis: The hypothesis being tested
            query_text: The query that produced this answer
            challenge: The challenge this evidence addresses (if any)

        Returns:
            List of Evidence items (usually 1, but may split complex answers)
        """
        # Extract answer data - handle both dict and QueryAnswer model
        if isinstance(query_answer, dict):
            answer_data = query_answer.get("answer", {})
            if isinstance(answer_data, dict):
                answer_text = answer_data.get("answer", str(answer_data))
                confidence = answer_data.get("confidence", 0.5)
            else:
                answer_text = str(answer_data)
                confidence = 0.5
        else:
            answer_text = str(query_answer.answer) if hasattr(query_answer, 'answer') else str(query_answer)
            confidence = getattr(query_answer, 'confidence', 0.5)

        if not answer_text or answer_text == "Placeholder - implement with answer_generator":
            return []

        # Use LLM to classify the answer as supporting/contradicting/neutral
        classification = await self._llm_classify(
            hypothesis_statement=hypothesis.statement,
            answer_text=answer_text,
            query_text=query_text,
        )

        evidence_type = EvidenceType(classification.get("type", "neutral"))
        description = classification.get("description", answer_text[:200])
        source = f"query: {query_text[:80]}"

        if challenge:
            description = f"[Challenge: {challenge.challenged_aspect}] {description}"
            source = f"challenge_query: {query_text[:80]}"

        return [Evidence(
            evidence_id=f"ev_{uuid.uuid4().hex[:8]}",
            evidence_type=evidence_type,
            description=description,
            source=source,
            confidence=confidence * classification.get("confidence", 0.7),
            raw_data={
                "query": query_text,
                "answer": answer_text[:500],
                "challenge_id": challenge.challenge_id if challenge else None,
            },
        )]

    async def _llm_classify(
        self,
        hypothesis_statement: str,
        answer_text: str,
        query_text: str,
    ) -> dict[str, Any]:
        """Use LLM to classify an answer as evidence for/against hypothesis."""

        prompt = f"""Classify whether this answer supports or contradicts the hypothesis.

Hypothesis: {hypothesis_statement}
Query: {query_text}
Answer: {answer_text[:500]}

Respond with JSON:
{{"type": "supporting" | "contradicting" | "neutral", "confidence": 0.0-1.0, "description": "brief summary"}}"""

        try:
            response = await self.agent.infer(
                prompt=prompt,
                temperature=0.1,
                max_tokens=200,
            )

            text = response.generated_text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            return json.loads(text.strip())
        except Exception as e:
            logger.warning(f"Failed to classify evidence: {e}")
            return {"type": "neutral", "confidence": 0.5, "description": answer_text[:200]}

    async def _generate_challenge_queries(
        self,
        hypothesis: Hypothesis,
        challenge: ChallengeRecord,
        context: HypothesisContext,
    ) -> list[PageQuery]:
        """Generate targeted queries for a specific challenge.

        Uses GroundingCapability if available, otherwise generates from challenge text.
        """
        queries = []

        grounding_cap = self._get_grounding_capability()
        if grounding_cap:
            # Use GroundingCapability to generate a targeted grounding query
            request = GroundingRequest(
                claim_id=challenge.challenge_id,
                claim=f"Regarding '{challenge.challenged_aspect}': {hypothesis.statement}",
                initial_pages=context.subject.related_subjects,
                context={"challenge": challenge.model_dump()},
                evidence_provided=[],
                requesting_agent_id=self.agent.agent_id,
            )
            result = await grounding_cap.generate_grounding_query(request=request)
            if result and isinstance(result, dict) and "query" in result:
                query = result["query"]
                if isinstance(query, PageQuery):
                    queries.append(query)
                else:
                    queries.append(PageQuery(
                        query_text=str(query),
                        metadata={"challenge_id": challenge.challenge_id},
                    ))
        else:
            # Generate query directly from challenge text
            queries.append(PageQuery(
                query_text=f"Find evidence about: {challenge.challenged_aspect}. Context: {challenge.reason}",
                metadata={
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "challenge_id": challenge.challenge_id,
                },
            ))

        return queries


class LLMReasoningEvidence(EvidenceGatheringStrategy):
    """Gather evidence using LLM reasoning over the agent's loaded context.

    Interprets ``Hypothesis.test_queries`` as reasoning prompts: the LLM considers
    each query against its current context (bound pages, conversation history) and
    identifies supporting or contradicting evidence from what it already knows.

    This strategy does NOT perform retrieval or external calls. It is always
    applicable (no special capabilities required) and complements retrieval-based
    strategies like ``QueryBasedEvidence``.
    """

    def __init__(
        self,
        agent: Agent,
        temperature: float = 0.2,
    ):
        super().__init__(agent)
        self.temperature = temperature

    @override
    async def gather_evidence(
        self,
        hypothesis: Hypothesis,
        context: HypothesisContext,
        challenges: list[ChallengeRecord] | None = None,
    ) -> list[Evidence]:
        """Use LLM to analyze agent's current context and identify evidence."""

        prompt = self._build_evidence_prompt(hypothesis, context, challenges)

        response = await self.agent.infer(
            prompt=prompt,
            temperature=self.temperature,
        )

        evidence = []
        try:
            text = response.generated_text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())
            if isinstance(data, list):
                for item in data:
                    ev_type_str = item.get("type", "neutral")
                    try:
                        ev_type = EvidenceType(ev_type_str)
                    except ValueError:
                        ev_type = EvidenceType.NEUTRAL
                    evidence.append(Evidence(
                        evidence_id=f"ev_{uuid.uuid4().hex[:8]}",
                        evidence_type=ev_type,
                        description=item.get("description", ""),
                        source=item.get("source", "llm_analysis"),
                        confidence=item.get("confidence", 0.5),
                        raw_data=item.get("raw_data", {}),
                    ))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse evidence response: {e}")

        return evidence

    def _build_evidence_prompt(
        self,
        hypothesis: Hypothesis,
        context: HypothesisContext,
        challenges: list[ChallengeRecord] | None,
    ) -> str:
        """Build prompt for evidence gathering."""
        challenges_text = ""
        if challenges:
            open_challenges = [c for c in challenges if c.status == "open"]
            if open_challenges:
                challenges_text = "\nOpen challenges to address:\n" + "\n".join(
                    f"- {c.challenged_aspect}: {c.reason}"
                    for c in open_challenges
                )

        test_queries_text = ""
        if hypothesis.test_queries:
            test_queries_text = "\nTest queries to consider:\n" + "\n".join(
                f"- {q}" for q in hypothesis.test_queries
            )

        return f"""Analyze the following context and identify evidence for or against the hypothesis.

Hypothesis: {hypothesis.statement}

Subject: {context.subject.subject_type.value} - {context.subject.subject_id}
Domain: {context.domain.value}

Existing supporting evidence: {hypothesis.supporting_evidence}
{test_queries_text}
{challenges_text}

Identify evidence that:
1. Supports the hypothesis
2. Contradicts the hypothesis
3. Is relevant but inconclusive

For each piece of evidence, provide:
- type: "supporting", "contradicting", or "neutral"
- description: What the evidence shows
- source: Where it was found
- confidence: How confident you are (0.0-1.0)

Format as JSON list of evidence objects."""


# ============================================================================
# Hypothesis Evaluation Strategy
# ============================================================================


class HypothesisEvaluationStrategy(ABC):
    """Strategy for evaluating hypotheses against evidence."""

    def __init__(self, agent: Agent):
        self.agent = agent

    @abstractmethod
    async def evaluate(
        self,
        hypothesis: Hypothesis,
        evidence: list[Evidence],
        challenges: list[ChallengeRecord] | None = None,
    ) -> EvaluationResult:
        """Evaluate hypothesis against evidence.

        Args:
            hypothesis: Hypothesis to evaluate
            evidence: Gathered evidence
            challenges: Outstanding challenges

        Returns:
            Evaluation result with recommendation
        """
        ...


class LLMEvaluation(HypothesisEvaluationStrategy):
    """Evaluate hypotheses using LLM reasoning."""

    def __init__(
        self,
        agent: Agent,
        temperature: float = 0.2,
    ):
        super().__init__(agent)
        self.temperature = temperature

    @override
    async def evaluate(
        self,
        hypothesis: Hypothesis,
        evidence: list[Evidence],
        challenges: list[ChallengeRecord] | None = None,
    ) -> EvaluationResult:
        """Use LLM to evaluate hypothesis against evidence."""

        prompt = self._build_evaluation_prompt(hypothesis, evidence, challenges)

        response = await self.agent.infer(
            prompt=prompt,
            temperature=self.temperature,
        )

        # Parse response
        try:
            text = response.generated_text
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            data = json.loads(text.strip())

            decision_str = data.get("decision", "need_more_evidence")
            decision = EvaluationDecision(decision_str) if decision_str in EvaluationDecision.__members__.values() else EvaluationDecision.NEED_MORE_EVIDENCE

            return EvaluationResult(
                decision=decision,
                confidence=data.get("confidence", 0.5),
                reasoning=data.get("reasoning", ""),
                supporting_evidence_count=len([e for e in evidence if e.evidence_type == EvidenceType.SUPPORTING]),
                contradicting_evidence_count=len([e for e in evidence if e.evidence_type == EvidenceType.CONTRADICTING]),
                unresolved_challenges=len([c for c in (challenges or []) if c.status == "open"]),
                suggestions=data.get("suggestions", []),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse evaluation response: {e}")
            return self._fallback_evaluation(hypothesis, evidence, challenges)

    def _build_evaluation_prompt(
        self,
        hypothesis: Hypothesis,
        evidence: list[Evidence],
        challenges: list[ChallengeRecord] | None,
    ) -> str:
        """Build prompt for evaluation."""
        supporting = [e for e in evidence if e.evidence_type == EvidenceType.SUPPORTING]
        contradicting = [e for e in evidence if e.evidence_type == EvidenceType.CONTRADICTING]

        supporting_text = "\n".join(
            f"- {e.description} (confidence: {e.confidence:.2f})"
            for e in supporting
        ) or "None"

        contradicting_text = "\n".join(
            f"- {e.description} (confidence: {e.confidence:.2f})"
            for e in contradicting
        ) or "None"

        challenges_text = ""
        if challenges:
            open_challenges = [c for c in challenges if c.status == "open"]
            if open_challenges:
                challenges_text = "\nUnresolved challenges:\n" + "\n".join(
                    f"- {c.challenged_aspect}: {c.reason}"
                    for c in open_challenges
                )

        return f"""Evaluate whether the following hypothesis should be accepted or rejected.

Hypothesis: {hypothesis.statement}

Supporting evidence:
{supporting_text}

Contradicting evidence:
{contradicting_text}
{challenges_text}

Provide your evaluation as JSON with:
- decision: "accept", "reject", "revise", or "need_more_evidence"
- confidence: How confident you are (0.0-1.0)
- reasoning: Explanation of your decision, is evidence sufficient, etc.
- suggestions: List of suggestions or additional evidence for improvement (if reject/revise)
"""

    def _fallback_evaluation(
        self,
        hypothesis: Hypothesis,
        evidence: list[Evidence],
        challenges: list[ChallengeRecord] | None,
    ) -> EvaluationResult:
        """Fallback heuristic evaluation."""
        supporting = len([e for e in evidence if e.evidence_type == EvidenceType.SUPPORTING])
        contradicting = len([e for e in evidence if e.evidence_type == EvidenceType.CONTRADICTING])
        unresolved = len([c for c in (challenges or []) if c.status == "open"])

        if contradicting > supporting:
            decision = EvaluationDecision.REJECT
            confidence = 0.6
        elif supporting >= 2 and unresolved == 0:
            decision = EvaluationDecision.ACCEPT
            confidence = 0.7
        elif unresolved > 0:
            decision = EvaluationDecision.NEED_MORE_EVIDENCE
            confidence = 0.5
        else:
            decision = EvaluationDecision.NEED_MORE_EVIDENCE
            confidence = 0.4

        return EvaluationResult(
            decision=decision,
            confidence=confidence,
            reasoning="Heuristic evaluation based on evidence counts",
            supporting_evidence_count=supporting,
            contradicting_evidence_count=contradicting,
            unresolved_challenges=unresolved,
        )


class RuleBasedEvaluation(HypothesisEvaluationStrategy):
    """Evaluate hypotheses using configurable rules."""

    def __init__(
        self,
        agent: Agent,
        accept_threshold: int = 2,
        reject_threshold: int = 1,
        max_unresolved_challenges: int = 1,
    ):
        """Initialize with thresholds.

        Args:
            accept_threshold: Min supporting evidence to accept
            reject_threshold: Max contradicting evidence before reject
            max_unresolved_challenges: Max unresolved challenges to accept
        """
        super().__init__(agent)
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self.max_unresolved_challenges = max_unresolved_challenges

    @override
    async def evaluate(
        self,
        hypothesis: Hypothesis,
        evidence: list[Evidence],
        challenges: list[ChallengeRecord] | None = None,
    ) -> EvaluationResult:
        """Apply rules to determine outcome."""
        supporting = len([e for e in evidence if e.evidence_type == EvidenceType.SUPPORTING])
        contradicting = len([e for e in evidence if e.evidence_type == EvidenceType.CONTRADICTING])
        unresolved = len([c for c in (challenges or []) if c.status == "open"])

        suggestions = []

        # Check rejection criteria first
        if contradicting >= self.reject_threshold:
            return EvaluationResult(
                decision=EvaluationDecision.REJECT,
                confidence=0.8,
                reasoning=f"Contradicting evidence ({contradicting}) exceeds threshold",
                supporting_evidence_count=supporting,
                contradicting_evidence_count=contradicting,
                unresolved_challenges=unresolved,
                suggestions=["Address contradicting evidence", "Consider revising hypothesis"],
            )

        # Check if too many unresolved challenges
        if unresolved > self.max_unresolved_challenges:
            return EvaluationResult(
                decision=EvaluationDecision.NEED_MORE_EVIDENCE,
                confidence=0.5,
                reasoning=f"Too many unresolved challenges ({unresolved})",
                supporting_evidence_count=supporting,
                contradicting_evidence_count=contradicting,
                unresolved_challenges=unresolved,
                suggestions=["Address open challenges"],
            )

        # Check acceptance criteria
        if supporting >= self.accept_threshold:
            return EvaluationResult(
                decision=EvaluationDecision.ACCEPT,
                confidence=0.7 + (0.1 * min(supporting - self.accept_threshold, 3)),
                reasoning=f"Sufficient supporting evidence ({supporting})",
                supporting_evidence_count=supporting,
                contradicting_evidence_count=contradicting,
                unresolved_challenges=unresolved,
            )

        # Need more evidence
        return EvaluationResult(
            decision=EvaluationDecision.NEED_MORE_EVIDENCE,
            confidence=0.4,
            reasoning=f"Insufficient supporting evidence ({supporting}/{self.accept_threshold})",
            supporting_evidence_count=supporting,
            contradicting_evidence_count=contradicting,
            unresolved_challenges=unresolved,
            suggestions=["Gather more supporting evidence"],
        )


# ============================================================================
# Composite Evidence Strategy
# ============================================================================


class CompositeEvidenceStrategy(EvidenceGatheringStrategy):
    """Combines multiple evidence gathering strategies.

    Tries each applicable sub-strategy (those where ``can_gather()`` returns
    True), aggregates their evidence, and logs any failures without propagating
    them.

    This is the recommended default evidence strategy: it automatically selects
    which mechanisms to use based on what capabilities are available on the
    agent.

    Example::

        # Default: tries page-graph queries, then LLM reasoning
        strategy = CompositeEvidenceStrategy(agent)

        # Custom composition
        strategy = CompositeEvidenceStrategy(agent, strategies=[
            QueryBasedEvidence(agent),
            LLMReasoningEvidence(agent),
            MyCustomExecutionEvidence(agent),
        ])
    """

    def __init__(
        self,
        agent: Agent,
        strategies: list[EvidenceGatheringStrategy] | None = None,
    ):
        """Initialize with sub-strategies.

        Args:
            agent: Agent for evidence gathering
            strategies: Sub-strategies to compose. Defaults to
                ``[QueryBasedEvidence, LLMReasoningEvidence]`` if None.
        """
        super().__init__(agent)
        if strategies is not None:
            self._strategies = list(strategies)
        else:
            self._strategies = [
                QueryBasedEvidence(agent),
                LLMReasoningEvidence(agent),
            ]

    def add_strategy(self, strategy: EvidenceGatheringStrategy) -> None:
        """Add a strategy to the composition.

        Args:
            strategy: Strategy to add
        """
        self._strategies.append(strategy)

    @override
    def can_gather(
        self,
        hypothesis: Hypothesis,
        context: HypothesisContext,
    ) -> bool:
        """Returns True if any sub-strategy can gather."""
        return any(s.can_gather(hypothesis, context) for s in self._strategies)

    @override
    async def gather_evidence(
        self,
        hypothesis: Hypothesis,
        context: HypothesisContext,
        challenges: list[ChallengeRecord] | None = None,
    ) -> list[Evidence]:
        """Gather evidence from all applicable sub-strategies.

        Tries each strategy where ``can_gather()`` returns True. Failures in
        individual strategies are logged and skipped; evidence from all
        successful strategies is aggregated.
        """
        all_evidence: list[Evidence] = []

        for strategy in self._strategies:
            if not strategy.can_gather(hypothesis, context):
                logger.debug(
                    f"Skipping {type(strategy).__name__}: can_gather() returned False"
                )
                continue

            try:
                evidence = await strategy.gather_evidence(
                    hypothesis, context, challenges
                )
                all_evidence.extend(evidence)
                logger.debug(
                    f"{type(strategy).__name__} gathered {len(evidence)} evidence items"
                )
            except Exception:
                logger.warning(
                    f"{type(strategy).__name__} failed during evidence gathering",
                    exc_info=True,
                )

        return all_evidence

