"""Attention mechanism for multi-shard inference.

This module implements the key-query-value attention pattern generalized to the page level,
allowing agents to intelligently select which pages to load based on relevance.

Core Concept:
    Just as transformers use attention to focus on relevant tokens, agents use attention
    to focus on relevant pages. Keys (page summaries) are matched against Queries
    (information needs) to determine which Values (full pages) warrant loading.

Design Philosophy:
    - Protocol-based: Pluggable implementations for different domains
    - Composable: Mix and match key generators, query generators, attention mechanisms
    - Cache-aware: Keys are computed once and reused
    - Domain-agnostic: Works for code, documents, papers, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
import numpy as np
from pydantic import BaseModel, Field
from overrides import override

from ....distributed.ray_utils import serving
from ....system import get_llm_cluster
from ....utils import setup_logger
from ...models import AttentionContext
from ....cluster.models import InferenceRequest

logger = setup_logger(__name__)


# ============================================================================
# Data Models
# ============================================================================


class PageKey(BaseModel):
    """Compact summary of a page (the 'Key' in key-query-value).

    Keys are small (1-4KB) representations of pages that can be efficiently
    matched against queries without loading the full page content.

    Examples:
        Code Analysis:
            - classes: List of class names
            - functions: List of function signatures
            - imports: List of imported modules
            - exports: List of exported symbols

        Document Understanding:
            - section_titles: List of section headings
            - key_terms: Important terms/concepts
            - summary: One-paragraph summary
            - topics: List of topics

        Scientific Papers:
            - title: Paper title
            - abstract: Abstract text
            - methods: List of methods used
            - findings: Key findings
            - citations: Referenced papers
    """

    page_id: str
    key_type: str = "structural"  # "structural", "semantic", "hybrid"
    structural_features: dict[str, Any] = Field(default_factory=dict)  # Domain-specific
    semantic_embedding: list[float] | None = None  # For embedding-based matching
    metadata: dict[str, Any] = Field(default_factory=dict)  # Extensible
    summary: str | None = None  # Human-readable summary

    class Config:
        # Allow numpy arrays to be serialized
        arbitrary_types_allowed = True


class PageQuery(BaseModel):
    """Query for finding relevant pages (the 'Query' in key-query-value).

    Queries express information needs that are matched against page keys
    to determine relevance.

    Examples:
        Code Analysis:
            - "Where is class AuthManager defined?"
            - "Which files import the jwt module?"
            - "Pages containing security-related code"

        Document Understanding:
            - "Sections discussing methodology"
            - "Pages mentioning climate change"
            - "Documents from 2020-2023"

        Scientific Papers:
            - "Papers using BERT for classification"
            - "Studies on COVID-19 vaccines"
            - "Authors from Stanford"
    """

    query_text: str = Field(description="Natural language query")
    query_type: str = Field(default="semantic", description="Type of query (semantic, keyword, structural, hybrid, dependency, etc.)")
    source_page_ids: list[str] = Field(default_factory=list, description="Pages generating query")
    filters: dict[str, Any] = Field(default_factory=dict, description="Optional filters")
    max_results: int = Field(default=10, description="Top-K results to return")
    min_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum relevance score (0.0-1.0)")
    query_embedding: list[float] | None = None  # For embedding-based queries
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional query metadata")  # Extensible


class AttentionScore(BaseModel):
    """Attention score for a page (result of matching query against key).

    Represents how relevant a page is to a query.
    """
    page_id: str = Field(
        description="Page identifier"
    )

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score (0.0-1.0, higher = more relevant)"
    )

    explanation: str | None = Field(default=None, description="Why this page is relevant")
    reasoning: str | None = Field(
        default=None,
        description="Why this page is relevant"
    )
    matched_features: dict[str, Any] = Field(default_factory=dict, description="What matched")

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )



# ============================================================================
# Policy Interfaces (Customization Points)
# ============================================================================


class KeyGenerator(ABC):
    """Policy for generating page keys.

    Different implementations for different domains:
    - StructuralKeyGenerator: Extract structural features (classes, functions, etc.)
    - SemanticKeyGenerator: Generate embedding-based semantic summary
    - HybridKeyGenerator: Combine structural and semantic features
    """

    @abstractmethod
    async def generate_key(
        self, page_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> PageKey:
        """Generate a compact key for a page.

        Args:
            page_id: Unique page identifier
            content: Full page content
            metadata: Optional metadata about the page

        Returns:
            Compact page key (1-4KB)
        """
        ...


class QueryGenerator(ABC):
    """Policy for generating queries from analysis context.

    Different implementations for different needs:
    - DependencyQueryGenerator: Find dependencies
    - ErrorQueryGenerator: Find potential error sources
    - SemanticQueryGenerator: Find semantically related pages
    """

    @abstractmethod
    async def generate_queries(
        self,
        context: AttentionContext,
        findings: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[PageQuery]:
        """Generate queries based on current analysis context.

        Args:
            context: Current analysis context
            findings: Findings from current analysis
            metadata: Optional metadata

        Returns:
            List of queries to find relevant pages
        """
        ...


class AttentionScoringMechanism(ABC):
    """Policy for matching context page queries against context page keys.

    Different implementations for different matching strategies:
    - EmbeddingAttention: Cosine similarity of embeddings
    - KeywordAttention: TF-IDF matching
    - StructuralAttention: Match structural features
    - LLMAttention: Use LLM to judge relevance
    - HybridAttention: Combine multiple signals
    """

    @abstractmethod
    async def score_attention(
        self,
        query: PageQuery,
        keys: list[PageKey],
        context: AttentionContext | None = None,
    ) -> list[AttentionScore]:
        """Compute attention scores for query against keys.

        Args:
            query: Query to match
            keys: Available page keys
            context: Optional context for attention computation

        Returns:
            List of attention scores sorted by relevance (highest first)
        """
        ...


# ============================================================================
# Default Implementations
# ============================================================================


class BatchedLLMAttention(AttentionScoringMechanism):
    """LLM-based attention with batched relevance scoring.

    Uses LLM to score relevance, with efficient batching:
    - One query against multiple keys (1-to-N)
    - One key against multiple queries (N-to-1)

    This reduces LLM calls by batching multiple comparisons in one inference.
    """

    def __init__(
        self,
        relevance_threshold: float = 0.5,
        top_k: int = 10,
        batch_size: int = 20
    ):
        """Initialize batched LLM attention.

        Args:
            relevance_threshold: Minimum relevance score (0.0-1.0)
            top_k: Maximum results to return
            batch_size: Number of keys to score per LLM call
        """
        self.relevance_threshold = relevance_threshold
        self.top_k = top_k
        self.batch_size = batch_size
        self._llm_cluster = None

    def _get_llm_cluster(self) -> serving.DeploymentHandle:
        """Get LLM cluster handle."""
        if self._llm_cluster is None:
            self._llm_cluster = get_llm_cluster()
        return self._llm_cluster

    @override
    async def score_attention(
        self,
        query: PageQuery,
        keys: list[PageKey],
        context: AttentionContext | None = None,
    ) -> list[AttentionScore]:
        """Compute attention using batched LLM relevance scoring.

        Scores query against multiple keys in batches to minimize LLM calls.
        """
        if not keys:
            return []

        llm_cluster = self._get_llm_cluster()

        # Process keys in batches
        all_scores = []
        for i in range(0, len(keys), self.batch_size):
            batch_keys = keys[i:i + self.batch_size]

            # Build prompt for batch scoring
            prompt = self._build_batch_scoring_prompt(query, batch_keys)

            # Score batch with one LLM call
            request = InferenceRequest(
                request_id=f"attention-batch-{i}",
                prompt=prompt,
                context_page_ids=[],  # No pages needed
                max_tokens=500 # TODO: Make fucking configurable
            )

            response = await llm_cluster.infer(request)

            # Parse scores
            batch_scores = self._parse_batch_scores(response.text, batch_keys)
            all_scores.extend(batch_scores)

        # Filter by threshold and sort
        filtered_scores = [
            score for score in all_scores
            if score.score >= self.relevance_threshold
        ]
        filtered_scores.sort(key=lambda x: x.score, reverse=True)

        return filtered_scores[:self.top_k]

    def _build_batch_scoring_prompt(
        self,
        query: PageQuery,
        keys: list[PageKey]
    ) -> str:
        """Build prompt for scoring query against batch of keys."""
        import json

        # Summarize each key
        key_summaries = []
        for idx, key in enumerate(keys):
            summary = {
                "index": idx,
                "page_id": key.page_id,
                "summary": key.summary or "No summary",
                "key_type": key.key_type
            }
            if key.structural_features:
                summary["structure"] = key.structural_features
            key_summaries.append(summary)

        prompt = f"""Score the relevance of each page to this query.

**Query**: {query.query_text}

**Pages to score**:
{json.dumps(key_summaries, indent=2)}

**Task**: For each page (by index), assign a relevance score from 0.0 (not relevant) to 1.0 (highly relevant).

**Output format** (JSON array):
[
    {{"index": 0, "score": 0.8, "reason": "Contains AuthManager implementation"}},
    {{"index": 1, "score": 0.3, "reason": "Tangentially related"}},
    ...
]

Output ONLY the JSON array, no extra text."""

        return prompt

    def _parse_batch_scores(
        self,
        response_text: str,
        keys: list[PageKey]
    ) -> list[AttentionScore]:
        """Parse LLM response into attention scores."""
        import json

        try:
            scores_data = json.loads(response_text)

            scores = []
            for item in scores_data:
                idx = item.get("index")
                score_value = item.get("score", 0.0)
                reason = item.get("reason", "")

                if idx is not None and 0 <= idx < len(keys):
                    key = keys[idx]
                    scores.append(AttentionScore(
                        page_id=key.page_id,
                        score=float(score_value),
                        explanation=reason,
                        matched_features={"llm_score": float(score_value)}
                    ))

            return scores

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM attention scores: {response_text[:200]}")
            # Return zero scores for all keys
            return [
                AttentionScore(
                    page_id=key.page_id,
                    score=0.0,
                    explanation="Failed to parse LLM response",
                    matched_features={}
                )
                for key in keys
            ]


class EmbeddingBasedAttention(AttentionScoringMechanism):
    """Attention mechanism using embedding cosine similarity.

    This is a general-purpose implementation that works well for many domains.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        top_k: int = 10,
    ):
        """Initialize embedding-based attention.

        Args:
            similarity_threshold: Minimum cosine similarity (0.0-1.0)
            top_k: Maximum number of results to return
        """
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self._llm_cluster = None

    def _get_llm_cluster(self) -> serving.DeploymentHandle:
        """Get LLM cluster handle."""
        if self._llm_cluster is None:
            self._llm_cluster = get_llm_cluster()
        return self._llm_cluster

    @override
    async def score_attention(
        self,
        query: PageQuery,
        keys: list[PageKey],
        context: AttentionContext | None = None,
    ) -> list[AttentionScore]:
        """Compute attention using embedding cosine similarity."""
        # Get query embedding
        if query.query_embedding is None:
            llm_cluster = self._get_llm_cluster()
            embeddings = await llm_cluster.embed([query.query_text])
            query_emb = np.array(embeddings[0])
        else:
            query_emb = np.array(query.query_embedding)

        # Compute similarity scores
        scores = []
        for key in keys:
            if key.semantic_embedding is None:
                # Skip keys without embeddings
                logger.warning(f"Key {key.page_id} has no embedding, skipping")
                continue

            key_emb = np.array(key.semantic_embedding)

            # Cosine similarity
            similarity = self._cosine_similarity(query_emb, key_emb)

            if similarity >= self.similarity_threshold:
                scores.append(
                    AttentionScore(
                        page_id=key.page_id,
                        score=float(similarity),
                        explanation=f"Semantic similarity: {similarity:.3f}",
                        matched_features={"embedding_similarity": float(similarity)},
                    )
                )

        # Sort by score (highest first) and limit to top_k
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[: min(len(scores), query.max_results or self.top_k)]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


class StructuralKeyGenerator(KeyGenerator):
    """Generate structural keys for code analysis.

    Extracts classes, functions, imports, etc. from code via VCM.
    """

    def __init__(self):
        self._llm_cluster = None

    def _get_llm_cluster(self) -> serving.DeploymentHandle:
        """Get LLM cluster handle."""
        if self._llm_cluster is None:
            self._llm_cluster = get_llm_cluster()
        return self._llm_cluster

    async def generate_key(
        self, page_id: str, metadata: dict[str, Any] | None = None
    ) -> PageKey:
        """Generate structural key using LLM to analyze page via VCM.

        Args:
            page_id: ID of page to analyze (content loaded from VCM automatically)
            metadata: Optional metadata to attach to key

        Returns:
            PageKey with structural features extracted from code
        """
        llm_cluster = self._get_llm_cluster()

        # Build prompt for structural extraction
        prompt = """Analyze this code and extract its structure.

Output JSON with:
- classes: List of class names
- functions: List of function names
- imports: List of imported modules
- exports: List of exported symbols
- summary: One-sentence summary

Output ONLY valid JSON, no extra text."""

        # Use infer() with context_page_ids - VCM loads content automatically

        request = InferenceRequest(
            request_id=f"key-gen-structural-{page_id}",
            prompt=prompt,
            context_page_ids=[page_id],  # VCM loads page content
            max_tokens=1000, # TODO: Make fucking configurable
        )
        response = await llm_cluster.infer(request)

        # Parse LLM response to extract features
        import json

        try:
            features = json.loads(response.text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response for {page_id}, using fallback")
            features = {"error": "Failed to parse structure"}

        return PageKey(
            page_id=page_id,
            key_type="structural",
            structural_features=features,
            metadata=metadata or {},
        )


class SemanticKeyGenerator(KeyGenerator):
    """Generate semantic keys using embeddings.

    Uses LLM to generate summary, then creates embedding via VCM.
    """

    def __init__(self):
        self._llm_cluster = None

    def _get_llm_cluster(self) -> serving.DeploymentHandle:
        """Get LLM cluster handle."""
        if self._llm_cluster is None:
            self._llm_cluster = get_llm_cluster()
        return self._llm_cluster

    async def generate_key(
        self, page_id: str, metadata: dict[str, Any] | None = None
    ) -> PageKey:
        """Generate semantic key using page summary embedding.

        Args:
            page_id: ID of page to analyze (content loaded from VCM automatically)
            metadata: Optional metadata to attach to key

        Returns:
            PageKey with semantic embedding and summary
        """
        llm_cluster = self._get_llm_cluster()

        # First, generate summary using LLM via VCM
        summary_prompt = "Summarize this code in 1-2 sentences, focusing on its purpose and main functionality."

        summary_request = InferenceRequest(
            request_id=f"key-gen-semantic-summary-{page_id}",
            prompt=summary_prompt,
            context_page_ids=[page_id],  # VCM loads content
            max_tokens=100, # TODO: Make fucking configurable
        )
        summary_response = await llm_cluster.infer(summary_request)
        summary = summary_response.text

        # Then embed the summary
        embeddings = await llm_cluster.embed([summary])

        return PageKey(
            page_id=page_id,
            key_type="semantic",
            semantic_embedding=embeddings[0],
            summary=summary,
            metadata=metadata or {},
        )


class HybridKeyGenerator(KeyGenerator):
    """Combine structural and semantic key generation.

    Uses both structural features and embeddings for richer representation.
    """

    def __init__(self):
        self.structural_generator = StructuralKeyGenerator()
        self.semantic_generator = SemanticKeyGenerator()

    async def generate_key(
        self, page_id: str, metadata: dict[str, Any] | None = None
    ) -> PageKey:
        """Generate hybrid key with both structural and semantic features.

        Args:
            page_id: ID of page to analyze (content loaded from VCM automatically)
            metadata: Optional metadata to attach to key

        Returns:
            PageKey with both structural features and semantic embedding
        """
        # Generate both keys (each accesses VCM independently)
        structural_key = await self.structural_generator.generate_key(page_id, metadata)
        semantic_key = await self.semantic_generator.generate_key(page_id, metadata)

        # Combine into hybrid key
        return PageKey(
            page_id=page_id,
            key_type="hybrid",
            structural_features=structural_key.structural_features,
            semantic_embedding=semantic_key.semantic_embedding,
            summary=semantic_key.summary,
            metadata=metadata or {},
        )



# ============================================================================
# QueryGenerator Implementations
# ============================================================================


class LLMQueryGenerator(QueryGenerator):
    """Generate queries using LLM to analyze current context.

    This is the most flexible QueryGenerator - uses LLM to understand
    findings and generate appropriate follow-up queries.

    Use cases:
    - General code analysis (extract dependencies, find related code)
    - Document understanding (find related sections, citations)
    - Exploratory analysis (generate queries based on current insights)
    """

    def __init__(self, max_queries: int = 5):
        """Initialize LLM-based query generator.

        Args:
            max_queries: Maximum number of queries to generate per call
        """
        self.max_queries = max_queries
        self._llm_cluster = None

    def _get_llm_cluster(self) -> serving.DeploymentHandle:
        """Get LLM cluster handle."""
        if self._llm_cluster is None:
            self._llm_cluster = get_llm_cluster()
        return self._llm_cluster

    @override
    async def generate_queries(
        self,
        context: AttentionContext,
        findings: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[PageQuery]:
        """Generate queries using LLM analysis of findings.

        Args:
            context: Current analysis context (current pages, goals, etc.)
            findings: Findings from current analysis
            metadata: Optional metadata

        Returns:
            List of PageQuery objects for finding relevant pages
        """
        if not findings:
            return []

        llm_cluster = self._get_llm_cluster()

        # Build prompt for query generation
        import json
        findings_text = json.dumps(findings, indent=2)
        context_text = json.dumps(context.model_dump(), indent=2)

        prompt = f"""Based on these analysis findings, generate follow-up queries to find relevant pages.

**Current Context**:
{context_text}

**Findings**:
{findings_text}

**Your Task**: Generate {self.max_queries} queries to find pages that would help:
1. Understand dependencies/imports mentioned in findings
2. Find definitions of referenced classes/functions
3. Locate related code that uses the same patterns
4. Find potential sources of issues/errors
5. Discover semantically related content

**Output Format** (JSON list):
[
    {{
        "query_text": "Where is class AuthManager defined?",
        "query_type": "structural",
        "explanation": "Need to find AuthManager definition referenced in findings"
    }},
    ...
]

Output ONLY valid JSON, no extra text."""

        # Use LLM to generate queries

        request = InferenceRequest(
            request_id=f"query-gen-{context.page_id}",
            prompt=prompt,
            context_page_ids=[],  # No pages needed, just reasoning
            max_tokens=1000, # TODO: Make fucking configurable
        )

        response = await llm_cluster.infer(request)

        # Parse response
        try:
            queries_data = json.loads(response.text)
            if not isinstance(queries_data, list):
                logger.warning(f"LLM returned non-list for query generation: {response.text[:200]}")
                return []

            # Convert to PageQuery objects
            queries = []
            for q in queries_data[:self.max_queries]:
                query = PageQuery(
                    query_text=q.get("query_text", ""),
                    query_type=q.get("query_type", "semantic"),
                    source_page_ids=context.current_pages,
                    max_results=5,
                    min_relevance=0.5,
                    metadata={"explanation": q.get("explanation", ""), **(metadata or {})}
                )
                queries.append(query)

            logger.info(f"Generated {len(queries)} queries from findings")
            return queries

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM query generation response: {response.text[:200]}")
            return []


class DependencyQueryGenerator(QueryGenerator):
    """Generate queries to find code dependencies.

    Specialized for code analysis: extracts imports, class references,
    function calls from findings and generates queries to find their definitions.

    Use cases:
    - "Where is class X defined?"
    - "Which files import module Y?"
    - "Find definition of function Z"
    """

    def __init__(self, max_queries: int = 10):
        """Initialize dependency query generator.

        Args:
            max_queries: Maximum number of queries to generate
        """
        self.max_queries = max_queries

    @override
    async def generate_queries(
        self,
        context: AttentionContext,
        findings: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[PageQuery]:
        """Generate dependency-finding queries from findings.

        Extracts:
        - imports: Generate queries like "Where is module X imported from?"
        - classes: "Where is class Y defined?"
        - functions: "Where is function Z defined?"
        - dependencies: Any referenced symbols

        Args:
            context: Current analysis context
            findings: Findings with imports, classes, functions
            metadata: Optional metadata

        Returns:
            List of PageQuery objects for finding dependencies
        """
        queries = []

        for finding in findings:
            # Extract imports
            imports = finding.get("imports", []) or finding.get("dependencies", [])
            for imp in imports[:self.max_queries // 3]:
                queries.append(PageQuery(
                    query_text=f"Where is {imp} defined or imported from?",
                    query_type="structural",
                    source_page_ids=context.current_pages,
                    filters={"import": imp},
                    max_results=3,
                    min_relevance=0.6,
                    metadata={"dependency_type": "import", "symbol": imp, **(metadata or {})}
                ))

            # Extract class references
            classes = finding.get("classes", []) or finding.get("referenced_classes", [])
            for cls in classes[:self.max_queries // 3]:
                queries.append(PageQuery(
                    query_text=f"Where is class {cls} defined?",
                    query_type="structural",
                    source_page_ids=context.current_pages,
                    filters={"class": cls},
                    max_results=2,
                    min_relevance=0.7,
                    metadata={"dependency_type": "class", "symbol": cls, **(metadata or {})}
                ))

            # Extract function references
            functions = finding.get("functions", []) or finding.get("called_functions", [])
            for func in functions[:self.max_queries // 3]:
                queries.append(PageQuery(
                    query_text=f"Where is function {func} defined?",
                    query_type="structural",
                    source_page_ids=context.current_pages,
                    filters={"function": func},
                    max_results=2,
                    min_relevance=0.7,
                    metadata={"dependency_type": "function", "symbol": func, **(metadata or {})}
                ))

            if len(queries) >= self.max_queries:
                break

        logger.info(f"Generated {len(queries)} dependency queries")
        return queries[:self.max_queries]


class SemanticQueryGenerator(QueryGenerator):
    """Generate queries to find semantically related pages.

    Uses findings to generate natural language queries for finding
    pages about similar topics/concepts.

    Use cases:
    - "Pages about authentication"
    - "Code related to error handling"
    - "Documents discussing methodology"
    """

    def __init__(self, max_queries: int = 5):
        """Initialize semantic query generator.

        Args:
            max_queries: Maximum number of queries to generate
        """
        self.max_queries = max_queries
        self._llm_cluster = None

    def _get_llm_cluster(self) -> serving.DeploymentHandle:
        """Get LLM cluster handle."""
        if self._llm_cluster is None:
            self._llm_cluster = get_llm_cluster()
        return self._llm_cluster

    @override
    async def generate_queries(
        self,
        context: AttentionContext,
        findings: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[PageQuery]:
        """Generate semantic queries from findings.

        Extracts key concepts/topics and generates natural language queries.

        Args:
            context: Current analysis context
            findings: Findings with summaries, topics, insights
            metadata: Optional metadata

        Returns:
            List of PageQuery objects for semantic search
        """
        llm_cluster = self._get_llm_cluster()
        queries = []

        for finding in findings[:self.max_queries]:
            # Extract summary or key insight
            summary = finding.get("summary") or finding.get("insight") or finding.get("description")
            if not summary:
                continue

            # Generate natural language query
            query_text = f"Pages related to: {summary}"

            # Get embedding for semantic search
            embeddings = await llm_cluster.embed([query_text])

            queries.append(PageQuery(
                query_text=query_text,
                query_type="semantic",
                source_page_ids=context.current_pages,
                max_results=5,
                min_relevance=0.6,
                query_embedding=embeddings[0],
                metadata={"topic": summary, **(metadata or {})}
            ))

        logger.info(f"Generated {len(queries)} semantic queries")
        return queries


class ErrorQueryGenerator(QueryGenerator):
    """Generate queries to find potential error sources.

    Specialized for debugging: extracts error messages, exceptions,
    failure patterns and generates queries to find related code.

    Use cases:
    - "Where is NullPointerException thrown?"
    - "Code that handles AuthenticationError"
    - "Functions that call validate()"
    """

    def __init__(self, max_queries: int = 5):
        """Initialize error query generator.

        Args:
            max_queries: Maximum number of queries to generate
        """
        self.max_queries = max_queries

    @override
    async def generate_queries(
        self,
        context: AttentionContext,
        findings: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[PageQuery]:
        """Generate error-related queries from findings.

        Extracts:
        - errors: Exception types, error messages
        - issues: Potential problems, warnings
        - failures: Test failures, runtime errors

        Args:
            context: Current analysis context
            findings: Findings with errors, issues, problems
            metadata: Optional metadata

        Returns:
            List of PageQuery objects for finding error-related pages
        """
        queries = []

        for finding in findings:
            # Extract errors
            errors = finding.get("errors", []) or finding.get("exceptions", [])
            for error in errors[:self.max_queries // 2]:
                error_name = error if isinstance(error, str) else error.get("type", "unknown")
                queries.append(PageQuery(
                    query_text=f"Where is {error_name} raised or handled?",
                    query_type="structural",
                    source_page_ids=context.current_pages,
                    filters={"error_type": error_name},
                    max_results=3,
                    min_relevance=0.6,
                    metadata={"query_category": "error", "error_type": error_name, **(metadata or {})}
                ))

            # Extract issues/problems
            issues = finding.get("issues", []) or finding.get("problems", []) or finding.get("warnings", [])
            for issue in issues[:self.max_queries // 2]:
                issue_desc = issue if isinstance(issue, str) else issue.get("description", "unknown")
                queries.append(PageQuery(
                    query_text=f"Code related to: {issue_desc}",
                    query_type="semantic",
                    source_page_ids=context.current_pages,
                    max_results=3,
                    min_relevance=0.5,
                    metadata={"query_category": "issue", "issue": issue_desc, **(metadata or {})}
                ))

            if len(queries) >= self.max_queries:
                break

        logger.info(f"Generated {len(queries)} error-related queries")
        return queries[:self.max_queries]

