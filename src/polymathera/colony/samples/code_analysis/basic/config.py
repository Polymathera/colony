"""Configuration models for code analysis agents.

All hardcoded values should be replaced with these configurable parameters.
"""

from pydantic import BaseModel, Field


class PageAnalyzerConfig(BaseModel):
    """Configuration for PageAnalyzer agent."""

    max_tokens_summary: int = Field(
        default=2000,
        ge=100,
        le=10000,
        description="Maximum tokens for page summary generation"
    )
    summary_size_limit: int = Field(
        default=4096,
        ge=1024,
        le=10240,
        description="Maximum size in bytes for page summary"
    )
    wait_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout for waiting for page to load"
    )
    request_priority: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Priority for page load requests"
    )


class ClusterAnalyzerConfig(BaseModel):
    """Configuration for ClusterAnalyzer agent."""

    quality_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Quality threshold for completion"
    )
    max_pages_per_iteration: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum pages to analyze per iteration"
    )
    attention_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum attention score for page relevance"
    )
    num_tokens_context: int = Field(
        default=8192,
        ge=1024,
        le=32768,
        description="Context window size for LLM inference"
    )
    num_tokens_generation: int = Field(
        default=2000,
        ge=100,
        le=10000,
        description="Maximum tokens for LLM generation"
    )


class CoordinatorConfig(BaseModel):
    """Configuration for CodeAnalysisCoordinator agent."""

    max_cluster_size: int = Field(
        default=10,
        ge=2,
        le=50,
        description="Maximum pages per cluster"
    )
    min_cluster_size: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum pages per cluster"
    )
    monitor_interval_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=60.0,
        description="Interval for monitoring child agents"
    )
    synthesis_max_tokens: int = Field(
        default=3000,
        ge=500,
        le=10000,
        description="Maximum tokens for synthesis generation"
    )


class AttentionConfig(BaseModel):
    """Configuration for attention mechanism."""

    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum pages to return from attention query"
    )
    min_relevance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for attention results"
    )
    key_cache_size: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum number of page keys to cache"
    )
    embedding_dimension: int = Field(
        default=1536,
        ge=128,
        le=4096,
        description="Embedding dimension for semantic attention"
    )


class ReasoningConfig(BaseModel):
    """Configuration for reasoning loop."""

    max_plan_actions: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum actions in a plan"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries for failed actions"
    )
    reflection_depth: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Depth of reflection on action results"
    )