from __future__ import annotations

import asyncio
import itertools
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar

import git
import numpy as np
import xxhash
from overrides import override

from ...config import register_polymathera_config
from .....distributed import get_polymathera
from ...metrics.common import BaseMetricsMonitor
from .....utils import setup_logger
from .....utils.retry import standard_retry
from .....utils.git.clients import GitHubClient, GitLabClient, GitClientBase
from .base import AnalyzerConfig, BaseAnalyzer, FileContentCache

logger = setup_logger(__name__)


# TODO: Implement more sophisticated temporal analysis
# TODO: Add commit message semantic analysis
# TODO: Implement more sophisticated normalization strategies
# TODO: Add more advanced file rename tracking
# TODO: Implement blame analysis for fine-grained relationships
# TODO: Add support for analyzing merge commits
# TODO: Implement commit pattern detection
# TODO: Add support for analyzing commit author relationships
# TODO: Add more commit classification types
# TODO: Enhance the weighting strategies
# TODO: Add more sophisticated temporal analysis
# TODO: Implement additional scoring methods
"""
TODO: Implement distributed caching for large repos
Distributed caching for large repos would involve implementing a more sophisticated
caching system that can handle massive repositories efficiently.
This system would be particularly useful for:
- Very large repositories (100k+ files)
- High-traffic analysis systems
- Multi-region deployments
- Systems requiring high availability

using AWS EFS would significantly simplify our caching needs since EFS already provides:
1. Distributed storage with automatic scaling
2. Built-in redundancy and high availability
3. Consistent performance across AZs
4. Automatic data replication
5. Shared access from multiple compute instances

Instead of implementing complex distributed caching, we should focus on optimizing the
commit analysis for EFS access patterns.
"""


@register_polymathera_config()
class CommitAnalysisConfig(AnalyzerConfig):
    """Configuration for commit history analysis"""

    # Analysis depth
    max_commits: int = 1000  # Maximum commits to analyze
    max_commit_age_days: int = 365  # Only analyze commits within this timeframe
    min_commit_count: int = 5  # Minimum commits needed for reliable analysis

    # Performance settings
    max_workers: int = 4  # Thread pool size for git operations
    commit_batch_size: int = 100  # Number of commits to process per batch
    max_files_per_commit: int = 1000  # Skip massive commits

    # Analysis strategies
    analysis_strategy: str = "hybrid"  # Options: basic, temporal, weighted, hybrid
    temporal_decay_factor: float = 0.1  # Weight recent commits more heavily

    # EFS optimizations
    optimize_for_efs: bool = True

    # Relationship scoring
    min_relationship_score: float = 0.2
    max_relationship_score: float = 0.95
    score_normalization: str = "sigmoid"  # Options: linear, sigmoid, percentile

    # Advanced features
    track_renames: bool = True  # Track file renames across commits
    analyze_commit_messages: bool = True  # Use commit messages for additional context
    enable_blame_analysis: bool = False  # More detailed but expensive analysis

    # Rate limiting
    commits_per_second: int | None = None  # Rate limit for commit processing

    # Metrics and monitoring
    enable_detailed_metrics: bool = True
    metrics_aggregation_interval: int = 60

    CONFIG_PATH: ClassVar[str] = "llms.sharding.analyzers.history"



class CommitHistoryAnalyzerMetricsMonitor(BaseMetricsMonitor):
    """Base class for Prometheus metrics monitoring using node-global HTTP server."""

    def __init__(self,
                 enable_http_server: bool = True,
                 service_name: str = "service"):
        super().__init__(enable_http_server, service_name)

        self.logger.info(f"Initializing CommitHistoryAnalyzerMetricsMonitor instance {id(self)}...")
        self.commit_processing_time = self.create_histogram(
            "commit_analysis_processing_seconds",
            "Time spent processing commits",
            labelnames=["strategy"],
        )
        self.commit_batch_size = self.create_histogram(
            "commit_analysis_batch_size",
            "Distribution of commit batch sizes"
        )
        self.relationship_scores = self.create_histogram(
            "commit_analysis_relationship_scores",
            "Distribution of relationship scores",
        )
        self.file_changes = self.create_counter(
            "commit_analysis_file_changes_total",
            "Number of file changes processed",
            labelnames=["change_type"],
        )
        self.skipped_commits = self.create_counter(
            "commit_analysis_skipped_commits_total",
            "Number of commits skipped",
            labelnames=["reason"],
        )
        self.active_analyses = self.create_gauge(
            "commit_analysis_active",
            "Number of active commit analyses"
        )


class CommitHistoryAnalyzer(BaseAnalyzer):
    """
    Analyzes git commit history to find related files.
    Supports multiple analysis strategies and performance optimizations.

    1. Multiple Analysis Strategies:
        - Basic: Simple co-occurrence counting
        - Temporal: Time-weighted analysis
        - Weighted: Commit type and size-based weighting
    2. Sophisticated Scoring:
        - Temporal decay
        - Commit type classification
        - Size-based adjustments
        - Type diversity bonuses
    3. Performance Optimizations:
        - Batch processing
        - Concurrent execution
        - Efficient data structures
    4. Robust Error Handling:
        - Per-batch error isolation
        - Fallback mechanisms
        - Detailed logging

    """

    def __init__(self, file_content_cache: FileContentCache, config: CommitAnalysisConfig | None = None):
        super().__init__("commit_history", file_content_cache)
        self.config = config
        self.thread_pool = None
        self.metrics = CommitHistoryAnalyzerMetricsMonitor()
        self._rate_limiter = None

        # Cache for file rename tracking
        self._rename_cache: dict[str, set[str]] = {}
        self._local_cache: dict[str, Any] | None = None

        # Precompute temporal decay weights
        self._temporal_weights = self._precompute_temporal_weights()

    async def initialize(self):
        self.config = await CommitAnalysisConfig.check_or_get_component(self.config)
        await super().initialize()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._rate_limiter = asyncio.Semaphore(
            self.config.commits_per_second if self.config.commits_per_second else 1000
        )

    @override
    async def _analyze_file_impl(
        self, file_path: str, content: str, language: str | None = None, **kwargs
    ) -> dict[str, Any]:
        raise NotImplementedError("Commit history analysis is not supported for files")

    async def analyze_repo(
        self, repo: git.Repo, files: list[str], **kwargs
    ) -> list[tuple[str, str, float]]:
        """
        Analyze repository commit history to find file relationships.
        Supports multiple analysis strategies, parallel processing and
        AWS EFS optimizations.

        Args:
            repo: Git repository
            files: List of files to analyze
            **kwargs: Additional options
                - strategy: Override default analysis strategy
                - max_commits: Override maximum commits to analyze
                - min_score: Override minimum relationship score

        Returns:
            List of tuples (file1, file2, relationship_score)
        """
        if self.config.optimize_for_efs:
            return await self._analyze_repo_efs(repo, files, **kwargs)
        else:
            return await self._analyze_repo(repo, files, **kwargs)

    def _generate_cache_key(
        self, repo: git.Repo, files: set[str], strategy: str
    ) -> str:
        """Generate unique cache key for analysis results"""
        # Get repo identifier
        repo_id = repo.head.commit.hexsha[:8]

        # Sort and hash files for consistent keys
        files_hash = xxhash.xxh64(",".join(sorted(files))).hexdigest()

        # Combine components
        return self._make_cache_key(
            "commit_history",
            repo_id,
            files_hash,
            strategy
        )

    async def _analyze_repo(
        self, repo: git.Repo, files: list[str], **kwargs
    ) -> list[tuple[str, str, float]]:
        try:
            self.metrics.active_analyses.inc()
            start_time = time.time()

            # Get analysis parameters
            strategy = kwargs.get("strategy", self.config.analysis_strategy)
            max_commits = kwargs.get("max_commits", self.config.max_commits)

            # Prepare file paths
            files = set(str(Path(f)) for f in files)  # Normalize paths

            # Check cache first
            cache_key = self._generate_cache_key(repo, files, strategy)
            cached = await self._get_cached_result(cache_key)
            if cached:
                return cached

            # Get commit history in batches
            commit_batches = await self._get_commit_batches(repo, files, max_commits)

            # Process commits using appropriate strategy
            if strategy == "basic":
                relationships = await self._analyze_basic(commit_batches, files)
            elif strategy == "temporal":
                relationships = await self._analyze_temporal(commit_batches, files)
            elif strategy == "weighted":
                relationships = await self._analyze_weighted(commit_batches, files)
            else:  # hybrid
                relationships = await self._analyze_hybrid(commit_batches, files)

            # Cache results
            await self._cache_breaker(self.results_cache.set)(cache_key, relationships)

            # Update metrics
            duration = time.time() - start_time
            self.metrics.commit_processing_time.labels(strategy).observe(duration)

            return relationships

        except git.GitCommandError as e:
            logger.error(f"Git command error: {e}", exc_info=True)
            self.metrics.errors.labels("git_command").inc()
            return []
        except Exception as e:
            logger.error(f"Commit history analysis error: {e}", exc_info=True)
            self.metrics.errors.labels("analysis").inc()
            return []
        finally:
            self.metrics.active_analyses.dec()

    async def _get_commit_batches(
        self, repo: git.Repo, files: set[str], max_commits: int
    ) -> list[list[git.Commit]]:
        """Get commit history in batches for efficient processing"""
        try:
            # Get initial commit list
            cutoff_date = datetime.now() - timedelta(
                days=self.config.max_commit_age_days
            )

            def _get_commits():
                return list(
                    repo.iter_commits(
                        max_count=max_commits, paths=list(files), since=cutoff_date
                    )
                )

            commits = await asyncio.to_thread(_get_commits)

            # Split into batches
            return [
                commits[i : i + self.config.commit_batch_size]
                for i in range(0, len(commits), self.config.commit_batch_size)
            ]

        except Exception as e:
            logger.error(f"Error getting commit batches: {e}", exc_info=True)
            return []

    async def _analyze_hybrid(
        self, commit_batches: list[list[git.Commit]], files: set[str]
    ) -> list[tuple[str, str, float]]:
        """
        Hybrid analysis combining multiple strategies.
        Processes commits in parallel and combines results.
        """
        try:
            # Track file relationships with temporal and semantic weights
            relationships = defaultdict(
                lambda: {
                    "count": 0,
                    "temporal_score": 0.0,
                    "semantic_score": 0.0,
                    "last_commit": None,
                }
            )

            # Process commit batches concurrently
            async with asyncio.TaskGroup() as tg:
                tasks = []
                for batch in commit_batches:
                    tasks.append(
                        tg.create_task(
                            self._process_commit_batch(batch, files, relationships)
                        )
                    )

            # Combine and normalize scores
            return self._combine_relationship_scores(relationships)

        except Exception as e:
            logger.error(f"Error in hybrid analysis: {e}", exc_info=True)
            return []

    @standard_retry(logger)
    async def _process_commit_batch(
        self, commits: list[git.Commit], files: set[str], relationships: dict
    ) -> None:
        """Process a batch of commits with rate limiting"""
        try:
            for commit in commits:
                async with self._rate_limiter:
                    await self._process_single_commit(commit, files, relationships)

        except Exception as e:
            logger.error(f"Error processing commit batch: {e}", exc_info=True)
            raise

    def _combine_relationship_scores(
        self, relationships: dict[tuple[str, str], dict[str, Any]]
    ) -> list[tuple[str, str, float]]:
        """Combine different score components into final relationships"""
        try:
            results = []
            scores = []

            # Calculate combined scores
            for (file1, file2), data in relationships.items():
                if data["count"] < self.config.min_commit_count:
                    continue

                # Combine temporal and semantic scores
                score = self._calculate_combined_score(data)
                scores.append(score)

                if score >= self.config.min_relationship_score:
                    results.append((file1, file2, score))

            # Normalize scores if needed
            if self.config.score_normalization == "percentile":
                results = self._normalize_scores_percentile(results, scores)

            return sorted(results, key=lambda x: x[2], reverse=True)

        except Exception as e:
            logger.error(f"Error combining scores: {e}", exc_info=True)
            return []

    def _calculate_combined_score(self, data: dict[str, Any]) -> float:
        """Calculate combined relationship score"""
        try:
            # Base Jaccard similarity
            base_score = data["count"] / (data["count"] + 1)

            # Apply temporal decay
            if data["last_commit"]:
                age_days = (datetime.now() - data["last_commit"]).days
                temporal_factor = np.exp(-self.config.temporal_decay_factor * age_days)
                base_score *= temporal_factor

            # Add semantic component if available
            if data["semantic_score"] > 0:
                base_score = 0.7 * base_score + 0.3 * data["semantic_score"]

            return min(base_score, self.config.max_relationship_score)

        except Exception:
            return 0.0

    def _precompute_temporal_weights(self) -> np.ndarray:
        """Precompute temporal decay weights"""
        days = np.arange(self.config.max_commit_age_days)
        return np.exp(-self.config.temporal_decay_factor * days)

    def _get_fallback_result(self) -> dict[str, Any]:
        """Return safe fallback result"""
        return {"relationships": []}

    async def _process_single_commit(
        self,
        commit: git.Commit,
        files: set[str],
        relationships: dict[tuple[str, str], dict[str, Any]]
    ) -> None:
        """Process a single commit and update relationships"""
        try:
            # Skip massive commits
            if len(commit.stats.files) > self.config.max_files_per_commit:
                self.metrics.skipped_commits.labels("too_large").inc()
                return

            # Get changed files that we care about
            changed_files = set()
            for file_path, stats in commit.stats.files.items():
                norm_path = str(Path(file_path))
                if norm_path in files:
                    changed_files.add(norm_path)
                    self.metrics.file_changes.labels(
                        "modified" if stats["insertions"] > 0 else "deleted"
                    ).inc()

            # Track renames if enabled
            if self.config.track_renames:
                await self._track_renames(commit, changed_files)

            # Update relationships for all file pairs
            commit_time = datetime.fromtimestamp(commit.committed_date)
            for file1, file2 in itertools.combinations(changed_files, 2):
                if file1 < file2:  # Maintain consistent ordering
                    key = (file1, file2)
                    rel_data = relationships[key]
                    rel_data["count"] += 1
                    rel_data["last_commit"] = commit_time

                    # Update temporal score
                    age_days = (datetime.now() - commit_time).days
                    if age_days < len(self._temporal_weights):
                        rel_data["temporal_score"] = max(
                            rel_data["temporal_score"], self._temporal_weights[age_days]
                        )

                    # Update semantic score if commit message analysis is enabled
                    if self.config.analyze_commit_messages:
                        semantic_score = await self._analyze_commit_message(
                            commit, file1, file2
                        )
                        rel_data["semantic_score"] = max(
                            rel_data["semantic_score"], semantic_score
                        )

        except Exception as e:
            logger.error(f"Error processing commit: {e}", exc_info=True)
            raise

    async def _track_renames(self, commit: git.Commit, changed_files: set[str]) -> None:
        """Track file renames across commits"""
        try:
            if not commit.parents:
                return

            # Get rename info from diff
            for diff in commit.diff(commit.parents[0]):
                if diff.renamed:
                    old_path = str(Path(diff.a_path))
                    new_path = str(Path(diff.b_path))

                    # Update rename cache
                    if old_path not in self._rename_cache:
                        self._rename_cache[old_path] = set()
                    self._rename_cache[old_path].add(new_path)

                    # Update changed files set
                    if old_path in changed_files:
                        changed_files.add(new_path)
                        self.metrics.file_changes.labels("renamed").inc()

        except Exception as e:
            logger.error(f"Error tracking renames: {e}", exc_info=True)

    async def _analyze_commit_message(
        self, commit: git.Commit, file1: str, file2: str
    ) -> float:
        """Analyze commit message for semantic relationship hints"""
        try:
            message = commit.message.lower()

            # Simple keyword-based scoring for now
            # TODO: Implement more sophisticated semantic analysis
            score = 0.0

            # Check for file paths in message
            if Path(file1).name in message and Path(file2).name in message:
                score += 0.3

            # Check for related terms
            related_terms = {"refactor", "move", "reorganize", "restructure"}
            if any(term in message for term in related_terms):
                score += 0.2

            return min(score, 1.0)

        except Exception as e:
            logger.error(f"Error analyzing commit message: {e}", exc_info=True)
            return 0.0

    def _normalize_scores_percentile(
        self, results: list[tuple[str, str, float]], scores: list[float]
    ) -> list[tuple[str, str, float]]:
        """Normalize scores using percentile-based scaling"""
        try:
            if not scores:
                return results

            # Calculate percentiles
            percentiles = np.percentile(scores, [25, 50, 75])

            # Normalize scores
            normalized = []
            for file1, file2, score in results:
                if score <= percentiles[0]:
                    norm_score = 0.25 * (score / percentiles[0])
                elif score <= percentiles[1]:
                    norm_score = 0.25 + 0.25 * (
                        (score - percentiles[0]) / (percentiles[1] - percentiles[0])
                    )
                elif score <= percentiles[2]:
                    norm_score = 0.5 + 0.25 * (
                        (score - percentiles[1]) / (percentiles[2] - percentiles[1])
                    )
                else:
                    norm_score = 0.75 + 0.25 * (
                        (score - percentiles[2]) / (max(scores) - percentiles[2])
                    )

                normalized.append((file1, file2, norm_score))

            return normalized

        except Exception as e:
            logger.error(f"Score normalization error: {e}", exc_info=True)
            return results

    async def _analyze_basic(
        self, commit_batches: list[list[git.Commit]], files: set[str]
    ) -> list[tuple[str, str, float]]:
        """
        Basic analysis using simple co-occurrence counting.
        Fastest but least sophisticated strategy.
        """
        try:
            # Track simple co-occurrence counts
            cooccurrences = defaultdict(int)
            file_counts = defaultdict(int)

            # Process commit batches
            async with asyncio.TaskGroup() as tg:
                tasks = []
                for batch in commit_batches:
                    tasks.append(
                        tg.create_task(
                            self._process_basic_batch(
                                batch, files, cooccurrences, file_counts
                            )
                        )
                    )

            # Calculate Jaccard similarity scores
            results = []
            for (file1, file2), count in cooccurrences.items():
                if count < self.config.min_commit_count:
                    continue

                # Jaccard similarity = intersection / union
                union = file_counts[file1] + file_counts[file2] - count
                score = count / union if union > 0 else 0

                if score >= self.config.min_relationship_score:
                    results.append((file1, file2, score))

            return sorted(results, key=lambda x: x[2], reverse=True)

        except Exception as e:
            logger.error(f"Basic analysis error: {e}", exc_info=True)
            return []

    async def _analyze_temporal(
        self, commit_batches: list[list[git.Commit]], files: set[str]
    ) -> list[tuple[str, str, float]]:
        """
        Temporal analysis weighing recent commits more heavily.
        Better for active repositories with evolving relationships.
        """
        try:
            # Track relationships with temporal weights
            relationships = defaultdict(
                lambda: {"count": 0, "weighted_count": 0.0, "last_commit": None}
            )

            # Process commit batches
            async with asyncio.TaskGroup() as tg:
                tasks = []
                for batch in commit_batches:
                    tasks.append(
                        tg.create_task(
                            self._process_temporal_batch(batch, files, relationships)
                        )
                    )

            # Calculate temporal scores
            results = []
            now = datetime.now()

            for (file1, file2), data in relationships.items():
                if data["count"] < self.config.min_commit_count:
                    continue

                # Base score from weighted counts
                score = data["weighted_count"] / data["count"]

                # Apply recency boost
                if data["last_commit"]:
                    days_old = (now - data["last_commit"]).days
                    recency_factor = np.exp(
                        -self.config.temporal_decay_factor * days_old
                    )
                    score *= recency_factor

                if score >= self.config.min_relationship_score:
                    results.append((file1, file2, score))

            return sorted(results, key=lambda x: x[2], reverse=True)

        except Exception as e:
            logger.error(f"Temporal analysis error: {e}", exc_info=True)
            return []

    async def _analyze_weighted(
        self, commit_batches: list[list[git.Commit]], files: set[str]
    ) -> list[tuple[str, str, float]]:
        """
        Weighted analysis considering commit size and type.
        Best for repositories with diverse commit patterns.
        """
        try:
            # Track relationships with commit weights
            relationships = defaultdict(
                lambda: {
                    "count": 0,
                    "weighted_score": 0.0,
                    "commit_types": defaultdict(int),
                }
            )

            # Process commit batches
            async with asyncio.TaskGroup() as tg:
                tasks = []
                for batch in commit_batches:
                    tasks.append(
                        tg.create_task(
                            self._process_weighted_batch(batch, files, relationships)
                        )
                    )

            # Calculate weighted scores
            results = []
            for (file1, file2), data in relationships.items():
                if data["count"] < self.config.min_commit_count:
                    continue

                # Calculate commit type diversity
                type_diversity = len(data["commit_types"]) / max(
                    sum(data["commit_types"].values()), 1
                )

                # Combine weighted score with type diversity
                score = data["weighted_score"] * (0.7 + 0.3 * type_diversity)

                if score >= self.config.min_relationship_score:
                    results.append((file1, file2, score))

            return sorted(results, key=lambda x: x[2], reverse=True)

        except Exception as e:
            logger.error(f"Weighted analysis error: {e}", exc_info=True)
            return []

    async def _process_basic_batch(
        self,
        commits: list[git.Commit],
        files: set[str],
        cooccurrences: dict,
        file_counts: dict,
    ) -> None:
        """Process a batch of commits for basic analysis"""
        try:
            for commit in commits:
                changed_files = set(f for f in commit.stats.files.keys() if f in files)

                # Update file counts
                for file in changed_files:
                    file_counts[file] += 1

                # Update co-occurrences
                for file1, file2 in itertools.combinations(changed_files, 2):
                    if file1 < file2:  # Maintain consistent ordering
                        cooccurrences[(file1, file2)] += 1

        except Exception as e:
            logger.error(f"Basic batch processing error: {e}", exc_info=True)
            raise

    async def _process_temporal_batch(
        self,
        commits: list[git.Commit],
        files: set[str],
        relationships: dict[tuple[str, str], dict[str, Any]]
    ) -> None:
        """Process a batch of commits for temporal analysis"""
        try:
            for commit in commits:
                changed_files = set(f for f in commit.stats.files.keys() if f in files)

                commit_time = datetime.fromtimestamp(commit.committed_date)
                weight = self._get_temporal_weight(commit_time)

                for file1, file2 in itertools.combinations(changed_files, 2):
                    if file1 < file2:
                        rel = relationships[(file1, file2)]
                        rel["count"] += 1
                        rel["weighted_count"] += weight
                        rel["last_commit"] = max(
                            commit_time, rel["last_commit"] or commit_time
                        )

        except Exception as e:
            logger.error(f"Temporal batch processing error: {e}", exc_info=True)
            raise

    async def _process_weighted_batch(
        self,
        commits: list[git.Commit],
        files: set[str],
        relationships: dict[tuple[str, str], dict[str, Any]]
    ) -> None:
        """Process a batch of commits for weighted analysis"""
        try:
            for commit in commits:
                changed_files = set(f for f in commit.stats.files.keys() if f in files)

                # Determine commit type and weight
                commit_type = self._classify_commit(commit)
                weight = self._get_commit_weight(commit, commit_type)

                for file1, file2 in itertools.combinations(changed_files, 2):
                    if file1 < file2:
                        rel = relationships[(file1, file2)]
                        rel["count"] += 1
                        rel["weighted_score"] += weight
                        rel["commit_types"][commit_type] += 1

        except Exception as e:
            logger.error(f"Weighted batch processing error: {e}", exc_info=True)
            raise

    def _classify_commit(self, commit: git.Commit) -> str:
        """Classify commit type based on message and changes"""
        msg = commit.message.lower()

        # Check for common commit types
        if any(word in msg for word in ["fix", "bug", "patch"]):
            return "bugfix"
        elif any(word in msg for word in ["feat", "feature", "add"]):
            return "feature"
        elif any(word in msg for word in ["refactor", "restructure"]):
            return "refactor"
        elif any(word in msg for word in ["test", "spec"]):
            return "test"
        elif any(word in msg for word in ["docs", "documentation"]):
            return "docs"
        else:
            return "other"

    def _get_commit_weight(self, commit: git.Commit, commit_type: str) -> float:
        """Calculate commit weight based on type and size"""
        # Base weights for different commit types
        type_weights = {
            "bugfix": 1.2,  # Bug fixes often indicate strong relationships
            "feature": 1.0,  # New features are baseline
            "refactor": 1.5,  # Refactoring suggests strong relationships
            "test": 0.8,  # Test changes are less significant
            "docs": 0.5,  # Documentation changes are least significant
            "other": 0.9,  # Default weight for unclassified commits
        }

        # Get base weight
        base_weight = type_weights.get(commit_type, 0.9)

        # Adjust for commit size (smaller commits are more significant)
        total_changes = sum(
            stats.get("insertions", 0) + stats.get("deletions", 0)
            for stats in commit.stats.files.values()
        )
        size_factor = 1.0 / (1.0 + np.log1p(total_changes / 100))

        return base_weight * size_factor

    ###########################################################################
    # TODO: Move EFS Optimizations to a separate class
    ###########################################################################
    async def _analyze_repo_efs(
        self, repo: git.Repo, files: list[str], **kwargs
    ) -> list[tuple[str, str, float]]:
        """Analyze repository with EFS optimizations"""
        try:
            # Apply EFS optimizations
            await self._optimize_for_efs(repo, files)

            # Use larger batch sizes for EFS
            commit_batch_size = kwargs.get("commit_batch_size", self.config.commit_batch_size)
            commit_batch_size = max(commit_batch_size, 100)  # Minimum batch size for EFS

            # Get commit batches with optimized access
            commit_batches = await self._get_commit_batches(
                repo, files, max_commits=commit_batch_size
            )

            # Process batches with EFS-optimized settings
            return await self._analyze_with_efs(commit_batches, files)

        except Exception as e:
            logger.error(f"EFS-optimized analysis error: {e}", exc_info=True)
            return self._get_fallback_result()
        finally:
            # Restore git config
            try:
                repo.git.config("gc.auto", "1")
            except Exception as e:
                logger.error(f"Error restoring git config: {e}", exc_info=True)

    async def _optimize_for_efs(self, repo: git.Repo, files: list[str]) -> None:
        """Optimize git operations for EFS access patterns"""
        try:
            # 1. Minimize file system calls by batching operations
            self.config.commit_batch_size = max(100, self.config.commit_batch_size)

            # 2. Use local caching for frequently accessed metadata
            self._setup_local_cache()

            # 3. Configure git to optimize for network filesystem
            repo.git.config("core.preloadindex", "true")
            repo.git.config("core.fscache", "true")
            repo.git.config("gc.auto", "0")  # Disable auto gc during analysis

            # 4. Warm up the local git object cache
            await self._warm_git_cache(repo, files)

        except Exception as e:
            logger.error(f"EFS optimization error: {e}", exc_info=True)

    def _setup_local_cache(self):
        """Setup local in-memory cache for frequently accessed data"""
        self._local_cache = {
            "commit_metadata": {},  # Cache for commit metadata
            "file_stats": {},  # Cache for file statistics
            "rename_history": {},  # Cache for file renames
            "ttl": 300,  # Cache TTL in seconds
            "max_size": 10000,  # Maximum entries
        }

    async def _warm_git_cache(
        self, repo: git.Repo, files: list[str], max_commits: int = 1000
    ):
        """Pre-fetch commonly accessed git objects"""
        try:
            # Batch prefetch commit metadata
            commits = list(repo.iter_commits(max_count=max_commits, paths=files))

            # Warm up commit metadata cache
            for commit in commits[:100]:  # Cache most recent 100 commits
                key = commit.hexsha
                self._local_cache["commit_metadata"][key] = {
                    "message": commit.message,
                    "author": commit.author.name,
                    "date": commit.committed_datetime,
                    "stats": commit.stats.files,
                }

        except Exception as e:
            logger.error(f"Cache warming error: {e}", exc_info=True)

    async def _analyze_with_efs(
        self, commit_batches: list[list[git.Commit]], files: set[str]
    ) -> list[tuple[str, str, float]]:
        """EFS-optimized analysis process"""
        try:
            # Use local caching for frequently accessed data
            relationships = defaultdict(
                lambda: {
                    "count": 0,
                    "temporal_score": 0.0,
                    "semantic_score": 0.0,
                    "last_commit": None,
                }
            )

            # Process in larger batches for better EFS performance
            async with asyncio.TaskGroup() as tg:
                tasks = []
                for batch in commit_batches:
                    tasks.append(
                        tg.create_task(
                            self._process_commit_batch_efs(batch, files, relationships)
                        )
                    )

            # Convert relationships dict to the format expected by _combine_relationship_scores
            relationship_tuples = {}
            for (file1, file2), data in relationships.items():
                relationship_tuples[(file1, file2)] = data

            return self._combine_relationship_scores(relationship_tuples)

        except Exception as e:
            logger.error(f"EFS analysis error: {e}", exc_info=True)
            return []

    async def _process_commit_batch_efs(
        self, commits: list[git.Commit], files: set[str], relationships: dict
    ) -> None:
        """Process commits with EFS optimizations"""
        try:
            for commit in commits:
                # Check local cache first
                cache_key = commit.hexsha
                commit_data = self._local_cache["commit_metadata"].get(cache_key)

                if commit_data:
                    # Use cached data
                    changed_files = set(
                        f for f in commit_data["stats"].keys() if f in files
                    )
                    commit_time = commit_data["date"]
                else:
                    # Compute and cache
                    changed_files = set(
                        f for f in commit.stats.files.keys() if f in files
                    )
                    commit_time = commit.committed_datetime

                    # Cache for future use
                    if (
                        len(self._local_cache["commit_metadata"])
                        < self._local_cache["max_size"]
                    ):
                        self._local_cache["commit_metadata"][cache_key] = {
                            "stats": commit.stats.files,
                            "date": commit_time,
                        }

                # Update relationships
                await self._update_relationships_efs(
                    changed_files, commit_time, relationships
                )

        except Exception as e:
            logger.error(f"EFS batch processing error: {e}", exc_info=True)
            raise

    async def _update_relationships_efs(
        self, changed_files: set[str], commit_time: datetime, relationships: dict
    ) -> None:
        """Update file relationships for EFS-optimized processing"""
        try:
            # Convert to list for indexing
            file_list = list(changed_files)

            # Update relationships between all pairs of files in this commit
            for i, file1 in enumerate(file_list):
                for file2 in file_list[i + 1:]:
                    # Create relationship key
                    key = (file1, file2) if file1 < file2 else (file2, file1)

                    # Initialize relationship data if not exists
                    if key not in relationships:
                        relationships[key] = {
                            "count": 0,
                            "temporal_score": 0.0,
                            "semantic_score": 0.0,
                            "last_commit": None,
                        }

                    # Update relationship metrics
                    relationships[key]["count"] += 1
                    relationships[key]["last_commit"] = commit_time

                    # Calculate temporal decay (more recent commits have higher weight)
                    time_diff = (datetime.now(timezone.utc) - commit_time).total_seconds()
                    temporal_weight = np.exp(-time_diff / (365 * 24 * 3600 * self.config.temporal_decay_factor))
                    relationships[key]["temporal_score"] += temporal_weight

                    # Add semantic score based on file types and names
                    semantic_weight = self._calculate_semantic_similarity(file1, file2)
                    relationships[key]["semantic_score"] += semantic_weight

        except Exception as e:
            logger.error(f"Error updating EFS relationships: {e}", exc_info=True)
            raise

    def _calculate_semantic_similarity(self, file1: str, file2: str) -> float:
        """Calculate semantic similarity between two files based on names and extensions"""
        # TODO: This is a placeholder. We already have a semantic analyzer.
        try:
            # Get file extensions
            ext1 = Path(file1).suffix.lower()
            ext2 = Path(file2).suffix.lower()

            # Same extension gets higher score
            extension_score = 1.0 if ext1 == ext2 else 0.5

            # Get file names without extensions
            name1 = Path(file1).stem.lower()
            name2 = Path(file2).stem.lower()

            # Calculate name similarity (simple approach)
            name_score = 0.0
            if name1 == name2:
                name_score = 1.0
            elif name1 in name2 or name2 in name1:
                name_score = 0.7
            elif any(part in name2 for part in name1.split('_')):
                name_score = 0.5

            # Get directory paths
            dir1 = str(Path(file1).parent)
            dir2 = str(Path(file2).parent)

            # Same directory gets higher score
            directory_score = 1.0 if dir1 == dir2 else 0.3

            # Combine scores with weights
            total_score = (
                extension_score * 0.4 +
                name_score * 0.4 +
                directory_score * 0.2
            )

            return min(total_score, 1.0)

        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.5  # Default neutral score




class GitHubScrapingTool:
    """
    A tool to scrape GitHub data to augment git history with issues,
    pull requests, comments, long-running branches, and tagged commits.
    """

    def __init__(self):
        self.github_client: GitHubClient | None = None
        self.gitlab_client: GitLabClient | None = None

    async def initialize(self):
        self.github_client = await GitHubClient()
        self.gitlab_client = await GitLabClient()
        await self.github_client.initialize()
        await self.gitlab_client.initialize()

    def _get_client(self, repo_url: str) -> GitClientBase:
        if "github.com" in repo_url:
            return self.github_client
        elif "gitlab.com" in repo_url:
            return self.gitlab_client
        else:
            raise ValueError(f"Unsupported repository: {repo_url}")

    async def _scrape_repo(self, repo_url: str) -> dict[str, Any]:
        client = self._get_client(repo_url)
        owner, repo = repo_url.split("/")[-2:]
        issues = await client.get_issues(owner, repo)
        pull_requests = await client.get_pull_requests(owner, repo)
        branches = await client.get_branches(owner, repo)
        tags = await client.get_tags(owner, repo)

        # Fetch comments for each issue and pull request
        for item in issues + pull_requests:
            item["comments_data"] = await client.get_comments(owner, repo, item["number"])

        return {
            "issues": issues,
            "pull_requests": pull_requests,
            "branches": branches,
            "tags": tags,
        }

    def _augment_git_history(
        self, git_history: list[dict[str, Any]], github_data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        # Create a timeline of all events
        timeline = []
        for commit in git_history:
            timeline.append(
                {
                    "type": "commit",
                    "date": datetime.fromisoformat(commit["date"]),
                    "data": commit,
                }
            )

        for issue in github_data["issues"]:
            timeline.append(
                {
                    "type": "issue",
                    "date": datetime.fromisoformat(issue["created_at"]),
                    "data": issue,
                }
            )

        for pr in github_data["pull_requests"]:
            timeline.append(
                {
                    "type": "pull_request",
                    "date": datetime.fromisoformat(pr["created_at"]),
                    "data": pr,
                }
            )

        # Sort the timeline by date
        timeline.sort(key=lambda x: x["date"])

        # Create the augmented history
        augmented_history = []
        for event in timeline:
            if event["type"] == "commit":
                augmented_history.append(event["data"])
            else:
                # Find the nearest commit before this event
                nearest_commit = next(
                    commit
                    for commit in reversed(augmented_history)
                    if datetime.fromisoformat(commit["date"]) <= event["date"]
                )

                # Augment the commit with the GitHub data
                if "github_data" not in nearest_commit:
                    nearest_commit["github_data"] = []
                nearest_commit["github_data"].append(event["data"])

        # Add branch and tag information
        for commit in augmented_history:
            commit["branches"] = [
                branch["name"]
                for branch in github_data["branches"]
                if branch["commit"]["sha"] == commit["sha"]
            ]
            commit["tags"] = [
                tag["name"]
                for tag in github_data["tags"]
                if tag["commit"]["sha"] == commit["sha"]
            ]

        return augmented_history

    def _get_long_running_branches(self, github_data: dict[str, Any]) -> list[str]:
        # Sort branches by number of commits (assuming more commits means longer-running)
        sorted_branches = sorted(
            github_data["branches"], key=lambda x: x["commit"]["sha"], reverse=True
        )
        # Return top 5 branches or all if less than 5
        return [
            branch["name"] for branch in sorted_branches[: min(5, len(sorted_branches))]
        ]

    async def analyze(self, repo_url: str, repo: git.Repo) -> dict[str, Any]:
        github_data = await self._scrape_repo(repo_url)

        git_history = list(repo.iter_commits())

        augmented_history = self._augment_git_history(git_history, github_data)
        long_running_branches = self._get_long_running_branches(github_data)

        return {
            "augmented_git_history": augmented_history,
            "github_data": github_data,
            "long_running_branches": long_running_branches,
        }

