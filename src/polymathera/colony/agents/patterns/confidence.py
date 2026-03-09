"""Confidence tracking and uncertainty propagation.

This module implements confidence-aware decision making:
- ConfidenceTracker: Aggregates confidence across results
- UncertaintyPropagator: Composes uncertainty across dependencies
- Confidence-based decision heuristics

This module provides utilities for:
- Tracking confidence across analysis results
- Propagating uncertainty through analysis chains
- Deciding when to continue analysis based on confidence
- Aggregating confidence from multiple sources

Confidence is a key signal for:
- Deciding whether to query for more context
- Determining if results need validation
- Prioritizing which analyses to run
- Triggering refinement loops
"""

from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel, Field

from .scope import ScopeAwareResult


class ConfidenceMetrics(BaseModel):
    """Aggregated confidence metrics across results."""

    average_confidence: float = Field(
        description="Average confidence across all results"
    )

    minimum_confidence: float = Field(
        description="Minimum confidence (worst case)"
    )

    maximum_confidence: float = Field(
        description="Maximum confidence (best case)"
    )

    weighted_confidence: float = Field(
        description="Confidence weighted by scope completeness"
    )

    confidence_variance: float = Field(
        description="Variance in confidence scores"
    )

    low_confidence_count: int = Field(
        description="Number of results with confidence < 0.7"
    )

    high_confidence_count: int = Field(
        description="Number of results with confidence >= 0.8"
    )

    total_count: int = Field(
        description="Total number of results"
    )


class ConfidenceTracker:
    """Tracks and reasons about confidence across analysis results.

    Provides:
    - Confidence aggregation (weighted, average, min, max)
    - Confidence-based decisions (should continue analysis, should validate result)
    - Prioritizing analyses by confidence gaps
    - Confidence trend/evolution tracking
    """

    def __init__(self):
        """Initialize tracker."""
        self.confidence_history: list[tuple[float, ConfidenceMetrics]] = []

    def calculate_aggregate_confidence(
        self,
        results: list[ScopeAwareResult]
    ) -> float:
        """Calculate aggregate confidence across multiple results.

        Weighted average based on scope completeness and quality.
        Complete results get more weight than incomplete ones.

        Args:
            results: Results to aggregate

        Returns:
            Aggregate confidence score (0.0 to 1.0)

        Examples:
            ```python
            results = [
                ScopeAwareResult(content=..., scope=AnalysisScope(
                    is_complete=True, confidence=0.9
                )),
                ScopeAwareResult(content=..., scope=AnalysisScope(
                    is_complete=False, confidence=0.6
                ))
            ]

            tracker = ConfidenceTracker()
            agg = tracker.calculate_aggregate_confidence(results)
            # agg ≈ 0.75 (weighted toward complete result)
            ```
        """
        if not results:
            return 0.0

        # Weight by completeness and quality
        total_weight = 0.0
        weighted_sum = 0.0

        for result in results:
            # Weight combines confidence, completeness, and quality
            completeness_weight = 1.0 if result.scope.is_complete else 0.5
            quality_weight = result.scope.quality_score
            weight = completeness_weight * quality_weight

            weighted_sum += result.scope.confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def should_continue_analysis(
        self,
        results: list[ScopeAwareResult],
        target_confidence: float = 0.8,
        min_high_confidence_ratio: float = 0.7
    ) -> bool:
        """Decide if more analysis is needed based on confidence.

        Args:
            results: Current results
            target_confidence: Target aggregate confidence
            min_high_confidence_ratio: Minimum ratio of high-confidence results

        Returns:
            True if more analysis is needed
        """
        metrics = self.calculate_metrics(results)

        # Continue if aggregate confidence below target
        if metrics.weighted_confidence < target_confidence:
            return True

        # Continue if too many low-confidence results
        high_confidence_ratio = metrics.high_confidence_count / metrics.total_count
        if high_confidence_ratio < min_high_confidence_ratio:
            return True

        # Continue if variance is high (inconsistent confidence)
        if metrics.confidence_variance > 0.05:
            return True

        return False

    def calculate_metrics(
        self,
        results: list[ScopeAwareResult]
    ) -> ConfidenceMetrics:
        """Calculate comprehensive confidence metrics.

        Args:
            results: Results to analyze

        Returns:
            Confidence metrics
        """
        if not results:
            return ConfidenceMetrics(
                average_confidence=0.0,
                minimum_confidence=0.0,
                maximum_confidence=0.0,
                weighted_confidence=0.0,
                confidence_variance=0.0,
                low_confidence_count=0,
                high_confidence_count=0,
                total_count=0
            )

        confidences = [r.scope.confidence for r in results]

        metrics = ConfidenceMetrics(
            average_confidence=statistics.mean(confidences),
            minimum_confidence=min(confidences),
            maximum_confidence=max(confidences),
            weighted_confidence=self.calculate_aggregate_confidence(results),
            confidence_variance=statistics.variance(confidences) if len(confidences) > 1 else 0.0,
            low_confidence_count=sum(1 for c in confidences if c < 0.7),
            high_confidence_count=sum(1 for c in confidences if c >= 0.8),
            total_count=len(results)
        )

        return metrics

    def should_validate(
        self,
        result: ScopeAwareResult,
        validation_threshold: float = 0.7
    ) -> bool:
        """Decide if result needs validation.

        Args:
            result: Result to check
            validation_threshold: Confidence below which validation is needed

        Returns:
            True if validation is needed
        """
        # Always validate if not already validated
        if not result.validated:
            return True

        # Validate if confidence is low
        if result.scope.confidence < validation_threshold:
            return True

        # Validate if incomplete
        if not result.scope.is_complete:
            return True

        return False

    def prioritize_for_validation(
        self,
        results: list[ScopeAwareResult]
    ) -> list[ScopeAwareResult]:
        """Prioritize results for validation (lowest confidence first).

        Args:
            results: Results to prioritize

        Returns:
            Sorted list (lowest confidence first)
        """
        return sorted(results, key=lambda r: r.scope.confidence)

    def record_snapshot(self, results: list[ScopeAwareResult]) -> None:
        """Record current confidence state.

        Args:
            results: Current results
        """
        metrics = self.calculate_metrics(results)
        self.confidence_history.append((time.time(), metrics.weighted_confidence))
        self.completeness_history.append((
            time.time(),
            sum(1 for r in results if r.scope.is_complete) / len(results) if results else 0.0
        ))


class UncertaintyPropagator:
    """Propagates and composes uncertainty across analysis steps.

    When results depend on other results, uncertainty compounds.
    This class handles uncertainty composition and propagation through dependency chains.

    Key insight: Confidence decreases as we chain uncertain results.
    We use conservative composition to avoid overconfidence.
    """

    def compose_uncertainties(
        self,
        upstream_results: list[ScopeAwareResult]
    ) -> float:
        """Compose confidence from multiple upstream sources.

        Uses geometric mean (more conservative than arithmetic mean).

        Args:
            upstream_results: Results this result depends on

        Returns:
            Composed confidence score

        Examples:
            ```python
            # Two upstream results with moderate confidence
            upstream = [
                ScopeAwareResult(content=..., scope=AnalysisScope(confidence=0.8)),
                ScopeAwareResult(content=..., scope=AnalysisScope(confidence=0.7))
            ]

            propagator = UncertaintyPropagator()
            composed = propagator.compose_uncertainties(upstream)
            # composed ≈ 0.75 (geometric mean, more pessimistic than 0.75 arithmetic)
            ```
        """
        if not upstream_results:
            return 0.0

        confidences = [r.scope.confidence for r in upstream_results]

        # Geometric mean - confidence decreases as we chain uncertain results
        product = 1.0
        for c in confidences:
            product *= c

        return product ** (1.0 / len(confidences))

    def propagate_through_chain(
        self,
        result_chain: list[ScopeAwareResult]
    ) -> float:
        """Propagate uncertainty through a chain of dependent results.

        For sequential dependencies, confidence compounds multiplicatively.

        Args:
            result_chain: Chain of results (each depends on previous)

        Returns:
            Final confidence after propagation
        """
        if not result_chain:
            return 0.0

        # Multiply confidences through chain
        confidence = 1.0
        for result in result_chain:
            confidence *= result.scope.confidence

        return confidence

    def should_request_validation(
        self,
        result: ScopeAwareResult,
        threshold: float = 0.7
    ) -> bool:
        """Decide if result needs validation due to uncertainty.

        Args:
            result: Result to check
            threshold: Confidence threshold

        Returns:
            True if validation needed
        """
        return result.scope.confidence < threshold

    def estimate_downstream_confidence(
        self,
        upstream_results: list[ScopeAwareResult],
        analysis_confidence: float = 0.9
    ) -> float:
        """Estimate confidence of downstream result given upstream dependencies.

        Args:
            upstream_results: Upstream dependencies
            analysis_confidence: Confidence in the analysis itself (if inputs were perfect)

        Returns:
            Estimated downstream confidence
        """
        # Compose upstream uncertainties
        upstream_confidence = self.compose_uncertainties(upstream_results)

        # Combine with analysis confidence
        return upstream_confidence * analysis_confidence

    def identify_weak_links(
        self,
        results: list[ScopeAwareResult],
        threshold: float = 0.6
    ) -> list[ScopeAwareResult]:
        """Identify results with weak confidence that should be refined.

        Args:
            results: Results to check
            threshold: Confidence threshold for "weak"

        Returns:
            List of weak results
        """
        return [r for r in results if r.scope.confidence < threshold]


# Utility functions

def calculate_confidence_interval(
    results: list[ScopeAwareResult],
    confidence_level: float = 0.95
) -> tuple[float, float]:
    """Calculate confidence interval for aggregate confidence.

    Args:
        results: Results to analyze
        confidence_level: Desired confidence level (0.95 = 95%)

    Returns:
        (lower_bound, upper_bound) tuple
    """
    if not results:
        return (0.0, 0.0)

    confidences = [r.scope.confidence for r in results]

    if len(confidences) == 1:
        return (confidences[0], confidences[0])

    # Calculate mean and standard error
    mean = statistics.mean(confidences)
    std_dev = statistics.stdev(confidences)
    std_error = std_dev / (len(confidences) ** 0.5)

    # Use normal approximation (z-score for 95% = 1.96)
    z_score = 1.96 if confidence_level == 0.95 else 2.576  # 99%
    margin = z_score * std_error

    lower = max(0.0, mean - margin)
    upper = min(1.0, mean + margin)

    return (lower, upper)


def combine_confidence_scores(
    scores: list[float],
    method: str = "geometric_mean"
) -> float:
    """Combine multiple confidence scores.

    Args:
        scores: Confidence scores to combine
        method: Combination method
            - "geometric_mean": Multiplicative (conservative)
            - "arithmetic_mean": Average
            - "harmonic_mean": Penalizes low scores
            - "minimum": Most conservative

    Returns:
        Combined confidence score
    """
    if not scores:
        return 0.0

    if method == "geometric_mean":
        product = 1.0
        for score in scores:
            product *= score
        return product ** (1.0 / len(scores))

    elif method == "arithmetic_mean":
        return statistics.mean(scores)

    elif method == "harmonic_mean":
        return statistics.harmonic_mean(scores)

    elif method == "minimum":
        return min(scores)

    else:
        raise ValueError(f"Unknown combination method: {method}")

