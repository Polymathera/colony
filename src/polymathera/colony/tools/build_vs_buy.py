"""``BuildVsBuyAdvisor`` — master §4.3 decision policy, formalised.

Captures the decision rule the per-domain dossiers' Appendix-D verdicts
distilled to:

> **Rebuild if all of the following hold:**
> 1. The external tool is Tier C or D (not headless / agent-mediocre).
> 2. The custom version solves a **strictly narrower** problem.
> 3. Validation against a gold standard is feasible.
> 4. The tool is in the inner loop of an agentic workflow.
> 5. The external tool has licensing / determinism / reproducibility
>    issues.
> 6. *(optional but powerful)* The custom version is **differentiable**.
>
> Otherwise, integrate.

With the C5 augment-vs-build refinement (master §3.5.1): when a
``RepoStateProvider.find_existing_tool`` returns a *writable* match in
the design monorepo, the verdict shifts from BUILD to AUGMENT — the
tool already exists and the cheapest path is to extend it on a
``tool/<name>/<feature>`` branch.

Decisions surface as:

- ``AUGMENT`` — extend a writable local tool match.
- ``BUY`` — integrate an existing external adapter.
- ``BUILD`` — build a custom narrower tool from first principles.
- ``HYBRID`` — integrate the external for breadth, build a custom
  narrow path for the inner loop.
- ``CROSS_CHECK_ONLY`` — keep the external as a benchmark; primary
  workflow runs on a custom narrow path.
- ``DENY`` — no acceptable option (e.g., licence-forbidden + custom
  not feasible because validation isn't possible).

The advisor's output is a typed ``BuildVsBuyVerdict`` carrying the
decision, a free-form rationale, the matched tool when applicable, an
effort estimate, and a per-rule evaluation trace so the user can see
exactly why a verdict came out the way it did.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .base import HITLFrequency, HeadlessReadiness, Licensing, ToolSpec


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict shape
# ---------------------------------------------------------------------------


class BuildVsBuyDecision(str, Enum):
    AUGMENT = "augment"
    """Extend an existing writable local tool. Master §3.5.1."""

    BUY = "buy"
    """Integrate an existing external adapter as-is."""

    BUILD = "build"
    """Build a custom narrower tool from first principles."""

    HYBRID = "hybrid"
    """Integrate the external for breadth, build a narrower custom
    inner-loop path. Common when one external tool has the right
    breadth but is too slow / expensive / coarse for the hot path."""

    CROSS_CHECK_ONLY = "cross_check_only"
    """Keep the external as a benchmark; primary workflow runs on a
    custom narrow path."""

    DENY = "deny"
    """No acceptable option under the supplied constraints."""


class RuleEvaluation(BaseModel):
    """Per-rule evaluation result, carried in the verdict for
    transparency. The user can see exactly why a decision came out
    the way it did."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    """Stable id (``"R1"`` … ``"R6"`` plus extras like
    ``"R0_local_match"``)."""

    description: str
    """Human-readable summary of the rule."""

    satisfied: bool
    """Whether the rule was satisfied."""

    reason: str = ""
    """Free-form explanation."""


class ToolMatchSummary(BaseModel):
    """Compact summary of a matched tool — either a writable local
    augment-target or an external integration candidate."""

    model_config = ConfigDict(frozen=True)

    name: str
    location: str = ""
    """E.g., ``subdir:tools/racer/laptime`` for a local match,
    ``github:project/tool`` for an external."""

    spec_summary: str = ""
    """One-line headless / HITL / licence summary."""

    writable: bool = False
    """True for local-monorepo augment targets."""


class BuildVsBuyVerdict(BaseModel):
    """Output of ``BuildVsBuyAdvisor.recommend``."""

    model_config = ConfigDict(frozen=True)

    decision: BuildVsBuyDecision
    rationale: str
    matched_tool: ToolMatchSummary | None = None
    """Set on AUGMENT (a writable local match) or BUY (the chosen
    external adapter); also surfaced on HYBRID / CROSS_CHECK_ONLY for
    the integrated reference. None on BUILD / DENY."""

    estimated_effort_months: float | None = None
    """Per-domain dossier estimates: 1–6 FTE-months for a custom
    narrow tool. None when not applicable (BUY / DENY)."""

    rules_evaluated: tuple[RuleEvaluation, ...] = Field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Advisor input
# ---------------------------------------------------------------------------


class TeamTrackRecord(BaseModel):
    """Compact summary of the team's prior agentic-codegen success.

    Master §4.3 condition (c) leans on this implicitly: validation is
    only feasible if you have the bandwidth to maintain the regression
    suite. We make it explicit so the advisor can downgrade BUILD to
    HYBRID when the team isn't track-record-ready.
    """

    model_config = ConfigDict(frozen=True)

    custom_tools_built: int = 0
    custom_tools_validated_against_gold: int = 0
    average_tool_build_months: float = 3.0
    """Calibration: 1–6 in dossier; 3 is the centre."""

    @property
    def codegen_competence(self) -> float:
        """Roughly: fraction of built tools that passed validation.
        Used as a soft signal in the BUILD/HYBRID escalation."""

        if self.custom_tools_built <= 0:
            return 0.5
        return min(
            1.0,
            self.custom_tools_validated_against_gold / max(1, self.custom_tools_built),
        )


class BuildVsBuyContext(BaseModel):
    """Inputs to ``BuildVsBuyAdvisor.recommend``."""

    model_config = ConfigDict(frozen=True)

    capability_query: str
    """E.g., ``"differentiable_laptime_simulation"``."""

    available_external_tools: tuple[ToolSpec, ...] = Field(default_factory=tuple)
    """Open-source / commercial alternatives the supervisor has surveyed."""

    inner_loop_call_frequency_per_workflow: int = 0
    """How many times per workflow this capability is invoked. ``≥ 50``
    is the doc's "inner loop" threshold."""

    validation_against_gold_feasible: bool = True
    """Can a custom narrower tool be regression-tested against a
    gold-standard tool on a benchmark?"""

    custom_can_be_differentiable: bool = False
    """Optional rule (R6): a differentiable narrower version unlocks
    optimisation paths the external can't support."""

    custom_problem_narrower_than_external: bool = True
    """The custom version solves a strictly smaller problem; generality
    is not the goal."""

    licence_forbidden: frozenset[Licensing] = Field(default_factory=frozenset)
    """Licences this deployment refuses. Defaults to empty."""

    require_determinism: bool = False
    """Rule R5 partial: when True, externals that aren't
    DETERMINISTIC are flagged as licensing/determinism issues."""

    team: TeamTrackRecord = Field(default_factory=TeamTrackRecord)


# ---------------------------------------------------------------------------
# Advisor implementation
# ---------------------------------------------------------------------------


# Threshold for "inner loop" (R4) — master §4.3 condition 4.
INNER_LOOP_FREQUENCY_THRESHOLD = 50


class BuildVsBuyAdvisor:
    """Apply master §4.3's six-rule policy + C5 augment-vs-build
    refinement.

    The advisor is intentionally a plain class (not an
    ``AgentCapability``) because its core is pure logic — it can be
    invoked by any agent's planner, by a tool-building pool's
    supervisor, or by a unit test, all under the same code path.

    Wiring an ``AgentCapability`` wrapper for ``BuildVsBuyAdvisor``
    around this is a small follow-up that the SessionAgent / supervisor
    will pick up in Phase C3.
    """

    def __init__(
        self,
        *,
        repo_state_provider: Any | None = None,
    ) -> None:
        """``repo_state_provider`` is the C5 ``RepoStateProvider``
        capability instance (or any object with the same
        ``find_existing_tool`` method). Optional: when absent the
        AUGMENT path is skipped — useful for environments without a
        design monorepo."""

        self._repo_state_provider = repo_state_provider

    async def recommend(
        self, context: BuildVsBuyContext,
    ) -> BuildVsBuyVerdict:
        rules: list[RuleEvaluation] = []

        # R0 — local writable match.
        local_match = await self._find_local_match(context)
        rules.append(
            RuleEvaluation(
                rule_id="R0_local_match",
                description="Writable local match in the design monorepo",
                satisfied=local_match is not None,
                reason=(
                    f"matched local tool {local_match.name!r} "
                    f"at {local_match.location!r}"
                    if local_match is not None
                    else "no writable local match (or no monorepo configured)"
                ),
            ),
        )
        if local_match is not None:
            return BuildVsBuyVerdict(
                decision=BuildVsBuyDecision.AUGMENT,
                rationale=(
                    f"A writable local tool matches "
                    f"{context.capability_query!r}; augment on a "
                    "tool/<name>/<feature> branch and merge back."
                ),
                matched_tool=local_match,
                estimated_effort_months=context.team.average_tool_build_months / 4.0,
                rules_evaluated=tuple(rules),
            )

        # Build the rule evaluations. We need *all* of R1..R5 (and R6
        # is optional but powerful) to pick BUILD; otherwise BUY/DENY.
        externals = list(context.available_external_tools)
        forbid_set = context.licence_forbidden | _DEFAULT_FORBID

        eligible_externals = [
            spec for spec in externals
            if spec.licensing not in forbid_set
        ]

        # R1 — external tool is Tier C/D (not headless / agent-mediocre).
        # Per master §4.2, Tier C/D is characterised by HUMAN_PRIMARY HITL
        # *or* a GUI-primary / non-headless interface. Tier A/B clear that
        # bar.
        any_not_tier_cd = any(
            not _is_tier_cd(spec) for spec in eligible_externals
        )
        r1_satisfied = (
            len(eligible_externals) == 0
            or not any_not_tier_cd
        )
        rules.append(
            RuleEvaluation(
                rule_id="R1_tier_cd",
                description="External tool is Tier C/D (not headless / agent-mediocre)",
                satisfied=r1_satisfied,
                reason=(
                    "no eligible external tool" if not eligible_externals
                    else (
                        "all eligible externals are Tier C/D"
                        if r1_satisfied
                        else "at least one eligible external is Tier A/B"
                    )
                ),
            ),
        )

        # R2 — strictly narrower custom version.
        rules.append(
            RuleEvaluation(
                rule_id="R2_narrower",
                description="Custom version solves a strictly narrower problem",
                satisfied=context.custom_problem_narrower_than_external,
                reason=(
                    "narrower" if context.custom_problem_narrower_than_external
                    else "custom would aim for the full external scope"
                ),
            ),
        )

        # R3 — validation feasible.
        rules.append(
            RuleEvaluation(
                rule_id="R3_validation",
                description="Validation against a gold standard is feasible",
                satisfied=context.validation_against_gold_feasible,
                reason=(
                    "gold-standard validation possible"
                    if context.validation_against_gold_feasible
                    else "no benchmark exists or validation infeasible"
                ),
            ),
        )

        # R4 — inner-loop frequency threshold.
        r4_satisfied = (
            context.inner_loop_call_frequency_per_workflow
            >= INNER_LOOP_FREQUENCY_THRESHOLD
        )
        rules.append(
            RuleEvaluation(
                rule_id="R4_inner_loop",
                description=f"Tool is invoked ≥{INNER_LOOP_FREQUENCY_THRESHOLD} times per workflow (inner loop)",
                satisfied=r4_satisfied,
                reason=(
                    f"frequency={context.inner_loop_call_frequency_per_workflow}"
                ),
            ),
        )

        # R5 — licensing / determinism / reproducibility issues with the external.
        license_issue = any(
            spec.licensing in {Licensing.COMMERCIAL, Licensing.RESTRICTED, Licensing.GPL, Licensing.AGPL}
            for spec in eligible_externals
        ) or any(spec.licensing in forbid_set for spec in externals)
        determinism_issue = (
            context.require_determinism
            and any(
                spec.determinism.value != "deterministic"
                for spec in eligible_externals
            )
        )
        r5_satisfied = (
            len(externals) == 0
            or license_issue
            or determinism_issue
        )
        rules.append(
            RuleEvaluation(
                rule_id="R5_licence_or_determinism",
                description="External tool has licensing / determinism / reproducibility issues",
                satisfied=r5_satisfied,
                reason=(
                    "no external surveyed (assume issue exists)"
                    if not externals
                    else "; ".join(
                        s for s in (
                            "licence concern" if license_issue else "",
                            "determinism concern" if determinism_issue else "",
                        )
                        if s
                    ) or "no licence / determinism issues"
                ),
            ),
        )

        # R6 — optional, custom version differentiable.
        rules.append(
            RuleEvaluation(
                rule_id="R6_differentiable",
                description="Custom version is differentiable (optional but strong)",
                satisfied=context.custom_can_be_differentiable,
                reason=(
                    "differentiable" if context.custom_can_be_differentiable
                    else "not differentiable"
                ),
            ),
        )

        # Decision.
        return self._verdict_from_rules(context, rules, externals, eligible_externals)

    # ---- Internals ----------------------------------------------------

    async def _find_local_match(
        self, context: BuildVsBuyContext,
    ) -> ToolMatchSummary | None:
        provider = self._repo_state_provider
        if provider is None or not hasattr(provider, "find_existing_tool"):
            return None
        try:
            matches = await provider.find_existing_tool(
                context.capability_query, require_writable=True,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "BuildVsBuyAdvisor: find_existing_tool failed for %s",
                context.capability_query,
            )
            return None
        for m in matches or ():
            entry = getattr(m, "entry", None)
            writable = bool(getattr(m, "writable", False))
            if entry is None or not writable:
                continue
            return ToolMatchSummary(
                name=getattr(entry, "name", "<unknown>"),
                location=getattr(entry, "location", ""),
                spec_summary=_spec_summary_from_entry(entry),
                writable=True,
            )
        return None

    def _verdict_from_rules(
        self,
        context: BuildVsBuyContext,
        rules: list[RuleEvaluation],
        externals: list[ToolSpec],
        eligible_externals: list[ToolSpec],
    ) -> BuildVsBuyVerdict:
        rules_by_id = {r.rule_id: r for r in rules}
        r1 = rules_by_id["R1_tier_cd"].satisfied
        r2 = rules_by_id["R2_narrower"].satisfied
        r3 = rules_by_id["R3_validation"].satisfied
        r4 = rules_by_id["R4_inner_loop"].satisfied
        r5 = rules_by_id["R5_licence_or_determinism"].satisfied
        r6 = rules_by_id["R6_differentiable"].satisfied

        # If no eligible external survives the licence filter and
        # validation isn't feasible, we can't custom-build (no gold)
        # and we can't integrate (no licence-clean external) — DENY.
        if not eligible_externals and not context.validation_against_gold_feasible:
            return BuildVsBuyVerdict(
                decision=BuildVsBuyDecision.DENY,
                rationale=(
                    "No external tool clears the licence filter, and a "
                    "custom version cannot be validated against a gold "
                    "standard. Acquire a benchmark or relax licence "
                    "constraints, then re-evaluate."
                ),
                rules_evaluated=tuple(rules),
            )

        all_required = r1 and r2 and r3 and r4 and r5
        # When all five mandatory rules hold, BUILD; if R6 also holds,
        # the doc calls it "powerful" but doesn't change the verdict.
        if all_required:
            effort = self._estimated_effort(context, more=False)
            external_for_cross_check = self._best_external(eligible_externals)
            if external_for_cross_check is not None and context.validation_against_gold_feasible:
                return BuildVsBuyVerdict(
                    decision=BuildVsBuyDecision.CROSS_CHECK_ONLY,
                    rationale=(
                        "All mandatory build conditions hold (R1–R5"
                        + (", with R6 strengthening the case" if r6 else "")
                        + "). Build the custom narrower tool; keep "
                        f"{external_for_cross_check.name!r} as a "
                        "benchmark cross-check on the regression suite."
                    ),
                    matched_tool=_external_to_summary(external_for_cross_check),
                    estimated_effort_months=effort,
                    rules_evaluated=tuple(rules),
                )
            return BuildVsBuyVerdict(
                decision=BuildVsBuyDecision.BUILD,
                rationale=(
                    "All mandatory build conditions hold (R1–R5"
                    + (", with R6 strengthening the case" if r6 else "")
                    + "). Build a custom narrower tool from first principles."
                ),
                estimated_effort_months=effort,
                rules_evaluated=tuple(rules),
            )

        # HYBRID — at least one of R1/R5 fails (so we can't BUILD), but
        # the inner-loop + validation + narrower triad holds AND we have
        # an integrable external. Use both: external for breadth, custom
        # narrow path for the hot loop.
        if r2 and r3 and r4 and eligible_externals:
            chosen = self._best_external(eligible_externals)
            return BuildVsBuyVerdict(
                decision=BuildVsBuyDecision.HYBRID,
                rationale=(
                    "Inner-loop frequency + feasible validation + narrower "
                    "scope justify a custom narrow path, but the external "
                    f"{chosen.name if chosen else '<none>'!r} covers "
                    "breadth the custom version intentionally won't."
                ),
                matched_tool=_external_to_summary(chosen) if chosen else None,
                estimated_effort_months=self._estimated_effort(context, more=False),
                rules_evaluated=tuple(rules),
            )

        # Otherwise — BUY when any eligible external exists; DENY when
        # none does.
        chosen = self._best_external(eligible_externals)
        if chosen is not None:
            return BuildVsBuyVerdict(
                decision=BuildVsBuyDecision.BUY,
                rationale=(
                    "Mandatory build conditions did not all hold; an "
                    "eligible external exists — integrate it."
                ),
                matched_tool=_external_to_summary(chosen),
                rules_evaluated=tuple(rules),
            )
        return BuildVsBuyVerdict(
            decision=BuildVsBuyDecision.DENY,
            rationale=(
                "No eligible external tool, and the build conditions "
                "did not all hold. Either lift a constraint (e.g., "
                "make the custom version narrower, secure a benchmark) "
                "or accept that this capability is currently out of reach."
            ),
            rules_evaluated=tuple(rules),
        )

    def _estimated_effort(
        self, context: BuildVsBuyContext, *, more: bool,
    ) -> float:
        # The doc's calibration: 1–6 FTE-months per custom narrow
        # tool. Scale by team's centre estimate, soft-modify by the
        # codegen-competence signal.
        base = context.team.average_tool_build_months
        competence = context.team.codegen_competence
        # Less competent teams take longer; more competent are faster.
        # competence in [0, 1]. A team at 0.5 (default) yields the base.
        scale = 1.5 - competence  # 0 → 1.5x, 0.5 → 1.0x, 1.0 → 0.5x
        return round(max(0.5, min(12.0, base * scale)), 2)

    @staticmethod
    def _best_external(eligible: list[ToolSpec]) -> ToolSpec | None:
        if not eligible:
            return None
        # Prefer Tier-A then lower-HITL then deterministic; ties by name.
        return sorted(
            eligible,
            key=lambda s: (
                -s.headless.order,
                s.hitl_frequency.order,
                _DETERMINISM_RANK[s.determinism.value],
                s.name,
            ),
        )[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DEFAULT_FORBID: frozenset[Licensing] = frozenset()
_DETERMINISM_RANK: dict[str, int] = {
    "deterministic": 0,
    "seeded": 1,
    "stochastic": 2,
}


def _is_tier_cd(spec: ToolSpec) -> bool:
    """True iff ``spec`` falls into Tier C or D per master §4.2.

    Tier C/D is characterised by HUMAN_PRIMARY HITL or a non-headless
    interface (GUI_PRIMARY / NONE). Tier A/B clear that bar."""

    if spec.hitl_frequency == HITLFrequency.HUMAN_PRIMARY:
        return True
    if spec.headless in (HeadlessReadiness.GUI_PRIMARY, HeadlessReadiness.NONE):
        return True
    return False


def _external_to_summary(spec: ToolSpec) -> ToolMatchSummary:
    return ToolMatchSummary(
        name=spec.name,
        location=spec.extends_repo or "",
        spec_summary=(
            f"headless={spec.headless.value} hitl={spec.hitl_frequency.value} "
            f"licence={spec.licensing.value}"
        ),
        writable=False,
    )


def _spec_summary_from_entry(entry: Any) -> str:
    headless = getattr(entry, "headless", "")
    licence = getattr(entry, "license", "")
    return f"headless={headless} licence={licence}".strip()


__all__ = (
    "BuildVsBuyAdvisor",
    "BuildVsBuyContext",
    "BuildVsBuyVerdict",
    "BuildVsBuyDecision",
    "RuleEvaluation",
    "ToolMatchSummary",
    "TeamTrackRecord",
    "INNER_LOOP_FREQUENCY_THRESHOLD",
)
