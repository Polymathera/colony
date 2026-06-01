
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from overrides import override

from ..agents.base import Agent
from ..agents.blackboard.protocol import (
    DesignInconsistencyProtocol,
    DesignSuggestionProtocol,
)

from ..agents.patterns.actions import action_executor
from .client import (
    DesignMonorepoError,
)
from .capabilities import DesignMonorepoCapabilityBase
from .repo_map import REPO_MAP_DIR, REPO_MAP_FILENAME
from ._internal import (
    SYSDES_KUZU_SCAN_LIMIT,
    DESIGN_CONTEXT_URI_SCHEME,
    SYSDES_MAX_FILES_PER_SOURCE_IN_SUMMARY,
    parse_design_context_uri,
    sysdes_list_files,
    sysdes_peek_headings,
    sysdes_grep_file,
)

if TYPE_CHECKING:
    from .materialize import KnowledgeMaterialisationReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SystemDesignCapability — read view + reasoning over the design context
# ---------------------------------------------------------------------------
#
# Co-located with the other design-monorepo capabilities (not in
# ``agents/patterns/capabilities/``) so it can inherit
# :class:`DesignMonorepoCapabilityBase` without creating an import cycle
# (``design_monorepo.capabilities`` already imports from
# ``agents.patterns.capabilities.human_approval``).


# Default predicate sets for the P3c analysis actions. Operators
# extend these by passing explicit arguments to the actions; the
# defaults match what the Phase P3d LLMClaimExtractor's prompt
# would surface as natural-language relations from design markdown.
_DEFAULT_CONTRADICT_PREDICATES = frozenset(
    {"contradicts", "conflicts_with", "is_incompatible_with"},
)
_DEFAULT_CONSISTENCY_RULE_PREDICATES = frozenset(
    {"defines_consistency_rule"},
)
_DEFAULT_HYPOTHESIS_PREDICATES = frozenset(
    {"hypothesizes", "conjectures", "assumes", "posits"},
)
_DEFAULT_VERIFY_PREDICATES = frozenset(
    {"verifies", "falsifies", "supports", "refutes", "tests", "validates"},
)



@dataclass(frozen=True)
class _DesignContextClaim:
    """A claim drawn from the design-context corner of the KG —
    edge + canonical subject/object surface + parsed source-URI
    metadata. Internal to the SystemDesignCapability action surface;
    not part of any public contract.
    """

    subject: str
    predicate: str
    object_: str
    confidence: float
    citation_uri: str
    source_name: str
    rel_path: str


@dataclass(frozen=True)
class _DesignContextClaimScan:
    """Return value of :meth:`SystemDesignCapability._scan_design_context_claims`.

    ``claims`` is the design-context-filtered subset of edges from
    the underlying ``MATCH (s)-[r]->(o) LIMIT N`` query. ``scan_cap_hit``
    is ``True`` when the query hit the per-call edge cap — callers
    should propagate this as ``truncated=True`` in their response so
    the planner knows there may be more claims it didn't see.
    """

    claims: tuple[_DesignContextClaim, ...]
    scan_cap_hit: bool


class SystemDesignCapability(DesignMonorepoCapabilityBase):
    """Read view + reasoning over the design context (objectives,
    constraints, alternatives, hypotheses, decisions — in arbitrary
    mixes per file) declared in the design monorepo's
    ``repo_map.yaml`` ``design_context_sources`` block.

    Phase 1 ships two actions:

    - :meth:`summarise_design_context` — per-source file inventory
      with first-headings peeks; the planner uses it to understand
      "what design context exists, and what's in it at a glance".
    - :meth:`search_design_context` — file-grep with snippets across
      the design-context sources. Phase 3 promotes this to KG /
      VCM-content search via the ``path`` argument's auto-routing.

    The ``path='kuzu'`` and ``path='vcm'`` arguments on
    :meth:`search_design_context` are kept in the signature for
    forward-compatibility but return a structured
    ``not_yet_available`` result; the planner falls through to
    the ``raw`` path automatically when called with ``path='auto'``.

    Phase 3 will:

    - ship the Kuzu claim-extraction path (path 1) — at that point
      ``find_inconsistencies``, ``propose_alternatives``,
      ``assess_artifact``, and ``audit_hypothesis_coverage``
      (outlined in the top-level design doc) become implementable,
    - promote ``search_design_context(path='auto')`` to query the
      KG first and fall through to the file-grep path on miss.

    Inherits :class:`DesignMonorepoCapabilityBase` so it shares the
    per-agent design-monorepo clone with the other design-monorepo
    capabilities (``RepoStateProvider``, ``DesignCheckpointer``,
    ``ToolBuilder``) — no second clone, no custom path resolution.
    Pure action surface — declares no ``@event_handler`` methods,
    so ``input_patterns=[]`` is passed explicitly (same discipline
    as ``RepoStateProvider``).

    See ``colony_docs/markdown/plans/design_top_level_design_process.md``
    §5 (three ingestion paths) + §7 (SystemDesignCapability).
    """

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        working_dir: Path | str | None = None,
        clone_scope_id: str | None = None,
        read_only: bool = True,
        capability_key: str | None = None,
        app_name: str | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            working_dir=working_dir,
            clone_scope_id=clone_scope_id,
            read_only=read_only,
            input_patterns=[],
            capability_key=capability_key,
            app_name=app_name,
        )

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"design_context", "system_design", "read"})

    @action_executor(
        planning_summary=(
            "Summarise the project's design context (one bucket per "
            "``design_context_sources`` row; file counts, first-heading "
            "peeks, last-modified times)."
        ),
    )
    async def summarise_design_context(
        self,
        *,
        source_names: list[str] | None = None,
        max_files_per_source: int | None = None,
    ) -> dict[str, Any]:
        """Return a structured summary of the design context.

        Phase 1 grouping: one bucket per ``design_context_sources``
        row (the LLM-extracted claim-type bucketing arrives with
        Phase 3 KG ingestion). For each row, reports:

        - ``file_count`` — total matching files,
        - ``files`` — up to
          :attr:`SYSDES_MAX_FILES_PER_SOURCE_IN_SUMMARY` files (or
          ``max_files_per_source`` if smaller), each with path /
          size / mtime / a peek of the top markdown headings,
        - ``hint`` — the operator-authored prose from the row,
        - ``pin_in_vcm`` — whether the row is configured for pinning.

        ``source_names`` restricts the report to specific rows.

        Returns
        ``{"sources": [...], "total_files": int, "message": str}``.
        On a repo with no ``design_context_sources``, returns an
        empty report with a helpful ``message``.
        """

        from .repo_map import RepoMap

        repo_root = self._working_dir
        repo_map = await self._sysdes_load_repo_map(RepoMap)

        rows = list(repo_map.design_context_sources)
        if source_names is not None:
            requested = set(source_names)
            rows = [r for r in rows if r.name in requested]
            missing = requested - {r.name for r in rows}
            if missing and not rows:
                return {
                    "sources": [],
                    "total_files": 0,
                    "message": (
                        f"None of the requested source_names "
                        f"({sorted(missing)}) match rows in "
                        f"design_context_sources."
                    ),
                }

        if not rows:
            return {
                "sources": [],
                "total_files": 0,
                "message": (
                    f"No ``design_context_sources:`` rows declared in "
                    f"{REPO_MAP_DIR}/{REPO_MAP_FILENAME}. Add a section "
                    f"to the repo map to opt this project in to "
                    f"design-context summarisation."
                ),
            }

        cap_files = (
            max_files_per_source or SYSDES_MAX_FILES_PER_SOURCE_IN_SUMMARY
        )
        sources_payload: list[dict[str, Any]] = []
        total_files = 0
        for src in rows:
            matched = await asyncio.to_thread(
                sysdes_list_files, repo_root, src,
            )
            total_files += len(matched)
            files_info: list[dict[str, Any]] = []
            for f in matched[:cap_files]:
                try:
                    rel = f.relative_to(repo_root).as_posix()
                except ValueError:
                    rel = str(f)
                try:
                    stat = f.stat()
                except OSError:
                    continue
                headings = await asyncio.to_thread(sysdes_peek_headings, f)
                files_info.append(
                    {
                        "path": rel,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "headings": headings,
                    },
                )
            sources_payload.append(
                {
                    "name": src.name,
                    "hint": src.hint,
                    "pin_in_vcm": src.pin_in_vcm,
                    "file_count": len(matched),
                    "truncated_at": (
                        cap_files if len(matched) > cap_files else None
                    ),
                    "files": files_info,
                },
            )

        return {
            "sources": sources_payload,
            "total_files": total_files,
            "message": "",
        }

    @action_executor(
        planning_summary=(
            "Search the design context for a query string. ``path='auto'`` "
            "queries the Kuzu KG first (claims from path-1 ingestion); "
            "falls through to file-grep (path-3) on miss."
        ),
    )
    async def search_design_context(
        self,
        *,
        query: str,
        path: str = "auto",  # "auto" | "kuzu" | "vcm" | "raw"
        source_names: list[str] | None = None,
        top_k: int = 10,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """Search the design context for ``query``.

        Routing (per top-level design doc §5):

        - ``path='kuzu'`` — KG claim search. Scans claims whose
          ``citation.source_uri`` starts with ``design_context://``
          (populated by Phase P3a's path-1 ingestion), returns hits
          shaped as ``{source_name, file, subject, predicate, object,
          confidence, citation_uri}``. Empty result when no design-
          context claims have been ingested yet (run
          ``materialize_design_context`` first).
        - ``path='vcm'`` — VCM-paginated content search; **future**.
          Returns a structured ``not_yet_available`` shape until then.
        - ``path='raw'`` — file-grep across matching files; returns
          up to ``top_k`` ``(source_name, file, line, snippet)`` hits.
        - ``path='auto'`` — tries ``kuzu`` first; on empty result
          (no KG hits — typically because path-1 ingestion hasn't
          run, or the deterministic extractor produced no claims for
          this query), falls through to ``raw``. ``path_used`` in the
          return tells the planner where the hits came from.

        ``source_names`` restricts the search to specific
        ``design_context_sources`` rows. ``case_sensitive`` defaults
        to ``False``.

        Returns ``{"query", "path_used", "results", "truncated",
        "error"}``. The shape of each ``results`` entry differs by
        ``path_used`` — claims (kuzu) vs. file-line snippets (raw).
        """

        if not query:
            return {
                "query": query,
                "path_used": path,
                "results": [],
                "truncated": False,
                "error": "empty query",
            }

        if path == "vcm":
            return {
                "query": query,
                "path_used": "vcm",
                "results": [],
                "truncated": False,
                "error": (
                    "VCM-paginated content search is not yet wired; "
                    "VCM pages exist (after materialize_design_context) "
                    "and are pinned, but content-search through them "
                    "is a future enhancement. Use path='raw' or "
                    "'auto' (which falls through to raw on KG miss) "
                    "for now."
                ),
            }

        if path == "kuzu":
            return await self._search_kuzu(
                query=query,
                source_names=source_names,
                top_k=top_k,
                case_sensitive=case_sensitive,
            )

        if path == "auto":
            kuzu_result = await self._search_kuzu(
                query=query,
                source_names=source_names,
                top_k=top_k,
                case_sensitive=case_sensitive,
            )
            if kuzu_result["results"]:
                return kuzu_result
            # Fall through to raw — the planner sees ``path_used=='raw'``
            # in the return and knows the KG didn't help on this query.

        # ``path='raw'`` or ``path='auto'`` post-fallthrough.
        return await self._search_raw(
            query=query,
            source_names=source_names,
            top_k=top_k,
            case_sensitive=case_sensitive,
        )

    async def _search_raw(
        self,
        *,
        query: str,
        source_names: list[str] | None,
        top_k: int,
        case_sensitive: bool,
    ) -> dict[str, Any]:
        """File-grep across the matching ``design_context_sources``
        files. The Phase-1 workhorse and the Phase-3 ``auto``
        fallback."""

        import re as _re

        from .repo_map import RepoMap

        repo_root = self._working_dir
        repo_map = await self._sysdes_load_repo_map(RepoMap)

        rows = list(repo_map.design_context_sources)
        if source_names is not None:
            requested = set(source_names)
            rows = [r for r in rows if r.name in requested]
        if not rows:
            return {
                "query": query,
                "path_used": "raw",
                "results": [],
                "truncated": False,
                "error": (
                    "No ``design_context_sources:`` rows to search "
                    "(empty section, or ``source_names`` filtered "
                    "everything out)."
                ),
            }

        pattern = (
            _re.compile(_re.escape(query))
            if case_sensitive
            else _re.compile(_re.escape(query), _re.IGNORECASE)
        )

        results: list[dict[str, Any]] = []
        truncated = False
        for src in rows:
            if len(results) >= top_k:
                truncated = True
                break
            matched = await asyncio.to_thread(
                sysdes_list_files, repo_root, src,
            )
            for f in matched:
                remaining = top_k - len(results)
                if remaining <= 0:
                    truncated = True
                    break
                hits = await asyncio.to_thread(
                    sysdes_grep_file, f, pattern, remaining + 1,
                )
                try:
                    rel = f.relative_to(repo_root).as_posix()
                except ValueError:
                    rel = str(f)
                for line_no, snippet in hits[:remaining]:
                    results.append(
                        {
                            "source_name": src.name,
                            "file": rel,
                            "line": line_no,
                            "snippet": snippet,
                        },
                    )
                if len(hits) > remaining:
                    truncated = True
                if len(results) >= top_k:
                    break

        return {
            "query": query,
            "path_used": "raw",
            "results": results,
            "truncated": truncated,
            "error": "",
        }

    async def _search_kuzu(
        self,
        *,
        query: str,
        source_names: list[str] | None,
        top_k: int,
        case_sensitive: bool,
    ) -> dict[str, Any]:
        """KG claim search — scan claims tagged with the
        ``design_context://`` URI scheme (written by Phase P3a's
        path-1 ingestion), filter by query text matching
        subject/predicate/object surface forms.

        DSL limitation: the existing :class:`GraphStore` query DSL
        (``MATCH (s)-[r]->(o) [WHERE …] [LIMIT n]``) doesn't support
        ``LIKE`` / prefix filters on edge properties, so the shared
        :func:`_iter_design_context_claims` helper pulls a bounded
        set of edges and post-filters by ``citation_uri`` prefix.
        ``SYSDES_KUZU_SCAN_LIMIT`` caps the per-query scan.
        """

        import re as _re

        claims_or_error = await self._scan_design_context_claims(
            source_names=source_names,
            on_error_phase="search_kuzu",
        )
        if isinstance(claims_or_error, dict):
            # Error envelope — propagate.
            claims_or_error.setdefault("query", query)
            claims_or_error["path_used"] = "kuzu"
            return claims_or_error

        pattern = (
            _re.compile(_re.escape(query))
            if case_sensitive
            else _re.compile(_re.escape(query), _re.IGNORECASE)
        )

        hits: list[dict[str, Any]] = []
        truncated = False
        for claim in claims_or_error.claims:
            if not (
                pattern.search(claim.subject)
                or pattern.search(claim.predicate)
                or pattern.search(claim.object_)
            ):
                continue
            if len(hits) >= top_k:
                truncated = True
                break
            hits.append(
                {
                    "source_name": claim.source_name,
                    "file": claim.rel_path,
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object": claim.object_,
                    "confidence": claim.confidence,
                    "citation_uri": claim.citation_uri,
                },
            )

        # Cap-hit propagation: scan cap reached implies we may have
        # missed claims further on; flag truncation regardless of
        # whether top_k was reached.
        if not truncated and claims_or_error.scan_cap_hit:
            truncated = True

        return {
            "query": query,
            "path_used": "kuzu",
            "results": hits,
            "truncated": truncated,
            "error": "",
        }

    # ------------------------------------------------------------------
    # P3c: KG-driven analysis actions
    # ------------------------------------------------------------------

    @action_executor(
        planning_summary=(
            "Surface inconsistencies in the design-context knowledge "
            "graph — explicit contradiction claims + (in P3d+) "
            "operator-authored consistency_rule claims. Emits one "
            "DesignInconsistencyProtocol per finding."
        ),
    )
    async def find_inconsistencies(
        self,
        *,
        source_names: list[str] | None = None,
        contradict_predicates: list[str] | None = None,
        consistency_rule_predicates: list[str] | None = None,
        emit_blackboard_events: bool = True,
    ) -> dict[str, Any]:
        """Find inconsistencies in the design-context KG.

        Two surfaces:

        - **Explicit contradictions** — claims whose ``predicate``
          matches ``contradict_predicates`` (default:
          ``{contradicts, conflicts_with, is_incompatible_with}``).
          The LLM extractor (Phase P3d) emits these when it reads
          two design statements that mutually negate; the
          deterministic extractor (current default) does not, so v1
          returns these only when an operator has wired the LLM
          extractor or manually inserted such claims.

        - **Consistency-rule discovery** — claims whose ``predicate``
          matches ``consistency_rule_predicates`` (default:
          ``{defines_consistency_rule, is_a}`` filtered to objects
          ``consistency_rule``). The action returns these for the
          planner to consider applying (full rule execution waits
          for richer claim types from Phase P3d).

        ``source_names`` filters by ``design_context_sources`` row.

        Returns
        ``{contradictions: [...], rules_discovered: [...], stats: {...}, error: str}``.

        ``emit_blackboard_events=True`` (default) writes a
        :class:`DesignInconsistencyProtocol` per contradiction so
        downstream subscribers (e.g. the Colony Status panel) can
        react.
        """

        import time as _time

        contradict_set = set(contradict_predicates or _DEFAULT_CONTRADICT_PREDICATES)
        rule_set = set(
            consistency_rule_predicates or _DEFAULT_CONSISTENCY_RULE_PREDICATES,
        )

        claims_or_error = await self._scan_design_context_claims(
            source_names=source_names,
            on_error_phase="find_inconsistencies",
        )
        if isinstance(claims_or_error, dict):
            return {
                "contradictions": [],
                "rules_discovered": [],
                "stats": {"scanned_claims": 0, "scan_cap_hit": False},
                "error": claims_or_error.get("error", ""),
            }

        contradictions: list[dict[str, Any]] = []
        rules: list[dict[str, Any]] = []
        for c in claims_or_error.claims:
            if c.predicate in contradict_set:
                contradictions.append(
                    {
                        "kind": "contradiction",
                        "source_name": c.source_name,
                        "file": c.rel_path,
                        "subject": c.subject,
                        "predicate": c.predicate,
                        "object": c.object_,
                        "confidence": c.confidence,
                        "citation_uri": c.citation_uri,
                    },
                )
            # Consistency-rule discovery — match either a direct
            # rule-defining predicate, or the ``is_a consistency_rule``
            # idiom the deterministic extractor would emit if the
            # operator's markdown said ``X is a consistency rule``.
            if c.predicate in rule_set or (
                c.predicate == "is_a"
                and c.object_.replace("_", " ").lower()
                in {"consistency rule", "consistency_rule"}
            ):
                rules.append(
                    {
                        "kind": "rule_discovered",
                        "rule_id": c.subject,
                        "source_name": c.source_name,
                        "file": c.rel_path,
                        "subject": c.subject,
                        "predicate": c.predicate,
                        "object": c.object_,
                        "confidence": c.confidence,
                        "citation_uri": c.citation_uri,
                    },
                )

        if emit_blackboard_events and contradictions:
            blackboard = await self._get_colony_blackboard()
            now = _time.time()
            millis_base = int(now * 1000)
            for idx, finding in enumerate(contradictions):
                key = DesignInconsistencyProtocol.event_key(
                    source_name=finding["source_name"],
                    kind="contradiction",
                    millis=millis_base + idx,
                )
                await blackboard.write(
                    key=key,
                    value={
                        **finding,
                        "detected_at": now,
                    },
                    created_by=(
                        f"system_design.find_inconsistencies:"
                        f"{self.capability_key}"
                    ),
                    tags={
                        "design_context",
                        "inconsistency",
                        "contradiction",
                    },
                )

        return {
            "contradictions": contradictions,
            "rules_discovered": rules,
            "stats": {
                "scanned_claims": len(claims_or_error.claims),
                "scan_cap_hit": claims_or_error.scan_cap_hit,
            },
            "error": "",
        }

    @action_executor(
        planning_summary=(
            "For each hypothesis claim in the design-context KG, "
            "find verifying/falsifying claims pointing at it. Reports "
            "hypotheses with no coverage as orphans + emits one "
            "DesignSuggestionProtocol per orphan."
        ),
    )
    async def audit_hypothesis_coverage(
        self,
        *,
        source_names: list[str] | None = None,
        hypothesis_predicates: list[str] | None = None,
        verify_predicates: list[str] | None = None,
        emit_blackboard_events: bool = True,
    ) -> dict[str, Any]:
        """Audit hypothesis coverage in the design-context KG.

        A claim is treated as a hypothesis when its ``predicate``
        matches ``hypothesis_predicates`` (default:
        ``{hypothesizes, conjectures, assumes, posits}``) OR it
        matches the ``X is_a hypothesis`` idiom the deterministic
        extractor would emit.

        For each hypothesis, the action looks for claims whose
        ``subject`` or ``object`` is the hypothesis's subject and
        whose ``predicate`` matches ``verify_predicates`` (default:
        ``{verifies, falsifies, supports, refutes, tests, validates}``).
        Hypotheses with zero matching coverage claims are flagged as
        **orphans**.

        ``source_names`` filters by ``design_context_sources`` row.

        Returns
        ``{hypotheses: [...], orphans: [...], stats: {...}, error: str}``.
        Each hypothesis entry includes its ``coverage`` list (claims
        that verify / falsify it).

        ``emit_blackboard_events=True`` (default) writes a
        :class:`DesignSuggestionProtocol` per orphan with
        ``kind='hypothesis_orphan'`` so the planner can surface
        them.
        """

        import time as _time

        hyp_pred_set = set(hypothesis_predicates or _DEFAULT_HYPOTHESIS_PREDICATES)
        verify_pred_set = set(verify_predicates or _DEFAULT_VERIFY_PREDICATES)

        claims_or_error = await self._scan_design_context_claims(
            source_names=source_names,
            on_error_phase="audit_hypothesis_coverage",
        )
        if isinstance(claims_or_error, dict):
            return {
                "hypotheses": [],
                "orphans": [],
                "stats": {"scanned_claims": 0, "scan_cap_hit": False},
                "error": claims_or_error.get("error", ""),
            }

        all_claims = list(claims_or_error.claims)

        # Pass 1: identify hypothesis claims. Subject of the hypothesis
        # claim is the hypothesis entity; we'll look for coverage
        # claims that touch this subject.
        hypothesis_subjects: dict[str, list[Any]] = {}
        for c in all_claims:
            is_hypothesis = c.predicate in hyp_pred_set or (
                c.predicate == "is_a"
                and c.object_.lower() == "hypothesis"
            )
            if is_hypothesis:
                hypothesis_subjects.setdefault(c.subject, []).append(c)

        # Pass 2: for each hypothesis, find coverage claims.
        coverage_by_subject: dict[str, list[Any]] = {
            subj: [] for subj in hypothesis_subjects
        }
        for c in all_claims:
            if c.predicate not in verify_pred_set:
                continue
            # A coverage claim "verifies/falsifies H" can mention H
            # as either subject (``H verifies …`` is unusual but valid)
            # or object (``Test-7 verifies H`` — the common shape).
            for ref in (c.subject, c.object_):
                if ref in coverage_by_subject:
                    coverage_by_subject[ref].append(c)

        hypotheses: list[dict[str, Any]] = []
        orphans: list[dict[str, Any]] = []
        for subj, claims in hypothesis_subjects.items():
            # Pick the first claim as the canonical surface for this
            # hypothesis (subject text, source, file). Multiple claims
            # with the same subject are summarised.
            canonical = claims[0]
            coverage = coverage_by_subject.get(subj, [])
            entry = {
                "subject": subj,
                "source_name": canonical.source_name,
                "file": canonical.rel_path,
                "citation_uri": canonical.citation_uri,
                "predicate": canonical.predicate,
                "object": canonical.object_,
                "coverage": [
                    {
                        "subject": cv.subject,
                        "predicate": cv.predicate,
                        "object": cv.object_,
                        "source_name": cv.source_name,
                        "file": cv.rel_path,
                        "citation_uri": cv.citation_uri,
                        "confidence": cv.confidence,
                    }
                    for cv in coverage
                ],
                "coverage_count": len(coverage),
            }
            hypotheses.append(entry)
            if not coverage:
                orphans.append(entry)

        if emit_blackboard_events and orphans:
            blackboard = await self._get_colony_blackboard()
            now = _time.time()
            millis_base = int(now * 1000)
            for idx, orphan in enumerate(orphans):
                key = DesignSuggestionProtocol.event_key(
                    source_name=orphan["source_name"],
                    kind="hypothesis_orphan",
                    millis=millis_base + idx,
                )
                await blackboard.write(
                    key=key,
                    value={
                        "kind": "hypothesis_orphan",
                        "target_claim_type": "hypothesis",
                        "summary": (
                            f"Hypothesis {orphan['subject']!r} has no "
                            f"verifying/falsifying claim in the "
                            f"design context."
                        ),
                        "evidence": [
                            {
                                "citation_uri": orphan["citation_uri"],
                                "snippet": (
                                    f"{orphan['subject']} "
                                    f"{orphan['predicate']} "
                                    f"{orphan['object']}"
                                ),
                            },
                        ],
                        "confidence": 1.0,
                        "detected_at": now,
                    },
                    created_by=(
                        f"system_design.audit_hypothesis_coverage:"
                        f"{self.capability_key}"
                    ),
                    tags={
                        "design_context",
                        "suggestion",
                        "hypothesis_orphan",
                    },
                )

        return {
            "hypotheses": hypotheses,
            "orphans": orphans,
            "stats": {
                "scanned_claims": len(all_claims),
                "scan_cap_hit": claims_or_error.scan_cap_hit,
                "hypothesis_count": len(hypotheses),
                "orphan_count": len(orphans),
            },
            "error": "",
        }

    # ------------------------------------------------------------------
    # Shared KG-scan helper
    # ------------------------------------------------------------------

    async def _scan_design_context_claims(
        self,
        *,
        source_names: list[str] | None,
        on_error_phase: str,
    ) -> "_DesignContextClaimScan | dict[str, Any]":
        """Pull design-context claims from the KG. Returns a
        :class:`_DesignContextClaimScan` on success, or an error
        envelope ``dict`` on infrastructure failures so callers can
        propagate the error through their action-specific return
        shape.

        ``on_error_phase`` is a short label included in logs so
        operators can trace failures back to the calling action
        (``search_kuzu``, ``find_inconsistencies``,
        ``audit_hypothesis_coverage``)."""

        from polymathera.colony.knowledge.deps import get_knowledge_deps

        try:
            deps = get_knowledge_deps()
            graph_store = deps.graph_store
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "SystemDesignCapability.%s: cannot resolve "
                "knowledge_deps; returning empty kuzu result.",
                on_error_phase,
            )
            return {
                "results": [],
                "truncated": False,
                "error": f"deps_unavailable: {exc}",
            }

        if graph_store is None:
            return {
                "results": [],
                "truncated": False,
                "error": (
                    "No graph_store wired in knowledge deps. Set "
                    "``knowledge.graph_db_path`` in the operator YAML "
                    "(or pass an explicit graph_store to "
                    "``set_knowledge_deps``) and re-run "
                    "``materialize_design_context`` to populate the KG."
                ),
            }

        try:
            result = await graph_store.query(
                f"MATCH (s)-[r]->(o) LIMIT {SYSDES_KUZU_SCAN_LIMIT}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "SystemDesignCapability.%s: graph_store.query failed; "
                "returning empty kuzu result.",
                on_error_phase,
            )
            return {
                "results": [],
                "truncated": False,
                "error": f"query_failed: {exc}",
            }

        nodes_by_id = {n.node_id: n for n in result.nodes}
        wanted_sources = set(source_names) if source_names else None

        claims: list[_DesignContextClaim] = []
        for edge in result.edges:
            citation_uri = getattr(edge, "citation_uri", None) or ""
            if not citation_uri.startswith(
                f"{DESIGN_CONTEXT_URI_SCHEME}://",
            ):
                continue
            source_name, rel_path = parse_design_context_uri(citation_uri)
            if wanted_sources is not None and source_name not in wanted_sources:
                continue
            s_node = nodes_by_id.get(edge.source_id)
            o_node = nodes_by_id.get(edge.target_id)
            if s_node is None or o_node is None:
                continue
            claims.append(
                _DesignContextClaim(
                    subject=(
                        str(s_node.properties.get("surface", ""))
                        or s_node.node_id
                    ),
                    predicate=edge.predicate,
                    object_=(
                        str(o_node.properties.get("surface", ""))
                        or o_node.node_id
                    ),
                    confidence=edge.confidence,
                    citation_uri=citation_uri,
                    source_name=source_name,
                    rel_path=rel_path,
                ),
            )

        scan_cap_hit = len(result.edges) >= SYSDES_KUZU_SCAN_LIMIT
        return _DesignContextClaimScan(
            claims=tuple(claims), scan_cap_hit=scan_cap_hit,
        )

    async def _sysdes_load_repo_map(self, repo_map_cls: type) -> Any:
        """Resolve the per-agent clone (lazy-cloning if needed) and
        parse its ``repo_map.yaml``."""

        repo_root = self._working_dir
        if not (repo_root / ".git").is_dir():
            self._lazy_clone_from_agent_metadata()
        if not (repo_root / ".git").is_dir():
            raise DesignMonorepoError(
                f"{repo_root} is not a git repository — set the "
                "colony's design-monorepo URL on the landing page "
                "and start a fresh session, or run "
                "``initialize_repo_map`` first.",
            )
        return await asyncio.to_thread(repo_map_cls.load, repo_root)


__all__ = (
    "SystemDesignCapability",
)

