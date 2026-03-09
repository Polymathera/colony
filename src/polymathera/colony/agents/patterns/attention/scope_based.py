"""Scope-based query generation from analysis gaps.

This module implements automatic query generation from scope metadata.
When an analysis result has missing context or related shards, this
automatically generates queries to fill those gaps.

This reduces the burden on agents to manually identify what context they need.
"""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from ...models import QueryContext
from ..scope import ScopeAwareResult, AnalysisScope
from .incremental import PageQuery

T = TypeVar('T')

class ScopeBasedQueryGenerator:
    """Generates queries automatically from scope metadata.

    Looks at:
    - missing_context: Generate queries to find missing items
    - related_shards: Generate queries to understand relationships
    - external_refs: Generate queries about external dependencies

    This enables systematic coverage of missing context.
    """

    def __init__(self, max_queries_per_scope: int = 5):
        """Initialize query generator.

        Args:
            max_queries_per_scope: Maximum queries to generate per scope
        """
        self.max_queries_per_scope = max_queries_per_scope

    async def generate_queries(
        self,
        result: ScopeAwareResult[T],
        context: QueryContext | None = None
    ) -> list[PageQuery]:
        """Generate queries from scope gaps.

        Args:
            result: Result with scope information
            context: Optional context (current pages, etc.)

        Returns:
            List of generated queries

        Examples:
            ```python
            result = ScopeAwareResult(
                content=...,
                scope=AnalysisScope(
                    is_complete=False,
                    missing_context=["AuthManager.validate()", "TokenService"],
                    related_shards=["page_042"]
                )
            )

            generator = ScopeBasedQueryGenerator()
            queries = await generator.generate_queries(result)

            # queries might be:
            # [
            #     PageQuery(query_text="Where is AuthManager.validate() defined?"),
            #     PageQuery(query_text="Where is TokenService defined or used?"),
            #     PageQuery(query_text="What is the relationship with page_042?")
            # ]
            ```
        """
        queries = []
        context = context or QueryContext()
        current_pages = context.current_pages

        scope = result.scope

        # Generate queries for missing context
        for missing_item in scope.missing_context[:self.max_queries_per_scope]:
            query = PageQuery(
                query_text=f"Where is {missing_item} defined or implemented or used?",  # TODO: Need smarter query generation
                query_type="structural",
                source_page_ids=current_pages,
                metadata={"missing_item": missing_item}
            )
            queries.append(query)

        # Generate queries for related shards
        remaining_quota = self.max_queries_per_scope - len(queries)
        for related_shard in scope.related_shards[:remaining_quota]:
            query = PageQuery(
                query_text=f"What is the relationship between current scope and {related_shard}?",  # TODO: Smarter query generation
                query_type="semantic",
                source_page_ids=current_pages,
                metadata={"related_shard": related_shard}
            )
            queries.append(query)

        # Generate queries for external refs if still under quota
        remaining_quota = self.max_queries_per_scope - len(queries)
        for external_ref in scope.external_refs[:remaining_quota]:
            query = PageQuery(
                query_text=f"What is {external_ref}, how does the code interact with it and how is it used?",  # TODO: Smarter query generation
                query_type="semantic",
                source_page_ids=current_pages,
                metadata={"external_ref": external_ref}
            )
            queries.append(query)

        return queries

    async def generate_queries_from_findings(
        self,
        findings: list[Any]
    ) -> list[str]:
        """Generate queries from analysis findings.

        Args:
            findings: Analysis findings

        Returns:
            List of query strings
        """
        queries = []

        # Extract scope information from findings
        for finding in findings:
            if hasattr(finding, 'scope'):
                scope_queries = await self.generate_queries_from_scope(
                    ScopeAwareResult(content=finding, scope=finding.scope)
                )
                queries.extend(scope_queries)

        return queries

    async def generate_targeted_query(
        self,
        missing_item: str,
        current_context: QueryContext
    ) -> PageQuery:
        """Generate a targeted query for a specific missing item.

        Args:
            missing_item: Specific item that's missing
            current_context: Current analysis context

        Returns:
            Targeted query
        """
        # Determine query type based on missing item
        query_type = "structural"

        # If it looks like a type name, use type query
        if missing_item[0].isupper():
            query_type = "type_definition"
            query_text = f"Where is type {missing_item} defined?"
        # If it has parentheses, it's a function
        elif "(" in missing_item:
            query_type = "function_definition"
            query_text = f"Where is function {missing_item} implemented?"
        # Otherwise, generic
        else:
            query_text = f"Where is {missing_item} defined or used?"

        return PageQuery(
            query_text=query_text,
            query_type=query_type,
            source_page_ids=current_context.get("current_pages", []),
            metadata={"missing_item": missing_item}
        )

