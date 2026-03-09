from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .costs import ProcessingCosts
from .insights import CodeInsights


class AnalysisRequest(BaseModel):
    request_type: str
    details: str


class AnalysisResponse(BaseModel):
    insights: CodeInsights
    processing_costs: ProcessingCosts
    analysis: dict[str, Any]


class LLMStaticCodeAnalysisDescription(BaseModel):
    analysis_type: str
    description: str
    examples: list[str]
    target_patterns: list[str]
    scope: str = Field(
        default="Repository",
        description="The scope of the analysis. Possible values: 'Repository', 'File', 'Function', 'Class', 'Method', 'Module', etc.",
    )


