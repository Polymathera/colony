from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CodeExecutionContext(BaseModel):
    code: str | Path
    build_dir: Path = Path("./build")
    env: dict[str, str] = Field(default_factory=dict)
    entrypoint: str = "main"
    command: list[str] = []
    working_directory: Path = Path("./")
    resources: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    container_registry: str = ""
    permissions: dict[str, Any] = Field(default_factory=dict)
    python_paths: list[str] = Field(default_factory=list)
    virtualenv: str = ""
    dependencies: list[str] = Field(default_factory=list)
    requirements_file: str = "requirements.txt"


class BugReport(BaseModel):
    description: str
    code_execution_context: CodeExecutionContext


class BugFix(BaseModel):
    description: str
    code_execution_context: CodeExecutionContext


class FeatureRequest(BaseModel):
    description: str
    code_execution_context: CodeExecutionContext


class VulnerabilityReport(BaseModel):
    description: str
    code_execution_context: CodeExecutionContext


class ExploitReport(BaseModel):
    description: str
    code_execution_context: CodeExecutionContext


class CodeDocumentation(BaseModel):
    description: str




class CodeInsights(BaseModel):
    code_quality: dict[str, Any] = Field(
        default_factory=dict, description="Code quality insights"
    )
    hidden_gems: dict[str, Any] = Field(default_factory=dict, description="Hidden gems insights")
    performance_insights: dict[str, Any] = Field(
        default_factory=dict, description="Performance insights"
    )
    security_insights: dict[str, Any] = Field(
        default_factory=dict, description="Security insights"
    )
    maintainability_insights: dict[str, Any] = Field(
        default_factory=dict, description="Maintainability insights"
    )
    complexity_insights: dict[str, Any] = Field(
        default_factory=dict, description="Complexity insights"
    )
    code_smells: dict[str, Any] = Field(default_factory=dict, description="Code smells insights")
    best_practices: dict[str, Any] = Field(
        default_factory=dict, description="Best practices insights"
    )
    design_patterns: dict[str, Any] = Field(
        default_factory=dict, description="Design patterns insights"
    )
    anti_patterns: dict[str, Any] = Field(
        default_factory=dict, description="Anti-patterns insights"
    )
    cyclomatic_complexity: dict[str, int] = Field(
        default_factory=dict, description="Cyclomatic complexity per language"
    )
    maintainability_index: dict[str, int] = Field(
        default_factory=dict, description="Maintainability index per language"
    )
    code_duplication: dict[str, int] = Field(
        default_factory=dict, description="Code duplication per language"
    )
    code_coverage: dict[str, int] = Field(
        default_factory=dict, description="Code coverage per language"
    )
    security_issues: dict[str, int] = Field(
        default_factory=dict, description="Number of security issues per language"
    )
    performance_issues: dict[str, int] = Field(
        default_factory=dict, description="Number of performance issues per language"
    )
    style_violations: dict[str, int] = Field(
        default_factory=dict, description="Number of style violations per language"
    )
    vulnerabilities: dict[str, int] = Field(
        default_factory=dict, description="Number of vulnerabilities per language"
    )
    test_coverage: dict[str, int] = Field(
        default_factory=dict, description="Test coverage per language"
    )
    code_quality_score: dict[str, int] = Field(
        default_factory=dict, description="Code quality score per language"
    )


class CodeChurn(BaseModel):
    lines_added: int = Field(description="Number of lines added")
    lines_deleted: int = Field(description="Number of lines deleted")


