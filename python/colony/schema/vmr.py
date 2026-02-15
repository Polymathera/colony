
from __future__ import annotations

import uuid
from typing import Any, Callable
import os
from pathlib import Path
import json
from datetime import datetime, timezone

import sqlmodel as sqlm
from sqlalchemy import JSON, DateTime, String, Integer, TypeDecorator
from sqlalchemy.dialects.postgresql import TIMESTAMP
from pydantic import BaseModel, Field, TypeAdapter, HttpUrl, field_validator, UUID4, field_serializer, ConfigDict
from logging import getLogger

from .base_types import (
    RepoId,
    VmrId,
    get_repo_fingerprint,
    get_repo_name_from_origin_url,
)
from .goals import (
    GoalGroup,
    get_default_intrinsic_goals,
)
from .costs import ProcessingCosts
from .analyses import LLMStaticCodeAnalysisDescription
from .aspects import PredefinedCodeAspects
from .constraints import ExplorationConstraints, get_default_exploration_constraints
from .insights import CodeChurn
from ..agents.models import ActionPlan

logger = getLogger(__name__)

# ---------------------------------------------------------------------------
# Important:
# - SQLModel does NOT work with from __future__ import annotations
# - Forward references within the same file must use string quotes
# - Use string quotes for type hints of function, relationships, fields, etc.
# ---------------------------------------------------------------------------

# Custom TypeDecorator for Pydantic models
class PydanticType(TypeDecorator):
    """A SQLAlchemy TypeDecorator that handles Pydantic models in JSON columns.

    This properly serializes Pydantic models to JSON for storage and
    deserializes them back to Pydantic models when loading from the database.

    Supports both single Pydantic models and lists of Pydantic models.
    """
    impl = JSON
    cache_ok = True

    def __init__(self, pydantic_model_class, is_list=False):
        super().__init__()
        self.pydantic_model_class = pydantic_model_class
        self.is_list = is_list

    def process_bind_param(self, value, dialect):
        """Convert Pydantic model(s) to dict/list for JSON storage."""
        if value is None:
            return None

        if self.is_list:
            if isinstance(value, list):
                result = []
                for item in value:
                    if hasattr(item, 'model_dump'):
                        result.append(item.model_dump(mode='json'))
                    elif isinstance(item, dict):
                        result.append(item)
                    else:
                        raise TypeError(f"Expected {self.pydantic_model_class.__name__} or dict in list, got {type(item)}")
                return result
            elif isinstance(value, dict):
                return value  # Pass through dicts for backwards compatibility
            else:
                raise TypeError(f"Expected list of {self.pydantic_model_class.__name__} or dict, got {type(value)}")
        else:
            if isinstance(value, self.pydantic_model_class):
                # Use mode='json' to handle datetime and other non-JSON types
                return value.model_dump(mode='json')
            # If it's already a dict, pass it through
            if isinstance(value, dict):
                return value
            raise TypeError(f"Expected {self.pydantic_model_class.__name__} or dict, got {type(value)}")

    def process_result_value(self, value, dialect):
        """Convert dict/list from JSON storage back to Pydantic model(s)."""
        if value is None:
            return None

        if self.is_list:
            if isinstance(value, list):
                result = []
                for item in value:
                    if isinstance(item, dict):
                        result.append(self.pydantic_model_class(**item))
                    else:
                        result.append(item)  # Pass through non-dict items
                return result
            else:
                return value  # Pass through non-list values for backwards compatibility
        else:
            if isinstance(value, dict):
                return self.pydantic_model_class(**value)
            return value

class RepositoryStats(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    size_mb: float = Field(default=0.0, description="Size of the repository in MB")
    lines_of_code: dict[str, int] = Field(
        default_factory=dict, description="Number of lines of code per language"
    )
    llm_tokens: dict[str, int] = Field(
        default_factory=dict, description="Number of tokens per language"
    )
    file_count: dict[str, int] = Field(
        default_factory=dict, description="Number of files per language"
    )
    binary_file_count: dict[str, int] = Field(
        default_factory=dict, description="Number of binary files per language"
    )
    text_file_count: dict[str, int] = Field(
        default_factory=dict, description="Number of text files per language"
    )
    total_commit_count: int = Field(default=0, description="Total number of commits")
    commits_last_month: int = Field(
        default=0, description="Number of commits in the last month"
    )
    commits_last_year: int = Field(
        default=0, description="Number of commits in the last year"
    )
    branch_count: int = Field(default=0, description="Number of branches")
    active_branches: list[str] = Field(default_factory=list, description="Active branches")
    contributor_count: int = Field(default=0, description="Number of contributors")
    top_contributors: list[str] = Field(default_factory=list, description="Top contributors")
    tags: list[str] = Field(default_factory=list, description="Tags")
    last_commit_date: datetime = Field(
        default=datetime.min.replace(tzinfo=timezone.utc), description="Date of the last commit"
    )
    language_breakdown: dict[str, float] = Field(
        default_factory=dict, description="Percentage breakdown of languages"
    )
    complexity_metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Complexity metrics per file"
    )
    dependencies: list["RepositoryDependency"] = Field(
        default_factory=list, description="Dependencies per language"
    )
    code_churn: "CodeChurn" = Field(
        default_factory=lambda: CodeChurn(lines_added=0, lines_deleted=0),
        description="Code churn metrics (lines added/deleted)"
    )

    # Ensure datetimes and nested models serialize safely for HTTP
    @field_serializer("last_commit_date")
    def _serialize_last_commit_date(self, v):
        # ISO 8601 string
        return v.isoformat() if v is not None else None



class RepositoryPointCloud(BaseModel):
    points: list[list[float]] = Field(
        description="The points in the point cloud.",
        default_factory=list,
    )
    metadata: list[dict] = Field(
        description="The metadata for the points.",
        default_factory=list,
    )

# ---------------------------------------------------------------------------
# Join table between VMRs and Repositories
# ---------------------------------------------------------------------------

class VMRRepositoryLink(sqlm.SQLModel, table=True):
    __tablename__ = "vmr_repository_link"

    vmr_id: uuid.UUID | None = sqlm.Field(foreign_key="vmrs.id", primary_key=True)
    repo_id: uuid.UUID | None = sqlm.Field(foreign_key="repositories.id", primary_key=True)

# ---------------------------------------------------------------------------
# Repository table (was pure Pydantic)
# ---------------------------------------------------------------------------

class Repository(sqlm.SQLModel, table=True):
    """
    A repository is a collection of code and related assets.
    A repository can be a member of many VMRs. So, VMRs reference repositories
    but not vice versa.
    It can also represent a "candidate" dependency that is not a direct member
    of a VMR but is a dependency of another repository.
    """

    __tablename__ = "repositories"

    # Unique constraint on origin_url, branch, and commit for non-candidate repositories
    __table_args__ = (
        sqlm.UniqueConstraint("origin_url", "branch", "commit", name="uq_repo_version"),
    )

    # Primary key – random UUID generated server-side.  This keeps the
    # relational schema simple and avoids foreign-key type mismatches.
    id: RepoId = sqlm.Field(
        default_factory=uuid.uuid4,
        primary_key=True,
        description="Unique identifier for the repository (UUID)",
    )

    # Deterministic fingerprint that uniquely identifies the logical repo
    # version (origin-URL, branch, commit).  Useful for de-duplication and
    # content-addressable caching.  The value is a fixed-length, URL-safe
    # 32-character string derived from SHA-256.
    fingerprint: str | None = sqlm.Field(
        default=None,
        sa_column=sqlm.Column(String(32), unique=True, nullable=False, index=True),
        description="Deterministic fingerprint of the repository (SHA-256 hash of origin/branch/commit)",
    )

    name: str = sqlm.Field(
        description="Name of the repository derived from its URL.",
        default=None,
    )
    description: str | None = sqlm.Field(
        description="Description of the repository", default=None
    )
    enabled: bool = sqlm.Field(
        description="Whether the repository is enabled", default=True
    )
    origin_url: str = sqlm.Field(
        description="URL of the repository", index=True, nullable=False
    )
    branch: str | None = sqlm.Field(description="Branch of the repository", default=None)
    commit: str | None = sqlm.Field(
        description="Commit of the repository", default=None
    )
    access_rights: str | None = sqlm.Field(
        description="Access rights of the repository", default=None
    )
    file_system_path: str | None = sqlm.Field(
        description="File system path of the repository", default=None
    )
    size_mb: float | None = sqlm.Field(
        description="Size of the repository in MB. Used for load balancing among tasks.",
        default=None,
    )
    polymathera_config: dict[str, Any] | None = sqlm.Field(
        sa_column=sqlm.Column(JSON),
        description="Polymathera configuration for the repository",
        default=None,
    )
    tags: list[str] = sqlm.Field(
        default_factory=list,
        sa_column=sqlm.Column(JSON),
        description="Tags of the repository",
    )
    badges: list[str] = sqlm.Field(
        default_factory=list,
        sa_column=sqlm.Column(JSON),
        description="Badges of the repository",
    )
    # insights: CodeInsights = sqlm.Field(default_factory=CodeInsights, sa_column=sqlm.Column(JSON)) # TODO: Remove this. Insights are stored at the VMR level and can span multiple repos.
    certificates: list[str] = sqlm.Field(
        default_factory=list, sa_column=sqlm.Column(JSON)
    )
    processing_costs: ProcessingCosts | None = sqlm.Field(
        default=None,
        sa_column=sqlm.Column(PydanticType(ProcessingCosts)),
    )
    stats: RepositoryStats | None = sqlm.Field(
        default=None,
        sa_column=sqlm.Column(PydanticType(RepositoryStats)),
    )
    is_direct_external_dependency: bool = sqlm.Field(
        default=False,
        index=True,
        description="True if this repo is an external dependency, not a direct VMR member. This is used to distinguish between direct and indirect dependencies.",
    )
    point_cloud: RepositoryPointCloud | None = sqlm.Field(
        default=None,
        sa_column=sqlm.Column(PydanticType(RepositoryPointCloud)),
    )

    # Many-to-many relationship back to VMRs (for regular member repositories)
    vmrs: list["VirtualMonorepo"] = sqlm.Relationship(
        back_populates="repositories",
        link_model=VMRRepositoryLink,
        sa_relationship_kwargs={"cascade": "all, delete"},
    )

    # Many-to-many relationship back to VMRs (for candidate/external dependency repositories)
    tentative_vmrs: list["VirtualMonorepo"] = sqlm.Relationship(
        back_populates="candidate_repositories",
        link_model=VMRRepositoryLink,
        sa_relationship_kwargs={"cascade": "all, delete", "overlaps": "vmrs"},
    )

    # The list of dependencies FOR this repository.
    # `RepositoryDependency` is the association object.
    dependencies: list["RepositoryDependency"] = sqlm.Relationship(
        back_populates="dependent_repo",
        sa_relationship_kwargs={
            "foreign_keys": "[RepositoryDependency.dependent_repo_id]",
            "cascade": "all, delete-orphan",
        },
    )

    # The list of repositories that depend ON this repository.
    dependents: list["RepositoryDependency"] = sqlm.Relationship(
        back_populates="dependency_repo",
        sa_relationship_kwargs={
            "foreign_keys": "[RepositoryDependency.dependency_repo_id]",
        },
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("origin_url", mode="before")
    @classmethod
    def _validate_origin_url(cls, v: Any) -> str:
        """Validate that *origin_url* is a well-formed HTTP (or HTTPS) URL but
        always store it as a plain string so that client code continues to
        treat it as such.

        Using :class:`pydantic.HttpUrl` here directly would yield an
        ``AnyUrl`` object at runtime and a ``HttpUrl`` type at static-type
        level, which breaks code that expects a ``str``.  We therefore run
        the validation manually and return the canonical string
        representation.
        """
        try:
            # TypeAdapter is the recommended way to run ad-hoc validation in
            # Pydantic v2.  We ignore the return value – we only care that the
            # validation succeeds – and then return the original string.
            TypeAdapter(HttpUrl).validate_python(v)  # type: ignore[arg-type]
        except Exception as exc:
            raise ValueError(f"Invalid origin_url '{v}': {exc}") from exc

        return str(v)

    # ------------------------------------------------------------------

    def __init__(self, **data: Any):
        # Ensure the deterministic fingerprint is always present.  Compute it
        # *before* calling super so validators that rely on it can run.
        fp: str | None = data.get("fingerprint")
        if fp is None:
            try:
                fp = get_repo_fingerprint(data)  # 32-char SHA-256/URL-safe
            except Exception:
                # Fallback – will be recomputed after super when all fields are set
                fp = None
            else:
                data["fingerprint"] = fp

        super().__init__(**data)

        # If fingerprint is still missing (e.g., because origin_url etc. came
        # from default values applied in super), compute it now.
        if self.fingerprint is None:
            self.fingerprint = get_repo_fingerprint(
                {
                    "origin_url": self.origin_url,
                    "branch": self.branch,
                    "commit": self.commit,
                }
            )

        # Derive a human-friendly name if not provided.
        if not self.name:
            self.name = get_repo_name_from_origin_url(self.origin_url)

    async def load_polymathera_config(self):
        config_path = Path(self.file_system_path) / ".polymathera"
        if config_path.exists():
            with config_path.open("r") as f:
                self.polymathera_config = json.load(f)
        else:
            self.polymathera_config = {}

        preload_aspects_default = [
            PredefinedCodeAspects.BUILD_SYSTEM,
            PredefinedCodeAspects.TESTING,
            PredefinedCodeAspects.DOCUMENTATION,
        ]
        self.polymathera_config.setdefault("preload_aspects", preload_aspects_default)

    def get_polymathera_ignore(self) -> list[Path]:
        return self.polymathera_config.get("polymathera_ignore", [])

    def get_preload_aspects(self) -> list[PredefinedCodeAspects]:
        return self.polymathera_config.get("preload_aspects", [])

    def get_llm_code_analysis_descriptions(
        self
    ) -> list[LLMStaticCodeAnalysisDescription]:
        return self.polymathera_config.get("code_analysis_descriptions", []) if self.polymathera_config else []

    def get_code_files(self) -> list[Path]:
        polymathera_ignore = self.get_polymathera_ignore()
        code_files = []
        for root, dirs, files in os.walk(self.file_system_path):
            for file in files:
                if file.endswith(".polymathera"):
                    continue
                file_path = Path(root) / file
                if polymathera_ignore and file_path in polymathera_ignore:
                    continue
                code_files.append(file_path)
        return code_files

    def get_code_file_contents(self) -> dict[Path, str]:
        """
        # Repositories with Code Generation Facilities
        Such repos contain code generation scripts (e.g., CMake or C++ macros).
        **The code slicing model needs to be able to reason about the code generation process**.
        Alternatively, code generation needs to be run before slicing starts.
        > If the slicing model is causal (consumes code blocks one at a time),
          autogenerated code (e.g., preprocessor macro expansion) needs to take place before
          passing the code base to the slicing model.
        > What about **JIT compilation**? For example, PyTorch can dynamcally compile C++/CUDA
          code and load it into the Python process.
        """
        # TODO: Placeholder for code retrieval logic
        # Walk the file system and read the code files
        # Skip the files that are ignored by the .polymathera file
        code_files = self.get_code_files()
        code_file_contents = {}
        for file_path in code_files:
            with file_path.open("r") as f:
                code_file_contents[file_path] = f.read()
        return code_file_contents

    def load_previous_session(self):
        """Loads data from a previous Polymathera session into this VMR."""
        pass


class RepositoryDependency(sqlm.SQLModel, table=True):
    __tablename__ = "repo_dependencies"

    # Ensure JSON-friendly serialization for HTTP/Serve responses
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    dependent_repo_id: uuid.UUID = sqlm.Field(
        foreign_key="repositories.id", primary_key=True
    )
    dependency_repo_id: uuid.UUID = sqlm.Field(
        foreign_key="repositories.id", primary_key=True
    )

    dependent_repo: "Repository" = sqlm.Relationship(
        back_populates="dependencies",
        sa_relationship_kwargs={
            "foreign_keys": "[RepositoryDependency.dependent_repo_id]"
        },
    )
    dependency_repo: "Repository" = sqlm.Relationship(
        back_populates="dependents",
        sa_relationship_kwargs={
            "foreign_keys": "[RepositoryDependency.dependency_repo_id]"
        },
    )

    # Metadata about the dependency itself
    name: str = sqlm.Field(
        sa_column=sqlm.Column(String(100), nullable=False),
        description="Name of the dependency",
    )
    version: str = sqlm.Field(
        sa_column=sqlm.Column(String(100), nullable=False),
        description="Version of the dependency",
    )
    type: str = sqlm.Field(
        sa_column=sqlm.Column(String(100), nullable=False),
        description="Type of the dependency (e.g., 'library', 'framework', 'tool', etc.)",
    )
    description: str = sqlm.Field(
        sa_column=sqlm.Column(String(100), nullable=False),
        description="Description of the dependency",
    )
    license: str = sqlm.Field(
        sa_column=sqlm.Column(String(100), nullable=False),
        description="License of the dependency",
    )
    ecosystem: str = sqlm.Field(
        sa_column=sqlm.Column(String(100), nullable=False),
        description="Ecosystem of the dependency: 'python', 'javascript', 'java', etc.'",
    )
    source: str = sqlm.Field(
        sa_column=sqlm.Column(String(100), nullable=False),
        description="Source of the dependency: 'requirements.txt', 'package.json', etc.'",
    )
    origin_url: HttpUrl = sqlm.Field(
        sa_column=sqlm.Column(String(255), nullable=True),
        description="Origin URL of the dependency, if known.",
    )

    # JSON serialization helpers
    @field_serializer("dependent_repo_id", "dependency_repo_id")
    def _serialize_uuid(self, v):
        return str(v) if v is not None else None

    @field_serializer("origin_url")
    def _serialize_origin_url(self, v):
        return str(v) if v is not None else None

    def __eq__(self, other: Any) -> bool:
        """
        Two dependencies are considered equal if they have the same origin_url, version,
        and dependent_repo_id. Other fields might differ (e.g., description, source).
        """
        if not isinstance(other, RepositoryDependency):
            raise NotImplementedError(
                f"Cannot compare RepositoryDependency with {type(other)}"
            )
        return (
            self.origin_url == other.origin_url
            and self.version == other.version
            and self.dependent_repo_id == other.dependent_repo_id
        )
        # return (
        #     self.dependent_repo_id == other.dependent_repo_id
        #     and self.dependency_repo_id == other.dependency_repo_id
        #     and self.name == other.name
        #     and self.version == other.version
        #     and self.ecosystem == other.ecosystem
        # )

    def __hash__(self) -> int:
        """
        Hash should be consistent with __eq__ and use the same fields.
        """
        return hash((self.origin_url, self.version, self.dependent_repo_id))
        # return hash(
        #     (
        #         self.dependent_repo_id,
        #         self.dependency_repo_id,
        #         self.name,
        #         self.version,
        #         self.ecosystem,
        #     )
        # )

# ---------------------------------------------------------------------------
# Database-backed VirtualMonorepo – shared Pydantic + SQLAlchemy definition
# VirtualMonorepo with relational repos, plus dict-facade property
# ---------------------------------------------------------------------------

class VirtualMonorepo(sqlm.SQLModel, table=True):
    """
    A virtual monorepo (VMR) is a collection of repositories.
    It can be user-created or polymathera-created.
    Polymathera creates VMRs dynamically during execution to break down
    complex tasks into smaller, more manageable subtasks over subsets of
    large VMRs.
    """

    __tablename__ = "vmrs"

    id: UUID4 | None = sqlm.Field(default_factory=uuid.uuid4, primary_key=True, description="Unique identifier for the VMR")
    name: str = sqlm.Field(sa_column=sqlm.Column(String(100), nullable=False), description="Name of the virtual monorepo.")
    # repositories: dict[UUID4, Repository] = sqlm.Field(  # type: ignore[valid-type]
    #     default_factory=dict,
    #     sa_column=sqlm.Column(JSON, nullable=False, default={}),
    # )
    # relationship holding Repository rows (regular members)
    repositories: list["Repository"] = sqlm.Relationship(
        back_populates="vmrs",
        link_model=VMRRepositoryLink,
        sa_relationship_kwargs={"cascade": "all, delete", "overlaps": "tentative_vmrs"},
    )

    status: str = sqlm.Field(default="", sa_column=sqlm.Column(String(100), nullable=False))

    # relationship holding CandidateRepository rows (external dependencies)
    candidate_repositories: list["Repository"] = sqlm.Relationship(
        back_populates="tentative_vmrs",
        link_model=VMRRepositoryLink,
        sa_relationship_kwargs={"cascade": "all, delete", "overlaps": "repositories,vmrs"},
    )
    # candidate_repositories: dict[HttpUrl, set[RepositoryDependency]] = sqlm.Field(  # type: ignore[valid-type]
    #     default_factory=lambda: defaultdict(set),
    #     sa_column=sqlm.Column(JSON, nullable=False),
    #     description="Candidate repositories of the virtual monorepo that can be used to grow the VMR.",
    # )
    # seed repos – optional simple JSON list of repository ids
    seed_repository_ids: list[RepoId] = sqlm.Field(
        default_factory=list,
        sa_column=sqlm.Column(JSON, nullable=False, default=[]),
        description="IDs of seed repositories used to construct VMR graph.",
    )
    intrinsic_goals: list[GoalGroup] = sqlm.Field(  # type: ignore[valid-type]
        default_factory=get_default_intrinsic_goals,
        sa_column=sqlm.Column(PydanticType(GoalGroup, is_list=True), nullable=False),
    )
    exploration_constraints: ExplorationConstraints = sqlm.Field(
        default_factory=get_default_exploration_constraints,
        sa_column=sqlm.Column(PydanticType(ExplorationConstraints), nullable=False),
    )
    tags: list[str] = sqlm.Field(default_factory=list, sa_column=sqlm.Column(JSON, nullable=False))
    description: str = sqlm.Field(default="", sa_column=sqlm.Column(String(1000), nullable=False))
    stars: int = sqlm.Field(default=0, sa_column=sqlm.Column(Integer, nullable=False))
    code_analysis_descriptions: list[LLMStaticCodeAnalysisDescription] = sqlm.Field(  # type: ignore[valid-type]
        default_factory=list,
        sa_column=sqlm.Column(PydanticType(LLMStaticCodeAnalysisDescription, is_list=True), nullable=False),
    )
    chat_channel_name: str | None = sqlm.Field(
        default=None, sa_column=sqlm.Column(String(100), nullable=True)
    )
    chat_channel_id: str | None = sqlm.Field(default=None, sa_column=sqlm.Column(String(36), nullable=True))
    chat_thread_id: str | None = sqlm.Field(default=None, sa_column=sqlm.Column(String(36), nullable=True))
    # insights: list[KnowledgeItem] = Field(default_factory=list) # TODO: Too much memory overhead
    action_plan: ActionPlan | None = sqlm.Field(
        default=None,
        sa_column=sqlm.Column(PydanticType(ActionPlan), nullable=True),
    )

    created_at: datetime = sqlm.Field(default_factory=lambda: datetime.now(timezone.utc), sa_column=sqlm.Column(TIMESTAMP(timezone=True), nullable=False))
    updated_at: datetime = sqlm.Field(default_factory=lambda: datetime.now(timezone.utc), sa_column=sqlm.Column(TIMESTAMP(timezone=True), nullable=False))

    def get_repo_dependencies(self, repo_id: RepoId) -> set[RepositoryDependency]:
        # Find the repository by ID in the list
        for repo in self.repositories:
            if repo.id == repo_id:
                return set(repo.dependencies) # repo.stats.dependencies
        return set()

    def get_repo_ids_by_origin_url(self, origin_url: str) -> set[RepoId]:
        return {
            repo.id
            for repo in self.repositories
            if repo.origin_url == origin_url
        }

    def filter_dependent_repos(
        self,
        repo_id: RepoId,
        condition: Callable[["Repository"], bool],
        transitive: bool = False,
    ) -> set[RepoId]:
        visited = set()
        dependent_ids = set()

        def visit_dependents(current_id: RepoId):
            if current_id in visited:
                return
            visited.add(current_id)

            # Find direct dependents that match the condition
            direct_dependents = set(
                repo.id
                for repo in self.repositories
                if condition(repo) and current_id in [dep.dependent_repo_id for dep in repo.dependencies] # repo.stats.dependencies
            )
            dependent_ids.update(direct_dependents)

            # Recursively visit transitive dependents
            if transitive:
                for dependent in direct_dependents:
                    visit_dependents(dependent)

        visit_dependents(repo_id)
        return dependent_ids

    def add_repository(self, repo_instance: Repository) -> bool:
        """Add a repository to this VMR if not already present.

        Returns:
            bool: True if the repository was added, False if it was already present (deduplicated)
        """
        # Check if repository is already in ANY of the relationship lists (by fingerprint for logical deduplication)
        all_existing_fingerprints = set()

        # Collect fingerprints from both repositories and candidate_repositories
        all_existing_fingerprints.update(repo.fingerprint for repo in self.repositories)
        all_existing_fingerprints.update(repo.fingerprint for repo in self.candidate_repositories)

        if repo_instance.fingerprint not in all_existing_fingerprints:
            # Add to the appropriate list based on is_direct_external_dependency
            if repo_instance.is_direct_external_dependency:
                self.candidate_repositories.append(repo_instance)
            else:
                self.repositories.append(repo_instance)
            return True
        else:
            logger.warning(f"Repository {repo_instance.fingerprint} already exists in VMR {self.id}")
        return False

    def get_repository_by_id(self, repo_id: UUID4) -> "Repository | None":
        # Search through the list of repositories
        repos = [repo for repo in self.repositories if str(repo.id) == str(repo_id)]
        if not repos:
            raise ValueError(f"Repository {repo_id} not found in VMR {self.id}: {json.dumps([(str(repo.id), repo.origin_url) for repo in self.repositories], indent=2)}")
        if len(repos) > 1:
            raise ValueError(f"Multiple repositories found for ID {repo_id} in VMR {self.id}")
        return repos[0]

    def clear_repositories(self) -> None:
        self.repositories.clear()

    def get_llm_code_analysis_descriptions(
        self
    ) -> list[LLMStaticCodeAnalysisDescription]:
        analysis_descriptions = self.code_analysis_descriptions
        for repo in self.repositories:
            if repo.polymathera_config:
                analysis_descriptions.extend(
                    repo.polymathera_config.get("code_analysis_descriptions", [])
                )
        return analysis_descriptions

    def clear_direct_external_dependencies(self) -> None:
        # This method is now a no-op or should be removed,
        # as candidates are just part of the main repositories list.
        # For now, we'll just filter them out.
        self.repositories = [
            repo for repo in self.repositories if not repo.is_direct_external_dependency
        ]


class UniversalCodeRepository(BaseModel):
    """
    A universal code repository (UCR) is a knowledge graph
    factored out of all existing software repositories.
    To enable working with multi-language repos, the UCR
    needs to be language-agnostic. The UCR will likely be
    a partially order set (directed acyclic graph) of code aspects
    described in a meta-language that is independent of any programming
    language and that offers a highly concise and expressive way to
    describe code itself in terms of woven aspects.
    """

    aspect_registry: list[str] = []



