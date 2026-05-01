"""``DataCurationAgent`` + ``DataCurationCapability``.

Owns dataset operations: registration, content-hash versioning, and
lineage tracking. The capability is colony-generic — it does not pull
in DVC, MLflow, or any specific dataset-versioning system; instead it
carries a small, typed in-memory store of dataset metadata + lineage
edges that deployments either use directly or back with their own
persistence.

(The earlier federated-learning round-coordination surface was
deleted: it was a speculative Protocol with a 25-line in-process
default and no real consumers. Real federated-learning deployments
plug FedML / Flower / NVFlare in directly; they don't need a colony
shim layer.)
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from overrides import override
from pydantic import BaseModel, ConfigDict, Field

from ..base import Agent, AgentCapability
from ..models import AgentSuspensionState
from ..patterns.actions import action_executor


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DatasetVersion(BaseModel):
    """One version of a dataset."""

    model_config = ConfigDict(frozen=True)

    version_id: str
    dataset_id: str
    parent_version_id: str | None = None
    """For derived versions; None for the original registration."""

    transform_description: str = ""
    """Free-form note on how this version was derived from its parent."""

    schema_summary: str = ""
    """One-line schema summary (free-form: e.g.,
    ``"columns: [doi, title, abstract, year]"``)."""

    content_hash: str
    """SHA-256 (or other) content hash. The capability supports any
    hex-encoded digest."""

    storage_uri: str = ""
    """Where the data lives (``"s3://...","file:///...","hdf5://..."``).
    Free-form; the capability only stores it."""

    size_bytes: int | None = Field(default=None, ge=0)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    created_by: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class DatasetLineageEdge(BaseModel):
    """An edge in the dataset DAG.

    For audit + reproducibility: a derived version has one or more
    parents (the standard is one, but a join across two source
    datasets has two). The runner enforces the DAG (no cycles).
    """

    model_config = ConfigDict(frozen=True)

    child_version_id: str
    parent_version_id: str
    relation: str = "derived_from"
    """Free-form: ``"derived_from"``, ``"joined_with"``,
    ``"filtered_from"``, ``"resampled_from"``."""


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


class DataCurationError(RuntimeError):
    pass


class UnknownDatasetVersionError(DataCurationError):
    pass


class ContentHashMismatchError(DataCurationError):
    pass


class DataCurationCapability(AgentCapability):
    """Dataset registration + content-hash versioning + lineage."""

    def __init__(
        self,
        agent: Agent | None = None,
        scope_id: str | None = None,
        *,
        capability_key: str | None = None,
        app_name: str | None = None,
    ) -> None:
        super().__init__(
            agent=agent,
            scope_id=scope_id,
            input_patterns=[],
            capability_key=capability_key,
            app_name=app_name,
        )
        self._versions: dict[str, DatasetVersion] = {}
        self._by_dataset: dict[str, list[str]] = {}
        self._lineage: list[DatasetLineageEdge] = []

    def get_capability_tags(self) -> frozenset[str]:
        return frozenset({"data", "versioning"})

    # ---- Registration / versioning -------------------------------------

    @action_executor(
        planning_summary=(
            "Register a new dataset (root version). Returns the version_id."
        ),
    )
    async def register_dataset(
        self,
        dataset_id: str,
        *,
        content_hash: str,
        storage_uri: str = "",
        schema_summary: str = "",
        size_bytes: int | None = None,
        created_by: str = "",
        extra: Mapping[str, Any] | None = None,
    ) -> DatasetVersion:
        version_id = self._next_version_id(dataset_id, parent=None)
        version = DatasetVersion(
            version_id=version_id,
            dataset_id=dataset_id,
            parent_version_id=None,
            schema_summary=schema_summary,
            content_hash=content_hash,
            storage_uri=storage_uri,
            size_bytes=size_bytes,
            created_by=created_by,
            extra=dict(extra or {}),
        )
        self._versions[version_id] = version
        self._by_dataset.setdefault(dataset_id, []).append(version_id)
        return version

    @action_executor(
        planning_summary=(
            "Create a new version derived from an existing parent version."
        ),
    )
    async def version_dataset(
        self,
        parent_version_id: str,
        *,
        content_hash: str,
        transform_description: str,
        storage_uri: str = "",
        schema_summary: str = "",
        size_bytes: int | None = None,
        created_by: str = "",
        relation: str = "derived_from",
        extra: Mapping[str, Any] | None = None,
    ) -> DatasetVersion:
        parent = self._versions.get(parent_version_id)
        if parent is None:
            raise UnknownDatasetVersionError(
                f"No version with id {parent_version_id!r}.",
            )
        version_id = self._next_version_id(parent.dataset_id, parent=parent_version_id)
        version = DatasetVersion(
            version_id=version_id,
            dataset_id=parent.dataset_id,
            parent_version_id=parent_version_id,
            transform_description=transform_description,
            schema_summary=schema_summary or parent.schema_summary,
            content_hash=content_hash,
            storage_uri=storage_uri,
            size_bytes=size_bytes,
            created_by=created_by,
            extra=dict(extra or {}),
        )
        self._versions[version_id] = version
        self._by_dataset[parent.dataset_id].append(version_id)
        self._lineage.append(
            DatasetLineageEdge(
                child_version_id=version_id,
                parent_version_id=parent_version_id,
                relation=relation,
            )
        )
        return version

    @action_executor(planning_summary="List all versions of a dataset.")
    async def list_versions(self, dataset_id: str) -> list[DatasetVersion]:
        ids = self._by_dataset.get(dataset_id, ())
        return [self._versions[i] for i in ids]

    @action_executor(planning_summary="Fetch one dataset version by id.")
    async def get_version(self, version_id: str) -> DatasetVersion:
        v = self._versions.get(version_id)
        if v is None:
            raise UnknownDatasetVersionError(
                f"No version with id {version_id!r}.",
            )
        return v

    @action_executor(planning_summary="List a version's lineage edges.")
    async def list_lineage(self, version_id: str) -> list[DatasetLineageEdge]:
        return [
            edge for edge in self._lineage
            if edge.child_version_id == version_id
            or edge.parent_version_id == version_id
        ]

    @action_executor(
        planning_summary=(
            "Re-hash a local file and verify against the recorded content_hash."
        ),
    )
    async def verify_content_hash(
        self,
        version_id: str,
        *,
        path: str,
        algorithm: str = "sha256",
    ) -> dict[str, Any]:
        version = await self.get_version(version_id)
        digest = self._hash_file(Path(path), algorithm=algorithm)
        ok = digest == version.content_hash
        return {
            "ok": ok,
            "version_id": version_id,
            "expected": version.content_hash,
            "observed": digest,
            "algorithm": algorithm,
        }

    # ---- Suspension hooks ---------------------------------------------

    @override
    async def serialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> AgentSuspensionState:
        if self._versions or self._lineage:
            state.custom_data["data_curation_capability"] = {
                "versions": [
                    v.model_dump(mode="json") for v in self._versions.values()
                ],
                "lineage": [
                    e.model_dump(mode="json") for e in self._lineage
                ],
            }
        return state

    @override
    async def deserialize_suspension_state(
        self, state: AgentSuspensionState
    ) -> None:
        payload = state.custom_data.get("data_curation_capability") or {}
        for raw in payload.get("versions") or ():
            try:
                v = DatasetVersion.model_validate(raw)
            except Exception:  # noqa: BLE001
                continue
            self._versions[v.version_id] = v
            self._by_dataset.setdefault(v.dataset_id, []).append(v.version_id)
        for raw in payload.get("lineage") or ():
            try:
                e = DatasetLineageEdge.model_validate(raw)
            except Exception:  # noqa: BLE001
                continue
            self._lineage.append(e)

    # ---- Internals ----------------------------------------------------

    def _next_version_id(self, dataset_id: str, *, parent: str | None) -> str:
        existing = self._by_dataset.get(dataset_id, [])
        idx = len(existing) + 1
        return f"{dataset_id}@v{idx}"

    @staticmethod
    def _hash_file(path: Path, *, algorithm: str = "sha256") -> str:
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist or is not a file.")
        if algorithm not in hashlib.algorithms_available:
            raise ValueError(
                f"Unsupported hash algorithm {algorithm!r}; "
                f"available: {sorted(hashlib.algorithms_available)}",
            )
        h = hashlib.new(algorithm)
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# DataCurationAgent
# ---------------------------------------------------------------------------


class DataCurationAgent(Agent):
    """Generic dataset-curation role (master §3.5)."""

    agent_type: str = (
        "polymathera.colony.agents.roles.data_curation.DataCurationAgent"
    )


__all__ = (
    "DataCurationAgent",
    "DataCurationCapability",
    "DataCurationError",
    "UnknownDatasetVersionError",
    "ContentHashMismatchError",
    "DatasetVersion",
    "DatasetLineageEdge",
)
