"""Base type definitions with no dependencies."""

from typing import TypeAlias, Any
import hashlib
import base64
import ray
from pydantic import UUID4

# Basic type aliases
NodeId: TypeAlias = str
# Repository identifiers are now UUIDs in the relational model.
# Use Pydantic's UUID4 type alias so validation works seamlessly while
# still materializing to ``uuid.UUID`` instances at runtime.
RepoId: TypeAlias = UUID4
VmrId: TypeAlias = str
ShardId: TypeAlias = str
QueryId: TypeAlias = str
GroupId: TypeAlias = str
SinglePromptJobId: TypeAlias = str
MultiPromptJobId: TypeAlias = str
GitRepoKey: TypeAlias = str
LLMClientId: TypeAlias = str
DeploymentIdx: TypeAlias = int
ResidentContextLLMClientHandle: TypeAlias = ray.actor.ActorHandle
LLMClusterManagerHandle: TypeAlias = ray.actor.ActorHandle
QueryBlackboardHandle: TypeAlias = ray.actor.ActorHandle
GitRepoInferenceEngineHandle: TypeAlias = ray.actor.ActorHandle
GitRepoDeploymentManagerHandle: TypeAlias = ray.actor.ActorHandle


def get_repo_fingerprint(repo: dict[str, Any]) -> str:
    # Create a list of key-value pairs to hash
    properties = [
        ("origin_url", repo["origin_url"]),
        ("branch", repo.get("branch", "")),
        ("commit", repo.get("commit", "")),
        # Add any additional properties here in the future
    ]

    # Sort the properties to ensure consistent ordering
    properties.sort(key=lambda x: x[0])

    # Create a string representation of the properties
    repo_signature = "|".join(f"{k}:{v}" for k, v in properties)

    # Use SHA256 for hashing (you could use a shorter hash if preferred)
    hash_object = hashlib.sha256(repo_signature.encode())

    # Use base64 encoding to make the hash shorter while keeping it URL-safe
    return base64.urlsafe_b64encode(hash_object.digest()).decode()[:32]


def get_repo_name_from_origin_url(origin_url: str) -> str:
    # return Path(self.origin_url).stem
    return origin_url.split("/")[-1]
