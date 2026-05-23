"""Private helpers for ``SandboxedShellCapability``.

Kept out of the top-level ``capabilities`` namespace so the public
action surface (``SandboxedShellCapability``) is the only entry point
users import from.
"""

from .backend import (
    ContainerBackend,
    ContainerHandle,
    ContainerSpec,
    DockerCLIBackend,
    ExecResult,
    NoSuchContainer,
)
from .registry import (
    DockerImageRegistry,
    DockerImageSpec,
    ScriptSpec,
)

__all__ = [
    "ContainerBackend",
    "ContainerHandle",
    "ContainerSpec",
    "DockerCLIBackend",
    "ExecResult",
    "NoSuchContainer",
    "DockerImageRegistry",
    "DockerImageSpec",
    "ScriptSpec",
]
