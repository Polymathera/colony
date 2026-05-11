"""Operator-YAML schema and resolver for the L1-G image-extension mechanism.

Per the alignment plan §2.1 L1-G.3, the cluster YAML carries three optional
fields that compose at container start (mirroring Ray's
``docker.image`` / ``setup_commands`` layering):

- ``cluster.docker.image`` — base image to run (defaults to the published
  ``polymathera/colony:<version>``; the consumer chooses the default).
- ``cluster.extensions.packages`` — Python packages pip-installed at
  container start against a persistent overlay volume (cache-keyed by hash).
- ``cluster.setup_commands`` / ``head_setup_commands`` /
  ``worker_setup_commands`` — arbitrary shell run after package install.
  Use only when ``extensions.packages`` cannot.

This module is the source-of-truth schema; ``colony-env up`` reads the YAML,
validates with these models, exports env vars to docker-compose, and writes
a container-start hook that installs the resolved package list and runs the
setup commands.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class DockerImageConfig(BaseModel):
    """``cluster.docker`` block. Optional override of the runtime image.

    ``None`` lets the consumer (``colony-env``) pick the default; the project
    intentionally does not bake a default image tag here so the CLI stays the
    single owner of "what does no override mean today".
    """

    model_config = ConfigDict(extra="forbid")

    image: str | None = None


class PackageSpec(BaseModel):
    """One entry in ``cluster.extensions.packages``.

    Two source variants:

    - ``source: "version"`` (default) — installs ``<name>[<extras>]<op><version>``
      using pip-compatible operators (``==``, ``>=``, ``~=``, …). A bare
      ``version`` is treated as ``==``. Poetry-style ``^`` / ``~``-with-caret
      operators are NOT translated; pip will reject them at install time.
    - ``source: "path"`` — installs from a local directory. The path is
      resolved relative to the YAML file's directory if not absolute. Mirrors
      how Colony itself is installed in the base image.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    source: Literal["version", "path"] = "version"
    version: str | None = None
    path: str | None = None
    extras: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_source(self) -> PackageSpec:
        if self.source == "path":
            if not self.path:
                raise ValueError(f"package {self.name!r}: source='path' requires 'path'")
            if self.version is not None:
                raise ValueError(
                    f"package {self.name!r}: 'version' is invalid for source='path'",
                )
        else:  # source == "version"
            if self.path is not None:
                raise ValueError(
                    f"package {self.name!r}: 'path' is invalid for source='version'",
                )
        return self


class ExtensionsConfig(BaseModel):
    """``cluster.extensions`` block."""

    model_config = ConfigDict(extra="forbid")

    packages: list[PackageSpec] = Field(default_factory=list)


class ClusterRuntimeConfig(BaseModel):
    """Schema for the L1-G operator-YAML fields under ``cluster.*``.

    Other keys under ``cluster:`` (LLM cluster config: ``app_name``,
    ``remote_deployments``, …) are owned by other parsers and never reach
    this model — :func:`load_cluster_runtime_config` extracts only the L1-G
    keys before validation.
    """

    model_config = ConfigDict(extra="forbid")

    docker: DockerImageConfig = Field(default_factory=DockerImageConfig)
    extensions: ExtensionsConfig = Field(default_factory=ExtensionsConfig)
    setup_commands: list[str] = Field(default_factory=list)
    head_setup_commands: list[str] = Field(default_factory=list)
    worker_setup_commands: list[str] = Field(default_factory=list)


def load_cluster_runtime_config(yaml_path: str | Path) -> ClusterRuntimeConfig:
    """Load and validate the L1-G fields from an operator YAML file.

    Reads only the keys this module owns (``cluster.docker``,
    ``cluster.extensions``, ``cluster.{,head_,worker_}setup_commands``).
    Other ``cluster.*`` fields (LLM cluster config etc.) are ignored.

    Returns a default-everything ``ClusterRuntimeConfig`` if the file has no
    ``cluster:`` block at all — the caller decides whether that's an error.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    cluster_raw = raw.get("cluster") or {}
    return ClusterRuntimeConfig.model_validate({
        "docker": cluster_raw.get("docker") or {},
        "extensions": cluster_raw.get("extensions") or {},
        "setup_commands": cluster_raw.get("setup_commands") or [],
        "head_setup_commands": cluster_raw.get("head_setup_commands") or [],
        "worker_setup_commands": cluster_raw.get("worker_setup_commands") or [],
    })


def dedupe_packages(packages: list[PackageSpec]) -> list[PackageSpec]:
    """Drop exact duplicates; raise on conflicting entries with the same name.

    Two entries are an exact duplicate if every field matches. Two entries
    sharing only ``name`` (different version, source, extras) are a conflict
    — the operator must pick one.
    """
    seen: dict[str, PackageSpec] = {}
    for pkg in packages:
        existing = seen.get(pkg.name)
        if existing is None:
            seen[pkg.name] = pkg
            continue
        if existing.model_dump() == pkg.model_dump():
            continue
        raise ValueError(
            f"package {pkg.name!r} declared twice with conflicting specs; "
            f"first: {existing.model_dump()}, second: {pkg.model_dump()}",
        )
    return list(seen.values())


def resolve_pip_args(
    packages: list[PackageSpec], *, yaml_dir: Path | None = None,
) -> list[str]:
    """Convert ``PackageSpec``s into the args that ``pip install`` consumes.

    Path-source entries are normalized to absolute paths relative to
    ``yaml_dir`` (the directory containing the operator YAML). Pass
    ``yaml_dir=None`` to require absolute paths.
    """
    args: list[str] = []
    for pkg in packages:
        extras = f"[{','.join(pkg.extras)}]" if pkg.extras else ""
        if pkg.source == "path":
            assert pkg.path is not None  # validated
            target = Path(pkg.path)
            if not target.is_absolute():
                if yaml_dir is None:
                    raise ValueError(
                        f"package {pkg.name!r} uses a relative path "
                        f"({pkg.path!r}) but no yaml_dir was supplied",
                    )
                target = (yaml_dir / target).resolve()
            args.append(f"{target}{extras}")
        else:
            if pkg.version is None:
                args.append(f"{pkg.name}{extras}")
            else:
                op = "" if pkg.version[0] in "<>=~!" else "=="
                args.append(f"{pkg.name}{extras}{op}{pkg.version}")
    return args


def resolved_hash(
    packages: list[PackageSpec],
    *,
    setup_commands: list[str] | None = None,
    head_setup_commands: list[str] | None = None,
    worker_setup_commands: list[str] | None = None,
) -> str:
    """Deterministic 16-char hex hash of the resolved runtime config.

    Used (a) as the cache key for the persistent overlay volume so identical
    package lists reuse the same pip-install layer, and (b) as the image tag
    for ``colony-env up --bake`` (``colony-local:<hash>``).

    Path-source ``PackageSpec``s are hashed as written (relative paths stay
    relative). Callers that need a *content-addressed* hash for path sources
    must resolve the path themselves and pass an absolute spec.
    """
    payload = json.dumps({
        "packages": [p.model_dump(mode="json") for p in packages],
        "setup_commands": setup_commands or [],
        "head_setup_commands": head_setup_commands or [],
        "worker_setup_commands": worker_setup_commands or [],
    }, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def lint_setup_commands(cfg: ClusterRuntimeConfig) -> list[str]:
    """Flag ``pip install`` calls in any of the setup_commands lists.

    Per Risk #5 in the alignment plan: ``setup_commands`` is imperative
    shell that re-runs every container start (cost multiplies on multi-
    node), bypasses Pydantic validation, and invites drift. Anything
    pip-installable belongs in ``cluster.extensions.packages`` where the
    overlay cache + hash-keyed install kick in.

    Returns one warning string per offending line. Empty list = clean.
    Call site decides severity (warn vs error).
    """
    warnings: list[str] = []
    sources = [
        ("setup_commands", cfg.setup_commands),
        ("head_setup_commands", cfg.head_setup_commands),
        ("worker_setup_commands", cfg.worker_setup_commands),
    ]
    for field, cmds in sources:
        for i, cmd in enumerate(cmds):
            # Tokenize on whitespace; check for ``pip install`` (or
            # ``pip3 install`` / ``python -m pip install``). Trim leading
            # ``sudo`` / ``env VAR=...`` / shell-builtin prefixes.
            tokens = cmd.split()
            for j in range(len(tokens) - 1):
                if tokens[j] in ("pip", "pip3") and tokens[j + 1] == "install":
                    warnings.append(
                        f"{field}[{i}]: looks like a pip install "
                        f"({cmd!r}); use extensions.packages instead",
                    )
                    break
                if (
                    tokens[j] in ("python", "python3", "python3.11")
                    and j + 3 < len(tokens)
                    and tokens[j + 1] == "-m"
                    and tokens[j + 2] == "pip"
                    and tokens[j + 3] == "install"
                ):
                    warnings.append(
                        f"{field}[{i}]: looks like a pip install "
                        f"({cmd!r}); use extensions.packages instead",
                    )
                    break
    return warnings


__all__ = (
    "ClusterRuntimeConfig",
    "DockerImageConfig",
    "ExtensionsConfig",
    "PackageSpec",
    "dedupe_packages",
    "lint_setup_commands",
    "load_cluster_runtime_config",
    "resolve_pip_args",
    "resolved_hash",
)
