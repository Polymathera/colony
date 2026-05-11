"""L1-G runtime artifact writer.

``colony-env up`` calls these helpers BEFORE invoking docker-compose to:

1. Resolve the operator YAML's ``cluster.docker`` / ``cluster.extensions`` /
   ``cluster.{,head_,worker_}setup_commands`` blocks into a
   ``cluster-runtime.json`` consumed by the container-start hook.
2. For ``source: path`` extensions, write a compose override file
   (``docker-compose.path-extensions.yml``) bind-mounting each host path
   into ``/mnt/path-extensions/<name>`` of every colony service. The
   ``pip_args`` in the runtime JSON reference these in-container paths,
   not the host paths.

Pure functions; no docker / async involvement so they're trivially testable.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .extensions import (
    ClusterRuntimeConfig,
    PackageSpec,
    dedupe_packages,
    load_cluster_runtime_config,
    resolve_pip_args,
    resolved_hash,
)


CONTAINER_PATH_EXTENSIONS_ROOT = "/mnt/path-extensions"
"""Path inside colony service containers where path-source extension
directories are bind-mounted."""


def _resolve_path_source_in_container(
    pkg: PackageSpec, *, yaml_dir: Path,
) -> tuple[Path, str]:
    """Translate a ``source: path`` :class:`PackageSpec` to a (host_path,
    container_path) pair.

    ``yaml_dir`` is the directory of the operator YAML; relative ``path``
    entries are resolved against it. The container path is
    ``/mnt/path-extensions/<package-name>``.
    """
    assert pkg.source == "path" and pkg.path is not None
    host = Path(pkg.path)
    if not host.is_absolute():
        host = (yaml_dir / host).resolve()
    if not host.exists():
        raise FileNotFoundError(
            f"package {pkg.name!r}: path {host} does not exist on host",
        )
    container = f"{CONTAINER_PATH_EXTENSIONS_ROOT}/{pkg.name}"
    return host, container


def _split_container_pip_args(
    packages: list[PackageSpec], *, yaml_dir: Path | None,
) -> tuple[list[str], list[str]]:
    """Like :func:`resolve_pip_args` but rewrites path sources to their
    in-container mount points and splits results into two lists by source
    type.

    Returns ``(version_args, path_args)``:

    - ``version_args`` — version-source specs, installed *with* dependency
      resolution so transitive deps land in the overlay.
    - ``path_args`` — path-source specs as in-container paths
      (``/mnt/path-extensions/<name>[<extras>]``), installed with
      ``--no-deps``. Path-source dev workflows assume the container's base
      image already supplies the dep tree (e.g., ``polymathera-colony`` is
      built into the base image; CPS's path-dep on it would otherwise
      trigger a redundant in-overlay reinstall and editable-install
      conflicts under ``pip install --target``).
    """
    version_args: list[str] = []
    path_args: list[str] = []
    for pkg in packages:
        extras = f"[{','.join(pkg.extras)}]" if pkg.extras else ""
        if pkg.source == "path":
            if yaml_dir is None and not Path(pkg.path or "").is_absolute():
                raise ValueError(
                    f"package {pkg.name!r}: relative path {pkg.path!r} "
                    f"requires yaml_dir",
                )
            container_path = f"{CONTAINER_PATH_EXTENSIONS_ROOT}/{pkg.name}"
            path_args.append(f"{container_path}{extras}")
        else:
            if pkg.version is None:
                version_args.append(f"{pkg.name}{extras}")
            else:
                op = "" if pkg.version[0] in "<>=~!" else "=="
                version_args.append(f"{pkg.name}{extras}{op}{pkg.version}")
    return version_args, path_args


def write_cluster_runtime(
    *,
    config_path: str | Path | None,
    runtime_dir: Path,
    bake_pip_inline: bool = False,
) -> ClusterRuntimeConfig:
    """Resolve the L1-G fields from the operator YAML and write
    ``runtime_dir/cluster-runtime.json``. The directory is created if missing.

    Always writes a file (a stub when ``config_path`` is None or has no L1-G
    fields) so the bind mount in ``docker-compose.yml`` always has a real
    file as its source — without it Docker would silently create the path
    as an empty directory and break containers on mount.

    Returns the parsed :class:`ClusterRuntimeConfig` for callers that need
    it (e.g., the bake builder needs the package list).

    ``bake_pip_inline=True`` writes the JSON with empty ``pip_args``; the
    caller has installed packages at image-build time and the container-
    start hook should be a no-op for installs (it still runs setup
    commands). The hash is unchanged so a bake-image cache key is stable.
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)
    json_path = runtime_dir / "cluster-runtime.json"

    if config_path is None:
        cfg = ClusterRuntimeConfig()
        yaml_dir: Path | None = None
    else:
        config_path = Path(config_path)
        cfg = load_cluster_runtime_config(config_path)
        yaml_dir = config_path.parent

    pkgs = dedupe_packages(cfg.extensions.packages)
    h = resolved_hash(
        pkgs,
        setup_commands=cfg.setup_commands,
        head_setup_commands=cfg.head_setup_commands,
        worker_setup_commands=cfg.worker_setup_commands,
    )
    if bake_pip_inline:
        version_args: list[str] = []
        path_args: list[str] = []
    else:
        version_args, path_args = _split_container_pip_args(pkgs, yaml_dir=yaml_dir)
    payload = {
        "hash": h,
        "pip_args": version_args,
        "pip_args_no_deps": path_args,
        "setup_commands": cfg.setup_commands,
        "head_setup_commands": cfg.head_setup_commands,
        "worker_setup_commands": cfg.worker_setup_commands,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    return cfg


def write_operator_config(
    *,
    config_path: str | Path | None,
    runtime_dir: Path,
) -> Path:
    """Materialize the operator YAML at ``runtime_dir/operator-config.yaml``
    so it can be bind-mounted into colony services as ``/etc/colony/cluster.yaml``
    *before* containers start.

    Without this, the ray-head entrypoint runs ``polymath deploy --config
    /mnt/shared/config.yaml`` before ``colony-env up``'s ``docker cp`` can
    populate the shared volume, and ``load_config_from_yaml`` exits the
    container with FileNotFoundError. The bind mount sidesteps the race
    entirely.

    Always writes a file (empty YAML stub when ``config_path`` is None) so
    the docker-compose bind mount never silently turns into a directory.
    Copies file contents rather than bind-mounting the host path directly
    so the source path is stable (under ``.runtime/``) and the operator can
    move / rename their --config arg between runs without breaking compose.
    """
    runtime_dir.mkdir(parents=True, exist_ok=True)
    out = runtime_dir / "operator-config.yaml"
    if config_path is None:
        out.write_text("# empty operator config — no --config arg passed to colony-env up\n")
    else:
        src = Path(config_path)
        if not src.is_file():
            raise FileNotFoundError(f"Config not found: {src}")
        out.write_bytes(src.read_bytes())
    return out


def write_path_extensions_override(
    *,
    packages: list[PackageSpec],
    yaml_dir: Path | None,
    runtime_dir: Path,
    services: tuple[str, ...] = ("ray-head", "ray-worker", "dashboard"),
) -> Path | None:
    """Generate a docker-compose override that bind-mounts ``source: path``
    extension directories into colony services at
    ``/mnt/path-extensions/<name>``.

    Returns the override path when at least one path-source package exists;
    ``None`` when no override is needed (caller should not pass ``-f`` for
    a non-existent file).

    ``runtime_dir`` is the directory the override is written to (and the
    bind-mount source paths are absolute, so the override file is location-
    independent).
    """
    path_pkgs = [p for p in packages if p.source == "path"]
    override_path = runtime_dir / "docker-compose.path-extensions.yml"
    if not path_pkgs:
        # Remove a stale override from a previous run with path sources so
        # ``up()`` does not accidentally pull in old mounts.
        if override_path.exists():
            override_path.unlink()
        return None

    if yaml_dir is None:
        # All path entries must be absolute; otherwise we cannot resolve.
        for pkg in path_pkgs:
            if not Path(pkg.path or "").is_absolute():
                raise ValueError(
                    f"package {pkg.name!r}: relative path {pkg.path!r} "
                    f"requires yaml_dir",
                )

    overlay: dict[str, list[str]] = {svc: [] for svc in services}
    for pkg in path_pkgs:
        host, container = _resolve_path_source_in_container(
            pkg, yaml_dir=yaml_dir or Path.cwd(),
        )
        mount = f"{host}:{container}:ro"
        for svc in services:
            overlay[svc].append(mount)

    doc: dict = {
        "services": {svc: {"volumes": mounts} for svc, mounts in overlay.items()},
    }
    runtime_dir.mkdir(parents=True, exist_ok=True)
    with open(override_path, "w") as f:
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)
    return override_path


__all__ = (
    "CONTAINER_PATH_EXTENSIONS_ROOT",
    "write_cluster_runtime",
    "write_operator_config",
    "write_path_extensions_override",
)
