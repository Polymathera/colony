"""Tests for the L1-G runtime artifact writer.

Three concerns:

1. :func:`write_cluster_runtime` always produces a file (stub when no L1-G
   fields are set) so the bind mount in docker-compose.yml never silently
   becomes a directory.
2. Path-source ``PackageSpec``s emit *in-container* pip args
   (``/mnt/path-extensions/<name>``) — pip runs inside the container and
   has no access to host paths.
3. :func:`write_path_extensions_override` emits a compose override
   bind-mounting host paths to the matching container paths, and removes a
   stale override when the new config has no path sources.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from polymathera.colony.cli.deploy.extensions import PackageSpec
from polymathera.colony.cli.deploy.runtime_writer import (
    CONTAINER_PATH_EXTENSIONS_ROOT,
    write_cluster_runtime,
    write_operator_config,
    write_path_extensions_override,
)


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# write_cluster_runtime
# ---------------------------------------------------------------------------


def test_write_runtime_emits_stub_when_no_config(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "rt"
    write_cluster_runtime(config_path=None, runtime_dir=runtime_dir)

    payload = json.loads((runtime_dir / "cluster-runtime.json").read_text())
    assert payload["pip_args"] == []
    assert payload["pip_args_no_deps"] == []
    assert payload["setup_commands"] == []
    assert "hash" in payload


def test_write_runtime_round_trips_version_packages(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, """
cluster:
  extensions:
    packages:
      - { name: polymathera-cps, version: "0.1.0", extras: [quantum] }
""")
    runtime_dir = tmp_path / "rt"
    write_cluster_runtime(config_path=yaml_path, runtime_dir=runtime_dir)
    payload = json.loads((runtime_dir / "cluster-runtime.json").read_text())
    assert payload["pip_args"] == ["polymathera-cps[quantum]==0.1.0"]
    assert payload["pip_args_no_deps"] == []


def test_write_runtime_translates_path_sources_to_container_paths(
    tmp_path: Path,
) -> None:
    """Path-source pip args land in ``pip_args_no_deps`` so the hook
    installs them with ``--no-deps``: their pyproject's dep tree is already
    in the base image, and ``pip install --target`` with editable deps
    would conflict."""
    cps_dir = tmp_path / "cps"
    cps_dir.mkdir()
    yaml_path = _write_yaml(tmp_path, """
cluster:
  extensions:
    packages:
      - { name: polymathera-cps, source: path, path: cps, extras: [quantum] }
""")
    runtime_dir = tmp_path / "rt"
    write_cluster_runtime(config_path=yaml_path, runtime_dir=runtime_dir)
    payload = json.loads((runtime_dir / "cluster-runtime.json").read_text())
    assert payload["pip_args"] == []
    assert payload["pip_args_no_deps"] == [
        f"{CONTAINER_PATH_EXTENSIONS_ROOT}/polymathera-cps[quantum]",
    ]


def test_write_runtime_splits_mixed_version_and_path_sources(
    tmp_path: Path,
) -> None:
    """Version + path entries in the same YAML must land in their
    respective lists; the hook runs two pip-install calls."""
    cps_dir = tmp_path / "cps"
    cps_dir.mkdir()
    yaml_path = _write_yaml(tmp_path, """
cluster:
  extensions:
    packages:
      - { name: polymathera-cps, source: path, path: cps }
      - { name: third-party, version: "1.2.3" }
""")
    runtime_dir = tmp_path / "rt"
    write_cluster_runtime(config_path=yaml_path, runtime_dir=runtime_dir)
    payload = json.loads((runtime_dir / "cluster-runtime.json").read_text())
    assert payload["pip_args"] == ["third-party==1.2.3"]
    assert payload["pip_args_no_deps"] == [
        f"{CONTAINER_PATH_EXTENSIONS_ROOT}/polymathera-cps",
    ]


def test_write_runtime_bake_inline_emits_empty_pip_args(tmp_path: Path) -> None:
    """``bake_pip_inline=True`` means the caller pre-installed packages at
    image-build time; the container-start hook must skip the install path
    but still run setup_commands."""
    yaml_path = _write_yaml(tmp_path, """
cluster:
  extensions:
    packages:
      - { name: polymathera-cps, version: "0.1.0" }
  setup_commands:
    - "echo hi"
""")
    runtime_dir = tmp_path / "rt"
    write_cluster_runtime(
        config_path=yaml_path, runtime_dir=runtime_dir, bake_pip_inline=True,
    )
    payload = json.loads((runtime_dir / "cluster-runtime.json").read_text())
    assert payload["pip_args"] == []
    assert payload["pip_args_no_deps"] == []
    assert payload["setup_commands"] == ["echo hi"]


def test_write_runtime_creates_dir_if_missing(tmp_path: Path) -> None:
    """``runtime_dir`` is created automatically — that's the contract."""
    runtime_dir = tmp_path / "deeply" / "nested" / "rt"
    write_cluster_runtime(config_path=None, runtime_dir=runtime_dir)
    assert (runtime_dir / "cluster-runtime.json").is_file()


# ---------------------------------------------------------------------------
# write_path_extensions_override
# ---------------------------------------------------------------------------


def test_override_returns_none_when_no_path_sources(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "rt"
    out = write_path_extensions_override(
        packages=[PackageSpec(name="x", version="1.0")],
        yaml_dir=tmp_path,
        runtime_dir=runtime_dir,
    )
    assert out is None
    # No stale file was created.
    assert not (runtime_dir / "docker-compose.path-extensions.yml").exists()


def test_override_removes_stale_file_when_no_path_sources(tmp_path: Path) -> None:
    """Previous run had path sources; this run doesn't. Stale override must
    be removed so it isn't accidentally re-loaded."""
    runtime_dir = tmp_path / "rt"
    runtime_dir.mkdir()
    stale = runtime_dir / "docker-compose.path-extensions.yml"
    stale.write_text("# stale\n")
    out = write_path_extensions_override(
        packages=[PackageSpec(name="x", version="1.0")],
        yaml_dir=tmp_path,
        runtime_dir=runtime_dir,
    )
    assert out is None
    assert not stale.exists()


def test_override_writes_volumes_for_each_service(tmp_path: Path) -> None:
    cps_dir = tmp_path / "cps"
    cps_dir.mkdir()
    runtime_dir = tmp_path / "rt"
    out = write_path_extensions_override(
        packages=[
            PackageSpec(name="polymathera-cps", source="path", path="cps"),
        ],
        yaml_dir=tmp_path,
        runtime_dir=runtime_dir,
    )
    assert out is not None
    doc = yaml.safe_load(out.read_text())
    expected_mount = f"{cps_dir.resolve()}:/mnt/path-extensions/polymathera-cps:ro"
    for svc in ("ray-head", "ray-worker", "dashboard"):
        assert doc["services"][svc]["volumes"] == [expected_mount]


# ---------------------------------------------------------------------------
# write_operator_config — bind-mount source for /etc/colony/cluster.yaml
# ---------------------------------------------------------------------------


def test_operator_config_copies_when_path_provided(tmp_path: Path) -> None:
    src = tmp_path / "my-config.yaml"
    body = "cluster:\n  app_name: smoke-test\n"
    src.write_text(body)
    runtime_dir = tmp_path / "rt"
    out = write_operator_config(config_path=src, runtime_dir=runtime_dir)
    assert out == runtime_dir / "operator-config.yaml"
    assert out.read_text() == body


def test_operator_config_writes_stub_when_no_config(tmp_path: Path) -> None:
    """An always-written stub keeps the docker-compose bind mount valid even
    when the operator runs ``colony-env up`` without ``--config``."""
    runtime_dir = tmp_path / "rt"
    out = write_operator_config(config_path=None, runtime_dir=runtime_dir)
    assert out.is_file()
    assert out.read_text().strip().startswith("#")


def test_operator_config_raises_on_missing_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write_operator_config(
            config_path=tmp_path / "does-not-exist.yaml",
            runtime_dir=tmp_path / "rt",
        )


def test_override_raises_on_missing_host_path(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "rt"
    with pytest.raises(FileNotFoundError):
        write_path_extensions_override(
            packages=[
                PackageSpec(name="x", source="path", path="does-not-exist"),
            ],
            yaml_dir=tmp_path,
            runtime_dir=runtime_dir,
        )
