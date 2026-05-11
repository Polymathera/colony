"""Tests for the L1-G operator-YAML schema + resolver.

Three concerns:

1. :class:`PackageSpec` validation — version-source vs path-source mutually
   exclude their respective fields; ``extra="forbid"`` rejects typos.
2. :func:`load_cluster_runtime_config` — extracts only the L1-G keys from a
   YAML file's ``cluster:`` block, ignores LLM-cluster fields that coexist.
3. :func:`resolve_pip_args` / :func:`dedupe_packages` / :func:`resolved_hash`
   — the resolver pipeline that the container-start hook consumes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polymathera.colony.cli.deploy.extensions import (
    ClusterRuntimeConfig,
    PackageSpec,
    dedupe_packages,
    lint_setup_commands,
    load_cluster_runtime_config,
    resolve_pip_args,
    resolved_hash,
)


# ---------------------------------------------------------------------------
# PackageSpec validation
# ---------------------------------------------------------------------------


def test_package_spec_version_default() -> None:
    pkg = PackageSpec(name="polymathera-cps", version="0.1.0")
    assert pkg.source == "version"
    assert pkg.path is None


def test_package_spec_path_requires_path() -> None:
    with pytest.raises(ValueError, match="source='path' requires 'path'"):
        PackageSpec(name="x", source="path")


def test_package_spec_version_rejects_path() -> None:
    with pytest.raises(ValueError, match="'path' is invalid for source='version'"):
        PackageSpec(name="x", version="1.0", path="/tmp/x")


def test_package_spec_path_rejects_version() -> None:
    with pytest.raises(ValueError, match="'version' is invalid for source='path'"):
        PackageSpec(name="x", source="path", path="/tmp/x", version="1.0")


def test_package_spec_rejects_unknown_field() -> None:
    """`extra="forbid"` catches typos like ``pakcages`` instead of ``packages``."""
    with pytest.raises(ValueError):
        PackageSpec.model_validate({"name": "x", "verison": "1.0"})


# ---------------------------------------------------------------------------
# load_cluster_runtime_config — YAML extraction
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


def test_load_yaml_extracts_only_l1g_keys(tmp_path: Path) -> None:
    """LLM-cluster fields under ``cluster:`` are silently ignored — they're
    owned by a different parser."""
    yaml_path = _write(tmp_path, """
cluster:
  app_name: polymathera
  remote_deployments:
    - model_name: claude-sonnet-4-6
      provider: anthropic
  docker:
    image: polymathera/colony:0.3.0
  extensions:
    packages:
      - { name: polymathera-cps, version: "0.1.0", extras: [quantum] }
  setup_commands:
    - "echo hello"
""")
    cfg = load_cluster_runtime_config(yaml_path)
    assert cfg.docker.image == "polymathera/colony:0.3.0"
    assert len(cfg.extensions.packages) == 1
    assert cfg.extensions.packages[0].name == "polymathera-cps"
    assert cfg.setup_commands == ["echo hello"]


def test_load_yaml_no_cluster_block_returns_defaults(tmp_path: Path) -> None:
    yaml_path = _write(tmp_path, "# empty\n")
    cfg = load_cluster_runtime_config(yaml_path)
    assert cfg == ClusterRuntimeConfig()


def test_load_yaml_path_source_round_trip(tmp_path: Path) -> None:
    yaml_path = _write(tmp_path, """
cluster:
  extensions:
    packages:
      - { name: polymathera-cps, source: path, path: ../cps, extras: [quantum] }
""")
    cfg = load_cluster_runtime_config(yaml_path)
    pkg = cfg.extensions.packages[0]
    assert pkg.source == "path"
    assert pkg.path == "../cps"
    assert pkg.version is None


# ---------------------------------------------------------------------------
# dedupe_packages
# ---------------------------------------------------------------------------


def test_dedupe_keeps_exact_duplicates_once() -> None:
    pkg = PackageSpec(name="x", version="1.0", extras=["a"])
    out = dedupe_packages([pkg, pkg, pkg])
    assert out == [pkg]


def test_dedupe_raises_on_conflict() -> None:
    a = PackageSpec(name="x", version="1.0")
    b = PackageSpec(name="x", version="2.0")
    with pytest.raises(ValueError, match="declared twice with conflicting specs"):
        dedupe_packages([a, b])


# ---------------------------------------------------------------------------
# resolve_pip_args
# ---------------------------------------------------------------------------


def test_resolve_version_default_uses_double_equals() -> None:
    pkg = PackageSpec(name="polymathera-cps", version="0.1.0")
    assert resolve_pip_args([pkg]) == ["polymathera-cps==0.1.0"]


def test_resolve_version_passes_through_explicit_operators() -> None:
    pairs = [
        (">=0.1", "polymathera-cps>=0.1"),
        ("<2.0", "polymathera-cps<2.0"),
        ("~=0.1.0", "polymathera-cps~=0.1.0"),
        ("!=0.5", "polymathera-cps!=0.5"),
    ]
    for ver, expected in pairs:
        assert resolve_pip_args([PackageSpec(name="polymathera-cps", version=ver)]) == [expected]


def test_resolve_version_unbounded_when_omitted() -> None:
    pkg = PackageSpec(name="polymathera-cps")
    assert resolve_pip_args([pkg]) == ["polymathera-cps"]


def test_resolve_extras() -> None:
    pkg = PackageSpec(name="polymathera-cps", version="0.1", extras=["quantum", "duv"])
    assert resolve_pip_args([pkg]) == ["polymathera-cps[quantum,duv]==0.1"]


def test_resolve_path_relative_to_yaml_dir(tmp_path: Path) -> None:
    cps_dir = tmp_path / "cps"
    cps_dir.mkdir()
    pkg = PackageSpec(name="polymathera-cps", source="path", path="cps", extras=["quantum"])
    [arg] = resolve_pip_args([pkg], yaml_dir=tmp_path)
    assert arg == f"{cps_dir.resolve()}[quantum]"


def test_resolve_path_absolute_passthrough(tmp_path: Path) -> None:
    abs_path = tmp_path / "cps"
    pkg = PackageSpec(name="polymathera-cps", source="path", path=str(abs_path))
    [arg] = resolve_pip_args([pkg])
    assert arg == str(abs_path)


def test_resolve_path_relative_without_yaml_dir_raises() -> None:
    pkg = PackageSpec(name="polymathera-cps", source="path", path="../cps")
    with pytest.raises(ValueError, match="relative path"):
        resolve_pip_args([pkg])


# ---------------------------------------------------------------------------
# resolved_hash — determinism + sensitivity
# ---------------------------------------------------------------------------


def test_hash_deterministic() -> None:
    pkgs = [PackageSpec(name="a", version="1"), PackageSpec(name="b", version="2")]
    cmds = ["echo x"]
    assert resolved_hash(pkgs, setup_commands=cmds) == resolved_hash(pkgs, setup_commands=cmds)


def test_hash_changes_on_package_difference() -> None:
    a = resolved_hash([PackageSpec(name="x", version="1.0")])
    b = resolved_hash([PackageSpec(name="x", version="2.0")])
    assert a != b


def test_hash_changes_on_setup_commands() -> None:
    pkgs = [PackageSpec(name="x", version="1.0")]
    a = resolved_hash(pkgs, setup_commands=[])
    b = resolved_hash(pkgs, setup_commands=["echo x"])
    assert a != b


def test_hash_distinguishes_role_specific_setup_commands() -> None:
    """Head-only and worker-only commands must produce different hashes — the
    image needs to be rebuilt when role-specific setup changes."""
    pkgs = [PackageSpec(name="x", version="1.0")]
    a = resolved_hash(pkgs, head_setup_commands=["echo head"])
    b = resolved_hash(pkgs, worker_setup_commands=["echo head"])
    assert a != b


# ---------------------------------------------------------------------------
# lint_setup_commands — flag pip install in shell commands
# ---------------------------------------------------------------------------


def test_lint_clean_when_no_pip() -> None:
    cfg = ClusterRuntimeConfig(setup_commands=["echo hello", "mkdir -p /tmp/x"])
    assert lint_setup_commands(cfg) == []


def test_lint_flags_pip_install() -> None:
    cfg = ClusterRuntimeConfig(setup_commands=["pip install foo"])
    [warning] = lint_setup_commands(cfg)
    assert "extensions.packages" in warning
    assert "setup_commands[0]" in warning


def test_lint_flags_python_m_pip() -> None:
    cfg = ClusterRuntimeConfig(
        head_setup_commands=["python -m pip install requests"],
    )
    [warning] = lint_setup_commands(cfg)
    assert "head_setup_commands[0]" in warning


def test_lint_flags_pip3_install() -> None:
    cfg = ClusterRuntimeConfig(worker_setup_commands=["pip3 install numpy"])
    [warning] = lint_setup_commands(cfg)
    assert "worker_setup_commands[0]" in warning


def test_lint_does_not_flag_unrelated_install() -> None:
    """``apt-get install`` / ``gem install`` are NOT pip and stay legitimate
    setup commands."""
    cfg = ClusterRuntimeConfig(
        setup_commands=["sudo apt-get install -y libfoo", "gem install bar"],
    )
    assert lint_setup_commands(cfg) == []


def test_lint_does_not_flag_pip_freeze() -> None:
    """Only ``pip install`` is a problem — ``pip freeze``, ``pip list`` are
    legitimate diagnostic commands."""
    cfg = ClusterRuntimeConfig(setup_commands=["pip freeze > /tmp/freeze.txt"])
    assert lint_setup_commands(cfg) == []
