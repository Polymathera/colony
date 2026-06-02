"""Orchestration tests for :class:`DockerComposeProvider`.

The provider shells out to ``docker`` and ``docker compose``; we mock those
calls (the ``_exec`` method) and assert on the orchestration shape:

- Base image build runs BEFORE the compose runtime build.
- ``cluster-runtime.json`` is always written before any compose call.
- ``--bake`` triggers a third docker build and overrides ``COLONY_IMAGE``
  in the compose subprocess env.
- Path-source packages cause a path-extensions compose override to be
  passed via ``-f``.

Real container-start behaviour (the hook reading the JSON, pip install
into the overlay, setup_commands execution) is exercised by manual smoke
tests against a running Docker daemon — out of scope for unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from polymathera.colony.cli.deploy.config import DeployConfig
from polymathera.colony.cli.deploy.providers.compose import (
    _DOCKER_DIR,
    _RUNTIME_DIR,
    _BASE_IMAGE_TAG,
    DockerComposeProvider,
)


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> DockerComposeProvider:
    """Provider with a tmp runtime dir and a recording _exec.

    Calls are appended to ``provider.calls`` as ``(args_tuple, env_keys)``;
    each returns ``(0, "", "")`` (success) by default.
    """
    # Redirect the runtime dir to tmp so we don't pollute the colony source tree.
    monkeypatch.setattr(
        "polymathera.colony.cli.deploy.providers.compose._RUNTIME_DIR",
        tmp_path / ".runtime",
    )

    p = DockerComposeProvider(DeployConfig(mode="compose"))
    p.calls = []  # type: ignore[attr-defined]

    async def fake_exec(*args: str, capture: bool = True, env: dict[str, str] | None = None) -> tuple[int, str, str]:
        p.calls.append((args, dict(env or {})))  # type: ignore[attr-defined]
        return (0, "", "")

    monkeypatch.setattr(p, "_exec", fake_exec)
    return p


def _docker_build_calls(calls: list[tuple]) -> list[tuple]:
    return [c for c in calls if c[0][:2] == ("docker", "build")]


def _compose_calls(calls: list[tuple]) -> list[tuple]:
    return [c for c in calls if c[0][:2] == ("docker", "compose")]


# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_up_builds_base_image_before_compose_build(
    provider: DockerComposeProvider, tmp_path: Path,
) -> None:
    # Stub healthchecks so up() returns without polling.
    async def _no_wait(*args: Any, **kwargs: Any) -> bool:
        return True
    import polymathera.colony.cli.deploy.providers.compose as m
    m_wait_until_ready = m.wait_until_ready
    m.wait_until_ready = _no_wait  # type: ignore[assignment]
    m_redis_ping = m.redis_ping
    m.redis_ping = lambda *a, **k: True  # type: ignore[assignment]
    try:
        await provider.up(build=True, workers=2, config_path=None)
    finally:
        m.wait_until_ready = m_wait_until_ready  # type: ignore[assignment]
        m.redis_ping = m_redis_ping  # type: ignore[assignment]

    builds = _docker_build_calls(provider.calls)  # type: ignore[attr-defined]
    composes = _compose_calls(provider.calls)  # type: ignore[attr-defined]

    # First docker build is the base, tagged with _BASE_IMAGE_TAG.
    assert builds, "expected at least one docker build call"
    base_args = builds[0][0]
    assert "-t" in base_args and _BASE_IMAGE_TAG in base_args, base_args
    assert any(arg.endswith("Dockerfile.base") for arg in base_args), base_args

    # Compose build comes AFTER the base build (orchestration order).
    base_idx = provider.calls.index(builds[0])  # type: ignore[attr-defined]
    compose_build_idx = next(
        i for i, c in enumerate(provider.calls) if c[0][:3] == ("docker", "compose", "-f") and "build" in c[0]  # type: ignore[attr-defined]
    )
    assert base_idx < compose_build_idx


@pytest.mark.asyncio
async def test_up_always_writes_cluster_runtime_json(
    provider: DockerComposeProvider, tmp_path: Path,
) -> None:
    """Even with no --config, a stub JSON is written so the bind mount in
    docker-compose.yml has a real source file."""
    import polymathera.colony.cli.deploy.providers.compose as m
    m.wait_until_ready = lambda *a, **k: _async(True)  # type: ignore[assignment]
    m.redis_ping = lambda *a, **k: True  # type: ignore[assignment]

    runtime_dir = tmp_path / ".runtime"
    await provider.up(build=False, workers=1, config_path=None)

    json_path = runtime_dir / "cluster-runtime.json"
    assert json_path.is_file()
    payload = json.loads(json_path.read_text())
    assert "hash" in payload
    assert payload["pip_args"] == []


def _async(value):
    """Wrap a value as a coroutine for monkeypatching async functions."""
    async def _coro(*a, **k):
        return value
    return _coro()


@pytest.mark.asyncio
async def test_up_with_path_source_passes_override_via_minus_f(
    provider: DockerComposeProvider, tmp_path: Path,
) -> None:
    cps_dir = tmp_path / "cps"
    cps_dir.mkdir()
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(f"""
cluster:
  extensions:
    packages:
      - {{ name: polymathera-cps, source: path, path: {cps_dir} }}
""")
    import polymathera.colony.cli.deploy.providers.compose as m
    async def _no_wait(*a, **k): return True
    m.wait_until_ready = _no_wait  # type: ignore[assignment]
    m.redis_ping = lambda *a, **k: True  # type: ignore[assignment]

    await provider.up(build=False, workers=1, config_path=str(yaml_path))

    composes = _compose_calls(provider.calls)  # type: ignore[attr-defined]
    # The override is appended as a second -f after the primary compose file.
    assert composes, "expected compose calls"
    args = composes[0][0]
    f_args = [args[i + 1] for i, a in enumerate(args) if a == "-f"]
    assert any("path-extensions" in str(f) for f in f_args), f_args


@pytest.mark.asyncio
async def test_up_with_bake_sets_colony_image_env(
    provider: DockerComposeProvider, tmp_path: Path,
) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("""
cluster:
  extensions:
    packages:
      - { name: polymathera-cps, version: "0.1.0" }
""")
    import polymathera.colony.cli.deploy.providers.compose as m
    async def _no_wait(*a, **k): return True
    m.wait_until_ready = _no_wait  # type: ignore[assignment]
    m.redis_ping = lambda *a, **k: True  # type: ignore[assignment]

    await provider.up(build=False, workers=1, config_path=str(yaml_path), bake=True)

    builds = _docker_build_calls(provider.calls)  # type: ignore[attr-defined]
    # With build=False, we still expect ONE bake build (the only docker build).
    assert builds, "bake should trigger a docker build"
    bake_args = builds[-1][0]
    assert any(arg.startswith("colony-local:") for arg in bake_args), bake_args
    bake_tag = next(arg for arg in bake_args if arg.startswith("colony-local:"))

    # Compose up after bake must have COLONY_IMAGE set to the bake tag.
    up_calls = [
        c for c in provider.calls  # type: ignore[attr-defined]
        if c[0][:3] == ("docker", "compose", "-f") and "up" in c[0]
    ]
    assert up_calls, "expected compose up call"
    _, env = up_calls[0]
    assert env.get("COLONY_IMAGE") == bake_tag


def test_active_profiles_off_when_smee_url_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No smee URL in env → no compose profiles active → no
    ``--profile`` arg appended to the compose command."""
    p = DockerComposeProvider(DeployConfig(mode="compose"))
    monkeypatch.setattr(
        p, "_compose_subprocess_env", lambda: {"PATH": "/usr/bin"},
    )
    assert p._active_profiles() == []
    cmd = p._compose_cmd("up", "-d")
    assert "--profile" not in cmd


def test_active_profiles_local_webhook_when_smee_url_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``POLYMATHERA_SMEE_FORWARDING_URL`` in env activates the
    ``local-webhook`` profile and injects ``--profile local-webhook``
    into every compose command."""
    p = DockerComposeProvider(DeployConfig(mode="compose"))
    monkeypatch.setattr(
        p, "_compose_subprocess_env",
        lambda: {
            "PATH": "/usr/bin",
            "POLYMATHERA_SMEE_FORWARDING_URL": "https://smee.io/abc123",
        },
    )
    assert p._active_profiles() == ["local-webhook"]
    cmd = p._compose_cmd("up", "-d")
    # Profile arg appears before the subcommand args, after -f / --env-file.
    assert "--profile" in cmd
    profile_idx = cmd.index("--profile")
    assert cmd[profile_idx + 1] == "local-webhook"
    up_idx = cmd.index("up")
    assert profile_idx < up_idx


def test_active_profiles_off_when_smee_url_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty string is treated as 'unset' — operators commenting out
    the value by setting it to '' shouldn't accidentally activate
    the sidecar."""
    p = DockerComposeProvider(DeployConfig(mode="compose"))
    monkeypatch.setattr(
        p, "_compose_subprocess_env",
        lambda: {"POLYMATHERA_SMEE_FORWARDING_URL": ""},
    )
    assert p._active_profiles() == []


@pytest.mark.asyncio
async def test_image_info_parses_baked_vs_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """image_info splits packages by which pip list invocation produced them."""
    p = DockerComposeProvider(DeployConfig(mode="compose"))

    overlay_out = "polymathera-cps==0.1.0\n"
    all_out = (
        "polymathera-colony==0.3.0\n"
        "polymathera-cps==0.1.0\n"
        "ray==2.49.0\n"
    )

    async def fake_exec(*args: str, **_: Any) -> tuple[int, str, str]:
        if "--path" in args:
            return (0, overlay_out, "")
        return (0, all_out, "")

    monkeypatch.setattr(p, "_exec", fake_exec)
    info = await p.image_info()
    assert info["overlay"] == ["polymathera-cps==0.1.0"]
    assert info["baked"] == ["polymathera-colony==0.3.0"]
