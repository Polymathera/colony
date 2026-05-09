"""Regression tests for the serving proxy's env-var propagation.

Spawned Ray actors run in separate processes from the driver and
do NOT inherit ``os.environ`` automatically — the proxy filters the
driver's env to a prefix allowlist and writes the result into the
actor's ``runtime_env.env_vars``. The contract:

1. **Bootstrap** propagates: ``POLYMATHERA_*`` (the path to the
   operator YAML, plus deployment-name overrides), ``RAY_*``,
   ``REDIS_*``. Without these, a worker can't reach its own
   :class:`ConfigurationManager` to load the typed config.

2. **Secrets** propagate: LLM provider API keys
   (``ANTHROPIC_*`` / ``OPENAI_*`` / ``OPENROUTER_*`` /
   ``GOOGLE_*`` / ``MISTRAL_*`` / ``LLAMA_*`` /
   ``HUGGING_FACE_*``). Reader / deployment client code reads
   these at call time — they don't go through the typed config
   because YAML must not carry secrets.

3. Everything else is dropped — raw shell variables, unrelated
   tooling, node-specific Ray internals.

4. Empty values are dropped — an unset compose passthrough like
   ``MISTRAL_API_KEY=${MISTRAL_API_KEY:-}`` shouldn't propagate
   an empty string to the actor.
"""

from __future__ import annotations

import pytest

from polymathera.colony.distributed.ray_utils.serving.proxy import (
    RAY_INTERNAL_ENV_VARS,
    RUNTIME_ENV_PREFIXES,
    runtime_env_vars_to_propagate,
)


# ---------------------------------------------------------------------------
# Allowlist contract — bootstrap + secrets, not config knobs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix",
    [
        # Bootstrap — required for the worker's ConfigurationManager
        # to load the operator YAML and reach Redis / Ray.
        "POLYMATH",
        "RAY_",
        "REDIS_",
        # Secrets — LLM provider API keys read directly by reader /
        # deployment client code.
        "ANTHROPIC_",
        "OPENROUTER_",
        "OPENAI_",
        "GOOGLE_",
        "MISTRAL_",
        "LLAMA_",
        "HUGGING_FACE_",
    ],
)
def test_prefix_in_allowlist(prefix: str) -> None:
    assert prefix in RUNTIME_ENV_PREFIXES


# ---------------------------------------------------------------------------
# Propagation behaviour
# ---------------------------------------------------------------------------


def test_polymathera_config_propagates() -> None:
    """``POLYMATHERA_CONFIG`` is the load-bearing bootstrap: it tells
    the worker's :class:`ConfigurationManager` where to find the YAML
    that carries every typed config component. Without this var,
    workers fall through to bare class defaults — no operator
    overrides take effect."""

    src = {
        "POLYMATHERA_CONFIG": "/mnt/shared/config.yaml",
        "POLYMATHERA_HEAD": "true",
        "POLYMATHERA_RUNNING_LOCALLY": "true",
    }
    assert runtime_env_vars_to_propagate(src) == src


def test_llm_provider_keys_propagate() -> None:
    """LLM provider API keys are secrets read directly by reader /
    deployment client code. They must reach actor processes."""

    src = {
        "ANTHROPIC_API_KEY": "sk-ant-fake",
        "GOOGLE_API_KEY": "gk-fake",
        "OPENAI_API_KEY": "sk-fake",
        "OPENROUTER_API_KEY": "or-fake",
        "MISTRAL_API_KEY": "msk-fake",
        "LLAMA_CLOUD_API_KEY": "llx-fake",
        "HUGGING_FACE_HUB_TOKEN": "hf-fake",
    }
    assert runtime_env_vars_to_propagate(src) == src


def test_raw_shell_vars_dropped() -> None:
    """``PATH``, ``HOME``, etc. are noise to actor processes — the
    actor inherits sensible defaults from its container's base
    environment, and we don't want stale shell vars baking in."""

    src = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/operator",
        "SHELL": "/bin/bash",
        "EDITOR": "vim",
        "USER": "ray",
        "TERM": "xterm-256color",
        "LANG": "en_US.UTF-8",
    }
    assert runtime_env_vars_to_propagate(src) == {}


def test_ray_internal_vars_dropped_even_when_prefix_matches() -> None:
    """``RAY_RAYLET_PID`` etc. start with ``RAY_`` (which IS in
    the allowlist) but propagating them across nodes makes the
    actor monitor the wrong PIDs and crash. They must be dropped
    despite the prefix match."""

    src = {
        "RAY_RAYLET_PID": "1234",
        "RAY_JOB_ID": "01000000",
        "RAY_LD_PRELOAD_ON_WORKERS": "/tmp/lib.so",
        "RAY_NODE_MANAGER_PORT": "12345",
        "RAY_OBJECT_STORE_MEMORY": "1000000",
        "RAY_RAYLET_SOCKET_NAME": "/tmp/raylet",
        # ``RAY_USEFUL`` is not in the internal blocklist and
        # SHOULD propagate — verifies the blocklist is exact-match,
        # not a second prefix filter.
        "RAY_USEFUL": "yes",
    }
    out = runtime_env_vars_to_propagate(src)
    assert out == {"RAY_USEFUL": "yes"}
    assert RAY_INTERNAL_ENV_VARS == frozenset({
        "RAY_RAYLET_PID",
        "RAY_JOB_ID",
        "RAY_LD_PRELOAD_ON_WORKERS",
        "RAY_NODE_MANAGER_PORT",
        "RAY_OBJECT_STORE_MEMORY",
        "RAY_RAYLET_SOCKET_NAME",
    })


def test_empty_values_dropped() -> None:
    """``MISTRAL_API_KEY=`` (e.g. compose's ``${MISTRAL_API_KEY:-}``
    passthrough when the host doesn't have the key set) should NOT
    end up in the actor — empty values shadow the actor's own
    defaults and confuse the reader's "is the API key set?" check.
    """
    src = {
        "POLYMATHERA_CONFIG": "/mnt/shared/config.yaml",
        "MISTRAL_API_KEY": "",
        "GOOGLE_API_KEY": "",
    }
    out = runtime_env_vars_to_propagate(src)
    assert out == {"POLYMATHERA_CONFIG": "/mnt/shared/config.yaml"}


def test_default_source_is_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling without an explicit source filters the live
    ``os.environ`` — convenience for the proxy's call site."""
    monkeypatch.setenv("POLYMATHERA_CONFIG", "/mnt/shared/config.yaml")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    monkeypatch.setenv("UNRELATED_TEST_VAR", "leak")
    out = runtime_env_vars_to_propagate()
    assert out.get("POLYMATHERA_CONFIG") == "/mnt/shared/config.yaml"
    assert out.get("ANTHROPIC_API_KEY") == "sk-fake"
    assert "UNRELATED_TEST_VAR" not in out


def test_explicit_source_does_not_mutate_environ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POLYMATHERA_CONFIG", raising=False)
    src = {"POLYMATHERA_CONFIG": "/from/arg.yaml"}
    runtime_env_vars_to_propagate(src)
    import os as _os
    assert "POLYMATHERA_CONFIG" not in _os.environ
