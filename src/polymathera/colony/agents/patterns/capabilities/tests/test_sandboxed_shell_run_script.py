"""Tests for the E-3 additions to ``SandboxedShellCapability``:

- ``AgentCapability.resolve_value`` default (returns None).
- ``DockerImageSpec`` new fields: ``required_env`` /
  ``script_template_packages`` / ``tags`` (parsed + round-tripped
  through ``to_summary``).
- ``list_images(role_allowlist=..., tags=...)`` filters.
- ``list_script_templates`` / ``get_script_template`` (reads via
  ``importlib.resources`` against ``script_template_packages``).
- ``run_script``: template-vs-literal validation, env-var resolution
  via sibling capabilities' ``resolve_value``, missing-resolver
  raise, conflicting-resolver raise, ``tool_call_id`` in return.

Uses a minimal in-memory ``_FakeBackend`` that records every
``exec`` so tests can assert the dispatched cmd / stdin / env.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.base import AgentCapability
from polymathera.colony.agents.patterns.capabilities._sandbox import (
    ContainerBackend, ContainerHandle, ExecResult, DockerImageRegistry,
)
from polymathera.colony.agents.patterns.capabilities._sandbox.registry import (
    DockerImageSpec,
)
from polymathera.colony.agents.patterns.capabilities.sandboxed_shell import (
    SandboxedShellCapability,
)
from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring, execution_context,
)


# ---------------------------------------------------------------------------
# Test fixtures: in-memory backend + a synthetic template package
# ---------------------------------------------------------------------------


class _RecordingBackend(ContainerBackend):
    """In-memory backend that records each ``exec`` so tests can
    assert the dispatched cmd / stdin / env. Implements the full
    ``ContainerBackend`` ABC with no-op stubs for methods
    ``run_script`` doesn't touch."""

    def __init__(self) -> None:
        self.execs: list[dict[str, Any]] = []
        self._counter = 0

    async def launch(self, spec):
        self._counter += 1
        return ContainerHandle(
            container_id=f"cid-{self._counter}", name=spec.name,
            image=spec.image,
        )

    async def stop(self, handle, *, timeout_s=10):
        pass

    async def restart(self, handle):
        pass

    async def is_running(self, handle):
        return True

    async def inspect(self, handle):
        return {"State": {"Status": "running", "Running": True,
                          "ExitCode": 0, "StartedAt": "0"}}

    async def exec(self, handle, cmd, *, timeout_seconds,
                   env=None, workdir=None, stdin=None):
        self.execs.append({
            "container_id": handle.container_id, "command": list(cmd),
            "env": dict(env or {}), "stdin": stdin,
            "timeout_seconds": timeout_seconds,
        })
        return ExecResult(
            exit_code=0, stdout="ok", stderr="", wall_time_ms=1,
        )

    async def exec_stream(self, handle, cmd, *, timeout_seconds,
                          env=None, workdir=None, stdin=None):
        async def _gen():
            yield "stdout", "ok\n"
        async for x in _gen():
            yield x

    async def copy_in(self, handle, *, src_host_path, dst_container_path):
        pass

    async def copy_out(self, handle, *, src_container_path, dst_host_path):
        pass

    async def list_by_label(self, labels):
        return []


_TEMPLATES_PACKAGE_NAME = "polymathera_test_templates_e3"


@pytest.fixture
def template_package(tmp_path, monkeypatch):
    """Build a tmp Python package with one .py template + register
    it in sys.modules so importlib.resources can find it."""
    pkg_dir = tmp_path / _TEMPLATES_PACKAGE_NAME
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text('"""Test templates."""\n')
    (pkg_dir / "hello.py").write_text(
        '"""Hello template.\n\nExtra detail.\n"""\n'
        "print('hello from template')\n",
    )
    (pkg_dir / "_private.py").write_text(
        '"""Should be hidden."""\nprint("private")\n',
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    # Force a clean re-import for each test.
    sys.modules.pop(_TEMPLATES_PACKAGE_NAME, None)
    yield _TEMPLATES_PACKAGE_NAME
    sys.modules.pop(_TEMPLATES_PACKAGE_NAME, None)


def _registry_yaml_for(template_pkg: str) -> str:
    return f"""
images:
  - role: data-analysis
    image: polymathera/data-analysis:0.1
    description: Analysis stack.
    required_env: [TEST_TOOL_CALL_ID, TEST_BACKEND_URI]
    script_template_packages: ["{template_pkg}"]
    tags: [data-analysis, scientific-python]
  - role: bayesian
    image: polymathera/bayesian:0.1
    description: Bayesian stack.
    required_env: []
    script_template_packages: []
    tags: [data-analysis, bayesian]
  - role: hpc
    image: polymathera/hpc:0.1
    description: HPC stack.
    required_env: []
    script_template_packages: []
    tags: [hpc]
"""


def _make_cap(
    backend: _RecordingBackend,
    *,
    registry_yaml: str,
    sibling_capabilities: list[AgentCapability] | None = None,
    agent_id: str = "agent-A",
) -> SandboxedShellCapability:
    agent = MagicMock()
    agent.agent_id = agent_id
    cap = SandboxedShellCapability(
        agent=agent,
        scope=BlackboardScope.SESSION,
        backend=backend,
        registry=DockerImageRegistry.from_yaml_text(registry_yaml),
        host_workspace_root="/tmp/cps_test_ws",
    )
    # Wire sibling capabilities into agent._capabilities (the shell
    # iterates this for resolve_value lookups + auto-includes self).
    # Use index-suffixed keys so tests with multiple stub caps of the
    # same type don't collide on insertion.
    caps_dict = {"sandboxed_shell": cap}
    for idx, sib in enumerate(sibling_capabilities or []):
        caps_dict[f"{type(sib).__name__}_{idx}"] = sib
    agent._capabilities = caps_dict
    return cap


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _execution_context():
    """Every test in this file runs inside the colony execution-context
    required by ``SandboxedShellCapability`` actions (tenant/session
    lookup is contextvar-keyed)."""
    with execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1", session_id="s1",
    ):
        yield


# ---------------------------------------------------------------------------
# AgentCapability.resolve_value default
# ---------------------------------------------------------------------------


def test_resolve_value_default_returns_none():
    """Base implementation returns None so unrelated capabilities
    don't trip the conflict-raise."""
    backend = _RecordingBackend()
    cap = _make_cap(backend, registry_yaml=_registry_yaml_for("ignored"))
    result = _run(cap.resolve_value("ANY_NAME", purpose="script_env"))
    assert result is None


# ---------------------------------------------------------------------------
# DockerImageSpec new fields
# ---------------------------------------------------------------------------


class TestImageSpecExtension:

    def test_from_dict_parses_new_fields(self):
        spec = DockerImageSpec.from_dict({
            "role": "x", "image": "y:1",
            "required_env": ["A", "B"],
            "script_template_packages": ["pkg.a", "pkg.b"],
            "tags": ["t1", "t2"],
        })
        assert spec.required_env == ("A", "B")
        assert spec.script_template_packages == ("pkg.a", "pkg.b")
        assert spec.tags == ("t1", "t2")

    def test_from_dict_defaults_empty_tuples(self):
        spec = DockerImageSpec.from_dict({"role": "x", "image": "y:1"})
        assert spec.required_env == ()
        assert spec.script_template_packages == ()
        assert spec.tags == ()

    def test_to_summary_round_trips_new_fields(self):
        spec = DockerImageSpec.from_dict({
            "role": "x", "image": "y:1",
            "required_env": ["A"], "script_template_packages": ["p"],
            "tags": ["t"],
        })
        summary = spec.to_summary()
        assert summary["required_env"] == ["A"]
        assert summary["script_template_packages"] == ["p"]
        assert summary["tags"] == ["t"]


# ---------------------------------------------------------------------------
# list_images filters
# ---------------------------------------------------------------------------


class TestListImagesFilters:

    def test_no_filter_returns_all(self):
        cap = _make_cap(
            _RecordingBackend(),
            registry_yaml=_registry_yaml_for("ignored"),
        )
        result = _run(cap.list_images())
        roles = [img["role"] for img in result["images"]]
        assert set(roles) == {"data-analysis", "bayesian", "hpc"}

    def test_role_allowlist_filters(self):
        cap = _make_cap(
            _RecordingBackend(),
            registry_yaml=_registry_yaml_for("ignored"),
        )
        result = _run(cap.list_images(
            role_allowlist=["data-analysis", "bayesian"],
        ))
        roles = {img["role"] for img in result["images"]}
        assert roles == {"data-analysis", "bayesian"}

    def test_tags_all_match_filter(self):
        cap = _make_cap(
            _RecordingBackend(),
            registry_yaml=_registry_yaml_for("ignored"),
        )
        # Only "bayesian" has BOTH "data-analysis" AND "bayesian" tags.
        result = _run(cap.list_images(tags=["data-analysis", "bayesian"]))
        roles = {img["role"] for img in result["images"]}
        assert roles == {"bayesian"}

    def test_tags_single_tag_matches_multiple(self):
        cap = _make_cap(
            _RecordingBackend(),
            registry_yaml=_registry_yaml_for("ignored"),
        )
        result = _run(cap.list_images(tags=["data-analysis"]))
        roles = {img["role"] for img in result["images"]}
        assert roles == {"data-analysis", "bayesian"}

    def test_role_allowlist_and_tags_compose_by_and(self):
        cap = _make_cap(
            _RecordingBackend(),
            registry_yaml=_registry_yaml_for("ignored"),
        )
        result = _run(cap.list_images(
            role_allowlist=["data-analysis", "hpc"],
            tags=["data-analysis"],
        ))
        roles = {img["role"] for img in result["images"]}
        assert roles == {"data-analysis"}


# ---------------------------------------------------------------------------
# list_script_templates / get_script_template
# ---------------------------------------------------------------------------


class TestScriptTemplateActions:

    def test_list_script_templates_reads_via_importlib_resources(
        self, template_package: str,
    ):
        cap = _make_cap(
            _RecordingBackend(),
            registry_yaml=_registry_yaml_for(template_package),
        )
        result = _run(cap.list_script_templates(image_role="data-analysis"))
        names = {t["name"] for t in result["templates"]}
        # __init__.py + _private.py excluded
        assert names == {"hello"}
        hello = result["templates"][0]
        assert hello["summary"] == "Hello template."
        assert hello["package"] == template_package

    def test_list_script_templates_unknown_role_raises(self):
        cap = _make_cap(
            _RecordingBackend(),
            registry_yaml=_registry_yaml_for("ignored"),
        )
        with pytest.raises(ValueError, match="unknown image_role"):
            _run(cap.list_script_templates(image_role="missing"))

    def test_get_script_template_returns_source(
        self, template_package: str,
    ):
        cap = _make_cap(
            _RecordingBackend(),
            registry_yaml=_registry_yaml_for(template_package),
        )
        result = _run(cap.get_script_template(
            image_role="data-analysis", name="hello",
        ))
        assert result["name"] == "hello"
        assert result["filename"] == "hello.py"
        assert "print('hello from template')" in result["source"]

    def test_get_script_template_unknown_name_raises(
        self, template_package: str,
    ):
        cap = _make_cap(
            _RecordingBackend(),
            registry_yaml=_registry_yaml_for(template_package),
        )
        with pytest.raises(FileNotFoundError, match="no template"):
            _run(cap.get_script_template(
                image_role="data-analysis", name="does_not_exist",
            ))

    @pytest.mark.parametrize("bad", ["..", ".", "sub/dir", "a\\b", ""])
    def test_get_script_template_path_traversal_rejected(
        self, template_package: str, bad: str,
    ):
        cap = _make_cap(
            _RecordingBackend(),
            registry_yaml=_registry_yaml_for(template_package),
        )
        with pytest.raises(ValueError, match="single template stem"):
            _run(cap.get_script_template(
                image_role="data-analysis", name=bad,
            ))


# ---------------------------------------------------------------------------
# run_script — the main attraction
# ---------------------------------------------------------------------------


class _StubResolverCap(AgentCapability):
    """A minimal AgentCapability that owns a set of resolver values.
    Used in run_script tests to simulate provider capabilities."""

    def __init__(self, values: dict[str, str]):
        # Skip the AgentCapability init (it requires agent/scope_id);
        # we won't call any inherited methods that need them.
        self._values = values

    async def resolve_value(self, name: str, **context: Any) -> Any | None:
        return self._values.get(name)

    async def serialize_suspension_state(self, state):
        return state

    async def deserialize_suspension_state(self, state):
        return None


def _launch(cap, role="data-analysis"):
    return _run(cap.launch_container(image_role=role))


class TestRunScriptValidation:

    def test_rejects_both_template_and_script(self, template_package: str):
        cap = _make_cap(
            _RecordingBackend(),
            registry_yaml=_registry_yaml_for(template_package),
            sibling_capabilities=[_StubResolverCap({
                "TEST_TOOL_CALL_ID": "ignored",
                "TEST_BACKEND_URI": "file://x",
            })],
        )
        launch = _launch(cap)
        result = _run(cap.run_script(
            container_id=launch["container_id"],
            template_name="hello", script="print(1)",
        ))
        assert result["exit_code"] == -1
        assert "exactly one" in result["message"]

    def test_rejects_neither_template_nor_script(self):
        cap = _make_cap(
            _RecordingBackend(),
            registry_yaml=_registry_yaml_for("ignored"),
            sibling_capabilities=[_StubResolverCap({
                "TEST_TOOL_CALL_ID": "x", "TEST_BACKEND_URI": "y",
            })],
        )
        launch = _launch(cap)
        result = _run(cap.run_script(container_id=launch["container_id"]))
        assert result["exit_code"] == -1
        assert "exactly one" in result["message"]


class TestRunScriptEnvResolution:

    def test_resolves_env_via_sibling_resolve_value(
        self, template_package: str,
    ):
        backend = _RecordingBackend()
        cap = _make_cap(
            backend,
            registry_yaml=_registry_yaml_for(template_package),
            sibling_capabilities=[_StubResolverCap({
                "TEST_TOOL_CALL_ID": "ignored",   # overwritten by run_script context
                "TEST_BACKEND_URI": "file:///tmp/mlruns",
            })],
        )
        launch = _launch(cap)
        result = _run(cap.run_script(
            container_id=launch["container_id"], template_name="hello",
        ))
        assert result.get("exit_code") == 0
        assert "tool_call_id" in result and len(result["tool_call_id"]) > 0
        exec_call = backend.execs[-1]
        assert exec_call["env"]["TEST_BACKEND_URI"] == "file:///tmp/mlruns"

    def test_run_script_passes_tool_call_id_to_resolvers(
        self, template_package: str,
    ):
        """Resolver context includes the freshly generated
        tool_call_id; resolvers can pull it via context.get(...)."""

        class _CtxClaimer(AgentCapability):
            def __init__(self):
                self.seen_context: dict[str, Any] = {}

            async def resolve_value(self, name, **context):
                if name == "TEST_TOOL_CALL_ID":
                    self.seen_context = dict(context)
                    return context.get("tool_call_id")
                return None

            async def serialize_suspension_state(self, state):
                return state

            async def deserialize_suspension_state(self, state):
                return None

        claimer = _CtxClaimer()
        backend = _RecordingBackend()
        cap = _make_cap(
            backend,
            registry_yaml=_registry_yaml_for(template_package),
            sibling_capabilities=[
                claimer,
                _StubResolverCap({"TEST_BACKEND_URI": "file:///tmp"}),
            ],
        )
        launch = _launch(cap)
        result = _run(cap.run_script(
            container_id=launch["container_id"], template_name="hello",
        ))
        assert claimer.seen_context["purpose"] == "script_env"
        assert claimer.seen_context["image_role"] == "data-analysis"
        assert claimer.seen_context["tool_call_id"] == result["tool_call_id"]
        # Echoed into the env dict the shell built.
        assert backend.execs[-1]["env"]["TEST_TOOL_CALL_ID"] == result["tool_call_id"]

    def test_missing_resolver_raises_with_var_name(
        self, template_package: str,
    ):
        cap = _make_cap(
            _RecordingBackend(),
            registry_yaml=_registry_yaml_for(template_package),
            sibling_capabilities=[_StubResolverCap({
                # TEST_BACKEND_URI missing — image requires it.
                "TEST_TOOL_CALL_ID": "x",
            })],
        )
        launch = _launch(cap)
        result = _run(cap.run_script(
            container_id=launch["container_id"], template_name="hello",
        ))
        assert result["exit_code"] == -1
        assert "TEST_BACKEND_URI" in result["message"]

    def test_conflicting_resolvers_raise_with_capability_names(
        self, template_package: str,
    ):
        cap = _make_cap(
            _RecordingBackend(),
            registry_yaml=_registry_yaml_for(template_package),
            sibling_capabilities=[
                _StubResolverCap({
                    "TEST_TOOL_CALL_ID": "from-A",   # claimed by A only
                    "TEST_BACKEND_URI": "file:///A",  # ALSO claimed by B
                }),
                _StubResolverCap({
                    "TEST_BACKEND_URI": "file:///B",  # collision with A
                }),
            ],
        )
        launch = _launch(cap)
        result = _run(cap.run_script(
            container_id=launch["container_id"], template_name="hello",
        ))
        assert result["exit_code"] == -1
        assert "TEST_BACKEND_URI" in result["message"]
        assert "_StubResolverCap" in result["message"]


class TestRunScriptDispatch:

    def test_literal_script_path_dispatches_via_stdin(
        self,
    ):
        backend = _RecordingBackend()
        cap = _make_cap(
            backend,
            registry_yaml=_registry_yaml_for("ignored"),
            sibling_capabilities=[],   # bayesian role has required_env=[]
        )
        launch = _run(cap.launch_container(image_role="bayesian"))
        _run(cap.run_script(
            container_id=launch["container_id"],
            script="import sys; print('inline')",
        ))
        last = backend.execs[-1]
        assert last["command"] == ["python", "-"]
        assert last["stdin"] == "import sys; print('inline')"

    def test_template_args_become_argparse_argv(self, template_package: str):
        backend = _RecordingBackend()
        cap = _make_cap(
            backend,
            registry_yaml=_registry_yaml_for(template_package),
            sibling_capabilities=[_StubResolverCap({
                "TEST_TOOL_CALL_ID": "x", "TEST_BACKEND_URI": "y",
            })],
        )
        launch = _launch(cap)
        _run(cap.run_script(
            container_id=launch["container_id"], template_name="hello",
            template_args={"input_uri": "artifact://fs/r/t/x.txt",
                           "formula": "y ~ x"},
        ))
        last = backend.execs[-1]
        assert last["command"][:2] == ["python", "-"]
        assert "--input-uri" in last["command"]
        assert "artifact://fs/r/t/x.txt" in last["command"]
        assert "--formula" in last["command"]
        assert "y ~ x" in last["command"]

    def test_extra_env_layers_on_top(self, template_package: str):
        backend = _RecordingBackend()
        cap = _make_cap(
            backend,
            registry_yaml=_registry_yaml_for(template_package),
            sibling_capabilities=[_StubResolverCap({
                "TEST_TOOL_CALL_ID": "x", "TEST_BACKEND_URI": "y",
            })],
        )
        launch = _launch(cap)
        _run(cap.run_script(
            container_id=launch["container_id"], template_name="hello",
            extra_env={"AD_HOC": "value"},
        ))
        env = backend.execs[-1]["env"]
        assert env["AD_HOC"] == "value"
        assert env["TEST_BACKEND_URI"] == "y"        # from resolver
