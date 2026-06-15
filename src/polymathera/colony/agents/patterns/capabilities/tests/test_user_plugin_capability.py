"""Unit tests for ``UserPluginCapability``.

No Docker, no filesystem assumptions beyond ``tmp_path`` — each test
lays down a synthetic ``SKILL.md`` / ``plugin.json`` and feeds the
paths into the capability via explicit ``skill_roots`` /
``plugin_roots``.

Sandbox execution is mocked: a tiny stub ``SandboxedShellCapability``
replacement records the calls and returns canned results so we can
prove the capability wires things through correctly without bringing
up a container.
"""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.patterns.capabilities._plugin import (
    parse_frontmatter,
    discover_skills,
    discover_plugins,
    SkillSource,
    SkillSpec,
)
from polymathera.colony.agents.patterns.capabilities.sandboxed_shell import (
    SandboxedShellCapability,
)
from polymathera.colony.agents.patterns.capabilities.user_plugin import (
    UserPluginCapability,
    _shell_quote,
)
from polymathera.colony.agents.scopes import BlackboardScope
from polymathera.colony.distributed.ray_utils.serving.context import (
    execution_context, Ring,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_skill(
    root: Path, name: str, *,
    frontmatter: str, body: str = "body text",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\n{frontmatter.strip()}\n---\n{body}\n",
        encoding="utf-8",
    )
    return skill_dir


def _write_plugin(
    root: Path, plugin_name: str, *,
    manifest: dict[str, Any],
    skills: dict[str, tuple[str, str]] | None = None,
) -> Path:
    plugin_dir = root / plugin_name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / ".claude-plugin").mkdir(exist_ok=True)
    import json as _json
    (plugin_dir / ".claude-plugin" / "plugin.json").write_text(
        _json.dumps(manifest), encoding="utf-8",
    )
    skills_root = plugin_dir / "skills"
    if skills:
        for sname, (fm, body) in skills.items():
            _write_skill(skills_root, sname, frontmatter=fm, body=body)
    return plugin_dir


class _SandboxStub(SandboxedShellCapability):
    """Subclass so ``isinstance(cap, SandboxedShellCapability)`` holds."""

    def __init__(self):
        # Bypass the real __init__ — we don't want backend/registry.
        self.launch_calls: list[dict] = []
        self.exec_calls: list[dict] = []
        self.stop_calls: list[str] = []
        self.exec_exit_code: int = 0
        self.launch_should_fail: bool = False

    async def launch_container(self, **kwargs):
        self.launch_calls.append(kwargs)
        if self.launch_should_fail:
            return {"started": False, "message": "simulated launch failure"}
        return {
            "started": True,
            "container_id": "stub_cid",
            "container_name": "stub",
            "image": "stub",
            "workspace_path": "/tmp/ws",
            "owner_agent_id": "agent-A",
            "shared": False,
            "message": "",
        }

    async def execute_command(self, **kwargs):
        self.exec_calls.append(kwargs)
        return {
            "container_id": kwargs.get("container_id"),
            "exec_id": "e",
            "command": kwargs.get("command"),
            "exit_code": self.exec_exit_code,
            "stdout": "ok",
            "stderr": "",
            "wall_time_ms": 1,
            "truncated": False,
            "stream_key": None,
            "message": "",
        }

    async def stop_container(self, cid, **_):
        self.stop_calls.append(cid)
        return {"stopped": True, "message": ""}


def _make_cap(
    *,
    skill_roots: list[Path] | None = None,
    plugin_roots: list[Path] | None = None,
    sandbox: _SandboxStub | None = None,
    allow_disabled: bool = False,
) -> UserPluginCapability:
    agent = MagicMock()
    agent.agent_id = "agent-A"
    agent.get_capability_by_type = MagicMock(return_value=sandbox)
    cap = UserPluginCapability(
        agent=agent,
        scope=BlackboardScope.SESSION,
        skill_roots=[str(p) for p in (skill_roots or [])],
        plugin_roots=[str(p) for p in (plugin_roots or [])],
        allow_model_invocation_override=allow_disabled,
    )
    return cap


def _run(coro):
    return asyncio.run(coro)


def _with_context():
    return execution_context(
        ring=Ring.USER, tenant_id="t1", colony_id="c1", session_id="s1",
    )


# ---------------------------------------------------------------------------
# Frontmatter + discovery
# ---------------------------------------------------------------------------

def test_parse_frontmatter_handles_no_frontmatter():
    fm, body = parse_frontmatter("just body\ncontent\n")
    assert fm == {}
    assert body == "just body\ncontent\n"


def test_parse_frontmatter_extracts_yaml_and_body():
    doc = textwrap.dedent("""
        ---
        name: mytool
        params:
          foo:
            type: string
            required: true
        ---
        # Title
        body
    """).strip() + "\n"
    fm, body = parse_frontmatter(doc)
    assert fm["name"] == "mytool"
    assert fm["params"]["foo"]["required"] is True
    assert body.startswith("# Title")


def test_parse_frontmatter_rejects_unterminated_block():
    with pytest.raises(ValueError, match="no closing"):
        parse_frontmatter("---\nname: x\nno close here\n")


def test_discover_skills_walks_roots_in_priority_order(tmp_path):
    # Two skills, same name, different sources → SESSION wins.
    workspace = tmp_path / "workspace"
    user = tmp_path / "user"
    workspace.mkdir(); user.mkdir()
    _write_skill(
        workspace, "alpha",
        frontmatter="name: alpha\ndescription: ws\nscript: run.sh",
    )
    _write_skill(
        user, "alpha",
        frontmatter="name: alpha\ndescription: user\nscript: run.sh",
    )
    result = discover_skills([
        (workspace, SkillSource.SESSION),
        (user, SkillSource.USER),
    ])
    assert list(result.skills.keys()) == ["alpha"]
    assert result.skills["alpha"].description == "ws"
    # The user-source skill is reported as a collision.
    [c] = result.collisions
    assert c == ("alpha", "session", "user")


def test_discover_skills_logs_errors_for_malformed_files(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    # No 'name' field.
    _write_skill(root, "bad", frontmatter="description: oops")
    # Unterminated frontmatter.
    (root / "worse").mkdir()
    (root / "worse" / "SKILL.md").write_text("---\nname: w\n")
    # Valid.
    _write_skill(
        root, "good",
        frontmatter="name: good\nscript: run.sh",
    )
    result = discover_skills([(root, SkillSource.SESSION)])
    assert set(result.skills.keys()) == {"good"}
    assert len(result.errors) == 2


def test_discover_plugins_namespaces_skills_by_plugin_name(tmp_path):
    plugins_root = tmp_path / "plugins"
    plugins_root.mkdir()
    _write_plugin(
        plugins_root, "my-plug",
        manifest={
            "name": "my-plug", "version": "0.1.0",
            "description": "d", "skills": ["foo"],
        },
        skills={
            "foo": (
                "name: foo\nscript: r.sh\ndescription: inside plugin",
                "body",
            ),
        },
    )
    result = discover_plugins([
        (plugins_root, SkillSource.SESSION),
    ])
    assert "my-plug/foo" in result.skills
    assert result.plugins["my-plug"].version == "0.1.0"


# ---------------------------------------------------------------------------
# Capability action surface
# ---------------------------------------------------------------------------

def test_list_skills_returns_loaded_skills(tmp_path):
    root = tmp_path / "skills"; root.mkdir()
    _write_skill(
        root, "alpha",
        frontmatter="name: alpha\ndescription: first\nscript: r.sh",
    )
    _write_skill(
        root, "beta",
        frontmatter="name: beta\ndescription: second\nscript: r.sh",
    )
    with _with_context():
        cap = _make_cap(skill_roots=[root])
        result = _run(cap.list_skills())
    names = [s["name"] for s in result["skills"]]
    assert sorted(names) == ["alpha", "beta"]


def test_list_skills_filters_by_source(tmp_path):
    workspace = tmp_path / "ws"; workspace.mkdir()
    system = tmp_path / "sys"; system.mkdir()
    _write_skill(
        workspace, "ws_skill",
        frontmatter="name: ws_skill\nscript: r.sh",
    )
    _write_skill(
        system, "sys_skill",
        frontmatter="name: sys_skill\nscript: r.sh",
    )
    agent = MagicMock(); agent.agent_id = "agent-A"
    agent.get_capability_by_type = MagicMock(return_value=None)
    with _with_context():
        cap = UserPluginCapability(
            agent=agent, scope=BlackboardScope.SESSION,
            skill_roots=None,       # use priority defaults
            plugin_roots=None,
        )
        # Override the resolver: session=workspace, system=system.
        cap._skill_roots = [
            (workspace, SkillSource.SESSION),
            (system, SkillSource.SYSTEM),
        ]
        cap._plugin_roots = []
        cap._load()
        session_only = _run(cap.list_skills(source="session"))
        system_only = _run(cap.list_skills(source="system"))
    assert {s["name"] for s in session_only["skills"]} == {"ws_skill"}
    assert {s["name"] for s in system_only["skills"]} == {"sys_skill"}


def test_list_skills_rejects_unknown_source(tmp_path):
    with _with_context():
        cap = _make_cap(skill_roots=[tmp_path])
        result = _run(cap.list_skills(source="nope"))
    assert result["count"] == 0
    assert "unknown source" in result["message"]


def test_get_skill_returns_detail_for_qualified_name(tmp_path):
    root = tmp_path / "skills"; root.mkdir()
    _write_skill(
        root, "foo",
        frontmatter=(
            "name: foo\n"
            "description: a description\n"
            "script: run.sh\n"
            "params:\n  x: {type: string, required: true}\n"
        ),
    )
    with _with_context():
        cap = _make_cap(skill_roots=[root])
        result = _run(cap.get_skill(name="foo"))
    assert result["found"] is True
    assert result["skill"]["name"] == "foo"
    assert result["skill"]["params"] == [
        {"name": "x", "type": "string", "required": True},
    ]


def test_search_skills_orders_name_match_before_description(tmp_path):
    root = tmp_path / "skills"; root.mkdir()
    _write_skill(
        root, "verilator_lint",
        frontmatter=(
            "name: verilator_lint\n"
            "description: Runs lint on systemverilog\n"
            "script: r.sh\n"
        ),
    )
    _write_skill(
        root, "general",
        frontmatter=(
            "name: general\n"
            "description: mentions verilator in passing\n"
            "script: r.sh\n"
        ),
    )
    with _with_context():
        cap = _make_cap(skill_roots=[root])
        result = _run(cap.search_skills(query="verilator"))
    assert [s["name"] for s in result["skills"]] == [
        "verilator_lint", "general",
    ]


def test_search_skills_rejects_empty_query(tmp_path):
    with _with_context():
        cap = _make_cap(skill_roots=[tmp_path])
        result = _run(cap.search_skills(query=""))
    assert result["count"] == 0
    assert "empty query" in result["message"]


def test_search_skills_skips_model_disabled_skills(tmp_path):
    root = tmp_path / "skills"; root.mkdir()
    _write_skill(
        root, "secret",
        frontmatter=(
            "name: secret\n"
            "description: only human-invocable\n"
            "script: r.sh\n"
            "disable-model-invocation: true\n"
        ),
    )
    with _with_context():
        cap = _make_cap(skill_roots=[root])
        result = _run(cap.search_skills(query="secret"))
    assert result["count"] == 0


def test_reload_skills_picks_up_newly_added_files(tmp_path):
    root = tmp_path / "skills"; root.mkdir()
    with _with_context():
        cap = _make_cap(skill_roots=[root])
        assert _run(cap.list_skills())["count"] == 0
        _write_skill(
            root, "fresh",
            frontmatter="name: fresh\nscript: r.sh",
        )
        reload_r = _run(cap.reload_skills())
        assert reload_r["skill_count"] == 1
        assert _run(cap.list_skills())["count"] == 1


# ---------------------------------------------------------------------------
# run_skill
# ---------------------------------------------------------------------------

def test_run_skill_launches_container_and_execs_script(tmp_path):
    root = tmp_path / "skills"; root.mkdir()
    _write_skill(
        root, "lint",
        frontmatter=(
            "name: lint\n"
            "description: lint files\n"
            "script: lint.sh\n"
            "sandbox_image_role: code_analysis\n"
            "params:\n  path: {type: string, required: true}\n"
            "timeout_seconds: 120\n"
        ),
    )
    sandbox = _SandboxStub()
    with _with_context():
        cap = _make_cap(skill_roots=[root], sandbox=sandbox)
        result = _run(cap.run_skill(
            name="lint", params={"path": "src/app.py"},
        ))
    assert result["exit_code"] == 0
    # Container launched with the declared role, skill dir mounted ro.
    [launch] = sandbox.launch_calls
    assert launch["image_role"] == "code_analysis"
    mounts = launch["extra_volumes"]
    assert any(
        m["dst"] == "/skill" and m["mode"] == "ro"
        for m in mounts
    )
    [exec_call] = sandbox.exec_calls
    cmd = exec_call["command"]
    # bash -lc 'cd /skill && bash lint.sh --path src/app.py'
    assert cmd[0] == "bash" and cmd[1] == "-lc"
    assert "cd /skill" in cmd[2]
    assert "lint.sh" in cmd[2]
    assert "--path" in cmd[2]
    assert exec_call["timeout_seconds"] == 120
    assert sandbox.stop_calls == ["stub_cid"]


def test_run_skill_respects_provided_container_id(tmp_path):
    root = tmp_path / "skills"; root.mkdir()
    _write_skill(
        root, "x",
        frontmatter="name: x\nscript: r.sh",
    )
    sandbox = _SandboxStub()
    with _with_context():
        cap = _make_cap(skill_roots=[root], sandbox=sandbox)
        _run(cap.run_skill(name="x", container_id="caller-owned"))
    assert sandbox.launch_calls == []
    assert sandbox.stop_calls == []
    assert sandbox.exec_calls[0]["container_id"] == "caller-owned"


def test_run_skill_validates_param_type(tmp_path):
    root = tmp_path / "skills"; root.mkdir()
    _write_skill(
        root, "counts",
        frontmatter=(
            "name: counts\n"
            "script: r.sh\n"
            "params:\n  n: {type: integer, required: true}\n"
        ),
    )
    sandbox = _SandboxStub()
    with _with_context():
        cap = _make_cap(skill_roots=[root], sandbox=sandbox)
        r = _run(cap.run_skill(name="counts", params={"n": "five"}))
    assert r["exit_code"] == -1
    assert "expected type integer" in r["message"]
    assert sandbox.exec_calls == []


def test_run_skill_rejects_missing_required_param(tmp_path):
    root = tmp_path / "skills"; root.mkdir()
    _write_skill(
        root, "needs",
        frontmatter=(
            "name: needs\n"
            "script: r.sh\n"
            "params:\n  foo: {type: string, required: true}\n"
        ),
    )
    sandbox = _SandboxStub()
    with _with_context():
        cap = _make_cap(skill_roots=[root], sandbox=sandbox)
        r = _run(cap.run_skill(name="needs", params={}))
    assert "missing required" in r["message"]


def test_run_skill_refuses_when_sandbox_missing(tmp_path):
    root = tmp_path / "skills"; root.mkdir()
    _write_skill(
        root, "simple",
        frontmatter="name: simple\nscript: r.sh",
    )
    with _with_context():
        cap = _make_cap(skill_roots=[root], sandbox=None)
        r = _run(cap.run_skill(name="simple"))
    assert r["exit_code"] == -1
    assert "requires SandboxedShellCapability" in r["message"]


def test_run_skill_surfaces_launch_failure(tmp_path):
    root = tmp_path / "skills"; root.mkdir()
    _write_skill(
        root, "x",
        frontmatter="name: x\nscript: r.sh",
    )
    sandbox = _SandboxStub()
    sandbox.launch_should_fail = True
    with _with_context():
        cap = _make_cap(skill_roots=[root], sandbox=sandbox)
        r = _run(cap.run_skill(name="x"))
    assert r["exit_code"] == -1
    assert "simulated launch failure" in r["message"]


def test_run_skill_rejects_disabled_invocation(tmp_path):
    root = tmp_path / "skills"; root.mkdir()
    _write_skill(
        root, "no_llm",
        frontmatter=(
            "name: no_llm\n"
            "script: r.sh\n"
            "disable-model-invocation: true\n"
        ),
    )
    sandbox = _SandboxStub()
    with _with_context():
        cap = _make_cap(skill_roots=[root], sandbox=sandbox)
        r = _run(cap.run_skill(name="no_llm"))
    assert r["exit_code"] == -1
    assert "disable-model-invocation" in r["message"]


def test_run_skill_honours_timeout_override(tmp_path):
    root = tmp_path / "skills"; root.mkdir()
    _write_skill(
        root, "x",
        frontmatter=(
            "name: x\nscript: r.sh\ntimeout_seconds: 60\n"
        ),
    )
    sandbox = _SandboxStub()
    with _with_context():
        cap = _make_cap(skill_roots=[root], sandbox=sandbox)
        _run(cap.run_skill(name="x", timeout_seconds=7))
    assert sandbox.exec_calls[0]["timeout_seconds"] == 7


# ---------------------------------------------------------------------------
# Quoting helper
# ---------------------------------------------------------------------------

def test_shell_quote_keeps_simple_strings_unchanged():
    assert _shell_quote("hello") == "hello"
    assert _shell_quote("path/to/file.txt") == "path/to/file.txt"


def test_shell_quote_protects_special_characters():
    assert _shell_quote("a b") == "'a b'"
    assert _shell_quote("it's") == "'it'\"'\"'s'"
    assert _shell_quote("") == "''"


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------

def test_bind_round_trips_through_cloudpickle():
    # Ray's vendored cloudpickle — see comment in
    # test_github_capability for why standalone PyPI cloudpickle is
    # not the right import here.
    from ray import cloudpickle
    bp = UserPluginCapability.bind(scope=BlackboardScope.SESSION)
    bp2 = cloudpickle.loads(cloudpickle.dumps(bp))
    assert bp2.cls is UserPluginCapability


def test_action_executors_are_registered():
    import inspect
    keys = {
        m._action_key for _, m in inspect.getmembers(
            UserPluginCapability, predicate=inspect.isfunction,
        ) if getattr(m, "_action_key", None)
    }
    assert keys == {
        "list_skills", "get_skill", "search_skills",
        "list_plugins", "reload_skills", "run_skill",
    }


def test_action_group_description_lists_loaded_skills(tmp_path):
    root = tmp_path / "skills"; root.mkdir()
    _write_skill(
        root, "alpha",
        frontmatter=(
            "name: alpha\n"
            "description: a short description for alpha\n"
            "script: r.sh\n"
        ),
    )
    with _with_context():
        cap = _make_cap(skill_roots=[root])
        desc = cap.get_action_group_description()
    assert "User Plugins" in desc
    assert "alpha" in desc
    assert "a short description for alpha" in desc
