"""Tests for the scaffold renderer + each shipped template."""

from __future__ import annotations

from pathlib import Path

import pytest

from polymathera.colony.design_monorepo.scaffolds import (
    AVAILABLE_TEMPLATES,
    ScaffoldRenderError,
    list_template_files,
    render_template,
)


def test_all_templates_listed_have_files() -> None:
    for tpl in AVAILABLE_TEMPLATES:
        files = list_template_files(tpl)
        assert files, f"Scaffold {tpl} declared no files."


def test_unknown_template_raises(tmp_path: Path) -> None:
    with pytest.raises(ScaffoldRenderError):
        render_template(
            "missing-template",  # type: ignore[arg-type]
            tmp_path / "out",
            name="x",
            purpose="test",
            license_id="MIT",
        )


def test_render_python_lib_substitutes(tmp_path: Path) -> None:
    target = tmp_path / "tool"
    files = render_template(
        "python_lib",
        target,
        name="widget_engine",
        purpose="shared/widgets",
        license_id="MIT",
        description="Widgets",
    )
    assert "src/widget_engine/__init__.py" in files
    pyproj = (target / "pyproject.toml").read_text("utf-8")
    assert "name = \"widget-engine\"" in pyproj
    init = (target / "src/widget_engine/__init__.py").read_text("utf-8")
    assert "widget_engine" in init


def test_render_refuses_nonempty(tmp_path: Path) -> None:
    target = tmp_path / "tool"
    target.mkdir()
    (target / "stub").write_text("x")
    with pytest.raises(ScaffoldRenderError):
        render_template(
            "python_lib", target,
            name="x", purpose="p", license_id="MIT",
        )


@pytest.mark.parametrize("template", AVAILABLE_TEMPLATES)
def test_each_template_renders(tmp_path: Path, template: str) -> None:
    target = tmp_path / template
    files = render_template(
        template, target,
        name="gizmo_smith", purpose="shared/test",
        license_id="MIT", description="Gizmo.",
    )
    assert files
    for rel in files:
        assert (target / rel).is_file(), rel


def test_initial_files_override(tmp_path: Path) -> None:
    target = tmp_path / "tool"
    files = render_template(
        "python_lib", target,
        name="x", purpose="p", license_id="MIT",
        initial_files={"src/x/core.py": "def run():\n    return 1\n"},
    )
    core = (target / "src/x/core.py").read_text("utf-8")
    assert core.strip().endswith("return 1")


def test_template_vars_passthrough(tmp_path: Path) -> None:
    target = tmp_path / "tool"
    render_template(
        "python_lib", target,
        name="x", purpose="p", license_id="MIT",
        template_vars={"author": "Custom Author"},
    )
    pyproj = (target / "pyproject.toml").read_text("utf-8")
    assert "Custom Author" in pyproj
