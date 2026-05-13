"""Language-agnostic scaffolds for tool-building pools.

A scaffold is a directory of files (with ``$variable`` placeholders) that
``ToolBuilder.bootstrap_repo`` materializes into ``tools/<purpose>/<name>/``
in the design monorepo. The set of templates the framework ships is the
language-agnostic minimum (master §9.4):

- ``python_lib`` — a Python package with pyproject.toml, tests, README,
  one entry-point module, MIT-by-default LICENCE.
- ``c_library`` — a C library with CMakeLists, src/, include/, a ctest
  smoke test.
- ``julia_module`` — a Julia package with Project.toml, src, runtests.
- ``rust_crate`` — a Rust crate with Cargo.toml, src/lib.rs, tests/.
- ``cmake_project`` — a generic CMake project skeleton (for non-C tools
  that ship a CMake build, e.g. C++ codes).

Higher-level *flavours* (Python lib in the ``polymathera.cps`` namespace,
FEniCS plug-in shape, Julia SciML module shape) live in
``cps/tools/scaffolds/`` per master §9.4.
"""

from __future__ import annotations

from .registry import (
    ExtensionScaffold,
    ExtensionScaffoldRegistryError,
    available_scaffolds,
    get_extension_scaffold,
    register_extension_scaffold,
    reset_registry,
)
from .renderer import (
    AVAILABLE_TEMPLATES,
    ScaffoldRenderError,
    list_template_files,
    render_extension_scaffold,
    render_template,
)


__all__ = (
    "AVAILABLE_TEMPLATES",
    "ExtensionScaffold",
    "ExtensionScaffoldRegistryError",
    "ScaffoldRenderError",
    "available_scaffolds",
    "get_extension_scaffold",
    "list_template_files",
    "register_extension_scaffold",
    "render_extension_scaffold",
    "render_template",
    "reset_registry",
)
