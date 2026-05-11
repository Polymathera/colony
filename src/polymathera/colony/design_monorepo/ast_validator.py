"""AST allow-list for files written under ``.colony/`` (and, for L1-F,
later under ``src/`` and ``tests/``).

Risk #5 in the CPS alignment plan calls for a single uniform AST gate
that applies to **both** agent-authored extensions (L1-E) and human-
edited ones — the discovery side (L1-A) cannot tell them apart, so the
write side must reject disallowed surfaces before they reach disk.

The check is intentionally narrow: it rejects the handful of names
that grant arbitrary shell / process / dynamic-code surface
(``os.system``, ``subprocess.*``, builtin ``eval`` / ``exec`` /
``compile``, dynamic ``__import__``, and imports of dangerous modules).
Everything else — including importing standard-library and Colony
modules — is allowed. The deeper sandbox is :class:`SandboxedShell`'s
container backend at run time; this validator is the cheap static gate
that catches the obvious cases at write time.

One module — one list — one validator function. Tests consume the same
constants the validator does, so the allow-list never gets enumerated
twice.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Single source of truth for the disallow-list
# ---------------------------------------------------------------------------


#: Module names that may not be imported at all in agent-authored or
#: human-edited ``.colony/`` extensions. ``subprocess`` is the obvious
#: shell-spawning surface; ``ctypes`` / ``cffi`` reach below Python's
#: type system into raw memory; ``pickle`` and ``marshal`` deserialise
#: into arbitrary code if fed adversarial bytes; ``importlib`` is the
#: dynamic-import surface ``__import__`` already covers.
DISALLOWED_IMPORT_MODULES: frozenset[str] = frozenset({
    "subprocess",
    "ctypes",
    "cffi",
    "pickle",
    "marshal",
    "importlib",
})


#: Names that may not be imported via ``from os import ...``. ``os``
#: itself is allowed (``os.path.join`` is universal), but the
#: shell-spawning subset is forbidden.
DISALLOWED_FROM_OS: frozenset[str] = frozenset({
    "system",
    "popen",
    "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "posix_spawn", "posix_spawnp",
    "fork", "forkpty",
})


#: Builtin functions that grant arbitrary code execution by name. These
#: are checked as ``ast.Call`` of an ``ast.Name`` (not attribute access),
#: which is how Python's actual builtins resolve.
DISALLOWED_BUILTIN_CALLS: frozenset[str] = frozenset({
    "eval",
    "exec",
    "compile",
    "__import__",
})


#: ``os.<name>`` and ``subprocess.<name>`` attribute accesses that grant
#: shell / process control. Stored as ``(module, attribute)`` tuples so
#: the validator can pattern-match on ``ast.Attribute`` nodes whose
#: ``.value`` is ``ast.Name``.
DISALLOWED_ATTRIBUTE_CALLS: frozenset[tuple[str, str]] = frozenset({
    ("os", "system"),
    ("os", "popen"),
    ("os", "execv"), ("os", "execve"), ("os", "execvp"), ("os", "execvpe"),
    ("os", "execl"), ("os", "execle"), ("os", "execlp"), ("os", "execlpe"),
    ("os", "spawnv"), ("os", "spawnve"), ("os", "spawnvp"), ("os", "spawnvpe"),
    ("os", "spawnl"), ("os", "spawnle"), ("os", "spawnlp"), ("os", "spawnlpe"),
    ("os", "posix_spawn"), ("os", "posix_spawnp"),
    ("os", "fork"), ("os", "forkpty"),
    ("subprocess", "run"), ("subprocess", "call"), ("subprocess", "check_call"),
    ("subprocess", "check_output"), ("subprocess", "Popen"),
    ("subprocess", "getoutput"), ("subprocess", "getstatusoutput"),
    ("importlib", "import_module"),
    ("importlib", "reload"),
    ("pickle", "loads"), ("pickle", "load"),
    ("marshal", "loads"), ("marshal", "load"),
})


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    """One disallowed-surface use in a Python source file."""

    line: int
    col: int
    surface: str
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of :func:`validate_python_source` — empty ``issues``
    means the file passed."""

    ok: bool
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)
    syntax_error: str | None = None


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    def _add(self, node: ast.AST, surface: str, detail: str) -> None:
        self.issues.append(
            ValidationIssue(
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                surface=surface,
                detail=detail,
            ),
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            # ``import a.b.c`` — ``alias.name`` is the full dotted path.
            # Reject if any prefix matches a disallowed module so that
            # ``import subprocess.foo`` (nonsense but parseable) is also
            # blocked alongside the plain ``import subprocess``.
            head = alias.name.split(".", 1)[0]
            if head in DISALLOWED_IMPORT_MODULES:
                self._add(node, "import", f"import {alias.name!r} is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        head = mod.split(".", 1)[0] if mod else ""
        if head and head in DISALLOWED_IMPORT_MODULES:
            self._add(
                node, "import",
                f"from {mod!r} import ... is not allowed",
            )
        elif mod == "os":
            for alias in node.names:
                if alias.name in DISALLOWED_FROM_OS:
                    self._add(
                        node, "import",
                        f"from os import {alias.name!r} is not allowed",
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in DISALLOWED_BUILTIN_CALLS:
            self._add(
                node, "builtin_call",
                f"call to builtin {func.id!r} is not allowed",
            )
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            pair = (func.value.id, func.attr)
            if pair in DISALLOWED_ATTRIBUTE_CALLS:
                self._add(
                    node, "attribute_call",
                    f"call to {pair[0]}.{pair[1]}(...) is not allowed",
                )
        self.generic_visit(node)


def validate_python_source(
    source: str, *, path: str | Path | None = None,
) -> ValidationReport:
    """Parse ``source`` and report disallowed surfaces.

    Returns a :class:`ValidationReport` with ``ok=False`` and a
    populated ``issues`` tuple when surfaces are rejected; or
    ``ok=False`` with ``syntax_error`` set when the source does not
    parse. Empty input is treated as ``ok=True``.
    """
    path_str = str(path) if path is not None else "<unknown>"
    try:
        tree = ast.parse(source, filename=path_str)
    except SyntaxError as exc:
        return ValidationReport(ok=False, syntax_error=str(exc))
    visitor = _Visitor()
    visitor.visit(tree)
    return ValidationReport(
        ok=not visitor.issues,
        issues=tuple(visitor.issues),
    )


def validate_python_file(path: Path) -> ValidationReport:
    """Read ``path`` and validate. UTF-8; non-text reads as syntax error."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return ValidationReport(ok=False, syntax_error=f"read {path}: {exc}")
    return validate_python_source(source, path=path)


__all__ = (
    "DISALLOWED_ATTRIBUTE_CALLS",
    "DISALLOWED_BUILTIN_CALLS",
    "DISALLOWED_FROM_OS",
    "DISALLOWED_IMPORT_MODULES",
    "ValidationIssue",
    "ValidationReport",
    "validate_python_file",
    "validate_python_source",
)
