# Project-substance authoring (L1-F)

The write surface for everything *outside* `.colony/` — `src/`, `tests/`, `data/`, `dossier/`, `models/`, `notebooks/`, `docs/`. Symmetric companion to L1-E (meta-tooling write surface under `.colony/`). Both halves share the same audit pipeline (per Risk #5 in [`CPS_ALIGNMENT_PLAN.md`](../../../cps/CPS_ALIGNMENT_PLAN.md)) — same AST allow-list, same `DesignCheckpointer`-style commit attribution, same blackboard event protocol.

## TL;DR

```python
from polymathera.colony.design_monorepo import ProjectAuthoringCapability

cap = ProjectAuthoringCapability(agent=..., working_dir=repo_root)

await cap.write_file("src/opm_meg/serf/calibration.py", body)
await cap.write_file("tests/serf/test_calibration.py", test_body)
await cap.write_file("dossier/510k/section_4_calibration.md", "# Section 4\n…")
await cap.edit_file("src/opm_meg/__init__.py", old_snippet, new_snippet)
await cap.insert_lines("src/opm_meg/serf/__init__.py", after_line=0, content=import_line)
```

Each call returns a [`ProjectArtifactAuthoredPayload`](../../src/polymathera/colony/design_monorepo/models.py) with `action_kind`, `affected_paths`, `commit_sha`, `pre_commit_validation_results`, `session_id`, `authored_at`.

## Seven low-level actions

Action kinds come from [`PROJECT_ACTION_KINDS`](../../src/polymathera/colony/design_monorepo/models.py) — the single source of truth for the set. The `ProjectArtifactAuthoredPayload.action_kind` `Literal` enumerates the same strings; a drift-detector test asserts they stay in sync.

| Method | Effect | Validators applied to |
|---|---|---|
| `write_file(path, content)` | Create or overwrite | resulting file |
| `edit_file(path, old_content, new_content)` | Unique-match string replace | resulting file |
| `delete_file(path)` | Remove | none (file is gone) |
| `move_file(src, dst)` | Rename | resulting file (`dst`) |
| `insert_lines(path, after_line, content)` | Insert (`after_line=0` for prepend, `=N` for append) | resulting file |
| `delete_lines(path, start_line, end_line)` | Drop inclusive 1-indexed range | resulting file |
| `replace_lines(path, start_line, end_line, content)` | Replace inclusive 1-indexed range | resulting file |

**Guiding principle** ([`CPS_ALIGNMENT_PLAN.md`](../../../cps/CPS_ALIGNMENT_PLAN.md)): a finite, validatable set of low-level operations beats either (a) shell access or (b) a sprawling catalog of high-level / language-specific ops. Higher-level outcomes — "add a Python module with tests and a dossier section" — are *sequences* of these emitted by the planner; CPS-shaped planner prompts live in L2-G (PR 6).

## Pipeline (every action)

The seven actions all funnel through one private helper `_run_action_sync` so the discipline can't drift between methods:

1. **Snapshot** — capture pre-action content (or absence) of every affected path. The rollback record.
2. **Mutate** — apply the in-process file operation.
3. **Validate** — run every registered validator whose matcher matches the resulting path. All must pass.
4. **Rollback on failure** — restore snapshots, then raise `DesignMonorepoError` with the validator's `detail`.
5. **Commit + event** — `client.commit_with_identity(identity, "L1-F <action_kind>: <primary_path>", paths=[...])`, then emit `ProjectArtifactAuthored` on the blackboard.

## Path safety

Every method validates its path arguments through `_resolve_safe_path` (in [`capabilities.py`](../../src/polymathera/colony/design_monorepo/capabilities.py)). It rejects:

- Absolute paths.
- Path-traversal (`..` that escapes `working_dir`).
- Anything under `.colony/` — that's L1-E's surface, route through `ToolBuilder`.
- Anything under `.git/` — git internals.
- Empty paths.

The forbidden top-levels are declared once as `_L1F_FORBIDDEN_TOP_LEVEL = frozenset({".colony", ".git"})` — single source of truth.

## Validator registry

[`colony.design_monorepo.artifact_validators`](../../src/polymathera/colony/design_monorepo/artifact_validators.py) holds one ordered list of `(matcher, validator)` triples. The capability queries `validators_for(path)`, runs every match, and aborts the commit on the first failure.

Built-in validators (`DEFAULT_VALIDATORS`):

| Validator | Matches | Source |
|---|---|---|
| `ast_allow_list` | `.py` under `src/` or `tests/` | reuses [`ast_validator.py`](../../src/polymathera/colony/design_monorepo/ast_validator.py) — same module L1-E goes through |
| `pytest_collect` | `tests/**/*.py` | subprocess `pytest --collect-only`; passes when pytest is absent |
| `dossier_markdown` | `dossier/**/*.md` | UTF-8 + non-empty + at least one heading |
| `ipynb_ast` | `*.ipynb` | parses JSON cells; runs AST allow-list on each Python code cell |
| `cad_non_empty` | `.step` / `.stp` / `.iges` / `.igs` | stub: non-empty file |
| `fea_input_non_empty` | `.inp` / `.med` | stub: non-empty file |
| `reqif_non_empty` | `.reqif` | stub: non-empty file |

CPS PR 6 (L2-G) plugs in real domain validators via `register_validator(ArtifactValidator(...))`. Multiple validators on the same path stack — all must pass.

```python
from polymathera.colony.design_monorepo.artifact_validators import (
    ArtifactValidator, register_validator,
)

def _validate_step(repo_root, rel):
    # parse via OpenCascade / pythonocc-core
    ...

register_validator(ArtifactValidator(
    name="cad_step_parser",
    matcher=lambda p: p.suffix in {".step", ".stp"},
    run=_validate_step,
))
```

The AST allow-list is the SAME module as L1-E. One uniform pipeline across both halves of L4 — Risk #5's stopgap defence.

## Audit trail

Symmetric with L1-E's [`ExtensionAuthored`](design-monorepo-authoring.md):

```python
from polymathera.colony.agents.blackboard.protocol import DesignMonorepoEventProtocol

key = DesignMonorepoEventProtocol.project_artifact_authored_key(
    "write_file", "src/opm_meg/calibration.py",
)
entry = await bb.read(key)
# entry.value: dict matching ProjectArtifactAuthoredPayload, including
# pre_commit_validation_results for the audit log.
```

Subscribe with the wildcard:

```python
@event_handler(pattern=DesignMonorepoEventProtocol.project_artifact_authored_pattern())
async def _on_authored(self, event, repl) -> EventProcessingResult | None:
    kind, primary_path = DesignMonorepoEventProtocol.parse_project_artifact_authored_key(event.key)
    ...
```

`session_id` is populated automatically from [`get_current_session_id()`](../../src/polymathera/colony/agents/sessions/context.py). `user_message_id` is reserved for a future provenance plumbing pass.

## Read-side companions

[`RepoStateProvider`](../../src/polymathera/colony/design_monorepo/capabilities.py) (extended by this PR) exposes three planner-friendly read methods:

- `list_packages()` — top-level Python packages under `src/` (directories with `__init__.py`).
- `list_design_artifacts(kind=...)` — files by kind: `cad`, `fea`, `reqif`, `notebook`, `dossier`, `test`, `python_module`. `kind=None` returns everything.
- `summarize_project_layout()` — `{"top_level": [...], "counts": {kind: n}}`. The "what's already here" snapshot the planner reads on its first turn.

All three exclude `.colony/` and `.git/`.

## Boundary with `SandboxedShellCapability`

The shell capability ships actions named [`write_file` / `edit_file`](../../src/polymathera/colony/agents/patterns/capabilities/sandboxed_shell.py) too — these operate *inside the container* (not the working tree) and are appropriate for:

- Debug prints written into a sandboxed scratch dir.
- Temporary fixtures the agent generates during exploration.
- Anything the agent intends to be **thrown away** at session end.

`ProjectAuthoringCapability` is the **authoritative** write surface for design substance — anything that should land in the monorepo, be committed under the agent's identity, and be visible to the next session. CPS coordinator planner guidance (L2-G) routes design-substance mutations through `ProjectAuthoringCapability`.

### Risk #9 lint

[`colony.design_monorepo.session_lint`](../../src/polymathera/colony/design_monorepo/session_lint.py) ships `lint_session_actions(actions: Iterable[dict]) -> list[LintFinding]` that flags any `SandboxedShellCapability.write_file` / `edit_file` action targeting a design-artifact top-level directory (`src/`, `tests/`, `dossier/`, `data/`, `models/`, `notebooks/`, `docs/`). The top-levels are declared once as `DESIGN_ARTIFACT_TOP_LEVELS` — single source of truth.

Findings are pure data; CI wraps the function and either fails the build or surfaces them as warnings. Conservative by design — only flags what we can identify with confidence.

## What's not in scope for PR 3

- Per-domain validators (CAD parser, FEA solver-input parser, ReqIF schema, dossier markdown lint) — PR 6 plugs these in via `register_validator(...)`. PR 3 ships the registry and non-empty stubs.
- CPS-shaped scaffolds for project substance (`project_layout`, `module_*`, `artifact_*`) — PR 6.
- Per-domain planner prompts that route requests through these actions — PR 6.
- Wiring the lint into a `polymath` CLI subcommand — defer until a session-transcript schema is finalised.
