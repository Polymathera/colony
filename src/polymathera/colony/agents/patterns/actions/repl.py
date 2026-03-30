"""Python REPL execution context for ActionPolicy dataflow.

This module implements PolicyPythonREPL, an IPython-based Python REPL execution
context that provides rich dataflow capabilities between actions in an ActionPolicy.

Key features:
- Variable storage with metadata (descriptions, timestamps, provenance)
- Large data references via external backing stores (blackboard, S3, etc.)
- Code execution with native async support via IPython
- Queryable variable catalog for LLM planning
- Recursive action generation (REPL code can submit Actions)
- Code export to procedural memory and planning context
- Magic commands for LLM self-profiling (%timeit, %who, etc.)

Design philosophy:
- IPython for native async and persistent state across executions
- BackingStore protocol for dependency injection (not hardcoded storage)
- LLM-controlled storage decisions via StorageHint
- Sandboxed execution with restricted builtins and import validation

Why IPython over code.InteractiveConsole:
- Native async support via autoawait and run_cell_async()
- Persistent state across executions (user_ns)
- Magic commands for LLM self-profiling (%timeit, %who)
- Structured ExecutionResult with success/error info
- Memory overhead (~25MB) is negligible vs LLM agent footprint on Ray
- Startup time (~1-2s) is negligible vs LLM inference time
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from overrides import override

from IPython.core.interactiveshell import InteractiveShell

from ....utils import setup_logger
from ...models import PolicyREPL
from .backing_store import BackingStore, BlackboardBackingStore, StorageHint

from ...base import AgentCapability
from ...scopes import ScopeUtils, BlackboardScope, get_scope_prefix

if TYPE_CHECKING:
    from ...base import Agent
    from ...models import Action, ActionResult, AgentSuspensionState

logger = setup_logger(__name__)


# =============================================================================
# Dangerous builtins to exclude from REPL namespace
# =============================================================================

# These builtins are excluded for security:
# - Code execution: eval, exec, compile, __import__
# - File/IO access: open, input
# - Introspection that can leak info or escape sandbox: globals, locals, vars, dir
# - Process control: breakpoint, exit, quit
# - Help (can access filesystem): help
DANGEROUS_BUILTINS = frozenset({
    "eval", "exec", "compile", "__import__",
    "open", "input",
    "globals", "locals", "vars", "dir",
    "breakpoint", "exit", "quit",
    "help",
})


@dataclass
class REPLVariable:
    """Variable with rich metadata for LLM context.

    Variables can be stored directly (small data) or referenced via
    external backing stores (large data).

    Attributes:
        name: Variable name (valid Python identifier)
        description: LLM-readable description of the value
        dtype: Type annotation string (e.g., "list[str]", "dict")
        created_at: Timestamp when variable was created
        created_by: action_id that created this variable, or "repl_code"
        access_count: Number of times this variable was accessed
        last_accessed: Timestamp of last access
        is_reference: Whether value is stored in external backing store
        value: For direct storage, stored here. None for references.
        backing_store: Name of backing store (e.g., "blackboard", "s3")
        backing_key: Key in the backing store
    """

    name: str
    description: str
    dtype: str
    created_at: float
    created_by: str
    access_count: int = 0
    last_accessed: float | None = None

    # Value storage
    is_reference: bool = False
    value: Any | None = None

    # External backing store reference (for large data)
    backing_store: str | None = None
    backing_key: str | None = None

    def get_preview(self) -> str:
        """Get preview for LLM context.

        Returns:
            For direct storage: repr(value)[:100]
            For references: "[REFERENCE: backing_store:backing_key]"
        """
        if self.is_reference:
            return f"[REFERENCE: {self.backing_store}:{self.backing_key}]"
        if self.value is not None:
            preview = repr(self.value)
            return preview[:100] + "..." if len(preview) > 100 else preview
        return "None"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "name": self.name,
            "description": self.description,
            "dtype": self.dtype,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "is_reference": self.is_reference,
            "value": self.value if not self.is_reference else None,
            "backing_store": self.backing_store,
            "backing_key": self.backing_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "REPLVariable":
        """Deserialize from dictionary."""
        return cls(**data)


class PolicyPythonREPL(PolicyREPL):
    """Python REPL execution context for ActionPolicy using IPython.

    Provides:
    - Variable storage with metadata
    - Large data references via external backing stores (blackboard, S3, etc.)
    - Code execution with native async support
    - Queryable variable catalog
    - Recursive action generation
    - Magic commands for LLM self-profiling

    Uses IPython.InteractiveShell for:
    - Native async/await via autoawait and run_cell_async()
    - Persistent state across executions (user_ns)
    - Structured ExecutionResult with success/error info
    - Magic commands (%timeit, %who) for self-profiling

    Example:
        ```python
        repl = PolicyPythonREPL(agent=agent)

        # Set variable with metadata
        await repl.set("query", "How does auth work?", description="User query")

        # Execute code (native async)
        result = await repl.execute('''
            high_priority = [p for p in page_ids if p.startswith("core/")]
            for page_id in high_priority:
                submit_action(Action(
                    action_type=ActionType.ANALYZE_PAGE,
                    parameters={"page_id": page_id}
                ))
        ''')

        # Get variable summary for LLM context
        summary = repl.get_variable_summary()

        # Export code history for procedural memory
        code_history = repl.export_code_history()
        ```
    """

    def __init__(
        self,
        agent: "Agent",
        backing_stores: dict[str, BackingStore] | None = None,
        allowed_imports: list[str] | None = None,
        max_execution_time: float = 5.0,
        restrict_builtins: bool = True,
    ):
        """Initialize REPL with IPython.

        Args:
            agent: Agent that owns this REPL
            backing_stores: Map of store name -> BackingStore implementation
                If None, creates default BlackboardBackingStore
            allowed_imports: List of allowed import module names
            max_execution_time: Max seconds for code execution
            restrict_builtins: If True, restrict dangerous builtins (default: True)
        """
        self._agent = agent
        self._variables: dict[str, REPLVariable] = {}
        self._pending_actions: list[Action] = []
        self._allowed_imports = allowed_imports or self._default_allowed_imports()
        self._max_execution_time = max_execution_time
        self._restrict_builtins = restrict_builtins
        self._code_history: list[dict[str, Any]] = []  # For procedural memory export

        # Action results storage
        self._results: dict[str, "ActionResult"] = {}

        # Backing stores are injected dependencies
        if backing_stores is None:
            self._backing_stores: dict[str, BackingStore] = {
                "blackboard": BlackboardBackingStore(agent)
            }
        else:
            self._backing_stores = backing_stores

        # Create IPython shell
        self._shell = self._create_shell()

    @property
    def max_execution_time(self) -> float:
        return self._max_execution_time

    @property
    def agent(self) -> "Agent":
        """Get the owning agent."""
        return self._agent

    @property
    def variables(self) -> dict[str, REPLVariable]:
        """Get all REPL variables (copy)."""
        return self._variables.copy()

    @property
    def namespace(self) -> dict[str, Any]:
        """Get the IPython execution namespace."""
        return self._shell.user_ns

    @property
    def backing_stores(self) -> dict[str, BackingStore]:
        """Get available backing stores."""
        return self._backing_stores

    # =========================================================================
    # IPython Shell Setup
    # =========================================================================

    def _create_shell(self) -> InteractiveShell:
        """Create and configure IPython InteractiveShell."""
        # Create shell with empty namespace
        shell = InteractiveShell.instance(user_ns={})

        # Configure for programmatic use
        shell.cache_size = 0  # Disable output caching
        shell.colors = 'NoColor'  # No ANSI colors
        shell.history_manager.enabled = False  # No history persistence

        # Set up namespace with restricted builtins
        self._setup_namespace(shell)

        return shell

    def _setup_namespace(self, shell: InteractiveShell) -> None:
        """Pre-populate namespace with utilities for REPL code.

        If restrict_builtins is True, provides a restricted __builtins__ dict
        that excludes dangerous functions like eval, exec, open, __import__.
        """
        import json
        import re
        import math
        import itertools
        import functools
        from collections import defaultdict, Counter
        from dataclasses import dataclass as dc_decorator

        # Import Action and related types
        from ...models import Action, ActionType, ActionResult, Ref

        # Optionally restrict __builtins__
        if self._restrict_builtins:
            restricted_builtins = {
                name: getattr(builtins, name)
                for name in dir(builtins)
                if not name.startswith("_") and name not in DANGEROUS_BUILTINS
            }
            shell.user_ns["__builtins__"] = restricted_builtins

        shell.user_ns.update({
            # Standard utilities (modules)
            "json": json,
            "re": re,
            "math": math,
            "itertools": itertools,
            "functools": functools,
            "defaultdict": defaultdict,
            "Counter": Counter,
            "dataclass": dc_decorator,

            # Agent system hooks
            "submit_action": self.submit_action,
            "Action": Action,
            "ActionType": ActionType,
            "ActionResult": ActionResult,
            "Ref": Ref,

            # Agent reference for backing store access
            "_agent": self._agent,
            "agent_id": self._agent.agent_id,

            # REPL self-reference for programmatic access
            "_repl": self,
        })

    # =========================================================================
    # Variable Management
    # =========================================================================

    async def set(
        self,
        name: str,
        value: Any,
        description: str = "",
        created_by: str = "repl",
        storage_hint: StorageHint | None = None,
    ) -> None:
        """Set a variable with metadata.

        Storage decisions are made by the caller (typically from LLM planner
        via StorageHint):
        - storage_type="value": Store directly in namespace (default)
        - storage_type="reference": Store in backing store, keep only metadata

        Args:
            name: Variable name (must be valid Python identifier)
            value: Value to store
            description: Human-readable description for LLM context
            created_by: action_id or "repl" or "repl_code"
            storage_hint: LLM-provided storage decision (optional)

        Raises:
            ValueError: If name is not a valid Python identifier
            ValueError: If backing store not found (when storage_type="reference")
        """
        if not name.isidentifier():
            raise ValueError(f"Invalid variable name: {name}")

        # Determine storage from hint, defaulting to direct storage
        if storage_hint is not None:
            store_as_reference = storage_hint.storage_type == "reference"
            backing_store_name = storage_hint.backing_store
            description = storage_hint.description or description
        else:
            store_as_reference = False
            backing_store_name = None

        if store_as_reference:
            if backing_store_name not in self._backing_stores:
                raise ValueError(f"Unknown backing store: {backing_store_name}")

            store = self._backing_stores[backing_store_name]
            backing_key = store.generate_key(self._agent.agent_id, name)

            # Store in backing store
            await store.store(backing_key, value, metadata={
                "var_name": name,
                "description": description,
                "created_by": created_by,
                "created_at": time.time(),
            })

            stored_value = None
            logger.debug(f"REPL: Stored {name} in {backing_store_name} at {backing_key}")
        else:
            backing_store_name = None
            backing_key = None
            stored_value = value
            self._shell.user_ns[name] = value

        self._variables[name] = REPLVariable(
            name=name,
            description=description or self._auto_describe(value),
            dtype=self._get_type_str(value),
            created_at=time.time(),
            created_by=created_by,
            is_reference=store_as_reference,
            value=stored_value,
            backing_store=backing_store_name,
            backing_key=backing_key,
        )

    async def get(self, name: str, default: Any = None) -> Any:
        """Get variable value.

        For direct storage: returns value from namespace.
        For references: retrieves from backing store asynchronously.

        Args:
            name: Variable name
            default: Default value if variable not found

        Returns:
            Variable value, or default if not found
        """
        var = self._variables.get(name)
        if var is None:
            return default

        var.access_count += 1
        var.last_accessed = time.time()

        if var.is_reference:
            # Retrieve from backing store
            if var.backing_store not in self._backing_stores:
                logger.error(f"Unknown backing store: {var.backing_store}")
                return default

            store = self._backing_stores[var.backing_store]
            value = await store.retrieve(var.backing_key)

            # Cache in namespace for subsequent access
            if value is not None:
                self._shell.user_ns[name] = value

            return value

        return var.value

    def get_sync(self, name: str, default: Any = None) -> Any:
        """Synchronous get for use in REPL code.

        For direct storage: returns value from namespace.
        For references: returns None (use async get() instead).

        Args:
            name: Variable name
            default: Default value if variable not found

        Returns:
            Variable value, or default if not found or is a reference
        """
        var = self._variables.get(name)
        if var is None:
            return self._shell.user_ns.get(name, default)

        var.access_count += 1
        var.last_accessed = time.time()

        if var.is_reference:
            # For references, check if cached in namespace
            return self._shell.user_ns.get(name, default)

        return var.value

    def set_sync(
        self,
        name: str,
        value: Any,
        description: str = "",
        created_by: str = "repl",
    ) -> None:
        """Synchronous set for simple value storage.

        Unlike async set(), this does NOT support:
        - StorageHint (backing store storage)
        - Reference storage

        Use this for event handlers and other sync contexts that need
        simple variable storage. Use async set() for full features.

        Args:
            name: Variable name (must be valid Python identifier)
            value: Value to store
            description: Human-readable description for LLM context
            created_by: Source of the variable

        Raises:
            ValueError: If name is not a valid Python identifier
        """
        if not name.isidentifier():
            raise ValueError(f"Invalid variable name: {name}")

        self._shell.user_ns[name] = value

        self._variables[name] = REPLVariable(
            name=name,
            description=description or self._auto_describe(value),
            dtype=self._get_type_str(value),
            created_at=time.time(),
            created_by=created_by,
            is_reference=False,
            value=value,
        )

    def has_variable(self, name: str) -> bool:
        """Check if variable exists (in _variables or namespace)."""
        return name in self._variables or name in self._shell.user_ns

    # =========================================================================
    # PolicyREPL-Compatible Interface
    # =========================================================================
    # These aliases allow code written for PolicyREPL to work with REPL

    @override
    def get(self, key: str, default: Any = None) -> Any:
        """Alias for get_sync() - PolicyREPL compatibility."""
        return self.get_sync(key, default)

    @override
    def set(self, key: str, value: Any) -> None:
        """Alias for set_sync() - PolicyREPL compatibility.

        Note: For full metadata and backing store support, use async set()
        via `await repl.set(name, value, description=..., storage_hint=...)`
        """
        self.set_sync(key, value)

    @override
    def has(self, key: str) -> bool:
        """Alias for has_variable() - PolicyREPL compatibility."""
        return self.has_variable(key)

    @override
    async def delete_variable(self, name: str) -> bool:
        """Delete a variable.

        Also deletes from backing store if it's a reference.

        Args:
            name: Variable name

        Returns:
            True if deleted, False if not found
        """
        var = self._variables.get(name)
        if var is None:
            return False

        # Delete from backing store if reference
        if var.is_reference and var.backing_store in self._backing_stores:
            store = self._backing_stores[var.backing_store]
            await store.delete(var.backing_key)

        # Remove from tracking
        del self._variables[name]

        # Remove from namespace
        if name in self._shell.user_ns:
            del self._shell.user_ns[name]

        return True

    @override
    def list_variables(self) -> list[dict[str, Any]]:
        """List all variables with metadata for LLM planning context.

        Returns:
            List of variable info dicts with name, type, description, etc.
        """
        return [
            {
                "name": var.name,
                "type": var.dtype,
                "description": var.description,
                "is_reference": var.is_reference,
                "backing_store": var.backing_store,
                "preview": var.get_preview(),
            }
            for var in self._variables.values()
        ]

    @override
    def get_variable_summary(self) -> str:
        """Get formatted summary of all variables for LLM context.

        Returns:
            Multi-line string describing all REPL variables
        """
        if not self._variables:
            return "# REPL Variables: (none)"

        lines = ["# REPL Variables:"]
        for var in self._variables.values():
            storage = f" [{var.backing_store}]" if var.is_reference else ""
            lines.append(f"- {var.name}: {var.dtype}{storage}")
            lines.append(f"    {var.description}")
            if var.is_reference:
                lines.append(f"    Access via: await _repl.get('{var.name}')")
        return "\n".join(lines)

    # =========================================================================
    # Action Results
    # =========================================================================

    @override
    def set_result(self, action_id: str, result: "ActionResult") -> None:
        """Store action result by ID.

        Args:
            action_id: Action ID
            result: Action result
        """
        self._results[action_id] = result

    @override
    def get_result(self, action_id: str) -> "ActionResult | None":
        """Get action result by ID.

        Args:
            action_id: Action ID

        Returns:
            ActionResult or None if not found
        """
        return self._results.get(action_id)

    @override
    def has_result(self, action_id: str) -> bool:
        """Check if result exists for action ID."""
        return action_id in self._results

    @property
    @override
    def results(self) -> dict[str, "ActionResult"]:
        """Get all action results (copy)."""
        return self._results.copy()

    # =========================================================================
    # Code Execution
    # =========================================================================

    async def execute(self, code: str) -> dict[str, Any]:
        """Execute Python code in the REPL namespace.

        Uses IPython's run_cell_async for native async support.
        Validates imports before execution.

        Args:
            code: Python code to execute

        Returns:
            Dict with keys:
                - success: bool
                - result: Last expression value (if any)
                - error: Error message (if failed)
                - stdout: Captured stdout (IPython handles this)
                - new_names: List of new names added to namespace
                - pending_actions: List of Actions submitted via submit_action()
        """
        self._validate_code(code)
        self._pending_actions = []

        pre_names = set(self._shell.user_ns.keys())

        try:
            # Execute with IPython (native async support)
            exec_result = await asyncio.wait_for(
                self._shell.run_cell_async(
                    code,
                    store_history=False,  # Don't save to IPython history
                    silent=False,  # Show output
                ),
                timeout=self._max_execution_time
            )

            # Track new names
            new_names = [
                n for n in (set(self._shell.user_ns.keys()) - pre_names)
                if not n.startswith("_")
            ]

            # Track new names in _variables for LLM visibility
            for name in new_names:
                if name not in self._variables:
                    value = self._shell.user_ns[name]
                    self._variables[name] = REPLVariable(
                        name=name,
                        description=self._auto_describe(value),
                        dtype=self._get_type_str(value),
                        created_at=time.time(),
                        created_by="repl_code",
                        is_reference=False,
                        value=value,
                    )

            # Record in code history for procedural memory export
            self._code_history.append({
                "code": code,
                "timestamp": time.time(),
                "success": exec_result.success,
                "new_names": new_names,
            })

            return {
                "success": exec_result.success,
                "result": exec_result.result,
                "error": str(exec_result.error_in_exec) if exec_result.error_in_exec else None,
                "new_names": new_names,
                "pending_actions": list(self._pending_actions),
            }

        except asyncio.TimeoutError:
            return {
                "success": False,
                "result": None,
                "error": f"Timeout ({self._max_execution_time}s):\n{type(e).__name__}: {e}",
                "new_names": [],
                "pending_actions": [],
            }
        except Exception as e:
            logger.exception(f"REPL execution failed: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"{type(e).__name__}: {e}",
                "new_names": [],
                "pending_actions": [],
            }

    # =========================================================================
    # Action Generation (Recursion)
    # =========================================================================

    def submit_action(self, action: "Action") -> None:
        """Submit an action to be executed by the agent system.

        This method is exposed in the REPL namespace, so LLM-generated code
        can call: submit_action(Action(...))

        Args:
            action: Action to submit for execution
        """
        self._pending_actions.append(action)

    def get_pending_actions(self) -> list["Action"]:
        """Get and clear pending actions.

        Returns:
            List of Actions that were submitted via submit_action()
        """
        actions = self._pending_actions
        self._pending_actions = []
        return actions

    # =========================================================================
    # Code Export (for Procedural Memory and AgentContextEngine)
    # =========================================================================

    def export_code_history(self) -> list[dict[str, Any]]:
        """Export code execution history for procedural memory.

        Returns:
            List of code execution records with:
                - code: The executed code
                - timestamp: When it was executed
                - success: Whether execution succeeded
                - new_names: Names defined by the code
        """
        return list(self._code_history)

    def export_defined_functions(self) -> list[dict[str, Any]]:
        """Export user-defined functions from the REPL namespace.

        Useful for procedural memory - these functions can be persisted
        and loaded into future REPL sessions.

        Returns:
            List of function info dicts with:
                - name: Function name
                - code: Function source (if available via inspect)
                - signature: Function signature string
                - docstring: Function docstring
        """
        import inspect

        functions = []
        for name, value in self._shell.user_ns.items():
            if callable(value) and not name.startswith("_"):
                # Skip built-in functions and imported modules
                if hasattr(value, "__module__") and value.__module__ == "__main__":
                    try:
                        source = inspect.getsource(value)
                    except (OSError, TypeError):
                        source = None

                    try:
                        sig = str(inspect.signature(value))
                    except (ValueError, TypeError):
                        sig = "()"

                    functions.append({
                        "name": name,
                        "code": source,
                        "signature": sig,
                        "docstring": inspect.getdoc(value) or "",
                    })

        return functions

    def export_for_context(self) -> dict[str, Any]:
        """Export REPL state for AgentContextEngine planning context.

        Returns:
            Dict with:
                - variables: Variable summary for LLM context
                - functions: List of defined function names
                - recent_code: Last N code executions
        """
        return {
            "variables": self.list_variables(),
            "functions": [f["name"] for f in self.export_defined_functions()],
            "recent_code": self._code_history[-5:] if self._code_history else [],
        }

    # =========================================================================
    # Validation
    # =========================================================================

    def _default_allowed_imports(self) -> list[str]:
        """Default allowed import modules."""
        return [
            "json", "re", "math", "itertools", "functools",
            "collections", "dataclasses", "typing", "datetime",
            "pydantic", "asyncio",
        ]

    def _validate_code(self, code: str) -> None:
        """Validate code for safety.

        Checks:
        - Syntax validity
        - Import restrictions (only allowed modules)

        Note: Dangerous builtins are already excluded from __builtins__
        during namespace setup, so we don't need to check calls here.

        Args:
            code: Python code to validate

        Raises:
            ValueError: If code contains disallowed imports
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error: {e}")

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = node.module if isinstance(node, ast.ImportFrom) else None
                names = [alias.name for alias in node.names]

                for name in ([module] if module else []) + names:
                    if name and not self._is_allowed_import(name):
                        raise ValueError(f"Import not allowed: {name}")

            # Check dangerous builtins
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "compile", "open", "__import__"):
                        raise ValueError(f"Builtin not allowed: {node.func.id}")

    def _is_allowed_import(self, name: str) -> bool:
        """Check if import is in allowed list."""
        return any(
            name == allowed or name.startswith(allowed + ".")
            for allowed in self._allowed_imports
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _auto_describe(self, value: Any) -> str:
        """Auto-generate description for a value.

        Args:
            value: Value to describe

        Returns:
            Human-readable description
        """
        if isinstance(value, list):
            if len(value) == 0:
                return "Empty list"
            sample = repr(value[0])[:30]
            return f"List with {len(value)} items (first: {sample}...)"
        if isinstance(value, dict):
            keys_preview = list(value.keys())[:5]
            return f"Dict with {len(value)} keys: {keys_preview}..."
        if isinstance(value, str):
            preview = value[:50] + "..." if len(value) > 50 else value
            return f"String: {preview!r}"
        if callable(value):
            return f"Function: {getattr(value, '__name__', 'anonymous')}"
        if hasattr(value, "__class__"):
            return f"Instance of {value.__class__.__name__}"
        return str(type(value).__name__)

    def _get_type_str(self, value: Any) -> str:
        """Get type annotation string.

        Args:
            value: Value to get type for

        Returns:
            Type annotation string
        """
        t = type(value)
        if t.__module__ == "builtins":
            return t.__name__
        return f"{t.__module__}.{t.__name__}"

    # =========================================================================
    # Serialization (for suspension/resumption)
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize REPL state for ActionPolicyExecutionState persistence.

        Note: Functions defined in namespace are NOT serialized.
        They must be re-defined on resumption if needed.
        Code history IS serialized for procedural memory continuity.

        Returns:
            Dict with serialized REPL state
        """
        return {
            "variables": {
                name: var.to_dict()
                for name, var in self._variables.items()
            },
            "results": {
                action_id: result.model_dump()
                for action_id, result in self._results.items()
            },
            "code_history": self._code_history,
            # NOTE: Functions defined in namespace are NOT serialized.
            # They must be re-defined on resumption if needed.
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], agent: "Agent") -> "PolicyPythonREPL":
        """Restore REPL state from serialized form.

        Args:
            data: Serialized REPL state from to_dict()
            agent: Agent that will own this REPL

        Returns:
            Restored PolicyPythonREPL instance
        """
        from ...models import ActionResult

        repl = cls(agent=agent)

        # Restore code history
        repl._code_history = data.get("code_history", [])

        # Restore variables
        for name, var_data in data.get("variables", {}).items():
            repl._variables[name] = REPLVariable.from_dict(var_data)
            # Restore non-reference values to namespace
            if not var_data.get("is_reference") and var_data.get("value") is not None:
                repl._shell.user_ns[name] = var_data["value"]

        # Restore action results
        for action_id, result_data in data.get("results", {}).items():
            repl._results[action_id] = ActionResult(**result_data)

        return repl


# =============================================================================
# REPLCapability - Agent Capability Wrapper
# =============================================================================


class REPLCapability(AgentCapability):
    """Capability that provides REPL execution context.

    Wraps PolicyPythonREPL and exposes it as an AgentCapability
    for agent integration and discovery.

    This is the recommended way to use PolicyPythonREPL with agents:
    1. Create REPLCapability and add to agent
    2. ActionDispatcher discovers REPL via capability
    3. EXECUTE_CODE actions are routed to this capability

    Example:
        ```python
        # In agent initialization
        repl_cap = REPLCapability(agent=self)
        self.add_capability(repl_cap)

        # ActionDispatcher discovers it automatically
        # LLM can now generate EXECUTE_CODE actions
        ```

    Agents can also enable REPL via metadata:
        ```python
        agent = Agent(metadata=AgentMetadata(enable_repl=True))
        # REPLCapability is created automatically during initialization
        ```
    """

    def __init__(
        self,
        agent: "Agent",
        scope: BlackboardScope = BlackboardScope.AGENT,
        namespace: str = "repl",
        capability_key: str = "repl",
        backing_stores: dict[str, BackingStore] | None = None,
        allowed_imports: list[str] | None = None,
        restrict_builtins: bool = True,
        max_execution_time: float = 5.0,
    ):
        """Initialize REPL capability.

        Args:
            agent: Agent that owns this capability
            scope: BlackboardScope for variable storage (default: AGENT)
            namespace: Namespace prefix for this capability (default: "repl")
            capability_key: Unique key for this capability within the agent (default: "repl")
            backing_stores: Map of store name -> BackingStore implementation
            allowed_imports: List of allowed import module names
            restrict_builtins: If True, restrict dangerous builtins
            max_execution_time: Max seconds for code execution
        """
        super().__init__(
            agent=agent,
            scope_id=get_scope_prefix(scope, agent, namespace=namespace),
            input_patterns=None,
            capability_key=capability_key
        )

        # Create underlying REPL
        self._repl = PolicyPythonREPL(
            agent=agent,
            backing_stores=backing_stores,
            allowed_imports=allowed_imports,
            restrict_builtins=restrict_builtins,
            max_execution_time=max_execution_time,
        )

        self.max_execution_time = max_execution_time
        # Track if initialized
        self._initialized = False

    @property
    def repl(self) -> PolicyPythonREPL:
        """Get the underlying REPL instance."""
        return self._repl

    @override
    async def initialize(self) -> None:
        """Initialize capability."""
        if self._initialized:
            return
        await super().initialize()
        self._initialized = True

    async def execute_code(self, code: str) -> dict[str, Any]:
        """Execute Python code in the REPL.

        This method can be used as an action executor.

        Args:
            code: Python code to execute

        Returns:
            Execution result dict with success, result, error, etc.
        """
        return await self._repl.execute(code)

    def get_variable_summary(self) -> str:
        """Get REPL variable summary for planning context."""
        return self._repl.get_variable_summary()

    def list_variables(self) -> list[dict[str, Any]]:
        """List REPL variables with metadata."""
        return self._repl.list_variables()

    async def set_variable(
        self,
        name: str,
        value: Any,
        description: str = "",
        created_by: str = "capability",
        storage_hint: StorageHint | None = None,
    ) -> None:
        """Set a REPL variable.

        Args:
            name: Variable name
            value: Value to store
            description: Human-readable description
            created_by: Source of the variable
            storage_hint: LLM-provided storage decision
        """
        await self._repl.set(name, value, description, created_by, storage_hint)

    async def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a REPL variable.

        Args:
            name: Variable name
            default: Default if not found

        Returns:
            Variable value or default
        """
        return await self._repl.get(name, default)

    def export_for_context(self) -> dict[str, Any]:
        """Export REPL state for AgentContextEngine."""
        return self._repl.export_for_context()

    # =========================================================================
    # Serialization for suspension
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize capability state for suspension."""
        return {
            "repl_state": self._repl.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], agent: "Agent") -> "REPLCapability":
        """Restore capability from serialized state.

        Args:
            data: Serialized state from to_dict()
            agent: Agent that will own this capability

        Returns:
            Restored REPLCapability instance
        """
        capability = cls(agent=agent)
        repl_data = data.get("repl_state", {})
        if repl_data:
            capability._repl = PolicyPythonREPL.from_dict(repl_data, agent)
        return capability

    @override
    async def serialize_suspension_state(self, state: "AgentSuspensionState") -> "AgentSuspensionState":
        state.custom_data[self.capability_key] = self.to_dict()
        return state

    @override
    async def deserialize_suspension_state(self, state: "AgentSuspensionState") -> None:
        data = state.custom_data.get(self.capability_key, {})
        repl_data = data.get("repl_state", {})
        if repl_data:
            self._repl = PolicyPythonREPL.from_dict(repl_data, self.agent)


# =============================================================================
# REPL Guidance for LLM Planners
# =============================================================================


def get_repl_guidance(repl: PolicyPythonREPL) -> str:
    """Get comprehensive REPL guidance for LLM planner.

    This should be included in action descriptions when REPL is available.

    Returns:
        Multi-line guidance string for LLM context
    """
    return '''
## REPL Execution Context

You have access to a Python REPL (IPython) for dataflow between actions.

### Using `result_var` to Store Action Results

When planning an action, set `result_var` to store the result in a REPL variable:
```json
{
    "action_type": "route_query",
    "parameters": {"query": "How does auth work?"},
    "result_var": "route_result"
}
```
The action's `output` will be stored in `route_result` in the REPL namespace.

### Using `storage_hint` for Large Data

For large results that would bloat context, use `storage_hint` to store in backing store:
```json
{
    "action_type": "analyze_codebase",
    "parameters": {"path": "/src"},
    "result_var": "analysis",
    "storage_hint": {
        "var_name": "analysis",
        "description": "Full codebase analysis with dependency graph",
        "storage_type": "reference",
        "backing_store": "blackboard"
    }
}
```
The result is stored in blackboard, and only metadata is kept in REPL.

### Referencing REPL Variables in Actions

Use `Ref.var("name")` to reference REPL variables in action parameters:
```json
{
    "action_type": "analyze_pages",
    "parameters": {
        "page_ids": {"$ref": "$route_result.output.page_ids"}
    }
}
```

### Executing Python Code (EXECUTE_CODE Action)

Use the EXECUTE_CODE action to run Python code for:
- Transforming data between actions
- Filtering/aggregating results
- Defining reusable functions
- Generating multiple actions programmatically
- Self-profiling with magic commands (%timeit, %who)

```json
{
    "action_type": "execute_code",
    "code": "high_priority = [p for p in route_result['output']['page_ids'] if 'core/' in p]"
}
```

### Generating Actions from REPL Code

REPL code can submit actions to the agent system:
```python
# In REPL code:
for page_id in high_priority[:5]:
    submit_action(Action(
        action_type=ActionType.ANALYZE_PAGE,
        parameters={"page_id": page_id},
        result_var=f"analysis_{page_id.replace('/', '_')}"
    ))
```

### IPython Magic Commands

Useful magic commands available:
- `%timeit expression` - Time an expression (self-profiling)
- `%who` - List variables in namespace
- `%whos` - Detailed variable listing
- `%time statement` - Time a single statement

### Restrictions on Generated Code

- **Allowed imports**: json, re, math, itertools, functools, collections, dataclasses, typing, datetime, pydantic, asyncio
- **Blocked builtins**: eval, exec, compile, open, __import__, input, globals, locals, vars, dir
''' + f'''
- **Timeout**: Code execution limited to {repl.max_execution_time} seconds
''' + '''
- **No file I/O**: Use backing stores (blackboard) for persistent data

### Available Utilities in REPL Namespace

- `submit_action(action)`: Submit an Action to execute
- `Action`, `ActionType`, `ActionResult`, `Ref`: Data model classes
- `agent_id`: Current agent's ID
- `_repl`: Reference to REPL instance for programmatic access
- `json`, `re`, `math`, `itertools`, `functools`: Standard library
- `defaultdict`, `Counter`, `dataclass`: Common utilities

## VCM Analysis Primitive Composition

When analyzing code using VCMAnalysisCapability (or its subclasses like IntentAnalysisCapability,
ContractAnalysisCapability, SlicingAnalysisCapability, ComplianceVCMCapability), you have
composable primitives. **YOU decide the strategy** - the system doesn't force a workflow.

### Available Primitives by Category

**Worker Lifecycle** (spawn_worker, spawn_workers, terminate_worker, get_idle_workers, get_busy_workers):
- Control when/how many workers to create
- Use `cache_affine=True` to route workers to nodes with pages cached

**Work Assignment** (assign_work, prioritize_work, get_pending_work, add_pending_work):
- Control work distribution and priority
- Reprioritize dynamically based on discoveries

**Results** (get_result, get_results, merge_results, detect_contradictions, synthesize_results):
- Merge results when YOU decide, not automatically
- Detect contradictions to inform revisit decisions

**State Queries** (get_analyzed_pages, get_unanalyzed_pages, get_pages_with_issues, get_outstanding_queries):
- Check progress and identify what needs attention
- Drive query-based analysis strategies

**Iteration/Revisit** (mark_for_revisit, get_pages_needing_revisit, revisit_page, clear_result):
- Control when to revisit pages with new context
- Clear and re-analyze based on discoveries

### Example Strategies (Emergent, Not Hardcoded)

**Strategy A: Cluster-Based with Incremental Merge**
```python
# 1. Get clusters from page graph
submit_action(Action(action_type="get_clusters", result_var="clusters"))
# 2. Process first cluster
submit_action(Action(action_type="spawn_workers",
    parameters={"page_ids": Ref.var("clusters[0]"), "cache_affine": True}))
# 3. Wait for workers, then merge cluster results
submit_action(Action(action_type="merge_results",
    parameters={"page_ids": Ref.var("clusters[0]"), "detect_conflicts": True}))
# 4. Check for contradictions, mark for revisit if needed
# 5. Move to next cluster
```

**Strategy B: Query-Driven Continuous Analysis**
```python
# 1. Check outstanding queries
submit_action(Action(action_type="get_outstanding_queries", result_var="queries"))
# 2. Process top query's target pages
for query in queries[:3]:
    submit_action(Action(action_type="spawn_worker",
        parameters={"page_id": query["target_pages"][0]}))
# 3. On completion, detect contradictions
submit_action(Action(action_type="detect_contradictions",
    parameters={"page_ids": completed_pages}))
# 4. If contradictions found, mark for revisit with new context
```

**Strategy C: Opportunistic with Revisits**
```python
# 1. Spawn workers for all pages (fast start)
submit_action(Action(action_type="spawn_workers",
    parameters={"page_ids": all_pages, "max_parallel": 10}))
# 2. As results come in, check confidence
submit_action(Action(action_type="get_pages_with_issues",
    parameters={"min_severity": "medium"}, result_var="issues"))
# 3. Mark low-confidence pages for revisit
for issue in issues:
    if issue["issue_type"] == "low_confidence":
        submit_action(Action(action_type="mark_for_revisit",
            parameters={"page_id": issue["page_id"], "reason": "low confidence"}))
# 4. Periodically process revisit queue
```

**Key Insight**: Cache-awareness is EMERGENT from your choices:
- Set `cache_affine=True` when you want workers placed near cached data
- Use `working_set.request_pages()` before `spawn_workers()` to pre-warm cache
- Use `page_graph.get_clusters()` to find related pages that benefit from being analyzed together
'''
