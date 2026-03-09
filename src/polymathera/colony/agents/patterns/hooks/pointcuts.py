"""Pointcut definitions for hook matching.

Pointcuts determine which methods and instances a hook applies to.
They can be combined with AND, OR, and NOT operators.
"""

from __future__ import annotations

import fnmatch
import weakref
from abc import ABC, abstractmethod
from typing import Any


class Pointcut(ABC):
    """Base class for pointcuts.

    Pointcuts determine which method invocations a hook should intercept.
    They match against both the join point string (e.g., "MyClass.method")
    and the actual instance being called.

    Pointcuts can be combined using operators:
        - `&` (AND): Both pointcuts must match
        - `|` (OR): Either pointcut must match
        - `~` (NOT): Inverts the match
    """

    @abstractmethod
    def matches(self, join_point: str, instance: Any) -> bool:
        """Check if this pointcut matches the given join point and instance.

        Args:
            join_point: Method identifier (e.g., "MyCapability.process")
            instance: The object whose method is being called

        Returns:
            True if the hook should be applied
        """
        ...

    def __and__(self, other: Pointcut) -> AndPointcut:
        """Combine with AND."""
        return AndPointcut(self, other)

    def __or__(self, other: Pointcut) -> OrPointcut:
        """Combine with OR."""
        return OrPointcut(self, other)

    def __invert__(self) -> NotPointcut:
        """Invert with NOT."""
        return NotPointcut(self)

    # Convenience factory methods
    @staticmethod
    def pattern(pattern: str) -> PatternPointcut:
        """Create a pattern-based pointcut.

        Args:
            pattern: Glob pattern (e.g., "*.process", "Memory*.*", "Agent.execute_*")

        Example:
            ```python
            Pointcut.pattern("*.infer")      # All infer methods
            Pointcut.pattern("Memory*.*")    # All methods on Memory* classes
            ```
        """
        return PatternPointcut(pattern)

    @staticmethod
    def cls(target_class: type) -> ClassPointcut:
        """Create a class-based pointcut.

        Args:
            target_class: Match all methods on instances of this class

        Example:
            ```python
            Pointcut.cls(ActionDispatcher)  # All methods on ActionDispatcher
            ```
        """
        return ClassPointcut(target_class)

    @staticmethod
    def instance(obj: Any) -> InstancePointcut:
        """Create an instance-based pointcut.

        Args:
            obj: Match only methods on this specific instance

        Example:
            ```python
            Pointcut.instance(self)  # Only this specific object
            ```
        """
        return InstancePointcut(obj)

    @staticmethod
    def decorated_with(marker: str) -> DecoratorPointcut:
        """Create a decorator-based pointcut.

        Args:
            marker: Attribute name set by the decorator (e.g., "_action_key")

        Example:
            ```python
            Pointcut.decorated_with("_action_key")  # All @action_executor methods
            ```
        """
        return DecoratorPointcut(marker)

    @staticmethod
    def method(method_name: str) -> PatternPointcut:
        """Create a pointcut for an exact method name.

        Args:
            method_name: Full method identifier (e.g., "Agent.execute_action")

        Example:
            ```python
            Pointcut.method("Agent.execute_action")
            ```
        """
        return PatternPointcut(method_name)


class PatternPointcut(Pointcut):
    """Match methods by glob pattern on join point string.

    Supports standard glob patterns:
    - `*` matches any characters except `.`
    - `?` matches single character
    - `[seq]` matches any character in seq
    - `[!seq]` matches any character not in seq
    """

    def __init__(self, pattern: str):
        self.pattern = pattern

    def matches(self, join_point: str, instance: Any) -> bool:
        return fnmatch.fnmatch(join_point, self.pattern)

    def __repr__(self) -> str:
        return f"Pointcut.pattern({self.pattern!r})"


class ClassPointcut(Pointcut):
    """Match all methods on instances of a specific class."""

    def __init__(self, target_class: type):
        self.target_class = target_class

    def matches(self, join_point: str, instance: Any) -> bool:
        return isinstance(instance, self.target_class)

    def __repr__(self) -> str:
        return f"Pointcut.cls({self.target_class.__name__})"


class InstancePointcut(Pointcut):
    """Match only methods on a specific instance.

    Uses weak reference to avoid preventing garbage collection.
    If the instance is garbage collected, the pointcut will never match.
    """

    def __init__(self, obj: Any):
        self._obj_id = id(obj)
        self._obj_ref = weakref.ref(obj)

    def matches(self, join_point: str, instance: Any) -> bool:
        # Check if the referenced object still exists
        ref_obj = self._obj_ref()
        if ref_obj is None:
            return False
        return id(instance) == self._obj_id

    def __repr__(self) -> str:
        ref_obj = self._obj_ref()
        if ref_obj is None:
            return "Pointcut.instance(<garbage collected>)"
        return f"Pointcut.instance({type(ref_obj).__name__}@{self._obj_id})"


class DecoratorPointcut(Pointcut):
    """Match methods that have a specific decorator marker attribute.

    Many decorators set marker attributes on the decorated function.
    For example, `@action_executor` sets `_action_key`.
    """

    def __init__(self, marker: str):
        self.marker = marker

    def matches(self, join_point: str, instance: Any) -> bool:
        # Extract method name from join point
        parts = join_point.rsplit(".", 1)
        if len(parts) != 2:
            return False
        method_name = parts[1]

        # Get the method from the instance
        method = getattr(instance, method_name, None)
        if method is None:
            return False

        # Check if the method (or its underlying function) has the marker
        # Handle both bound methods and their underlying functions
        func = getattr(method, "__func__", method)
        return hasattr(func, self.marker)

    def __repr__(self) -> str:
        return f"Pointcut.decorated_with({self.marker!r})"


class AndPointcut(Pointcut):
    """Logical AND of two pointcuts."""

    def __init__(self, left: Pointcut, right: Pointcut):
        self.left = left
        self.right = right

    def matches(self, join_point: str, instance: Any) -> bool:
        return self.left.matches(join_point, instance) and self.right.matches(
            join_point, instance
        )

    def __repr__(self) -> str:
        return f"({self.left!r} & {self.right!r})"


class OrPointcut(Pointcut):
    """Logical OR of two pointcuts."""

    def __init__(self, left: Pointcut, right: Pointcut):
        self.left = left
        self.right = right

    def matches(self, join_point: str, instance: Any) -> bool:
        return self.left.matches(join_point, instance) or self.right.matches(
            join_point, instance
        )

    def __repr__(self) -> str:
        return f"({self.left!r} | {self.right!r})"


class NotPointcut(Pointcut):
    """Logical NOT of a pointcut."""

    def __init__(self, inner: Pointcut):
        self.inner = inner

    def matches(self, join_point: str, instance: Any) -> bool:
        return not self.inner.matches(join_point, instance)

    def __repr__(self) -> str:
        return f"~{self.inner!r}"

