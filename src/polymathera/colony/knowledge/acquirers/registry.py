"""Registry of :class:`AcquirerStrategy` instances keyed by method."""

from __future__ import annotations

from .base import AcquirerStrategy
from .todo_stubs import (
    _TODO_ArxivAcquirer,
    _TODO_DoiAcquirer,
    _TODO_HttpAcquirer,
    _TODO_IeeeXploreAcquirer,
    _TODO_SaeMobilusAcquirer,
    _TODO_SemanticScholarAcquirer,
)


class AcquirerRegistry:
    """Method-string → :class:`AcquirerStrategy` map.

    Implemented as a thin wrapper over a dict so tests can construct
    a registry with a swapped-in fake strategy without monkeypatching
    a module-level singleton.
    """

    def __init__(self) -> None:
        self._by_method: dict[str, AcquirerStrategy] = {}

    def register(self, strategy: AcquirerStrategy) -> None:
        self._by_method[strategy.method] = strategy

    def get(self, method: str) -> AcquirerStrategy | None:
        return self._by_method.get(method)

    def methods(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_method))


def default_registry() -> AcquirerRegistry:
    """Build a registry pre-populated with every TODO stub.

    Real acquirers register themselves in this function as they ship.
    """
    reg = AcquirerRegistry()
    for cls in (
        _TODO_HttpAcquirer,
        _TODO_ArxivAcquirer,
        _TODO_DoiAcquirer,
        _TODO_IeeeXploreAcquirer,
        _TODO_SaeMobilusAcquirer,
        _TODO_SemanticScholarAcquirer,
    ):
        reg.register(cls())
    return reg


__all__ = ("AcquirerRegistry", "default_registry")
