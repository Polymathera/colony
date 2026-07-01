"""Deterministic subsampling shared by snapshot mixing and balancing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def evenly_spaced(seq: Sequence[T], count: int) -> list[T]:
    """``count`` items from ``seq`` at a uniform stride (never up-samples).

    Evenly spaced rather than first-``count`` so a down-sample does not
    bias toward the start of the sequence.
    """
    n = len(seq)
    if count >= n:
        return list(seq)
    if count <= 0:
        return []
    step = n / count
    return [seq[int(i * step)] for i in range(count)]


__all__ = ("evenly_spaced",)
