"""Tests for ``WriteRateLimiter``."""

from __future__ import annotations

import pytest

from polymathera.colony.vcm.convergence import WriteRateLimiter


def test_first_call_allowed() -> None:
    rl = WriteRateLimiter(min_interval_s=1.0, burst_size=1)
    assert rl.allow("p", now=0.0) is True


def test_strict_rate_blocks_second_immediate() -> None:
    rl = WriteRateLimiter(min_interval_s=1.0, burst_size=1)
    assert rl.allow("p", now=0.0) is True
    assert rl.allow("p", now=0.1) is False


def test_strict_rate_allows_after_interval() -> None:
    rl = WriteRateLimiter(min_interval_s=1.0, burst_size=1)
    assert rl.allow("p", now=0.0) is True
    assert rl.allow("p", now=1.5) is True


def test_burst_capacity() -> None:
    rl = WriteRateLimiter(min_interval_s=1.0, burst_size=3)
    # 3 immediate allows.
    assert rl.allow("p", now=0.0) is True
    assert rl.allow("p", now=0.0) is True
    assert rl.allow("p", now=0.0) is True
    # Bucket exhausted.
    assert rl.allow("p", now=0.0) is False


def test_keys_independent() -> None:
    rl = WriteRateLimiter(min_interval_s=1.0, burst_size=1)
    assert rl.allow("p1", now=0.0) is True
    assert rl.allow("p2", now=0.0) is True


def test_reset() -> None:
    rl = WriteRateLimiter(min_interval_s=1.0, burst_size=1)
    rl.allow("p", now=0.0)
    rl.reset("p")
    assert rl.allow("p", now=0.0) is True


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        WriteRateLimiter(min_interval_s=0.0)
    with pytest.raises(ValueError):
        WriteRateLimiter(burst_size=0)
