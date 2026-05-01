"""Tests for ``ConvergenceDamper``."""

from __future__ import annotations

import pytest

from polymathera.colony.vcm.convergence import ConvergenceDamper, NumericTolerance


def test_first_call_not_converged() -> None:
    d = ConvergenceDamper()
    tol = NumericTolerance(mode="absolute", value=0.5)
    assert d.is_converged(
        subscription_id="s", page_id="p", new_output=1.0, tolerance=tol,
    ) is False


def test_within_absolute_tolerance() -> None:
    d = ConvergenceDamper()
    tol = NumericTolerance(mode="absolute", value=0.5)
    d.is_converged(subscription_id="s", page_id="p", new_output=1.0, tolerance=tol)
    assert d.is_converged(
        subscription_id="s", page_id="p", new_output=1.4, tolerance=tol,
    ) is True


def test_outside_absolute_tolerance() -> None:
    d = ConvergenceDamper()
    tol = NumericTolerance(mode="absolute", value=0.5)
    d.is_converged(subscription_id="s", page_id="p", new_output=1.0, tolerance=tol)
    assert d.is_converged(
        subscription_id="s", page_id="p", new_output=2.0, tolerance=tol,
    ) is False


def test_relative_tolerance() -> None:
    d = ConvergenceDamper()
    tol = NumericTolerance(mode="relative", value=0.05)
    d.is_converged(subscription_id="s", page_id="p", new_output=100.0, tolerance=tol)
    # Within 5% of 100 == 5.0
    assert d.is_converged(
        subscription_id="s", page_id="p", new_output=104.0, tolerance=tol,
    ) is True
    # Outside 5%
    assert d.is_converged(
        subscription_id="s", page_id="p", new_output=110.0, tolerance=tol,
    ) is False


def test_no_tolerance_passthrough() -> None:
    d = ConvergenceDamper()
    assert d.is_converged(
        subscription_id="s", page_id="p", new_output=1.0, tolerance=None,
    ) is False
    # Cache is updated even without tolerance.
    tol = NumericTolerance(mode="absolute", value=0.5)
    assert d.is_converged(
        subscription_id="s", page_id="p", new_output=1.1, tolerance=tol,
    ) is True


def test_vector_output() -> None:
    d = ConvergenceDamper()
    tol = NumericTolerance(mode="absolute", value=0.5)
    d.is_converged(subscription_id="s", page_id="p", new_output=[1.0, 2.0, 3.0], tolerance=tol)
    assert d.is_converged(
        subscription_id="s", page_id="p", new_output=[1.4, 2.0, 3.3], tolerance=tol,
    ) is True
    assert d.is_converged(
        subscription_id="s", page_id="p", new_output=[2.0, 2.0, 3.3], tolerance=tol,
    ) is False


def test_shape_change_not_converged() -> None:
    d = ConvergenceDamper()
    tol = NumericTolerance(mode="absolute", value=0.5)
    d.is_converged(subscription_id="s", page_id="p", new_output=[1.0, 2.0], tolerance=tol)
    # Different length always counts as not-converged.
    assert d.is_converged(
        subscription_id="s", page_id="p", new_output=[1.0, 2.0, 3.0], tolerance=tol,
    ) is False


def test_non_numeric_falls_through() -> None:
    d = ConvergenceDamper()
    tol = NumericTolerance(mode="absolute", value=0.5)
    assert d.is_converged(
        subscription_id="s", page_id="p", new_output="a string", tolerance=tol,
    ) is False
    assert d.is_converged(
        subscription_id="s", page_id="p", new_output={"a": 1}, tolerance=tol,
    ) is False


def test_reset_targeted() -> None:
    d = ConvergenceDamper()
    tol = NumericTolerance(mode="absolute", value=0.5)
    d.is_converged(subscription_id="s1", page_id="p", new_output=1.0, tolerance=tol)
    d.is_converged(subscription_id="s2", page_id="p", new_output=10.0, tolerance=tol)
    d.reset(subscription_id="s1")
    # s1 is fresh: comparing 1.4 to nothing => not converged.
    assert d.is_converged(
        subscription_id="s1", page_id="p", new_output=1.4, tolerance=tol,
    ) is False
    # s2 still cached: comparing 10.4 to 10 => converged.
    assert d.is_converged(
        subscription_id="s2", page_id="p", new_output=10.4, tolerance=tol,
    ) is True
