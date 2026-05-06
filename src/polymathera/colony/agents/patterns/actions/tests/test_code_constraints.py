"""Tests for ``IterationShapeValidator`` — path-aware ``run()``
counting and the discovery-abuse / size limits.

The path-aware counter is the load-bearing change: a textual walker
would conflate ``if/else`` branches and reject the natural
"acknowledge → action → branched-response" pattern, which trips the
SessionAgent's REPL loop on routine commands. These tests pin the
intended semantics (max across mutually-exclusive branches, sum
elsewhere) so a future refactor can't silently regress to textual
counting.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from polymathera.colony.agents.patterns.actions.code_constraints import (
    IterationShapeValidator,
)


@pytest.mark.asyncio
async def test_branched_response_counts_as_one_path() -> None:
    """ack + action + (success-branch | failure-branch) must validate
    at the default ``max_actions=3`` because only ONE response branch
    runs per execution. This is the canonical SessionAgent shape."""

    code = (
        'await run("ack", content="working on it")\n'
        'r = await run("the_action")\n'
        "if r.success:\n"
        '    await run("respond_to_user", content="ok")\n'
        "else:\n"
        '    await run("respond_to_user", content="failed")\n'
    )
    result = await IterationShapeValidator().validate(code, agent=MagicMock())
    assert result.valid, result.errors


@pytest.mark.asyncio
async def test_linear_calls_still_sum() -> None:
    """No branching → every ``run()`` is on the single path → the
    validator must add them up. Four straight-line calls fails at
    ``max_actions=3``."""

    code = (
        'await run("a")\n'
        'await run("b")\n'
        'await run("c")\n'
        'await run("d")\n'
    )
    result = await IterationShapeValidator().validate(code, agent=MagicMock())
    assert not result.valid
    assert any("Too many actions" in e for e in result.errors)


@pytest.mark.asyncio
async def test_try_except_takes_max_of_body_and_handlers() -> None:
    """Either the ``try`` body completes (body+else) or an exception
    routes through one handler — so the worst-case path is
    ``body + max(handlers, else)``. ``finally`` always runs and adds
    on top.
    """

    code = (
        "try:\n"
        '    await run("a")\n'
        '    await run("b")\n'
        "except Exception:\n"
        '    await run("recover_a")\n'
        '    await run("recover_b")\n'
        "finally:\n"
        '    await run("cleanup")\n'
    )
    # Any path: body (2) + max(handlers=2, else=0) + finally (1) = 5.
    # That exceeds ``max_actions=3``, so this must fail.
    result = await IterationShapeValidator().validate(code, agent=MagicMock())
    assert not result.valid

    code_ok = (
        "try:\n"
        '    await run("the_action")\n'
        "except Exception:\n"
        '    await run("respond_err")\n'
        "else:\n"
        '    await run("respond_ok")\n'
    )
    # body (1) + max(handlers=1, else=1) = 2. Validates.
    result = await IterationShapeValidator().validate(code_ok, agent=MagicMock())
    assert result.valid, result.errors


@pytest.mark.asyncio
async def test_nested_if_else_takes_max_at_each_level() -> None:
    """Nested branches collapse path-by-path. The deepest path here
    runs at most ``ack + outer-then(inner-then) = 1 + 1 + 1 = 3``."""

    code = (
        'await run("ack")\n'
        "if outer:\n"
        "    if inner:\n"
        '        await run("path_aa")\n'
        "    else:\n"
        '        await run("path_ab")\n'
        "else:\n"
        '    await run("path_b")\n'
    )
    result = await IterationShapeValidator().validate(code, agent=MagicMock())
    assert result.valid, result.errors


@pytest.mark.asyncio
async def test_browse_count_is_textual_not_path_aware() -> None:
    """browse() is a discovery anti-pattern — multiple browses are
    excessive regardless of which branch they sit in. Text-counted."""

    code = (
        "if x:\n"
        '    browse("a")\n'
        "else:\n"
        '    browse("b")\n'
    )
    # Two textual browse() calls — fails at default ``max_browse=1``
    # even though only one runs per path.
    result = await IterationShapeValidator().validate(code, agent=MagicMock())
    assert not result.valid
    assert any("Too many browse" in e for e in result.errors)
