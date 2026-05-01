"""Core entry point for ${name}.

Replace this stub with the tool's actual implementation. The
tool-building pool is expected to keep ``run`` as the canonical
externally-callable surface; the colony tool adapter binds to it.
"""

from __future__ import annotations


def run(*, payload: dict) -> dict:
    """Execute the tool against ``payload`` and return its result.

    The default implementation is a pass-through used as a smoke test
    for the scaffold; the tool-building pool replaces it.
    """

    return {"echo": payload}
