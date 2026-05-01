"""Tests for ``WebhookEventBuilder``."""

from __future__ import annotations

import pytest

from polymathera.colony.vcm.page_events import PageChangeKind
from polymathera.colony.vcm.watchers import WebhookEventBuilder


def test_github_payload_basic() -> None:
    builder = WebhookEventBuilder(
        scope_id="program-1",
        source_uri="git:git@example/foo.git",
    )
    payload = {
        "after": "abc123",
        "commits": [
            {"added": ["a.txt"], "modified": ["b.txt"], "removed": []},
            {"added": [], "modified": ["c.txt"], "removed": ["d.txt"]},
        ],
    }
    events = builder.build(payload, provider="github")
    kinds = {e.kind for e in events}
    paths = {e.extra["relative_path"] for e in events}
    assert PageChangeKind.PAGE_ADDED in kinds
    assert PageChangeKind.PAGE_REPLACED in kinds
    assert PageChangeKind.PAGE_INVALIDATED in kinds
    assert paths == {"a.txt", "b.txt", "c.txt", "d.txt"}
    # source URI carries the after sha.
    assert all(e.source.endswith("@abc123") for e in events)


def test_gitlab_payload() -> None:
    builder = WebhookEventBuilder(
        scope_id="program-1",
        source_uri="git:gitlab.example/foo.git",
    )
    payload = {
        "checkout_sha": "deadbeef",
        "commits": [
            {"added": ["a.py"], "modified": [], "removed": []},
        ],
    }
    events = builder.build(payload, provider="gitlab")
    assert len(events) == 1
    assert events[0].source.endswith("@deadbeef")


def test_unknown_provider_raises() -> None:
    builder = WebhookEventBuilder(
        scope_id="program", source_uri="git:foo",
    )
    with pytest.raises(ValueError):
        builder.build({}, provider="bitbucket")


def test_path_collision_resolution() -> None:
    builder = WebhookEventBuilder(
        scope_id="program", source_uri="git:foo",
    )
    payload = {
        "after": "x",
        "commits": [
            {"added": ["a.txt"], "modified": ["a.txt"], "removed": []},
        ],
    }
    events = builder.build(payload, provider="github")
    # "added" wins over "modified" for the same path.
    kinds = {e.kind for e in events}
    paths_to_kinds = {e.extra["relative_path"]: e.kind for e in events}
    assert paths_to_kinds["a.txt"] == PageChangeKind.PAGE_ADDED


def test_remove_overrides_others() -> None:
    builder = WebhookEventBuilder(scope_id="program", source_uri="git:foo")
    payload = {
        "after": "x",
        "commits": [
            {"added": ["a.txt"], "modified": ["a.txt"], "removed": ["a.txt"]},
        ],
    }
    events = builder.build(payload, provider="github")
    kinds_for_path = {e.extra["relative_path"]: e.kind for e in events}
    assert kinds_for_path["a.txt"] == PageChangeKind.PAGE_INVALIDATED
