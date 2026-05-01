"""Tests for ``SubscriptionIndex``."""

from __future__ import annotations

import pytest

from polymathera.colony.vcm.convergence import (
    PageMetadataPredicate,
    PageSubscription,
    SubscriptionIndex,
    SubscriptionRegistryFull,
)


def _sub(**kwargs) -> PageSubscription:
    pred = kwargs.pop("predicate")
    return PageSubscription(
        predicate=pred,
        dispatch_scope=kwargs.pop("dispatch_scope", "scope"),
        dispatch_key=kwargs.pop("dispatch_key", "k"),
        capability_key=kwargs.pop("capability_key", "Cap"),
    )


def test_add_and_get() -> None:
    idx = SubscriptionIndex()
    s = _sub(predicate=PageMetadataPredicate(data_type="code"))
    idx.add(s)
    assert idx.get(s.subscription_id) == s
    assert len(idx) == 1


def test_duplicate_id_rejected() -> None:
    idx = SubscriptionIndex()
    s = _sub(predicate=PageMetadataPredicate(data_type="code"))
    idx.add(s)
    with pytest.raises(ValueError):
        idx.add(s)


def test_capacity_cap() -> None:
    idx = SubscriptionIndex(max_subscriptions=2)
    idx.add(_sub(predicate=PageMetadataPredicate(data_type="a")))
    idx.add(_sub(predicate=PageMetadataPredicate(data_type="b")))
    with pytest.raises(SubscriptionRegistryFull):
        idx.add(_sub(predicate=PageMetadataPredicate(data_type="c")))


def test_remove() -> None:
    idx = SubscriptionIndex()
    s = _sub(predicate=PageMetadataPredicate(data_type="code"))
    idx.add(s)
    assert idx.remove(s.subscription_id) is True
    assert idx.get(s.subscription_id) is None
    assert idx.remove(s.subscription_id) is False


def test_candidates_for_data_type() -> None:
    idx = SubscriptionIndex()
    a = _sub(predicate=PageMetadataPredicate(data_type="code"))
    b = _sub(predicate=PageMetadataPredicate(data_type="requirements"))
    c = _sub(predicate=PageMetadataPredicate(data_type="code"))
    idx.add(a); idx.add(b); idx.add(c)
    cands = idx.candidates_for(data_type="code", source="git:r:main:1")
    ids = {s.subscription_id for s in cands}
    assert a.subscription_id in ids
    assert c.subscription_id in ids
    assert b.subscription_id not in ids


def test_candidates_for_source_prefix() -> None:
    idx = SubscriptionIndex()
    git = _sub(predicate=PageMetadataPredicate(source_prefix="git:"))
    arxiv = _sub(predicate=PageMetadataPredicate(source_prefix="arxiv:"))
    idx.add(git); idx.add(arxiv)
    cands = idx.candidates_for(data_type=None, source="git:repo:main:1")
    ids = {s.subscription_id for s in cands}
    assert git.subscription_id in ids
    assert arxiv.subscription_id not in ids


def test_candidates_include_unindexed() -> None:
    idx = SubscriptionIndex()
    bare = _sub(predicate=PageMetadataPredicate(scope_id="program-1"))
    idx.add(bare)
    cands = idx.candidates_for(data_type="anything", source="git:r:main:1")
    assert any(s.subscription_id == bare.subscription_id for s in cands)


def test_no_duplicates_in_candidates() -> None:
    idx = SubscriptionIndex()
    s = _sub(predicate=PageMetadataPredicate(
        data_type="code", source_prefix="git:",
    ))
    idx.add(s)
    cands = idx.candidates_for(data_type="code", source="git:r:main:1")
    assert sum(1 for c in cands if c.subscription_id == s.subscription_id) == 1
