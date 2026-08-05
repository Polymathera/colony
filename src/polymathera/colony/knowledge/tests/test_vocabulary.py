"""Predicate vocabulary — registry, operations, resolution, stats
(``colony/predicate_vocabulary_plan.md``).

Pins the OBO/SKOS discipline: terms are never deleted; merges leave the
old name as an alias with a ``replaced_by`` pointer; destructive ops
without an approver refuse to apply; cycles fail loud.
"""

from __future__ import annotations

from collections import Counter

import pytest

from polymathera.colony.knowledge.vocabulary import (
    VocabError,
    VocabFile,
    VocabOperation,
    VocabOpType,
    VocabTerm,
    VocabTermStatus,
    apply_operation,
    broader_closure,
    merge_mapping,
    narrower_closure,
    register_provisional,
    resolve,
    top_active_predicates,
    vocab_stats,
)


def _vocab(*names: str, status: VocabTermStatus = VocabTermStatus.ACTIVE) -> VocabFile:
    return VocabFile(terms={
        n: VocabTerm(name=n, status=status) for n in names
    })


def _merge(a: str, b: str) -> VocabOperation:
    return VocabOperation(
        op_type=VocabOpType.MERGE, term=a, target=b,
        approved_by="operator",
    )


# ---- registration ----------------------------------------------------


def test_register_provisional_is_idempotent_and_alias_aware() -> None:
    vocab = _vocab("has_author")
    vocab.terms["has_author"].aliases.append("authored_by")
    added = register_provisional(vocab, ["has_author", "authored_by", "uses"])
    assert added == 1  # only "uses" is new
    assert vocab.terms["uses"].status is VocabTermStatus.PROVISIONAL
    assert register_provisional(vocab, ["uses"]) == 0


# ---- merge / rename --------------------------------------------------


def test_merge_deprecates_source_and_absorbs_aliases() -> None:
    vocab = _vocab("authored", "has_author")
    vocab.terms["authored"].aliases.append("wrote")
    apply_operation(vocab, _merge("authored", "has_author"))

    src = vocab.terms["authored"]
    assert src.status is VocabTermStatus.DEPRECATED  # never deleted
    assert src.replaced_by == "has_author"
    assert set(vocab.terms["has_author"].aliases) == {"authored", "wrote"}
    assert resolve(vocab, "authored") == "has_author"
    assert resolve(vocab, "wrote") == "has_author"
    assert merge_mapping(vocab)["authored"] == "has_author"
    assert vocab.operations[-1].applied_at  # audit trail stamped


def test_merge_chain_resolves_transitively_and_cycle_refused() -> None:
    vocab = _vocab("a", "b", "c")
    apply_operation(vocab, _merge("a", "b"))
    apply_operation(vocab, _merge("b", "c"))
    assert resolve(vocab, "a") == "c"
    with pytest.raises(VocabError, match="cycle"):
        apply_operation(vocab, _merge("c", "a"))


def test_destructive_op_without_approver_refuses() -> None:
    vocab = _vocab("a", "b")
    op = VocabOperation(op_type=VocabOpType.MERGE, term="a", target="b")
    with pytest.raises(VocabError, match="approver"):
        apply_operation(vocab, op)
    assert vocab.terms["a"].status is VocabTermStatus.ACTIVE  # untouched


def test_rename_mints_active_target() -> None:
    vocab = _vocab("was_published_in")
    apply_operation(vocab, VocabOperation(
        op_type=VocabOpType.RENAME, term="was_published_in",
        target="published_in", approved_by="operator",
    ))
    assert vocab.terms["published_in"].status is VocabTermStatus.ACTIVE
    assert resolve(vocab, "was_published_in") == "published_in"


# ---- hierarchy -------------------------------------------------------


def test_broader_closure_and_narrower_expansion() -> None:
    vocab = _vocab("has_soldering_temperature", "has_thermal_property")
    apply_operation(vocab, VocabOperation(
        op_type=VocabOpType.ADD_BROADER,
        term="has_soldering_temperature", target="has_thermal_property",
    ))
    apply_operation(vocab, VocabOperation(
        op_type=VocabOpType.ADD_BROADER,
        term="has_thermal_property", target="has_physical_property",
    ))
    # add_broader minted the organizational parent as ACTIVE.
    assert vocab.terms["has_physical_property"].status is VocabTermStatus.ACTIVE
    assert broader_closure(vocab, "has_soldering_temperature") == {
        "has_thermal_property", "has_physical_property",
    }
    assert narrower_closure(vocab, "has_physical_property") == {
        "has_thermal_property", "has_soldering_temperature",
    }


def test_broader_cycle_fails_loud_and_rolls_back() -> None:
    vocab = _vocab("x", "y")
    apply_operation(vocab, VocabOperation(
        op_type=VocabOpType.ADD_BROADER, term="x", target="y",
    ))
    with pytest.raises(VocabError):
        apply_operation(vocab, VocabOperation(
            op_type=VocabOpType.ADD_BROADER, term="y", target="x",
        ))
    assert "x" not in vocab.terms["y"].broader  # rolled back


# ---- lifecycle -------------------------------------------------------


def test_deprecated_term_cannot_be_promoted() -> None:
    vocab = _vocab("junk")
    apply_operation(vocab, VocabOperation(
        op_type=VocabOpType.DEPRECATE, term="junk", approved_by="operator",
    ))
    with pytest.raises(VocabError, match="deprecated"):
        apply_operation(vocab, VocabOperation(
            op_type=VocabOpType.PROMOTE, term="junk",
        ))


# ---- stats + soft prior ----------------------------------------------


def test_stats_surface_decision_signals() -> None:
    vocab = _vocab("is_a", "uses")
    register_provisional(vocab, ["one_off_pred"])
    usage = Counter({
        "is_a": 100, "uses": 40, "one_off_pred": 1, "unregistered_pred": 1,
    })
    stats = vocab_stats(vocab, usage)
    assert stats.total_terms == 3
    assert stats.provisional == 1
    assert stats.singleton_predicates == 2
    assert stats.unregistered_predicates == 1
    assert stats.top_predicates[0] == ("is_a", 100)


def test_soft_prior_is_active_only_usage_ranked() -> None:
    vocab = _vocab("is_a", "uses")
    register_provisional(vocab, ["fresh_mint"])  # provisional — excluded
    usage = Counter({"uses": 50, "is_a": 10, "fresh_mint": 99})
    prior = top_active_predicates(vocab, usage, k=10)
    assert [t.name for t in prior] == ["uses", "is_a"]


# ---- round trip ------------------------------------------------------


def test_vocab_file_json_round_trip_stable() -> None:
    vocab = _vocab("b_pred", "a_pred")
    apply_operation(vocab, _merge("b_pred", "a_pred"))
    text = vocab.to_json()
    assert VocabFile.from_json(text).to_json() == text
    # Terms serialize sorted for clean git diffs.
    assert text.index('"a_pred"') < text.index('"b_pred"')
