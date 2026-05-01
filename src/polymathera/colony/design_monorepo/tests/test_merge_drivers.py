"""Tests for the custom git merge drivers.

Each driver is a tiny script invoked as ``python -m <module> %A %O %B %P``.
These tests call the modules' ``main`` directly to avoid forking
subprocesses; that exercises the same code path the git workflow does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polymathera.colony.design_monorepo.git_merge import (
    budget_merge,
    corpus_merge,
    decisions_merge,
    kg_merge,
    page_graph_merge,
    reqif_merge,
)


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _drive(driver, ours: Path, base: Path, theirs: Path, path: str) -> int:
    return driver.main([str(ours), str(base), str(theirs), path])


# ---------- decisions-merge -------------------------------------------------


def test_decisions_field_disjoint_merges(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    ours = tmp_path / "ours.json"
    theirs = tmp_path / "theirs.json"
    _write(base, json.dumps({"decision_id": "d1", "title": "B", "summary": "S0"}))
    _write(ours, json.dumps({"decision_id": "d1", "title": "O", "summary": "S0"}))
    _write(theirs, json.dumps({"decision_id": "d1", "title": "B", "summary": "S2"}))
    rc = _drive(decisions_merge, ours, base, theirs, "design/decisions/d1.json")
    assert rc == 0
    merged = json.loads(ours.read_text("utf-8"))
    assert merged == {"decision_id": "d1", "title": "O", "summary": "S2"}


def test_decisions_same_field_conflicts(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    ours = tmp_path / "ours.json"
    theirs = tmp_path / "theirs.json"
    _write(base, json.dumps({"decision_id": "d1", "title": "B"}))
    _write(ours, json.dumps({"decision_id": "d1", "title": "O"}))
    _write(theirs, json.dumps({"decision_id": "d1", "title": "T"}))
    rc = _drive(decisions_merge, ours, base, theirs, "design/decisions/d1.json")
    assert rc != 0
    merged = json.loads(ours.read_text("utf-8"))
    assert merged["title"]["_conflict"] is True


def test_decisions_distinct_ids_merge(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    ours = tmp_path / "ours.json"
    theirs = tmp_path / "theirs.json"
    _write(base, json.dumps([{"decision_id": "d1", "title": "B"}]))
    _write(ours, json.dumps([
        {"decision_id": "d1", "title": "B"},
        {"decision_id": "d2", "title": "Added by ours"},
    ]))
    _write(theirs, json.dumps([
        {"decision_id": "d1", "title": "B"},
        {"decision_id": "d3", "title": "Added by theirs"},
    ]))
    rc = _drive(decisions_merge, ours, base, theirs, "design/decisions/decisions.json")
    assert rc == 0
    merged = json.loads(ours.read_text("utf-8"))
    ids = {item["decision_id"] for item in merged}
    assert ids == {"d1", "d2", "d3"}


def test_decisions_invalid_json_falls_back(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    ours = tmp_path / "ours.json"
    theirs = tmp_path / "theirs.json"
    _write(base, "{}")
    _write(ours, "{not json}")
    _write(theirs, "{}")
    rc = _drive(decisions_merge, ours, base, theirs, "design/decisions/x.json")
    assert rc != 0
    text = ours.read_text("utf-8")
    assert "<<<<<<<" in text


# ---------- kg-merge --------------------------------------------------------


def test_kg_set_merge_disjoint(tmp_path: Path) -> None:
    base = tmp_path / "b.kg.json"
    ours = tmp_path / "o.kg.json"
    theirs = tmp_path / "t.kg.json"
    _write(base, json.dumps({"triples": [["a", "rdf:type", "b"]]}))
    _write(ours, json.dumps({"triples": [["a", "rdf:type", "b"], ["a", "p1", "x"]]}))
    _write(theirs, json.dumps({"triples": [["a", "rdf:type", "b"], ["a", "p2", "y"]]}))
    rc = _drive(kg_merge, ours, base, theirs, "x.kg.json")
    assert rc == 0
    merged = json.loads(ours.read_text("utf-8"))
    triples = {tuple(t) for t in merged["triples"]}
    assert ("a", "rdf:type", "b") in triples
    assert ("a", "p1", "x") in triples
    assert ("a", "p2", "y") in triples


def test_kg_functional_predicate_contradicts(tmp_path: Path) -> None:
    base = tmp_path / "b.kg.json"
    ours = tmp_path / "o.kg.json"
    theirs = tmp_path / "t.kg.json"
    _write(base, json.dumps({"triples": [["a", "rdf:type", "b"]]}))
    _write(ours, json.dumps({"triples": [["a", "rdf:type", "b"], ["a", "rdf:type", "alt"]]}))
    _write(theirs, json.dumps({"triples": [["a", "rdf:type", "b"], ["a", "rdf:type", "different"]]}))
    rc = _drive(kg_merge, ours, base, theirs, "x.kg.json")
    assert rc != 0


def test_kg_deletion_propagates(tmp_path: Path) -> None:
    base = tmp_path / "b.kg.json"
    ours = tmp_path / "o.kg.json"
    theirs = tmp_path / "t.kg.json"
    _write(base, json.dumps({"triples": [["a", "p", "x"]]}))
    _write(ours, json.dumps({"triples": []}))
    _write(theirs, json.dumps({"triples": [["a", "p", "x"]]}))
    rc = _drive(kg_merge, ours, base, theirs, "x.kg.json")
    assert rc == 0
    merged = json.loads(ours.read_text("utf-8"))
    triples = [tuple(t) for t in merged["triples"]]
    assert ("a", "p", "x") not in triples


# ---------- corpus-merge ----------------------------------------------------


def test_corpus_appends_new_papers(tmp_path: Path) -> None:
    base = tmp_path / "b.json"
    ours = tmp_path / "o.json"
    theirs = tmp_path / "t.json"
    _write(base, json.dumps({"papers": [{"doi": "x", "version": "1", "title": "T"}]}))
    _write(ours, json.dumps({"papers": [
        {"doi": "x", "version": "1", "title": "T"},
        {"doi": "y", "version": "1", "title": "Y"},
    ]}))
    _write(theirs, json.dumps({"papers": [
        {"doi": "x", "version": "1", "title": "T"},
        {"doi": "z", "version": "1", "title": "Z"},
    ]}))
    rc = _drive(corpus_merge, ours, base, theirs, "metadata.json")
    assert rc == 0
    out = json.loads(ours.read_text("utf-8"))
    dois = {p["doi"] for p in out["papers"]}
    assert dois == {"x", "y", "z"}


def test_corpus_disagreement_conflicts(tmp_path: Path) -> None:
    base = tmp_path / "b.json"
    ours = tmp_path / "o.json"
    theirs = tmp_path / "t.json"
    _write(base, json.dumps({"papers": [{"doi": "x", "version": "1", "title": "T0"}]}))
    _write(ours, json.dumps({"papers": [{"doi": "x", "version": "1", "title": "TO"}]}))
    _write(theirs, json.dumps({"papers": [{"doi": "x", "version": "1", "title": "TT"}]}))
    rc = _drive(corpus_merge, ours, base, theirs, "metadata.json")
    assert rc != 0


# ---------- budget-merge ----------------------------------------------------


def _yaml_available() -> bool:
    try:
        import yaml  # noqa: F401
        return True
    except Exception:
        try:
            from ruamel.yaml import YAML  # noqa: F401
            return True
        except Exception:
            return False


@pytest.mark.skipif(not _yaml_available(), reason="No YAML library installed")
def test_budget_max_on_increase_merges(tmp_path: Path) -> None:
    base = tmp_path / "b.yaml"
    ours = tmp_path / "o.yaml"
    theirs = tmp_path / "t.yaml"
    _write(base, "metadata:\n  policy: max_on_increase\ntree:\n  chassis: 320.0\n  power: 410.0\n")
    _write(ours, "metadata:\n  policy: max_on_increase\ntree:\n  chassis: 340.0\n  power: 410.0\n")
    _write(theirs, "metadata:\n  policy: max_on_increase\ntree:\n  chassis: 320.0\n  power: 415.0\n")
    rc = _drive(budget_merge, ours, base, theirs, "budget.yaml")
    assert rc == 0
    text = ours.read_text("utf-8")
    assert "340" in text
    assert "415" in text


@pytest.mark.skipif(not _yaml_available(), reason="No YAML library installed")
def test_budget_mixed_direction_conflicts(tmp_path: Path) -> None:
    base = tmp_path / "b.yaml"
    ours = tmp_path / "o.yaml"
    theirs = tmp_path / "t.yaml"
    _write(base, "metadata:\n  policy: max_on_increase\ntree:\n  chassis: 320.0\n")
    _write(ours, "metadata:\n  policy: max_on_increase\ntree:\n  chassis: 340.0\n")
    _write(theirs, "metadata:\n  policy: max_on_increase\ntree:\n  chassis: 310.0\n")
    rc = _drive(budget_merge, ours, base, theirs, "budget.yaml")
    assert rc != 0


# ---------- page-graph-merge ------------------------------------------------


def test_page_graph_merge_drops_marker(tmp_path: Path) -> None:
    base = tmp_path / "b.parquet"
    ours = tmp_path / "o.parquet"
    theirs = tmp_path / "t.parquet"
    base.write_bytes(b"BASE")
    ours.write_bytes(b"OURS")
    theirs.write_bytes(b"THEIRS")
    rc = _drive(page_graph_merge, ours, base, theirs, "design/page_graph.parquet")
    assert rc == 0
    assert not ours.exists()
    marker = tmp_path / "o.parquet.regenerate_required"
    assert marker.is_file()


# ---------- reqif-merge -----------------------------------------------------


def test_reqif_disjoint_changes_merge(tmp_path: Path) -> None:
    base = tmp_path / "b.reqif"
    ours = tmp_path / "o.reqif"
    theirs = tmp_path / "t.reqif"
    _write(base, '<root><req IDENTIFIER="r1"><title>Old</title></req></root>')
    _write(ours, '<root><req IDENTIFIER="r1"><title>New-Ours</title></req></root>')
    _write(theirs, '<root><req IDENTIFIER="r1"><title>Old</title></req><req IDENTIFIER="r2"><title>Theirs added</title></req></root>')
    rc = _drive(reqif_merge, ours, base, theirs, "req.reqif")
    assert rc == 0
    text = ours.read_text("utf-8")
    assert "New-Ours" in text
    assert "r2" in text


def test_reqif_identifier_conflict(tmp_path: Path) -> None:
    base = tmp_path / "b.reqif"
    ours = tmp_path / "o.reqif"
    theirs = tmp_path / "t.reqif"
    _write(base, '<root><req IDENTIFIER="r1"><title>Old</title></req></root>')
    _write(ours, '<root><req IDENTIFIER="r1"><title>OursTitle</title></req></root>')
    _write(theirs, '<root><req IDENTIFIER="r1"><title>TheirsTitle</title></req></root>')
    rc = _drive(reqif_merge, ours, base, theirs, "req.reqif")
    assert rc != 0
