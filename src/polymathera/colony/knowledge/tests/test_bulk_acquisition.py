"""Tests for ``BulkAcquisitionCapability`` + the corpus manifest
schema (master §6.6 / Q-S0b in
``cps/phase_q_opm_meg_demo_plan.md``).

The tests use :class:`LocalPathAcquirer` so no network is required;
they verify:

- The default policy is ``SKIP_IF_PRESENT`` and second runs over the
  same manifest are no-ops (the user-seeded corpus is safe).
- ``UPGRADE_TIER`` is monotone (a higher tier upgrades, a lower tier
  is preserved).
- ``OVERWRITE`` re-ingests cleanly.
- The manifest YAML round-trips and the schema-version gate rejects
  manifests authored against newer schemas.
- Unsupported methods return a typed ``unsupported_method`` row
  instead of crashing the whole batch.
- TODO stubs raise ``NotImplementedError`` with a dossier reference.
- Hash mismatch between ``expected_sha256`` and the acquired file is
  surfaced as a typed failure.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from polymathera.colony.knowledge import (
    AcquisitionEntry,
    BulkAcquisitionCapability,
    BulkAcquisitionError,
    BulkAcquisitionReport,
    CorpusManifest,
    CorpusTier,
    InMemoryEmbedder,
    InMemoryVectorStore,
    Ingestor,
    IngestionPolicy,
    LocalPathAcquirer,
    ManifestEntry,
    _TODO_ArxivAcquirer,
    _TODO_DoiAcquirer,
    _TODO_HttpAcquirer,
    _TODO_IeeeXploreAcquirer,
    _TODO_NeuroImageAcquirer,
    _TODO_SaeMobilusAcquirer,
    _TODO_SemanticScholarAcquirer,
)


pytestmark = pytest.mark.asyncio


SAMPLE_TEXT_A = (
    "# Optical Magnetometry — chapter 1\n\n"
    "Optically-Pumped Magnetometers (OPMs) are atomic sensors.\n"
)
SAMPLE_TEXT_B = (
    "# Active shielding for OPM-MEG\n\n"
    "Magnetoencephalography (MEG) systems benefit from active "
    "magnetic shielding.\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ingestor() -> Ingestor:
    return Ingestor(
        embedder=InMemoryEmbedder(),
        vector_store=InMemoryVectorStore(),
        review_sample_rate=0.0,
    )


@pytest.fixture
def cap(ingestor: Ingestor, tmp_path: Path) -> BulkAcquisitionCapability:
    return BulkAcquisitionCapability(
        agent=None,
        scope_id="bulk_acq",
        ingestor=ingestor,
        cache_dir=tmp_path / "cache",
    )


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _manifest(
    tmp_path: Path,
    *,
    domain: str = "quantum",
    entries: list[ManifestEntry] | None = None,
) -> CorpusManifest:
    return CorpusManifest(
        schema_version=1,
        domain=domain,
        description="Test manifest",
        entries=tuple(entries or ()),
    )


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


async def test_manifest_yaml_round_trip(tmp_path: Path) -> None:
    paper = _write(tmp_path / "paper.md", SAMPLE_TEXT_A)
    manifest = _manifest(
        tmp_path,
        entries=[
            ManifestEntry(
                uri="paper:opm:budker_romalis_2013",
                tier=CorpusTier.TIER_1_FOUNDATIONS,
                method="local_path",
                acquirer_args={"path": str(paper)},
                notes="Budker & Romalis, Optical Magnetometry (Cambridge 2013).",
            ),
        ],
    )
    text = manifest.to_yaml()
    restored = CorpusManifest.from_yaml(text)
    assert restored.domain == "quantum"
    assert restored.entries[0].uri == "paper:opm:budker_romalis_2013"
    assert restored.entries[0].tier is CorpusTier.TIER_1_FOUNDATIONS
    assert restored.entries[0].method == "local_path"
    assert "Budker" in restored.entries[0].notes


async def test_manifest_rejects_newer_schema_version(tmp_path: Path) -> None:
    text = (
        "schema_version: 999\n"
        "domain: quantum\n"
        "entries: []\n"
    )
    with pytest.raises(ValueError, match="schema_version"):
        CorpusManifest.from_yaml(text)


async def test_manifest_tolerates_unknown_keys(tmp_path: Path) -> None:
    """Forward-compat: unknown YAML keys must round-trip via
    ``model_config = extra='allow'``."""

    text = (
        "schema_version: 1\n"
        "domain: quantum\n"
        "future_field: hello\n"
        "entries: []\n"
    )
    manifest = CorpusManifest.from_yaml(text)
    assert manifest.domain == "quantum"


# ---------------------------------------------------------------------------
# LocalPathAcquirer — happy path
# ---------------------------------------------------------------------------


async def test_acquire_one_local_path_ingests(
    cap: BulkAcquisitionCapability, tmp_path: Path,
) -> None:
    paper = _write(tmp_path / "paper.md", SAMPLE_TEXT_A)
    entry = ManifestEntry(
        uri="paper:a",
        tier=CorpusTier.TIER_3_RESEARCH,
        method="local_path",
        acquirer_args={"path": str(paper)},
    )
    out = await cap.acquire_one(entry)
    assert isinstance(out, AcquisitionEntry)
    assert out.outcome == "ingested"
    assert out.chunks_produced > 0
    assert out.cached is True  # local files always come from disk


async def test_acquire_one_local_path_missing_path_arg(
    cap: BulkAcquisitionCapability,
) -> None:
    """A manifest entry that forgets ``path`` is a fetch_failed
    outcome — typed, not a crash."""

    entry = ManifestEntry(
        uri="paper:a",
        tier=CorpusTier.TIER_3_RESEARCH,
        method="local_path",
        acquirer_args={},
    )
    out = await cap.acquire_one(entry)
    assert out.outcome == "fetch_failed"
    assert "path" in out.error.lower()


async def test_acquire_one_local_path_nonexistent_file(
    cap: BulkAcquisitionCapability, tmp_path: Path,
) -> None:
    entry = ManifestEntry(
        uri="paper:a",
        tier=CorpusTier.TIER_3_RESEARCH,
        method="local_path",
        acquirer_args={"path": str(tmp_path / "ghost.md")},
    )
    out = await cap.acquire_one(entry)
    assert out.outcome == "fetch_failed"


# ---------------------------------------------------------------------------
# Idempotency — bulk acquisition must respect Q-S0a
# ---------------------------------------------------------------------------


async def test_acquire_manifest_skip_default_on_second_run(
    cap: BulkAcquisitionCapability, tmp_path: Path,
) -> None:
    """The user-seed-safety property: re-running a manifest is a
    no-op under the default policy."""

    paper = _write(tmp_path / "paper.md", SAMPLE_TEXT_A)
    manifest = _manifest(
        tmp_path,
        entries=[
            ManifestEntry(
                uri="paper:a", tier=CorpusTier.TIER_1_FOUNDATIONS,
                method="local_path", acquirer_args={"path": str(paper)},
            ),
        ],
    )

    first = await cap.acquire_corpus(manifest)
    assert first.ingested == 1
    assert first.skipped_present == 0

    second = await cap.acquire_corpus(manifest)
    assert second.ingested == 0
    assert second.skipped_present == 1
    assert second.failed == 0


async def test_acquire_manifest_upgrade_tier_path(
    cap: BulkAcquisitionCapability, tmp_path: Path,
) -> None:
    """First ingest at Tier 3, then re-acquire under UPGRADE_TIER
    with Tier 1 — expect a tier_upgraded outcome."""

    paper = _write(tmp_path / "paper.md", SAMPLE_TEXT_A)

    seed_manifest = _manifest(
        tmp_path,
        entries=[
            ManifestEntry(
                uri="paper:a", tier=CorpusTier.TIER_3_RESEARCH,
                method="local_path", acquirer_args={"path": str(paper)},
            ),
        ],
    )
    await cap.acquire_corpus(seed_manifest)

    upgrade_manifest = _manifest(
        tmp_path,
        entries=[
            ManifestEntry(
                uri="paper:a", tier=CorpusTier.TIER_1_FOUNDATIONS,
                method="local_path", acquirer_args={"path": str(paper)},
            ),
        ],
    )
    report = await cap.acquire_corpus(
        upgrade_manifest, policy=IngestionPolicy.UPGRADE_TIER,
    )
    assert report.tier_upgraded == 1
    assert report.entries[0].outcome == "tier_upgraded"


async def test_acquire_manifest_overwrite(
    cap: BulkAcquisitionCapability, tmp_path: Path,
) -> None:
    """OVERWRITE re-runs the pipeline and treats the new content as
    canonical."""

    paper = _write(tmp_path / "paper.md", SAMPLE_TEXT_A)

    first_manifest = _manifest(
        tmp_path,
        entries=[
            ManifestEntry(
                uri="paper:a", tier=CorpusTier.TIER_3_RESEARCH,
                method="local_path", acquirer_args={"path": str(paper)},
            ),
        ],
    )
    await cap.acquire_corpus(first_manifest)

    # Simulate a paper revision.
    paper.write_text(
        SAMPLE_TEXT_A + "\n\n## Addendum\n\nRevised content.\n",
        encoding="utf-8",
    )
    overwrite_manifest = _manifest(
        tmp_path,
        entries=[
            ManifestEntry(
                uri="paper:a", tier=CorpusTier.TIER_2_STANDARDS,
                method="local_path", acquirer_args={"path": str(paper)},
            ),
        ],
    )
    report = await cap.acquire_corpus(
        overwrite_manifest, policy=IngestionPolicy.OVERWRITE,
    )
    assert report.ingested == 1
    assert report.entries[0].outcome == "ingested"


# ---------------------------------------------------------------------------
# Hash verification
# ---------------------------------------------------------------------------


async def test_expected_sha256_match(
    cap: BulkAcquisitionCapability, tmp_path: Path,
) -> None:
    paper = _write(tmp_path / "paper.md", SAMPLE_TEXT_A)
    digest = hashlib.sha256(paper.read_bytes()).hexdigest()
    out = await cap.acquire_one(
        ManifestEntry(
            uri="paper:a", tier=CorpusTier.TIER_1_FOUNDATIONS,
            method="local_path", acquirer_args={"path": str(paper)},
            expected_sha256=digest,
        ),
    )
    assert out.outcome == "ingested"


async def test_expected_sha256_mismatch_is_typed_failure(
    cap: BulkAcquisitionCapability, tmp_path: Path,
) -> None:
    paper = _write(tmp_path / "paper.md", SAMPLE_TEXT_A)
    out = await cap.acquire_one(
        ManifestEntry(
            uri="paper:a", tier=CorpusTier.TIER_1_FOUNDATIONS,
            method="local_path", acquirer_args={"path": str(paper)},
            expected_sha256="0" * 64,  # definitely wrong
        ),
    )
    assert out.outcome == "hash_mismatch"
    assert "expected_sha256" in out.error


# ---------------------------------------------------------------------------
# Unsupported method + multi-entry resilience
# ---------------------------------------------------------------------------


async def test_unsupported_method_typed_outcome(
    cap: BulkAcquisitionCapability,
) -> None:
    out = await cap.acquire_one(
        ManifestEntry(
            uri="paper:a", tier=CorpusTier.TIER_3_RESEARCH,
            method="ghost_method", acquirer_args={},
        ),
    )
    assert out.outcome == "unsupported_method"
    assert "ghost_method" in out.error


async def test_one_failure_does_not_poison_batch(
    cap: BulkAcquisitionCapability, tmp_path: Path,
) -> None:
    """A bad entry mid-manifest must not stop the rest from
    ingesting."""

    good = _write(tmp_path / "good.md", SAMPLE_TEXT_B)
    manifest = _manifest(
        tmp_path,
        entries=[
            ManifestEntry(
                uri="paper:bad", tier=CorpusTier.TIER_3_RESEARCH,
                method="ghost", acquirer_args={},
            ),
            ManifestEntry(
                uri="paper:good", tier=CorpusTier.TIER_3_RESEARCH,
                method="local_path", acquirer_args={"path": str(good)},
            ),
        ],
    )
    report = await cap.acquire_corpus(manifest)
    outcomes = sorted(r.outcome for r in report.entries)
    assert outcomes == ["ingested", "unsupported_method"]
    assert report.ingested == 1
    assert report.failed == 1


# ---------------------------------------------------------------------------
# Manifest-on-disk + acquire_manifest action
# ---------------------------------------------------------------------------


async def test_acquire_manifest_loads_from_yaml_path(
    cap: BulkAcquisitionCapability, tmp_path: Path,
) -> None:
    paper = _write(tmp_path / "paper.md", SAMPLE_TEXT_A)
    manifest_yaml = _write(
        tmp_path / "manifest.yaml",
        CorpusManifest(
            schema_version=1,
            domain="quantum",
            entries=(
                ManifestEntry(
                    uri="paper:a", tier=CorpusTier.TIER_1_FOUNDATIONS,
                    method="local_path", acquirer_args={"path": str(paper)},
                ),
            ),
        ).to_yaml(),
    )
    report = await cap.acquire_manifest(str(manifest_yaml))
    assert isinstance(report, BulkAcquisitionReport)
    assert report.domain == "quantum"
    assert report.ingested == 1


# ---------------------------------------------------------------------------
# TODO stubs (master §6.6 / dossier §II.N gap markers)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stub_cls,expected_method",
    [
        (_TODO_ArxivAcquirer, "arxiv_id"),
        (_TODO_DoiAcquirer, "doi"),
        (_TODO_HttpAcquirer, "http_url"),
        (_TODO_IeeeXploreAcquirer, "ieee_xplore"),
        (_TODO_NeuroImageAcquirer, "neuroimage"),
        (_TODO_SaeMobilusAcquirer, "sae_mobilus"),
        (_TODO_SemanticScholarAcquirer, "semantic_scholar"),
    ],
)
async def test_todo_stubs_raise_not_implemented(
    stub_cls, expected_method: str, tmp_path: Path,
) -> None:
    """Each TODO stub:
    - exposes the right ``method`` key,
    - raises ``NotImplementedError`` on use,
    - the message names the dossier reference (so the gap is
      visible to a reviewer)."""

    stub = stub_cls()
    assert stub.method == expected_method
    entry = ManifestEntry(
        uri="paper:a", tier=CorpusTier.TIER_3_RESEARCH,
        method=expected_method, acquirer_args={},
    )
    with pytest.raises(NotImplementedError, match="(?i)dossier|master"):
        await stub.acquire(entry, cache_dir=tmp_path)


async def test_todo_stub_routed_through_capability_is_typed_failure(
    ingestor: Ingestor, tmp_path: Path,
) -> None:
    """When a TODO stub is registered, going through the capability
    surfaces a typed ``fetch_failed`` (with the NotImplementedError
    in the error string) — the batch keeps moving."""

    cap = BulkAcquisitionCapability(
        agent=None, scope_id="bulk",
        ingestor=ingestor,
        cache_dir=tmp_path / "cache",
        acquirers=[_TODO_HttpAcquirer()],
    )
    out = await cap.acquire_one(
        ManifestEntry(
            uri="paper:a", tier=CorpusTier.TIER_3_RESEARCH,
            method="http_url",
            acquirer_args={"url": "https://example.com/paper.pdf"},
        ),
    )
    assert out.outcome == "fetch_failed"
    assert "TODO_HttpAcquirer" in out.error or "Master §6.6" in out.error


# ---------------------------------------------------------------------------
# register_acquirer + LocalPathAcquirer always available
# ---------------------------------------------------------------------------


async def test_local_path_acquirer_always_registered(
    ingestor: Ingestor, tmp_path: Path,
) -> None:
    """Even when the caller doesn't pass ``acquirers=[...]``, local_path
    must work — it's the default path for user-seeded corpora."""

    cap = BulkAcquisitionCapability(
        agent=None, scope_id="bulk",
        ingestor=ingestor,
        cache_dir=tmp_path / "cache",
    )
    paper = _write(tmp_path / "paper.md", SAMPLE_TEXT_A)
    out = await cap.acquire_one(
        ManifestEntry(
            uri="paper:a", tier=CorpusTier.TIER_1_FOUNDATIONS,
            method="local_path", acquirer_args={"path": str(paper)},
        ),
    )
    assert out.outcome == "ingested"


async def test_register_acquirer_replaces_method(
    ingestor: Ingestor, tmp_path: Path,
) -> None:
    """``register_acquirer`` last-write-wins lets a real
    implementation shadow a TODO stub at runtime."""

    cap = BulkAcquisitionCapability(
        agent=None, scope_id="bulk",
        ingestor=ingestor,
        cache_dir=tmp_path / "cache",
        acquirers=[_TODO_HttpAcquirer()],
    )
    # Confirm the stub is wired.
    out = await cap.acquire_one(
        ManifestEntry(
            uri="paper:a", tier=CorpusTier.TIER_3_RESEARCH,
            method="http_url", acquirer_args={"url": "x"},
        ),
    )
    assert out.outcome == "fetch_failed"

    # Replace with a working in-test acquirer that re-uses
    # LocalPathAcquirer's implementation but under the http_url method
    # so we can observe the swap.
    class _FakeHttp(LocalPathAcquirer):
        @property
        def method(self) -> str:
            return "http_url"

    paper = _write(tmp_path / "paper.md", SAMPLE_TEXT_A)
    cap.register_acquirer(_FakeHttp())
    out2 = await cap.acquire_one(
        ManifestEntry(
            uri="paper:b", tier=CorpusTier.TIER_3_RESEARCH,
            method="http_url", acquirer_args={"path": str(paper)},
        ),
    )
    assert out2.outcome == "ingested"
