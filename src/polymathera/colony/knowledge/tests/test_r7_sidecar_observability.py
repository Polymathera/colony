"""R7-FIX-E observability: ``MonorepoPersistedIngestor._write_sidecar``
emits an INFO log on every successful sidecar persistence. Run7
forensic had to inspect the agent's clone filesystem to verify
whether sidecars were written, because the persistence path was
log-silent. Pin the INFO so the substance is observable from logs
alone going forward (per
``[[concise-diagnostics-no-speculation]]`` — observability is
durable infrastructure, not a forensic-round band-aid)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest


def test_write_sidecar_emits_info_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The sidecar-write site must surface its outcome at INFO. The
    log line includes source_uri + extractor + section/page counts +
    md byte size so a single grep on
    ``MonorepoPersistedIngestor: sidecar persisted`` answers "did
    paper X actually get persisted, with what extractor, how
    many sections" without filesystem inspection."""

    from polymathera.colony.knowledge.monorepo_persisted_ingestor import (
        MonorepoPersistedIngestor,
        SidecarManifest,
    )

    # ``_write_sidecar`` is a pure file-write method; we don't need
    # the full wrapper machinery (Ingestor + ReaderRegistry) to
    # exercise it.
    wrapper = MonorepoPersistedIngestor.__new__(MonorepoPersistedIngestor)
    sidecar_dir = tmp_path / ".ingested" / "doc"
    manifest = SidecarManifest(
        source_uri="file:///tmp/doc.pdf",
        pdf_sha256="abc" * 21 + "d",  # 64 chars
        extractor="mistral_ocr",
        extracted_at="2026-06-25T00:00:00Z",
        section_count=3,
        page_count=42,
    )

    with caplog.at_level(
        logging.INFO,
        logger="polymathera.colony.knowledge.monorepo_persisted_ingestor",
    ):
        MonorepoPersistedIngestor._write_sidecar(
            wrapper,
            sidecar_dir=sidecar_dir,
            extracted_md="# Section 1\n\nbody text\n",
            manifest=manifest,
        )

    info_lines = [r for r in caplog.records if r.levelno == logging.INFO]
    assert info_lines, "no INFO log emitted by _write_sidecar"
    msg = info_lines[-1].getMessage()
    assert "sidecar persisted" in msg
    assert "file:///tmp/doc.pdf" in msg
    assert "mistral_ocr" in msg
    assert "sections=3" in msg
    assert "pages=42" in msg

    # Files actually landed on disk.
    assert (sidecar_dir / "extracted.md").exists()
    assert (sidecar_dir / "ingestion.json").exists()


def test_agent_infer_submit_log_is_info_not_warning() -> None:
    """R7-FIX-E: the per-call ``📡 agent_infer ... submitting`` and
    ``... responded`` traces are routine plumbing; they belong at
    INFO. Run7 emitted ~10k of these at WARNING level, contributing
    to log spam without surfacing real problems."""

    import inspect

    from polymathera.colony.agents.base import AgentManagerBase

    # The site lives on ``AgentManagerBase.agent_infer``; pin via
    # source inspection so a future re-promotion to WARNING surfaces.
    src = inspect.getsource(AgentManagerBase.agent_infer)
    # Both submit + response lines must use logger.info, not
    # logger.warning. We don't pin the exact emoji to avoid making
    # the test fragile to cosmetic edits.
    assert "logger.info" in src
    assert "submitting to LLM cluster" in src
    assert "LLM cluster responded" in src
    # Specifically: no logger.warning() near those two lines. We
    # check the substring "logger.warning" doesn't immediately
    # precede the trace text.
    for needle in ("submitting to LLM cluster", "LLM cluster responded"):
        idx = src.find(needle)
        assert idx >= 0
        # Look back 60 chars; the logger call sits within ~30 chars
        # of the message.
        window = src[max(0, idx - 60): idx]
        assert "logger.warning" not in window, (
            f"agent_infer trace near {needle!r} reverted to "
            f"logger.warning — see R7-FIX-E"
        )


def test_handle_py_logs_remote_error_at_warning_not_error() -> None:
    """R7-FIX-E: the serving handle layer logs deployment-side
    errors at WARNING (the exception is re-raised to the caller who
    decides the right severity). Logging at ERROR here was the
    primary contributor to the run7 log spam (107k lines)."""

    import inspect

    from polymathera.colony.distributed.ray_utils.serving import handle

    src = inspect.getsource(handle)
    # The Error-in-X.Y format string must exist.
    assert "Error in {self.deployment_name}" in src, (
        "expected error-message format string not found"
    )
    # The log call for the response-error branch must be
    # ``logger.warning(error_msg)``, not ``logger.error(error_msg)``.
    # Pin via file-wide substring: the only ``error_msg`` symbol in
    # the file lives in this branch.
    assert "logger.warning(error_msg)" in src, (
        "handle.py reverted to logger.error(error_msg) for the "
        "remote-error log site — re-introduces the run7 107k-line spam"
    )
    assert "logger.error(error_msg)" not in src, (
        "handle.py still has logger.error(error_msg) at the remote-"
        "error log site"
    )


def test_decorators_py_logs_handle_error_at_warning_not_error(
) -> None:
    """R7-FIX-E sibling: the deployment-side decorator wraps the
    inner method and logs handle-errors before returning a typed
    error response. Same WARNING-not-ERROR contract."""

    import inspect

    from polymathera.colony.distributed.ray_utils.serving import decorators

    src = inspect.getsource(decorators)
    idx = src.find("Error handling request {request.request_id}")
    assert idx >= 0
    # Search backward ~400 chars for the logger call wrapping the
    # f-string. The logger call sits BEFORE the f-string opening.
    window = src[max(0, idx - 400): idx]
    assert "logger.warning" in window, (
        "decorators.py reverted to logger.error for the handle-error "
        "log site — re-introduces the run7 81k-line spam"
    )
