"""Integration tests against a real GROBID service.

Skipped unless ``POLYMATHERA_GROBID_URL`` is set (e.g.,
``http://localhost:8070``). Run GROBID locally with::

    docker compose -f src/polymathera/colony/cli/deploy/docker/docker-compose.yml \\
        up -d grobid
    POLYMATHERA_GROBID_URL=http://localhost:8070 \\
        pytest src/polymathera/colony/knowledge/tests/integration/

These tests require a real PDF; the fixture builds a minimal valid
PDF in pure Python (no extra dependency) so the integration test can
run in a fresh environment.
"""

from __future__ import annotations

import io

import pytest

from polymathera.colony.knowledge import (
    GrobidPdfReader,
    KnowledgeFormat,
    RawDocument,
)


pytestmark = pytest.mark.asyncio


def _minimal_pdf(text: str = "Hello GROBID") -> bytes:
    """Build a minimal valid PDF carrying ``text`` on a single page.

    Hand-rolled to avoid pulling reportlab / fpdf into the test deps;
    ~600 bytes total. Sufficient for GROBID to return a TEI XML
    response (even if its content extraction yields little for such a
    trivial document — we just need GROBID to ack the request).
    """

    # PDF objects.
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
        ),
        f"<< /Length {44 + len(text)} >>\nstream\nBT /F1 24 Tf 100 700 Td ({text}) Tj ET\nendstream".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref_offset = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(b"trailer\n")
    out.write(f"<< /Size {len(objs) + 1} /Root 1 0 R >>\n".encode())
    out.write(f"startxref\n{xref_offset}\n%%EOF\n".encode())
    return out.getvalue()


async def test_real_grobid_returns_tei(grobid_url_or_skip: str) -> None:
    reader = GrobidPdfReader(base_url=grobid_url_or_skip, timeout_s=60.0)
    pdf_bytes = _minimal_pdf("Real GROBID round-trip test")
    doc = RawDocument(
        source_uri="file:///tmp/integration.pdf",
        detected_format=KnowledgeFormat.PDF,
        payload=pdf_bytes,
    )
    sections = await reader.read_async(doc)
    # GROBID always returns *some* TEI XML for a valid PDF; section
    # count is implementation-defined but the reader must not raise.
    assert isinstance(sections, tuple)
    # Citations all point at our source URI.
    for section in sections:
        assert section.citation.source_uri == doc.source_uri
