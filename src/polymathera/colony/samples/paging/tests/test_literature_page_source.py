"""Unit tests for :class:`LiteratureContextPageSource` — covers the
offline-extractable parts (path filter + chunk extractor). End-to-end
``initialize`` is exercised by the integration test which boots a real
VCM."""

from __future__ import annotations

from pathlib import Path

import git
import pytest

from polymathera.colony.distributed.ray_utils.serving.context import (
    Ring,
    execution_context,
)
from polymathera.colony.samples.paging._walk import walk_repo
from polymathera.colony.samples.paging.literature_page_source import (
    LiteratureContextPageSource,
    _DEFAULT_INCLUDE_GLOBS,
)
from polymathera.colony.vcm.models import MmapConfig


def _make_source(scope_id: str = "lit", **kwargs) -> LiteratureContextPageSource:
    """Build a source instance without entering ``initialize`` — the
    ctor only stores config; we test that surface in isolation."""

    return LiteratureContextPageSource(
        scope_id=scope_id,
        mmap_config=MmapConfig(),
        origin_url="file:///tmp/no-such-repo",
        **kwargs,
    )


def _make_repo(root: Path, files: dict[str, bytes]) -> Path:
    repo = git.Repo.init(root, initial_branch="main")
    repo.config_writer().set_value("user", "email", "t@t").release()
    repo.config_writer().set_value("user", "name", "t").release()
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    repo.git.add(all=True)
    repo.index.commit("initial")
    return root


@pytest.fixture
def _user_ctx():
    """LiteratureContextPageSource captures an execution context in
    ``__init__``; provide one for the tests that build instances."""

    with execution_context(
        ring=Ring.USER, tenant_id="t", colony_id="c", session_id="s",
    ) as ctx:
        yield ctx


def test_default_path_filter_keeps_binaries(_user_ctx) -> None:
    src = _make_source()
    assert src._path_filter.binary_policy == "include"
    assert src._path_filter.include_globs == _DEFAULT_INCLUDE_GLOBS
    assert src._path_filter.start_dir == "literature"


def test_constructor_overrides_apply(_user_ctx) -> None:
    src = _make_source(
        start_dir=None,
        include_globs=["**/*.pdf"],
        exclude_globs=["**/draft/**"],
    )
    assert src._path_filter.start_dir is None
    assert src._path_filter.include_globs == ("**/*.pdf",)
    assert src._path_filter.exclude_globs == ("**/draft/**",)
    # binary_policy is forced to "include" — caller cannot override.
    assert src._path_filter.binary_policy == "include"


@pytest.mark.asyncio
async def test_extract_chunks_markdown(tmp_path: Path, _user_ctx) -> None:
    md = tmp_path / "paper.md"
    md.write_text(
        "# Title\n\nThis is paragraph one.\n\nThis is paragraph two.\n",
        encoding="utf-8",
    )
    src = _make_source()
    chunks = list(await src._extract_chunks(md))
    assert chunks, "expected at least one chunk from a non-empty markdown file"
    assert any("paragraph one" in c.text for c in chunks)


@pytest.mark.asyncio
async def test_extract_chunks_plain_text(tmp_path: Path, _user_ctx) -> None:
    txt = tmp_path / "note.txt"
    txt.write_text("Just a single short note.\n", encoding="utf-8")
    src = _make_source()
    chunks = list(await src._extract_chunks(txt))
    # Short notes may fall below ProseChunker.min_tokens; the test only
    # asserts the call does not raise. Behaviour-wise: a sub-min-token
    # text yields zero chunks, which is correct (avoids polluting the
    # index).
    assert isinstance(chunks, list)


@pytest.mark.asyncio
async def test_extract_chunks_unsupported_extension(
    tmp_path: Path, _user_ctx,
) -> None:
    py = tmp_path / "script.py"
    py.write_text("print('hi')\n", encoding="utf-8")
    src = _make_source()
    chunks = await src._extract_chunks(py)
    assert tuple(chunks) == ()


def test_walk_repo_with_default_filter_picks_literature_only(
    tmp_path: Path, _user_ctx,
) -> None:
    root = _make_repo(tmp_path / "r", {
        "literature/paper.md": b"# Paper\n\nbody\n",
        "literature/note.txt": b"hello\n",
        "tools/main.py": b"x = 1\n",
        "README.md": b"top-level readme\n",
    })
    src = _make_source()
    matched = walk_repo(str(root), src._path_filter)
    rel = {str(Path(p).relative_to(root)) for p in matched}
    # start_dir="literature" + default globs ⇒ only literature/* files
    assert rel == {"literature/paper.md", "literature/note.txt"}
