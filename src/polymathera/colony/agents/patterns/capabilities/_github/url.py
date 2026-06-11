"""GitHub URL parsing helpers.

Pure functions; no IO; no dependency on the rest of the codebase.
Lives under ``_github/`` (alongside ``auth.py`` + ``client.py``) so
both the capability layer and the higher ``design_monorepo`` /
``web_ui`` layers can import without cycles.
"""

from __future__ import annotations


def parse_owner_repo_from_url(url: str) -> str | None:
    """Extract ``owner/repo`` from a github.com clone URL.

    Handles the common shapes:

    - ``https://github.com/owner/repo.git``
    - ``https://github.com/owner/repo``
    - ``git@github.com:owner/repo.git``

    Returns ``None`` for non-github URLs (gitlab, internal forges) or
    malformed input — caller surfaces a clean error rather than guess.
    Pure; no IO. Tested in isolation.
    """

    if not url:
        return None
    s = url.strip()
    # SSH form: ``git@github.com:owner/repo[.git]``
    if s.startswith("git@github.com:"):
        path = s[len("git@github.com:"):]
    elif "github.com/" in s:
        path = s.split("github.com/", 1)[1]
    else:
        return None
    if path.endswith(".git"):
        path = path[:-4]
    path = path.strip("/")
    parts = path.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return f"{parts[0]}/{parts[1]}"
