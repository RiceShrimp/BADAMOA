"""Release notes shown in the UI, parsed from the repository's single CHANGELOG.md.

The file ships inside the wheel (``badamoa/CHANGELOG.md``); in a source checkout
it is read from the repository root.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_RELEASE = re.compile(r"^##\s+\[(?P<version>[^\]]+)\](?:\s*-\s*(?P<date>\S+))?")
_PACKAGE_DIR = Path(__file__).resolve().parent.parent


def changelog_path() -> Path | None:
    for candidate in (_PACKAGE_DIR / "CHANGELOG.md", _PACKAGE_DIR.parent.parent / "CHANGELOG.md"):
        if candidate.is_file():
            return candidate
    return None


def parse_release_notes(text: str) -> list[dict[str, Any]]:
    """Parse ``## [version] - date`` / ``### section`` / ``- item`` into a list of releases."""
    releases: list[dict[str, Any]] = []
    release: dict[str, Any] | None = None
    section: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        match = _RELEASE.match(line)
        if match:
            release = {"version": match["version"], "date": match["date"], "summary": "", "sections": []}
            releases.append(release)
            section = None
        elif release is None:
            continue  # preamble before the first release
        elif line.startswith("### "):
            section = {"title": line[4:].strip(), "items": []}
            release["sections"].append(section)
        elif line.startswith("- ") and section is not None:
            section["items"].append(line[2:].strip())
        elif line.startswith("  ") and section is not None and section["items"]:
            section["items"][-1] += " " + line.strip()  # wrapped item
        elif line.strip() and section is None:
            release["summary"] = (release["summary"] + " " + line.strip()).strip()
    return releases


@lru_cache(maxsize=1)
def _load(path: str, mtime: float) -> list[dict[str, Any]]:
    return parse_release_notes(Path(path).read_text(encoding="utf-8"))


def load_release_notes() -> list[dict[str, Any]]:
    path = changelog_path()
    if path is None:
        return []
    return _load(str(path), path.stat().st_mtime)
