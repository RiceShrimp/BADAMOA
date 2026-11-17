"""One version's release notes from CHANGELOG.md, for a GitHub release. Standard library only.

    python scripts/release_notes.py 0.37.1            # the notes (Markdown) of that version
    python scripts/release_notes.py 0.37.1 --title    # a one-line title: "v0.37.1 — <its summary>"
    python scripts/release_notes.py --versions        # every version in the file, newest first

The text is the CHANGELOG section as written, so the release page and the app's release notes match.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"
HEAD = re.compile(r"^## \[(?P<version>[^\]]+)\](?:\s*-\s*(?P<date>\S+))?\s*$", re.M)
TITLE_MAX = 90


def sections(text: str) -> dict[str, tuple[str | None, str]]:
    """version -> (date, the Markdown under its heading)."""
    heads = list(HEAD.finditer(text))
    found: dict[str, tuple[str | None, str]] = {}
    for i, head in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        found[head["version"]] = (head["date"], text[head.end() : end].strip())
    return found


def title(version: str, body: str) -> str:
    """The first sentence of the paragraph above the first ### heading, when there is one."""
    summary = body.split("\n### ", 1)[0].strip() if not body.startswith("### ") else ""
    summary = " ".join(summary.split())
    first = re.split(r"(?<=[.!?。])\s", summary, maxsplit=1)[0] if summary else ""
    first = first.replace("**", "")
    if len(first) > TITLE_MAX:
        first = first[: TITLE_MAX - 1].rstrip() + "…"
    return f"v{version} — {first}" if first else f"v{version}"


def main(argv: list[str]) -> int:
    found = sections(CHANGELOG.read_text(encoding="utf-8"))
    if argv == ["--versions"]:
        print("\n".join(found))
        return 0
    if not argv or argv[0] not in found:
        print(f"no section for {argv[0] if argv else '(none)'} in {CHANGELOG.name}", file=sys.stderr)
        return 1
    date, body = found[argv[0]]
    if "--title" in argv[1:]:
        print(title(argv[0], body))
    else:
        footer = f"\n\n<sub>{date} · 앱 안의 릴리즈 노트(오른쪽 위 버전 표시)와 같은 내용입니다.</sub>" if date else ""
        print(body + footer)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
