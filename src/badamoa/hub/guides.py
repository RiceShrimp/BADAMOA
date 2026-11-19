"""Guides shown in the UI: the user manual and the template guide, as markdown.

They ship inside the wheel (``badamoa/docs/…``, see ``pyproject.toml``); in a source checkout they are read
from the repository's ``docs/``. A guide whose file is not there yet is listed as not ready.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .services import ServiceError

GUIDES: dict[str, dict[str, str]] = {
    "manual": {
        "title": "사용 설명서",
        "file": "MANUAL.md",
        "summary": "화면마다 할 수 있는 것과 순서, 관리자 설정, 단축키",
    },
    "templates": {
        "title": "템플릿 가이드",
        "file": "TEMPLATE_GUIDE.md",
        "summary": "팀의 학습 코드를 모델 종류로 만들어 확인하고 올리는 방법",
    },
}
_PACKAGE_DIR = Path(__file__).resolve().parent.parent


def _path(file: str) -> Path | None:
    for candidate in (_PACKAGE_DIR / "docs" / file, _PACKAGE_DIR.parent.parent / "docs" / file):
        if candidate.is_file():
            return candidate
    return None


def listing() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "title": g["title"],
            "summary": g["summary"],
            "file": g["file"],
            "ready": _path(g["file"]) is not None,
        }
        for name, g in GUIDES.items()
    ]


def read(name: str) -> dict[str, Any]:
    guide = GUIDES.get(name)
    if guide is None:
        raise ServiceError(404, "그런 설명서가 없습니다.")
    path = _path(guide["file"])
    if path is None:
        raise ServiceError(404, f"{guide['title']}는 아직 준비되지 않았습니다.")
    return {"name": name, "title": guide["title"], "file": guide["file"], "markdown": path.read_text(encoding="utf-8")}
