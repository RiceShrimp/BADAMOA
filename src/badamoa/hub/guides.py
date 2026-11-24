"""Guides shown in the UI: the user manual and the template guide, as markdown.

They ship inside the wheel (``badamoa/docs/…``, see ``pyproject.toml``); in a source checkout they are read
from the repository's ``docs/``. A guide whose file is not there yet is listed as not ready.

The manual's pictures (``docs/img/manual/*.png|jpg``, written ``![…](img/manual/x.png)``) are served by name and read
only when someone scrolls to them; their sizes come with the guide so the page keeps its place while they load.
"""

from __future__ import annotations

import re
import struct
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


IMAGE_DIR = "img/manual"
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg"}  # screens as PNG, mostly-photo screens as JPEG
PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
_IMAGE_NAME = re.compile(r"[a-z0-9][a-z0-9-]*\.(?:png|jpg)")
_IMAGE_REF = re.compile(r"!\[[^\]]*\]\(img/manual/([a-z0-9][a-z0-9-]*\.(?:png|jpg))")
# JPEG frame headers (SOF0–SOF15 but DHT, JPG and DAC), where the size is
_JPEG_FRAMES = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _path(file: str) -> Path | None:
    for candidate in (_PACKAGE_DIR / "docs" / file, _PACKAGE_DIR.parent.parent / "docs" / file):
        if candidate.is_file():
            return candidate
    return None


def image_path(file: str) -> Path:
    """A picture of the guides by its bare name (``nodes.png``); nothing outside the pictures' folder."""
    path = _path(f"{IMAGE_DIR}/{file}") if _IMAGE_NAME.fullmatch(file) else None
    if path is None:
        raise ServiceError(404, "그런 그림이 없습니다.")
    return path


def _png_size(data: bytes) -> list[int] | None:
    if len(data) < 24 or data[:8] != PNG_SIGNATURE:
        return None
    return list(struct.unpack(">II", data[16:24]))


def _jpeg_size(data: bytes) -> list[int] | None:
    at = 2 if data[:2] == b"\xff\xd8" else len(data)
    while at + 9 <= len(data):
        if data[at] != 0xFF:
            return None
        marker, length = data[at + 1], struct.unpack(">H", data[at + 2 : at + 4])[0]
        if marker in _JPEG_FRAMES:
            height, width = struct.unpack(">HH", data[at + 5 : at + 9])
            return [width, height]
        at += 2 + length
    return None


def image_size(path: Path) -> list[int] | None:
    """Width and height of a guide's picture, from its header (no image library)."""
    data = path.read_bytes()
    return _png_size(data) if path.suffix == ".png" else _jpeg_size(data)


def _image_sizes(markdown: str) -> dict[str, list[int]]:
    sizes: dict[str, list[int]] = {}
    for file in set(_IMAGE_REF.findall(markdown)):
        path = _path(f"{IMAGE_DIR}/{file}")
        size = image_size(path) if path else None
        if size:
            sizes[file] = size
    return sizes


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
    markdown = path.read_text(encoding="utf-8")
    return {
        "name": name,
        "title": guide["title"],
        "file": guide["file"],
        "markdown": markdown,
        "images": _image_sizes(markdown),  # name → [width, height]
    }
