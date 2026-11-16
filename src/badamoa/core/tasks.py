"""What a dataset is labelled for: four tasks built in, and any other a team names.

A dataset's task is chosen when it is imported ("자동" reads it from the folder). The built-in ones
are shown by short English names; a named task (say "OCR") keeps every value of its labels and is
trained by a template that declares the same task.
"""

from __future__ import annotations

import re

BUILTIN = ("detect", "segment", "pose", "classify")
SHORT = {"detect": "Det", "segment": "Seg", "pose": "Kpt", "classify": "Cls"}
_ALIASES = {
    **{a: "detect" for a in ("det", "detect", "detection", "bbox", "box", "boxes")},
    **{a: "segment" for a in ("seg", "segment", "segmentation", "mask", "masks")},
    **{a: "pose" for a in ("kpt", "kpts", "pose", "keypoint", "keypoints")},
    **{a: "classify" for a in ("cls", "classify", "classification")},
}
_NAMED = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,23}$")


def normalize(text: str | None) -> str:
    """A task as stored: a built-in id, a named task as typed, or "" (decide from the data).

    Raises ValueError for a name that cannot be one.
    """
    name = (text or "").strip()
    if not name or name.lower() in ("auto", "자동"):
        return ""
    if name.lower() in _ALIASES:
        return _ALIASES[name.lower()]
    if not _NAMED.match(name):
        raise ValueError("태스크 이름은 영문으로 시작하는 영문 · 숫자 · `-` · `_` 1~24자여야 합니다 (예: OCR, Depth).")
    return name


def short(task: str | None) -> str:
    """How a task is shown: Det, Seg, Kpt, Cls, or a named task as typed."""
    return SHORT.get(task or "", task or "Det")


def same(a: str | None, b: str | None) -> bool:
    return (a or "").lower() == (b or "").lower()
