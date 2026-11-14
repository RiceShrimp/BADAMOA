"""Write a release as a YOLO-format folder for a trainer.

The release lives in the Hub's catalog (which images, which split, which label
revision); the image files live on this node. The export links the images into
``images/<split>/`` (no copy when the filesystem allows it) and writes the
pinned labels to ``labels/<split>/``, so training always sees the labels the
release recorded — not whatever label files sit next to the images today.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "test")


@dataclass
class ExportOptions:
    classes: list[int] = field(default_factory=list)  # class ids to keep; empty = all
    unlabelled: str = "exclude"  # exclude | background
    val_fraction: float = 0.2  # held out of train when the release has no val split


def select_boxes(item: dict[str, Any], keep: dict[int, int], options: ExportOptions) -> tuple[list[list[float]] | None, str]:
    """Boxes to train on (class ids remapped to 0..n-1), or None to leave the image out.

    The second value says why an image was left out or downgraded: ``unlabelled``
    (nobody labelled it) or ``partial`` (labelled, but not for every class being trained).
    """
    state = item["state"]
    boxes = [[keep[int(b[0])], *b[1:5]] for b in item.get("boxes", []) if int(b[0]) in keep]
    if state == "background":
        return [], ""
    if state == "unlabelled":
        return ([], "unlabelled") if options.unlabelled == "background" else (None, "unlabelled")
    covered = set(item.get("labelled_class_ids") or keep)
    if not set(keep) <= covered:  # a missing box of an uncovered class does not mean "not there"
        return (boxes, "partial") if options.unlabelled == "background" else (None, "partial")
    return boxes, ""


def _held_out(sha256: str, fraction: float) -> bool:
    """Deterministic per image, so the same release always gives the same val set."""
    return int(sha256[:8], 16) / 0xFFFFFFFF < fraction


def _may_be_rewritten(source: Path) -> bool:
    """A JPEG without its end marker: Ultralytics "restores" such a file by saving over it.

    Through a link that would change the registered original, so these are copied instead.
    """
    try:
        with source.open("rb") as f:
            if f.read(2) != b"\xff\xd8":
                return False
            f.seek(-2, os.SEEK_END)
            return f.read(2) != b"\xff\xd9"
    except OSError:
        return False


def _place(source: Path, target: Path) -> str:
    """Hard link, else symlink, else copy. Returns which one worked.

    Registered images are never modified: a file the trainer might write to gets its own copy.
    """
    if target.exists():
        target.unlink()
    links = () if _may_be_rewritten(source) else (("hardlink", os.link), ("symlink", os.symlink))
    for name, make in links:
        try:
            make(source, target)
            return name
        except OSError:
            continue
    shutil.copy2(source, target)
    return "copy"


def build(
    items: Iterable[dict[str, Any]],
    classes: list[dict[str, Any]],
    locations: dict[str, str],
    out: Path,
    options: ExportOptions,
) -> dict[str, Any]:
    """Write ``out`` as a YOLO dataset. ``items`` come from the Hub's training manifest.

    Returns a summary: image counts per split, what was left out and why, the
    class names in training order, and how the val split was obtained.
    """
    all_ids = [int(c["id"]) for c in classes]
    chosen = [cid for cid in all_ids if cid in set(options.classes)] if options.classes else all_ids
    if not chosen:
        raise ValueError("학습할 클래스가 없습니다.")
    keep = {cid: i for i, cid in enumerate(chosen)}
    names = {int(c["id"]): str(c["name"]) for c in classes}

    selected: list[tuple[dict[str, Any], str, list[list[float]]]] = []
    excluded = {"unlabelled": 0, "partial": 0, "missing": 0}
    as_background = 0
    for item in items:
        boxes, reason = select_boxes(item, keep, options)
        if boxes is None:
            excluded[reason] += 1
            continue
        if reason:
            as_background += 1
        split = item.get("split") if item.get("split") in SPLITS else "train"
        selected.append((item, split, boxes))

    val_source = "release"
    if not any(split == "val" for _, split, _ in selected):
        if options.val_fraction > 0:
            val_source = "holdout"
            selected = [
                (item, "val" if split == "train" and _held_out(item["sha256"], options.val_fraction) else split, boxes)
                for item, split, boxes in selected
            ]
        if not any(split == "val" for _, split, _ in selected):
            val_source = "train"  # too few images to hold any out: validate on train

    counts = {split: 0 for split in SPLITS}
    methods: dict[str, int] = {}
    box_total = 0
    for item, split, boxes in selected:
        root = Path(locations.get(str(item.get("location_id")), "")).expanduser()
        source = root / item["rel_path"]
        try:
            ok = source.is_file() and source.stat().st_size == item["bytes"]
        except OSError:
            ok = False
        if not ok:  # moved, deleted or replaced since it was registered
            excluded["missing"] += 1
            continue
        stem = item["sha256"]
        image = out / "images" / split / f"{stem}{source.suffix.lower()}"
        label = out / "labels" / split / f"{stem}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        method = _place(source, image)
        methods[method] = methods.get(method, 0) + 1
        rows = (f"{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in boxes)
        label.write_text("".join(rows), encoding="utf-8")
        counts[split] += 1
        box_total += len(boxes)

    if counts["train"] == 0:
        raise ValueError("학습에 쓸 이미지가 없습니다. 릴리스의 라벨 상태와 파일 위치를 확인하세요.")

    lines = [f"path: {json.dumps(out.resolve().as_posix(), ensure_ascii=False)}", "train: images/train"]
    lines.append(f"val: images/{'val' if counts['val'] else 'train'}")
    if counts["test"]:
        lines.append("test: images/test")
    lines.append("names:")
    lines += [f"  {i}: {json.dumps(names[cid], ensure_ascii=False)}" for cid, i in keep.items()]
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "items": counts,
        "boxes": box_total,
        "excluded": excluded,
        "as_background": as_background,
        "classes": [names[cid] for cid in chosen],
        "class_ids": chosen,
        "val_source": val_source,
        "files": methods,
        "data_yaml": str(out / "data.yaml"),
    }
