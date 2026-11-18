"""Write a release in the format a template's code reads: YOLO (detection) or one folder per class (classification).

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
    # Evaluation wants the release's splits exactly as they are: no stand-in for a missing val split.
    keep_splits: bool = False
    masks: bool = False  # write outlines (segmentation) instead of boxes where a label has one
    symlinks: bool = True  # False: hard links or copies only (code in a container that cannot follow a link)


def select_boxes(item: dict[str, Any], keep: dict[int, int], options: ExportOptions) -> tuple[list[list[float]] | None, str]:
    """Boxes to train on (class ids remapped to 0..n-1), or None to leave the image out.

    The second value says why an image was left out or downgraded: ``unlabelled``
    (nobody labelled it) or ``partial`` (labelled, but not for every class being trained).
    """
    state = item["state"]
    boxes = [[keep[int(b[0])], *(b[1:] if options.masks else b[1:5])] for b in item.get("boxes", []) if int(b[0]) in keep]
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


def _place(source: Path, target: Path, symlinks: bool = True) -> str:
    """Hard link, else symlink, else copy. Returns which one worked.

    Registered images are never modified: a file the trainer might write to gets its own copy.
    ``symlinks=False``: no symbolic links (code in a container that cannot follow them).
    """
    if target.exists():
        target.unlink()
    links: tuple = () if _may_be_rewritten(source) else (("hardlink", os.link), ("symlink", os.symlink))
    if not symlinks:
        links = tuple(link for link in links if link[0] != "symlink")
    for name, make in links:
        try:
            make(source, target)
            return name
        except OSError:
            continue
    shutil.copy2(source, target)
    return "copy"


def _label_line(row: list[float], masks: bool) -> str:
    """One YOLO label line: the box, or for segmentation the outline that follows it in the stored label."""
    values = row[5:] if masks and len(row) > 5 else row[1:5]
    return f"{int(row[0])} " + " ".join(f"{v:.6f}" for v in values) + "\n"


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
    if not options.keep_splits and not any(split == "val" for _, split, _ in selected):
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
    registered: dict[str, str] = {}  # exported file stem → the registered path, to name images in results
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
        method = _place(source, image, options.symlinks)
        methods[method] = methods.get(method, 0) + 1
        label.write_text("".join(_label_line(row, options.masks) for row in boxes), encoding="utf-8")
        registered[stem] = item["rel_path"]
        counts[split] += 1
        box_total += len(boxes)

    if counts["train"] == 0 and not options.keep_splits:
        raise ValueError("학습에 쓸 이미지가 없습니다. 버전의 라벨 상태와 파일 위치를 확인하세요.")

    lines = [f"path: {json.dumps(out.resolve().as_posix(), ensure_ascii=False)}", "train: images/train"]
    lines.append(f"val: images/{'val' if counts['val'] else 'train'}")
    if counts["test"]:
        lines.append("test: images/test")
    lines.append("names:")
    lines += [f"  {i}: {json.dumps(names[cid], ensure_ascii=False)}" for cid, i in keep.items()]
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "files.json").write_text(json.dumps(registered, ensure_ascii=False), encoding="utf-8")

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


def build_imagefolder(
    items: Iterable[dict[str, Any]],
    classes: list[dict[str, Any]],
    locations: dict[str, str],
    out: Path,
    options: ExportOptions,
) -> dict[str, Any]:
    """Write ``out`` as ``<split>/<class>/<image>`` for classification (torchvision ``ImageFolder``).

    Class folders are named ``0000_<name>``, ``0001_<name>`` … so that their alphabetical
    order is the class order. A release without a val split is validated on its test
    split when it has one (the usual shape of classification datasets), else on a
    held-out share of train.
    """
    all_ids = [int(c["id"]) for c in classes]
    chosen = [cid for cid in all_ids if cid in set(options.classes)] if options.classes else all_ids
    if not chosen:
        raise ValueError("학습할 클래스가 없습니다.")
    keep = {cid: i for i, cid in enumerate(chosen)}
    names = {int(c["id"]): str(c["name"]) for c in classes}
    safe = {cid: "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in names[cid]) for cid in keep}
    folders = {cid: f"{i:04d}_{safe[cid]}" for cid, i in keep.items()}

    selected: list[tuple[dict[str, Any], str, int]] = []
    excluded = {"unlabelled": 0, "partial": 0, "missing": 0}
    for item in items:
        label = int(item["boxes"][0][0]) if item["state"] == "labelled" and item.get("boxes") else None
        if label is None:
            excluded["unlabelled"] += 1  # nobody said which class it is
        elif label in keep:
            selected.append((item, item.get("split") if item.get("split") in SPLITS else "train", label))

    val_source = "release"
    if not options.keep_splits and not any(split == "val" for _, split, _ in selected):
        if any(split == "test" for _, split, _ in selected):
            val_source = "test"
            selected = [(item, "val" if split == "test" else split, label) for item, split, label in selected]
        elif options.val_fraction > 0:
            val_source = "holdout"
            selected = [
                (item, "val" if split == "train" and _held_out(item["sha256"], options.val_fraction) else split, label)
                for item, split, label in selected
            ]
        if not any(split == "val" for _, split, _ in selected):
            val_source = "train"

    counts = {split: 0 for split in SPLITS}
    methods: dict[str, int] = {}
    registered: dict[str, str] = {}  # exported file stem → the registered path, to name images in results
    for item, split, label in selected:
        root = Path(locations.get(str(item.get("location_id")), "")).expanduser()
        source = root / item["rel_path"]
        try:
            ok = source.is_file() and source.stat().st_size == item["bytes"]
        except OSError:
            ok = False
        if not ok:
            excluded["missing"] += 1
            continue
        target = out / split / folders[label] / f"{item['sha256']}{source.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        method = _place(source, target, options.symlinks)
        methods[method] = methods.get(method, 0) + 1
        registered[item["sha256"]] = item["rel_path"]
        counts[split] += 1
    if counts["train"] == 0 and not options.keep_splits:
        raise ValueError("학습에 쓸 이미지가 없습니다. 버전의 라벨 상태와 파일 위치를 확인하세요.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "files.json").write_text(json.dumps(registered, ensure_ascii=False), encoding="utf-8")
    for split in ("train", "val"):  # every class folder exists in both, so class indices line up
        if counts[split]:
            for folder in folders.values():
                (out / split / folder).mkdir(parents=True, exist_ok=True)

    return {
        "items": counts,
        "boxes": sum(counts.values()),  # one label per image
        "excluded": excluded,
        "as_background": 0,
        "classes": [names[cid] for cid in chosen],
        "class_ids": chosen,
        "val_source": val_source,
        "files": methods,
        "data_yaml": str(out),  # the dataset path handed to the code
    }


def build_segmentation(
    items: Iterable[dict[str, Any]],
    classes: list[dict[str, Any]],
    locations: dict[str, str],
    out: Path,
    options: ExportOptions,
) -> dict[str, Any]:
    """The YOLO layout with outlines instead of boxes (YOLO segmentation format)."""
    options.masks = True
    return build(items, classes, locations, out, options)


BUILDERS = {"yolo": build, "yolo-seg": build_segmentation, "imagefolder": build_imagefolder}


# ---- downloads: a release as one zip in a common format ------------------------------------
#
# Unlike the training layouts above, a download keeps what the release recorded as it is:
# every labelled and background image (only the unlabelled ones stay out), the original file
# names, and every class.


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text)).strip("._") or "release"


def _registered(item: dict[str, Any], locations: dict[str, str]) -> Path | None:
    """The image file, if it is still the one that was registered."""
    source = Path(locations.get(str(item.get("location_id")), "")).expanduser() / item["rel_path"]
    try:
        return source if source.is_file() and source.stat().st_size == item["bytes"] else None
    except OSError:
        return None


def _outline(row: list[float]) -> list[float]:
    """A label's outline; a box without one becomes its four corners (segmentation formats take polygons)."""
    if len(row) > 5:
        return [float(v) for v in row[5:]]
    cx, cy, w, h = (float(v) for v in row[1:5])
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cx - w / 2, cy + h / 2]


def _area(points: list[float]) -> float:
    xs, ys = points[0::2], points[1::2]
    return abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(xs)))) / 2


def write_archive(
    items: Iterable[dict[str, Any]],
    header: dict[str, Any],
    target: Path,
    fmt: str,
    task: str = "detect",
    images: bool = True,
    progress: Any = None,
) -> dict[str, Any]:
    """Write a release as a zip: ``yolo``, ``coco`` or ``imagefolder``; with ``images=False`` the labels only.

    Every zip has ``files.csv`` (split, path in the zip, sha256, the registered path) to match
    labels to images, and a ``README.txt``. Items without a split go to train.
    """
    import csv
    import io
    import zipfile

    names = {int(c["id"]): str(c["name"]) for c in header["classes"]}
    number = {cid: i for i, cid in enumerate(sorted(names))}  # YOLO class numbers, 0..n-1 in id order
    segments = task == "segment"
    counts: dict[str, int] = {}
    skipped = {"unlabelled": 0, "missing": 0}
    unsplit = 0
    taken: dict[str, set[str]] = {}
    coco: dict[str, dict[str, list[dict[str, Any]]]] = {}
    listing = io.StringIO()
    rows = csv.writer(listing)
    rows.writerow(["split", "file", "sha256", "registered_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", allowZip64=True) as bundle:
        for n, item in enumerate(items, 1):
            if progress and n % 500 == 0:
                progress(n)
            boxes = [r for r in item.get("boxes") or [] if int(r[0]) in number]
            if item["state"] == "unlabelled" or (fmt == "imagefolder" and not boxes):
                skipped["unlabelled"] += 1  # nobody said what is in it
                continue
            source = _registered(item, header["locations"]) if images else None
            if images and source is None:
                skipped["missing"] += 1
                continue
            split = item.get("split") if item.get("split") in SPLITS else "train"
            unsplit += item.get("split") not in SPLITS
            original = Path(item["rel_path"]).name
            name = original
            if name.lower() in taken.setdefault(split, set()):  # the same name in another folder
                name = f"{Path(original).stem}-{item['sha256'][:8]}{Path(original).suffix}"
            taken[split].add(name.lower())

            if fmt == "imagefolder":
                path = f"{split}/{_safe(names[int(boxes[0][0])])}/{name}"
            else:
                path = f"images/{split}/{name}"
            if fmt == "yolo":
                lines = "".join(
                    f"{number[int(r[0])]} " + " ".join(f"{v:.6f}" for v in (_outline(r) if segments else r[1:5])) + "\n"
                    for r in boxes
                )
                bundle.writestr(f"labels/{split}/{Path(name).stem}.txt", lines, compress_type=zipfile.ZIP_DEFLATED)
            elif fmt == "coco":
                w, h = int(item.get("width") or 0), int(item.get("height") or 0)
                bucket = coco.setdefault(split, {"images": [], "annotations": []})
                image_id = len(bucket["images"]) + 1
                bucket["images"].append({"id": image_id, "file_name": name, "width": w, "height": h})
                for r in boxes:
                    cx, cy, bw, bh = (float(v) for v in r[1:5])
                    box = [round((cx - bw / 2) * w, 2), round((cy - bh / 2) * h, 2), round(bw * w, 2), round(bh * h, 2)]
                    note: dict[str, Any] = {
                        "id": len(bucket["annotations"]) + 1,
                        "image_id": image_id,
                        "category_id": number[int(r[0])] + 1,
                        "bbox": box,
                        "area": round(box[2] * box[3], 2),
                        "iscrowd": 0,
                    }
                    if segments:
                        polygon = [round(v * (w if i % 2 == 0 else h), 2) for i, v in enumerate(_outline(r))]
                        note["segmentation"], note["area"] = [polygon], round(_area(polygon), 2)
                    bucket["annotations"].append(note)
            if images:
                bundle.write(source, path, compress_type=zipfile.ZIP_STORED)  # images are compressed already
            rows.writerow([split, path, item["sha256"], item["rel_path"]])
            counts[split] = counts.get(split, 0) + 1

        if not counts:
            raise ValueError("담을 항목이 없습니다. 버전의 라벨 상태와 이미지 파일 위치를 확인하세요.")
        ordered = [names[cid] for cid in sorted(names)]
        if fmt == "yolo":
            lines = [f"{s}: images/{s}" for s in SPLITS if counts.get(s)]
            if not counts.get("val"):
                lines.append("val: images/train")
            lines += ["names:"] + [f"  {i}: {json.dumps(n, ensure_ascii=False)}" for i, n in enumerate(ordered)]
            bundle.writestr("data.yaml", "\n".join(lines) + "\n", compress_type=zipfile.ZIP_DEFLATED)
        elif fmt == "coco":
            categories = [{"id": i + 1, "name": n} for i, n in enumerate(ordered)]
            for split, bucket in coco.items():
                text = json.dumps({**bucket, "categories": categories}, ensure_ascii=False)
                bundle.writestr(f"annotations/instances_{split}.json", text, compress_type=zipfile.ZIP_DEFLATED)
        bundle.writestr("files.csv", listing.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
        what = "" if images else " (라벨만 — 이미지는 files.csv의 registered_path)"
        readme = [
            f"{header['dataset']} {header['version']} — BADAMOA에서 내보냄",
            f"형식: {fmt}{what}",
            "항목: " + ", ".join(f"{s} {counts[s]}" for s in SPLITS if counts.get(s)),
            f"클래스 {len(ordered)}개: " + ", ".join(ordered),
            f"빠진 항목: 미라벨 {skipped['unlabelled']}개, 파일이 바뀌었거나 없음 {skipped['missing']}개",
        ]
        if unsplit:
            readme.append(f"split이 없던 {unsplit}개는 train에 넣었습니다.")
        bundle.writestr("README.txt", "\n".join(readme) + "\n", compress_type=zipfile.ZIP_DEFLATED)
    return {"counts": counts, "skipped": skipped, "unsplit": unsplit, "classes": ordered}


def run_download(params: dict[str, Any]) -> dict[str, Any]:
    """The ``data.export`` job: the release as a zip, uploaded to the Hub for downloading."""
    import httpx

    from badamoa.node import trainer

    hub, token, job_id, home = (
        os.environ.get(k) for k in ("BADAMOA_HUB_URL", "BADAMOA_NODE_TOKEN", "BADAMOA_JOB_ID", "BADAMOA_NODE_HOME")
    )
    if not (hub and token and job_id and home):
        raise RuntimeError("이 작업은 Node Agent가 실행해야 합니다.")
    fmt, images, task = params.get("format", "yolo"), bool(params.get("images", True)), params.get("task", "detect")
    work = Path(home).resolve() / "runs" / job_id
    shutil.rmtree(work, ignore_errors=True)
    base = f"/api/agent/jobs/{job_id}/export"
    with httpx.Client(base_url=hub, headers={"Authorization": f"Bearer {token}"}, timeout=120) as client:
        header, items = trainer.manifest_stream(client, base)
        suffix = "" if images else "-labels"
        archive = work / f"{_safe(header['dataset'])}-{_safe(header['version'])}-{fmt}{suffix}.zip"
        kind = "이미지 포함" if images else "라벨만"
        print(f"{header['dataset']} {header['version']} → {fmt} ({kind})", flush=True)
        summary = write_archive(
            items, header, archive, fmt, task, images, progress=lambda n: print(f"{n:,}개 담음", flush=True)
        )
        size = archive.stat().st_size
        split_text = ", ".join(f"{s} {n}" for s, n in summary["counts"].items())
        print(f"zip {size / 1024**2:.1f} MB · {split_text}", flush=True)
        with archive.open("rb") as stream:
            resp = client.put(f"{base}/file", params={"name": archive.name}, content=stream, timeout=None)
        if resp.status_code != 200:
            raise RuntimeError(f"Hub에 올리지 못했습니다 ({resp.status_code}): {resp.text[:300]}")
        stored = resp.json()
    shutil.rmtree(work, ignore_errors=True)
    print("Hub에 올렸습니다. 버전 화면에서 내려받을 수 있습니다.", flush=True)
    return {**summary, **stored}
