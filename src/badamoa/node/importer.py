"""Register a dataset that already sits on this node, without copying it.

The node walks a YOLO or COCO detection dataset, or a classification dataset laid
out as one folder per class, hashes every image, makes a small thumbnail, and
sends the *index* (hash, size, split, labels, thumbnail) to the Hub in chunks.
The images themselves stay where they are.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
CHUNK = 100
THUMB_SIDE = 320


@dataclass
class Entry:
    """One image found in the dataset, before it is read."""

    path: Path
    split: str = ""
    state: str = "unlabelled"  # labelled | background | unlabelled
    boxes: list[list[float]] = field(default_factory=list)


@dataclass
class Scan:
    format: str
    classes: list[str]
    entries: list[Entry]


# ---- YOLO ----------------------------------------------------------------------


def parse_yolo_yaml(text: str) -> dict[str, Any]:
    """Read the handful of keys a YOLO data.yaml uses (no YAML dependency on nodes)."""
    data: dict[str, Any] = {}
    names: dict[int, str] | list[str] | None = None
    in_names = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if in_names and (line.startswith((" ", "\t", "-"))):
            item = line.strip()
            if item.startswith("-"):
                assert isinstance(names, list)
                names.append(item[1:].strip().strip("'\""))
            elif ":" in item:
                key, value = item.split(":", 1)
                if not isinstance(names, dict):
                    names = {}
                names[int(key.strip())] = value.strip().strip("'\"")
            continue
        in_names = False
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key == "names":
            if value.startswith("["):
                names = [v.strip().strip("'\"") for v in value.strip("[]").split(",") if v.strip()]
            elif value.startswith("{"):
                pairs = [p.split(":", 1) for p in value.strip("{}").split(",") if ":" in p]
                names = {int(k.strip()): v.strip().strip("'\"") for k, v in pairs}
            else:
                names, in_names = [], True
        elif key in ("path", "train", "val", "test"):
            data[key] = value.strip("'\"")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    data["names"] = names or []
    return data


def _images_under(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())


def _yolo_label_path(image: Path) -> Path:
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _read_yolo_label(path: Path) -> tuple[str, list[list[float]]]:
    if not path.is_file():
        return "unlabelled", []
    boxes = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            values = [round(min(1.0, max(0.0, float(v))), 5) for v in parts[1:]]
        except ValueError:
            continue
        if len(values) >= 6 and len(values) % 2 == 0:
            # A segmentation label: the outline as x1 y1 x2 y2 …. Stored as its box followed by the outline.
            xs, ys = values[0::2], values[1::2]
            box = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
            boxes.append([cls, *(round(v, 5) for v in box), *values])
        else:
            boxes.append([cls, *values[:4]])
    return ("labelled" if boxes else "background"), boxes


def scan_yolo(root: Path) -> Scan:
    yaml_files = [p for p in (root / "data.yaml", root / "dataset.yaml") if p.is_file()] or sorted(root.glob("*.yaml"))
    config = parse_yolo_yaml(yaml_files[0].read_text(encoding="utf-8", errors="replace")) if yaml_files else {}
    base = root / config["path"] if config.get("path") and not Path(config["path"]).is_absolute() else root
    if config.get("path") and Path(config["path"]).is_absolute() and Path(config["path"]).is_dir():
        base = Path(config["path"])

    split_dirs: list[tuple[str, Path]] = []
    for key in ("train", "val", "test"):
        if config.get(key):
            candidate = (base / config[key]).resolve()
            if not candidate.is_dir():
                candidate = (root / config[key]).resolve()
            if candidate.is_dir():
                split_dirs.append((key, candidate))
    if not split_dirs:
        images = root / "images" if (root / "images").is_dir() else root
        # Folder names decide the split; COCO-style names such as "train2017" count too.
        named = [(_split_from_name(d.name), d) for d in sorted(images.iterdir()) if d.is_dir() and _split_from_name(d.name)]
        nested = [
            (_split_from_name(d.name), d / "images")
            for d in sorted(root.iterdir())
            if d.is_dir() and _split_from_name(d.name) and (d / "images").is_dir()
        ]
        split_dirs = named or nested or [("", images)]

    entries = []
    for split, directory in split_dirs:
        for image in _images_under(directory):
            state, boxes = _read_yolo_label(_yolo_label_path(image))
            entries.append(Entry(path=image, split=split, state=state, boxes=boxes))

    classes = list(config.get("names") or [])
    if not classes and (root / "classes.txt").is_file():
        classes = [line.strip() for line in (root / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    top = max((int(b[0]) for e in entries for b in e.boxes), default=-1)
    classes += [f"class_{i}" for i in range(len(classes), top + 1)]
    return Scan(format="yolo", classes=classes, entries=entries)


# ---- COCO ----------------------------------------------------------------------


def _coco_files(root: Path) -> list[Path]:
    found = []
    for path in sorted([*root.glob("*.json"), *(root / "annotations").glob("*.json")]):
        try:
            with path.open(encoding="utf-8") as f:
                head = f.read(4096)
        except OSError:
            continue
        if '"images"' in head or '"annotations"' in head or '"categories"' in head or path.stat().st_size > 4096:
            found.append(path)
    return found


def _split_from_name(name: str) -> str:
    low = name.lower()
    for alias, split in SPLIT_ALIASES.items():
        if re.search(rf"(^|[^a-z]){alias}([^a-z]|$)", low):
            return split
    return ""


def _polygon_area(points: list[float]) -> float:
    xs, ys = points[0::2], points[1::2]
    return abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(xs)))) / 2


def _coco_outline(segmentation: Any, width: float, height: float) -> list[float]:
    """The outline of a COCO annotation as normalised x1, y1, x2, y2, …; empty when there is none.

    An object drawn as several polygons keeps its largest one; RLE masks are not read.
    """
    if not isinstance(segmentation, list):
        return []
    polygons = [p for p in segmentation if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0]
    if not polygons:
        return []
    largest = max(polygons, key=_polygon_area)
    return [round(min(1.0, max(0.0, float(v) / (width if n % 2 == 0 else height))), 5) for n, v in enumerate(largest)]


def scan_coco(root: Path) -> Scan:
    categories: dict[int, str] = {}
    parsed = []
    for path in _coco_files(root):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict) or "images" not in data:
            continue
        for cat in data.get("categories", []):
            categories[int(cat["id"])] = str(cat.get("name", cat["id"]))
        parsed.append((path, data))
    class_ids = sorted(categories)
    index = {cid: i for i, cid in enumerate(class_ids)}

    entries = []
    for path, data in parsed:
        split = _split_from_name(path.stem)
        has_annotations = "annotations" in data
        boxes_by_image: dict[Any, list[list[float]]] = {}
        sizes = {img["id"]: (img.get("width") or 0, img.get("height") or 0) for img in data["images"]}
        for ann in data.get("annotations", []):
            w, h = sizes.get(ann.get("image_id"), (0, 0))
            bbox = ann.get("bbox")
            if not w or not h or not bbox or ann.get("category_id") not in index:
                continue
            x, y, bw, bh = (float(v) for v in bbox[:4])
            box = [index[ann["category_id"]], (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h]
            row = [box[0], *(round(min(1.0, max(0.0, v)), 5) for v in box[1:])]
            # Kept only in segmentation projects; a detection project uses the box.
            boxes_by_image.setdefault(ann["image_id"], []).append([*row, *_coco_outline(ann.get("segmentation"), w, h)])
        for img in data["images"]:
            name = img["file_name"]
            candidates = [root / name, root / "images" / name, root / "images" / split / name, root / split / name]
            candidates += [root / f"{split}2017" / name, path.parent / name, root / Path(name).name]
            image = next((c for c in candidates if c.is_file()), None)
            if image is None:
                continue
            boxes = boxes_by_image.get(img["id"], [])
            state = "labelled" if boxes else ("background" if has_annotations else "unlabelled")
            entries.append(Entry(path=image, split=split, state=state, boxes=boxes))
    return Scan(format="coco", classes=[categories[c] for c in class_ids], entries=entries)


# ---- classification: one folder per class ----------------------------------------


def _class_dirs(directory: Path) -> list[Path]:
    """Sub-folders that hold images directly: the classes of an image-folder dataset."""
    return [
        d
        for d in sorted(directory.iterdir())
        if d.is_dir() and any(p.suffix.lower() in IMAGE_SUFFIXES for p in d.iterdir() if p.is_file())
    ]


def _imagefolder_splits(root: Path) -> list[tuple[str, Path]]:
    """``train/<class>/…`` + ``val/<class>/…`` (split folders), or ``<class>/…`` right under the root."""
    splits = [
        (_split_from_name(d.name), d)
        for d in sorted(root.iterdir())
        if d.is_dir() and _split_from_name(d.name) and _class_dirs(d)
    ]
    if splits:
        return splits
    return [("", root)] if _class_dirs(root) else []


def scan_imagefolder(root: Path) -> Scan:
    """A classification dataset. Each image carries one label, written as ``[[class_id]]``."""
    splits = _imagefolder_splits(root)
    classes = sorted({d.name for _, base in splits for d in _class_dirs(base)})
    index = {name: i for i, name in enumerate(classes)}
    entries = [
        Entry(path=image, split=split, state="labelled", boxes=[[index[folder.name]]])
        for split, base in splits
        for folder in _class_dirs(base)
        for image in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    ]
    return Scan(format="imagefolder", classes=classes, entries=entries)


def scan(root: Path, fmt: str = "auto") -> Scan:
    if not root.is_dir():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {root}")
    detection_layout = (root / "images").is_dir() or (root / "labels").is_dir() or list(root.glob("*.yaml"))
    looks_like_classes = not detection_layout and not _coco_files(root) and _imagefolder_splits(root)
    if fmt == "imagefolder" or (fmt == "auto" and looks_like_classes):
        return scan_imagefolder(root)
    if fmt == "coco" or (fmt == "auto" and not list(root.glob("*.yaml")) and _coco_files(root)):
        result = scan_coco(root)
        if result.entries or fmt == "coco":
            return result
    return scan_yolo(root)


# ---- reading images and uploading ------------------------------------------------


def describe(entry: Entry, root: Path) -> dict[str, Any] | None:
    """Hash the file, read its size, and build a thumbnail. None if it is not a readable image."""
    from PIL import Image, UnidentifiedImageError

    try:
        raw = entry.path.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            image.thumbnail((THUMB_SIDE, THUMB_SIDE))
            out = io.BytesIO()
            image.convert("RGB").save(out, format="JPEG", quality=75)
    except (OSError, UnidentifiedImageError, ValueError):
        return None
    try:
        rel = entry.path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        rel = entry.path.as_posix()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "file_name": entry.path.name,
        "rel_path": rel,
        "width": width,
        "height": height,
        "bytes": len(raw),
        "split": entry.split,
        "state": entry.state,
        "boxes": entry.boxes,
        "thumb_b64": base64.b64encode(out.getvalue()).decode(),
    }


def chunks(scan_result: Scan, root: Path) -> Iterator[tuple[list[dict[str, Any]], int]]:
    """Yield (items, skipped) batches of CHUNK described images."""
    batch: list[dict[str, Any]] = []
    skipped = 0
    for entry in scan_result.entries:
        item = describe(entry, root)
        if item is None:
            skipped += 1
            continue
        batch.append(item)
        if len(batch) >= CHUNK:
            yield batch, skipped
            batch, skipped = [], 0
    if batch or skipped:
        yield batch, skipped


def run(params: dict[str, Any]) -> dict[str, Any]:
    """Job entry point (``data.import``)."""
    hub, token, job_id = (os.environ.get(k) for k in ("BADAMOA_HUB_URL", "BADAMOA_NODE_TOKEN", "BADAMOA_JOB_ID"))
    if not (hub and token and job_id):
        raise RuntimeError("이 작업은 Node Agent가 실행해야 합니다.")
    root = Path(params["path"]).expanduser()
    print(f"폴더 확인: {root}")
    found = scan(root, params.get("format") or "auto")
    total = len(found.entries)
    print(f"형식 {found.format} · 클래스 {len(found.classes)}개 · 이미지 {total:,}장")
    if total == 0:
        raise RuntimeError("이미지를 찾지 못했습니다. 경로와 형식을 확인하세요.")

    base = f"/api/agent/jobs/{job_id}/import"
    with httpx.Client(base_url=hub, headers={"Authorization": f"Bearer {token}"}, timeout=120) as client:

        def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
            resp = client.post(base + path, json=body)
            if resp.status_code != 200:
                raise RuntimeError(f"Hub가 거부했습니다 ({resp.status_code}): {resp.text[:300]}")
            return resp.json()

        post("/begin", {"classes": found.classes, "total": total, "format": found.format})
        sent = skipped = 0
        for batch, bad in chunks(found, root):
            skipped += bad
            if batch:
                post("/items", {"items": batch})
                sent += len(batch)
                print(f"등록 {sent:,} / {total:,}", flush=True)
        result = post("/finish", {})
    if skipped:
        print(f"읽을 수 없는 파일 {skipped}개는 건너뛰었습니다.")
    print(f"완료: 버전 {result.get('version')} · 항목 {result.get('items'):,}개")
    return {**result, "skipped": skipped, "format": found.format}
