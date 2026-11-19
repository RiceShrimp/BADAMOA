"""Register a dataset that already sits on this node, without copying it.

The node walks a YOLO or COCO detection dataset, or a classification dataset laid
out as one folder per class, hashes every image, makes a small thumbnail, and
sends the *index* (hash, size, split, labels, thumbnail) to the Hub in chunks.
The images themselves stay where they are.

Datasets of hundreds of thousands of images often sit on a network folder, where
every call per file is a round trip. Folders are listed with ``os.scandir`` (the
file type comes with the listing, so no call per image), label files are read
many at a time, and how far each step got goes to the Hub as it goes:
finding the images, reading their labels, then reading and sending the images.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from badamoa.core import tasks as task_names

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}
CHUNK = 200
WORKERS = min(16, os.cpu_count() or 4)  # images read at once: decoding and hashing let go of the GIL
REPORT_EVERY_S = 5.0
THUMB_SIDE = 320
# Label files read at once: each is a small read, so on a network folder the round trip is most of the time.
LABEL_READERS = 32
PROGRESS_EVERY_S = 2.0  # how often the Hub hears how far finding images and reading labels got
LOG_EVERY_S = 30.0  # and the job's log

# A step's progress: (step, done, total). Steps: "find" (images found so far, total 0), "labels" (label files read).
Report = Callable[[str, int, int], None]


def _quiet(step: str, done: int, total: int) -> None:
    """No one to tell (tests, scans outside a job)."""


@dataclass
class Entry:
    """One image found in the dataset, before it is read."""

    path: Path
    split: str = ""
    # labelled | background (nothing to find in it) | unlabelled (nobody has labelled it yet)
    state: str = "background"
    boxes: list[list[float]] = field(default_factory=list)


@dataclass
class Scan:
    format: str
    classes: list[str]
    entries: list[Entry]
    # Keypoint (pose) datasets: {"count": K, "names": [...], "skeleton": [[a, b], ...]} (0-based pairs)
    keypoints: dict[str, Any] | None = None


# COCO's 17 body keypoints: the names and limbs used when a dataset gives none for 17 points.
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]  # fmt: skip
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
    [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
]  # fmt: skip


def keypoint_info(count: int, names: list[str] | None = None, skeleton: list[list[int]] | None = None) -> dict[str, Any]:
    names = [str(n) for n in names or []][:count]
    if not names and count == 17:
        names, skeleton = COCO_KEYPOINTS, skeleton or COCO_SKELETON
    names += [str(i + 1) for i in range(len(names), count)]
    pairs = [[int(a), int(b)] for a, b in skeleton or [] if 0 <= int(a) < count and 0 <= int(b) < count]
    return {"count": count, "names": names, "skeleton": pairs}


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
        elif key == "kpt_shape" and value.startswith("["):
            try:
                data["kpt_shape"] = [int(v) for v in value.strip("[]").split(",") if v.strip()]
            except ValueError:
                pass
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    data["names"] = names or []
    return data


def _is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_SUFFIXES


def _images_under(directory: Path, report: Report = _quiet, found_before: int = 0) -> list[Path]:
    """Every image under ``directory``, in order. Listed with ``os.scandir``: no call per file, which on a
    network folder of hundreds of thousands of images is the difference between minutes and an hour.
    Linked folders are followed, each folder once. ``found_before``: images other folders gave, for the count."""
    found: list[str] = []
    folders, visited = [str(directory)], set()
    while folders:
        folder = folders.pop()
        try:
            st = os.stat(folder)
            if (st.st_dev, st.st_ino) in visited:
                continue
            visited.add((st.st_dev, st.st_ino))
            with os.scandir(folder) as listing:
                entries = list(listing)
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir():
                    folders.append(entry.path)
                elif _is_image(entry.name):
                    found.append(entry.path)
            except OSError:
                continue
        report("find", found_before + len(found), 0)
    return sorted(Path(f) for f in found)


def _yolo_label_path(image: Path) -> Path:
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _clamp(v: float) -> float:
    return round(min(1.0, max(0.0, v)), 5)


def _read_yolo_label(
    path: Path, kpt_shape: list[int] | None = None, as_is: bool = False
) -> tuple[str, list[list[float]]]:
    """Boxes of a YOLO label file: [class, cx, cy, w, h], then an outline (segmentation) or keypoints (pose).

    ``as_is`` (a task a team named): the box, then every value after it exactly as written —
    nothing is taken for an outline, cut off or moved into 0..1.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return "unlabelled", []  # no label file: background or not labelled yet, by the import's rule (settle)
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            raw = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if as_is:
            boxes.append([cls, *(_clamp(v) for v in raw[:4]), *raw[4:]])
            continue
        if kpt_shape:  # pose: box, then K points of (x, y) or (x, y, visibility)
            count, dims = kpt_shape[0], (kpt_shape[1] if len(kpt_shape) > 1 else 3)
            points = raw[4:]
            keypoints: list[float] = []
            for k in range(count):
                x, y = (points[k * dims], points[k * dims + 1]) if len(points) >= (k + 1) * dims else (0.0, 0.0)
                seen = points[k * dims + 2] if dims >= 3 and len(points) >= (k + 1) * dims else (2.0 if x or y else 0.0)
                keypoints += [_clamp(x), _clamp(y), float(int(seen))]
            boxes.append([cls, *(_clamp(v) for v in raw[:4]), *keypoints])
            continue
        values = [_clamp(v) for v in raw]
        if len(values) >= 6 and len(values) % 2 == 0:
            # A segmentation label: the outline as x1 y1 x2 y2 …. Stored as its box followed by the outline.
            xs, ys = values[0::2], values[1::2]
            box = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
            boxes.append([cls, *(round(v, 5) for v in box), *values])
        else:
            boxes.append([cls, *values[:4]])
    return ("labelled" if boxes else "background"), boxes


def scan_yolo(root: Path, as_is: bool = False, report: Report = _quiet) -> Scan:
    """``as_is``: label values kept as written (see ``_read_yolo_label``)."""
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

    kpt_shape = config.get("kpt_shape") or None
    images: list[tuple[str, Path]] = []
    for split, directory in split_dirs:
        images += [(split, image) for image in _images_under(directory, report, len(images))]
    report("find", len(images), 0)

    def read(pair: tuple[str, Path]) -> Entry:
        split, image = pair
        state, boxes = _read_yolo_label(_yolo_label_path(image), kpt_shape, as_is)
        return Entry(path=image, split=split, state=state, boxes=boxes)

    entries = []
    with ThreadPoolExecutor(LABEL_READERS) as pool:
        for entry in _ordered(pool, read, images, LABEL_READERS * 8):
            entries.append(entry)
            if len(entries) % 500 == 0:
                report("labels", len(entries), len(images))
    report("labels", len(entries), len(images))

    classes = list(config.get("names") or [])
    if not classes and (root / "classes.txt").is_file():
        classes = [line.strip() for line in (root / "classes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    top = max((int(b[0]) for e in entries for b in e.boxes), default=-1)
    classes += [f"class_{i}" for i in range(len(classes), top + 1)]
    keypoints = keypoint_info(kpt_shape[0]) if kpt_shape else None
    return Scan(format="yolo", classes=classes, entries=entries, keypoints=keypoints)


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


class _Listings:
    """Which files a folder holds, listed once: looking a name up costs nothing after that."""

    def __init__(self) -> None:
        self.folders: dict[Path, set[str]] = {}

    def has(self, path: Path) -> bool:
        names = self.folders.get(path.parent)
        if names is None:
            try:
                with os.scandir(path.parent) as listing:
                    names = {e.name for e in listing if not e.is_dir()}
            except OSError:
                names = set()
            self.folders[path.parent] = names
        return path.name in names


def scan_coco(root: Path, report: Report = _quiet) -> Scan:
    categories: dict[int, str] = {}
    keypoints: dict[str, Any] | None = None
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
            if cat.get("keypoints") and keypoints is None:  # 1-based limbs in COCO
                pairs = [[int(a) - 1, int(b) - 1] for a, b in cat.get("skeleton") or []]
                keypoints = keypoint_info(len(cat["keypoints"]), list(cat["keypoints"]), pairs)
        parsed.append((path, data))
    class_ids = sorted(categories)
    index = {cid: i for i, cid in enumerate(class_ids)}

    entries = []
    listings = _Listings()
    total = sum(len(data["images"]) for _, data in parsed)
    looked = 0
    for path, data in parsed:
        split = _split_from_name(path.stem)
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
            if keypoints:  # pose: the points (x, y, visibility), as fractions of the image
                points = list(ann.get("keypoints") or [])
                extra: list[float] = []
                for k in range(keypoints["count"]):
                    x, y, seen = (points[k * 3 : k * 3 + 3] + [0, 0, 0])[:3]
                    extra += [_clamp(float(x) / w), _clamp(float(y) / h), float(int(seen))]
            else:  # kept only in segmentation projects; a detection project uses the box
                extra = _coco_outline(ann.get("segmentation"), w, h)
            boxes_by_image.setdefault(ann["image_id"], []).append([*row, *extra])
        for img in data["images"]:
            name = img["file_name"]
            candidates = [root / name, root / "images" / name, root / "images" / split / name, root / split / name]
            candidates += [root / f"{split}2017" / name, path.parent / name, root / Path(name).name]
            image = next((c for c in candidates if listings.has(c)), None)
            looked += 1
            if looked % 500 == 0:
                report("labels", looked, total)
            if image is None:
                continue
            boxes = boxes_by_image.get(img["id"], [])
            state = "labelled" if boxes else "background"
            entries.append(Entry(path=image, split=split, state=state, boxes=boxes))
    report("labels", total, total)
    return Scan(format="coco", classes=[categories[c] for c in class_ids], entries=entries, keypoints=keypoints)


# ---- classification: one folder per class ----------------------------------------


def _class_dirs(directory: Path) -> list[Path]:
    """Sub-folders that hold images directly: the classes of an image-folder dataset."""
    return [d for d in sorted(directory.iterdir()) if d.is_dir() and _holds_images(d)]


def _holds_images(folder: Path) -> bool:
    with os.scandir(folder) as listing:
        return any(_is_image(e.name) and not e.is_dir() for e in listing)


def _images_in(folder: Path) -> list[Path]:
    """The images right in ``folder`` (not below it), in order."""
    with os.scandir(folder) as listing:
        return sorted(Path(e.path) for e in listing if _is_image(e.name) and not e.is_dir())


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


def scan_imagefolder(root: Path, report: Report = _quiet) -> Scan:
    """A classification dataset. Each image carries one label, written as ``[[class_id]]``."""
    splits = [(split, _class_dirs(base)) for split, base in _imagefolder_splits(root)]
    classes = sorted({d.name for _, folders in splits for d in folders})
    index = {name: i for i, name in enumerate(classes)}
    entries = []
    for split, folders in splits:
        for folder in folders:
            entries += [
                Entry(path=image, split=split, state="labelled", boxes=[[index[folder.name]]])
                for image in _images_in(folder)
            ]
            report("find", len(entries), 0)
    return Scan(format="imagefolder", classes=classes, entries=entries)


def settle(found: Scan, rule: str = "auto") -> Scan:
    """Images without a label file are background (the YOLO custom), or not labelled yet: with the rule
    "unlabelled", or "auto" when no image of the folder has a label file (a folder of images to label)."""
    missing = [e for e in found.entries if e.state == "unlabelled"]
    if missing and not (rule == "unlabelled" or (rule == "auto" and len(missing) == len(found.entries))):
        for entry in missing:
            entry.state = "background"
    return found


def scan(root: Path, fmt: str = "auto", unlabelled: str = "auto", task: str = "", report: Report = _quiet) -> Scan:
    """The images and labels of a dataset folder. ``unlabelled``: see ``settle``.

    ``task``: what the dataset is labelled for. A task a team named (not Det · Seg · Kpt · Cls) keeps
    the values of its YOLO labels as written. ``report``: told how far finding images and reading labels got.
    """
    try:
        named = bool(task_names.normalize(task)) and task_names.normalize(task) not in task_names.BUILTIN
    except ValueError:
        named = False
    return settle(_scan(root, fmt, as_is=named, report=report), unlabelled)


def _scan(root: Path, fmt: str = "auto", as_is: bool = False, report: Report = _quiet) -> Scan:
    if not root.is_dir():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {root}")
    detection_layout = (root / "images").is_dir() or (root / "labels").is_dir() or list(root.glob("*.yaml"))
    looks_like_classes = not detection_layout and not _coco_files(root) and _imagefolder_splits(root)
    if fmt == "imagefolder" or (fmt == "auto" and looks_like_classes):
        return scan_imagefolder(root, report)
    if fmt == "coco" or (fmt == "auto" and not list(root.glob("*.yaml")) and _coco_files(root)):
        result = scan_coco(root, report)
        if result.entries or fmt == "coco":
            return result
    return scan_yolo(root, as_is=as_is, report=report)


# ---- reading images and uploading ------------------------------------------------


def describe(entry: Entry, root: Path) -> dict[str, Any] | None:
    """Hash the file, read its size, and build a thumbnail. None if it is not a readable image."""
    from PIL import Image, UnidentifiedImageError

    try:
        raw = entry.path.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            # JPEG: decode at 1/2 … 1/8 scale straight away, enough for a thumbnail and many times faster
            image.draft("RGB", (THUMB_SIDE, THUMB_SIDE))
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


def chunks(scan_result: Scan, root: Path, workers: int = WORKERS) -> Iterator[tuple[list[dict[str, Any]], int]]:
    """Yield (items, skipped) batches of CHUNK described images, read ``workers`` at a time, in order."""
    with ThreadPoolExecutor(max(1, workers)) as pool:
        yield from _batches(_ordered(pool, lambda entry: describe(entry, root), scan_result.entries, workers * 4))


def _ordered(pool: ThreadPoolExecutor, fn: Any, items: list[Any], ahead: int) -> Iterator[Any]:
    """``pool.map`` that reads only ``ahead`` items ahead: a slow Hub must not pile thumbnails up in memory."""
    pending: deque[Any] = deque()
    for item in items:
        pending.append(pool.submit(fn, item))
        if len(pending) >= ahead:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def _batches(described: Iterator[dict[str, Any] | None]) -> Iterator[tuple[list[dict[str, Any]], int]]:
    batch: list[dict[str, Any]] = []
    skipped = 0
    for item in described:
        if item is None:
            skipped += 1
            continue
        batch.append(item)
        if len(batch) >= CHUNK:
            yield batch, skipped
            batch, skipped = [], 0
    if batch or skipped:
        yield batch, skipped


class Progress:
    """Tells the Hub how far finding images and reading labels got (every few seconds), and the log now and then.

    Reading and sending the images needs none of this: the Hub counts what it is sent."""

    WORDS = {"find": "이미지 찾는 중: {done:,}장", "labels": "라벨 읽는 중: {done:,} / {total:,}"}

    def __init__(self, client: httpx.Client, url: str) -> None:
        self.client, self.url = client, url
        self.step, self.sent, self.logged = "", 0.0, time.monotonic()

    def __call__(self, step: str, done: int, total: int) -> None:
        now = time.monotonic()
        if step == self.step and now - self.sent < PROGRESS_EVERY_S:
            return
        self.step, self.sent = step, now
        try:
            self.client.post(self.url, json={"step": step, "done": done, "total": total}, timeout=15)
        except httpx.HTTPError:
            pass  # a Hub older than this, or a moment's trouble: the import goes on
        if now - self.logged >= LOG_EVERY_S:
            self.logged = now
            print(self.WORDS.get(step, step).format(done=done, total=total), flush=True)


def run(params: dict[str, Any]) -> dict[str, Any]:
    """Job entry point (``data.import``)."""
    hub, token, job_id = (os.environ.get(k) for k in ("BADAMOA_HUB_URL", "BADAMOA_NODE_TOKEN", "BADAMOA_JOB_ID"))
    if not (hub and token and job_id):
        raise RuntimeError("이 작업은 Node Agent가 실행해야 합니다.")
    root = Path(params["path"]).expanduser()
    print(f"폴더 확인: {root}", flush=True)
    base = f"/api/agent/jobs/{job_id}/import"
    with httpx.Client(base_url=hub, headers={"Authorization": f"Bearer {token}"}, timeout=120) as client:
        report = Progress(client, base + "/progress")
        began = time.monotonic()
        found = scan(
            root,
            params.get("format") or "auto",
            params.get("unlabelled") or "auto",
            params.get("task") or "",
            report=report,
        )
        total = len(found.entries)
        print(
            f"형식 {found.format} · 클래스 {len(found.classes)}개 · 이미지 {total:,}장"
            f" (찾고 라벨 읽는 데 {time.monotonic() - began:,.0f}초)",
            flush=True,
        )
        waiting = sum(1 for e in found.entries if e.state == "unlabelled")
        if waiting:
            print(f"라벨 파일이 없는 이미지 {waiting:,}장은 '아직 라벨 안 함'으로 등록합니다.")
        if total == 0:
            raise RuntimeError("이미지를 찾지 못했습니다. 경로와 형식을 확인하세요.")

        def post(path: str, body: dict[str, Any]) -> dict[str, Any]:
            resp = client.post(base + path, json=body)
            if resp.status_code != 200:
                raise RuntimeError(f"Hub가 거부했습니다 ({resp.status_code}): {resp.text[:300]}")
            return resp.json()

        # Outlines (segmentation data) are told apart here, so "자동" on the Hub can pick Seg.
        outlines = not found.keypoints and any(len(b) >= 11 and len(b) % 2 == 1 for e in found.entries for b in e.boxes)
        post(
            "/begin",
            {
                "classes": found.classes,
                "total": total,
                "format": found.format,
                "keypoints": found.keypoints,
                "outlines": outlines,
            },
        )
        if found.keypoints:
            print(f"키포인트 {found.keypoints['count']}개 (연결선 {len(found.keypoints['skeleton'])}개)", flush=True)
        # Reading the next batch goes on while the Hub stores the last one.
        outbox: queue.Queue[tuple[list[dict[str, Any]], int] | None] = queue.Queue(maxsize=2)
        failure: list[BaseException] = []
        sent = skipped = 0
        started = last = time.monotonic()

        def send() -> None:
            nonlocal sent, skipped, last
            while (piece := outbox.get()) is not None:
                if failure:
                    continue  # drain: the reader stops on its own
                batch, bad = piece
                try:
                    post("/items", {"items": batch, "skipped": bad})
                except BaseException as e:  # noqa: BLE001 - raised again in the main thread
                    failure.append(e)
                    continue
                sent, skipped = sent + len(batch), skipped + bad
                now = time.monotonic()
                if now - last >= REPORT_EVERY_S or sent + skipped >= total:
                    last = now
                    rate = sent / max(now - started, 1e-6)
                    left = (total - sent - skipped) / rate if rate else 0
                    print(f"등록 {sent:,} / {total:,} · 초당 {rate:,.0f}장 · 남은 시간 약 {left / 60:.0f}분", flush=True)

        sender = threading.Thread(target=send, name="import-send", daemon=True)
        sender.start()
        print(f"이미지 {WORKERS}장씩 동시에 읽습니다.", flush=True)
        for piece in chunks(found, root):
            if failure:
                break
            outbox.put(piece)
        outbox.put(None)
        sender.join()
        if failure:
            raise failure[0]
        seconds = time.monotonic() - started
        print(f"읽기 · 보내기 {seconds:,.0f}초 (초당 {sent / max(seconds, 1e-6):,.0f}장)", flush=True)
        result = post("/finish", {})
    if skipped:
        print(f"읽을 수 없는 파일 {skipped}개는 건너뛰었습니다.")
    print(f"완료: 버전 {result.get('version')} · 항목 {result.get('items'):,}개")
    return {**result, "skipped": skipped, "format": found.format}
