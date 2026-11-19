"""A small dataset the node draws itself for a trial run: coloured shapes and their labels.

No network and nothing to download, so it works on a closed network too. Each class has its own
colour. Detection (and any task written as YOLO boxes) gets one to three boxes per image,
segmentation the same boxes with their outlines, classification one class per image. The items
have the shape of the Hub's training manifest, so the node writes them with the same exporter
as a real version.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

NAMES = ["red", "blue", "green", "orange", "purple", "teal", "brown", "pink"]
COLORS = [
    (214, 69, 65),
    (42, 120, 214),
    (27, 175, 122),
    (236, 146, 37),
    (142, 82, 196),
    (22, 160, 170),
    (130, 90, 60),
    (220, 110, 170),
]
LOCATION = "sample"  # the location id the items name: the sample's own folder


def kind_of(data_format: str) -> str:
    """What labels the format takes: classify (one class per image), segment (outlines) or detect (boxes)."""
    return {"imagefolder": "classify", "yolo-seg": "segment"}.get(data_format, "detect")


def class_count(kind: str, minimum: int = 0, maximum: int = 0) -> int:
    """Three classes, or what the template says it needs."""
    count = max(3, minimum or 0)
    return min(count, maximum) if maximum else count


def _color(index: int) -> tuple[int, int, int]:
    if index < len(COLORS):
        return COLORS[index]
    rng = random.Random(index)
    return (rng.randint(30, 230), rng.randint(30, 230), rng.randint(30, 230))


def _boxes(rng: random.Random, classes: int) -> list[list[float]]:
    boxes = []
    for _ in range(rng.randint(1, 3)):
        w, h = round(rng.uniform(0.2, 0.4), 4), round(rng.uniform(0.2, 0.4), 4)
        cx, cy = round(rng.uniform(w / 2, 1 - w / 2), 4), round(rng.uniform(h / 2, 1 - h / 2), 4)
        boxes.append([rng.randrange(classes), cx, cy, w, h])
    return boxes


def _outline(box: list[float]) -> list[float]:
    """The box's corners as an outline (x, y pairs, 0–1), as a segmentation label stores it after the box."""
    _, cx, cy, w, h = box
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    return [round(v, 4) for v in (x0, y0, x1, y0, x1, y1, x0, y1)]


def make(
    root: Path,
    kind: str,
    classes: int = 3,
    train: int = 8,
    val: int = 4,
    size: int = 160,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Draw the images under ``root`` and return (manifest items, classes, locations) for the exporter."""
    rng = random.Random(seed)
    names = [NAMES[i] if i < len(NAMES) else f"class{i}" for i in range(classes)]
    if kind == "classify":  # every class in both splits
        train, val = max(train, classes * 2), max(val, classes)
    items: list[dict[str, Any]] = []
    for split, count in (("train", train), ("val", val)):
        for i in range(count):
            image = Image.new("RGB", (size, size), (rng.randint(40, 80), rng.randint(50, 90), rng.randint(60, 100)))
            draw = ImageDraw.Draw(image)
            if kind == "classify":
                label = i % classes
                draw.rectangle([size * 0.15, size * 0.15, size * 0.85, size * 0.85], fill=_color(label))
                boxes: list[list[float]] = [[label, 0.5, 0.5, 1.0, 1.0]]
            else:
                boxes = _boxes(rng, classes)
                for cls, cx, cy, w, h in boxes:
                    x0, y0 = (cx - w / 2) * size, (cy - h / 2) * size
                    draw.rectangle([x0, y0, x0 + w * size, y0 + h * size], fill=_color(int(cls)))
                if kind == "segment":
                    boxes = [[*box, *_outline(box)] for box in boxes]
            rel = f"{split}/{split}_{i:03d}.png"
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            data = path.read_bytes()
            items.append(
                {
                    "location_id": LOCATION,
                    "rel_path": rel,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                    "width": size,
                    "height": size,
                    "state": "labelled",
                    "split": split,
                    "boxes": boxes,
                    "labelled_class_ids": list(range(classes)),
                }
            )
    return items, [{"id": i, "name": name} for i, name in enumerate(names)], {LOCATION: str(root)}
