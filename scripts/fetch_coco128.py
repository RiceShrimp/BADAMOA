"""Download COCO128 (128 images from COCO train2017 with YOLO labels, about 7 MB) for light real-data checks.

    uv run python scripts/fetch_coco128.py [target-dir]

The archive comes from the Ultralytics assets release. It has no data.yaml, so
this script writes one with the 80 COCO class names. Used by the opt-in test
``tests/test_real_data.py`` and for trying the dataset screens by hand.
"""

from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO_NAMES = (
    "person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, "
    "parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, "
    "handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, "
    "surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, "
    "broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, "
    "laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, "
    "scissors, teddy bear, hair drier, toothbrush"
)


def coco_names() -> list[str]:
    return [name.strip() for name in COCO_NAMES.split(",")]


def fetch(target: Path) -> Path:
    root = target / "coco128"
    if (root / "images").is_dir():
        print(f"이미 있습니다: {root}")
    else:
        target.mkdir(parents=True, exist_ok=True)
        archive = target / "coco128.zip"
        print(f"내려받는 중: {URL}")
        urllib.request.urlretrieve(URL, archive)  # noqa: S310 - fixed https URL
        with zipfile.ZipFile(archive) as zf:
            base = target.resolve()
            for member in zf.namelist():  # refuse paths that would escape the target directory
                if base not in (base / member).resolve().parents and (base / member).resolve() != base:
                    raise RuntimeError(f"unexpected path in archive: {member}")
            zf.extractall(target)
        archive.unlink()
    names = coco_names()
    assert len(names) == 80, len(names)
    lines = "\n".join(f"  {i}: {name}" for i, name in enumerate(names))
    (root / "data.yaml").write_text(f"path: .\ntrain: images/train2017\nnames:\n{lines}\n", encoding="utf-8")
    images = len(list((root / "images").rglob("*.jpg")))
    print(f"준비됨: {root} (이미지 {images}장, 클래스 {len(names)}개)")
    return root


if __name__ == "__main__":
    fetch(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".badamoa/data"))
