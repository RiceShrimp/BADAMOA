"""What a new Hub starts with: a "샘플" project holding COCO8 for detection, segmentation and keypoints, and a
detection version whose labels were made wrong on purpose, so 검수 has something to show from the first day.

The images ship with BADAMOA (``badamoa/samples``: Ultralytics' COCO8, 8 images each). When the first admin is
made, they are copied into the Hub's folder (``<home>/samples``) and imported by this PC's own node like any
folder; a Hub without its own node starts without them. The project can be deleted like any other.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import catalog
from .config import HubSettings
from .models import Job, Node, User

log = logging.getLogger("badamoa.hub")

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
PROJECT = "샘플 (COCO8)"
NOTE = "MOA 샘플: Ultralytics COCO8 (COCO train2017의 처음 8장)"
# (folder, dataset, task)
SETS = [
    ("coco8", "COCO8 검출", "Det"),
    ("coco8-seg", "COCO8 분할", "Seg"),
    ("coco8-pose", "COCO8 키포인트", "Kpt"),
]
REVIEW = ("coco8-review", "COCO8 검수 예시", "Det")

# The labels made wrong in the 검수 example, one kind of mistake per image (``plant``).
PLANTED = {
    "train/000000000009": "duplicate",  # the same object boxed twice (박스 오류)
    "train/000000000025": "tiny",  # a box too small to be anything (박스 오류)
    "train/000000000030": "huge",  # a box covering the whole image (박스 오류)
    "train/000000000034": "class_overlap",  # one object with two classes (오라벨)
    "val/000000000036": "unlabelled",  # nobody labelled it (미라벨)
}


def plant(folder: Path) -> None:
    """Make the labels of a COCO8 copy wrong on purpose: what 검수's rules then find."""
    for name, kind in PLANTED.items():
        split, stem = name.split("/")
        path = folder / "labels" / split / f"{stem}.txt"
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        first = lines[0].split()
        if kind == "duplicate":
            moved = [first[0], f"{float(first[1]) + 0.004:.6f}", *first[2:5]]
            lines.append(" ".join(moved))
        elif kind == "tiny":
            lines.append(f"{first[0]} 0.5 0.5 0.002 0.05")
        elif kind == "huge":
            lines.append(f"{first[0]} 0.5 0.5 1.0 1.0")
        elif kind == "class_overlap":
            other = "0" if first[0] != "0" else "1"
            lines.append(" ".join([other, *first[1:5]]))
        elif kind == "unlabelled":
            path.unlink()
            continue
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy(target: Path) -> None:
    """The samples in the Hub's folder (the node reads them there), the 검수 example made from the detection set."""
    for folder, *_ in SETS:
        if not (target / folder).is_dir():
            shutil.copytree(SAMPLES / folder, target / folder)
    review = target / REVIEW[0]
    if not review.is_dir():
        shutil.copytree(SAMPLES / "coco8", review)
        plant(review)


def seed(db: Session, user: User, settings: HubSettings) -> list[Job]:
    """The "샘플" project and its imports, queued on this PC's node (none when the Hub runs without one)."""
    local = db.scalar(select(Node).where(Node.is_local.is_(True)))
    if local is None or not (SAMPLES / "coco8").is_dir():
        return []
    target = Path(settings.home) / "samples"
    target.mkdir(parents=True, exist_ok=True)
    _copy(target)
    about = "처음 설치 때 넣은 샘플입니다. 필요 없으면 지워도 됩니다."
    project = catalog.create_project(db, user, PROJECT, description=about)
    jobs = []
    for folder, name, task in [*SETS, REVIEW]:
        review = folder == REVIEW[0]
        jobs.append(
            catalog.start_import(
                db,
                user,
                project_id=project.id,
                node_id=local.id,
                path=str(target / folder),
                dataset=name,
                version="v1",
                task=task,
                # the example's image without labels is one nobody labelled yet (미라벨), not a background
                unlabelled="unlabelled" if review else "auto",
                note=NOTE + (" — 검수 예시: 일부러 틀린 라벨을 넣음" if review else ""),
                via="setup",
            )
        )
        if len(jobs) > 1:  # one after another: the sets share images, which a project keeps once
            jobs[-1].requirements = {**(jobs[-1].requirements or {}), "after": jobs[-2].id}
    log.info("samples: %s, %d imports on %s", PROJECT, len(jobs), local.name)
    return jobs


def safely(db: Session, user: User, settings: HubSettings) -> dict[str, Any]:
    """``seed``, never in the way of the first admin's account: what went wrong is logged."""
    try:
        with db.begin_nested():
            jobs = seed(db, user, settings)
        return {"project": PROJECT if jobs else "", "imports": len(jobs)}
    except Exception:  # noqa: BLE001 - a Hub without samples still works
        log.exception("samples")
        return {"project": "", "imports": 0}
