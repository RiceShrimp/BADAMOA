"""A screen's list of images as a file: what the version screen, 검수 or 이미지 탐색 shows under its conditions, as
CSV (a row per image: where it is and what it holds) or TXT (the images' paths on their server, one per line).

The version screen and 검수 ask for their list with the same conditions they page through (``catalog.list_items``),
so the file holds every image they would show, in their order, not only the page on screen. 이미지 탐색's file is
its search again with the same words or image, version and lowest score.
"""

from __future__ import annotations

import csv
import os
import re
import tempfile
from collections.abc import Iterable, Iterator
from types import SimpleNamespace
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import catalog
from .catalog_models import Dataset, Item, LabelSpace, Release, ReleaseItem, RevisionClass, StorageLocation
from .config import HubSettings
from .models import Node
from .services import ServiceError

FORMATS = ("csv", "txt")
COLUMNS = [
    "dataset", "version", "split", "file_name", "path", "server", "state", "boxes", "classes",
    "width", "height", "item_id", "sha256",
]  # fmt: skip
SEARCH_COLUMNS = ["rank", "score", *COLUMNS]
BATCH = 2000  # images read at once
CHUNK = 900  # ids asked about at once (SQLite's limit on bound parameters is 999)
_DRIVE = re.compile(r"^[A-Za-z]:")


def check_format(fmt: str) -> str:
    if fmt not in FORMATS:
        raise ServiceError(400, "목록 형식은 csv 또는 txt입니다.")
    return fmt


class _Places:
    """Where each storage location's files are: its folder on its server, and the server's name."""

    def __init__(self, db: Session) -> None:
        names = dict(db.execute(select(Node.id, Node.name)).all())
        self.where = {loc.id: (loc.root, names.get(loc.node_id, "")) for loc in db.scalars(select(StorageLocation))}

    def of(self, location_id: str | None, rel_path: str) -> tuple[str, str]:
        """The file's full path on its server (its path in the folder when the folder is not known), and the server."""
        root, server = self.where.get(location_id or "", ("", ""))
        if not root:
            return rel_path, server
        if _DRIVE.match(root) or ("\\" in root and "/" not in root):  # a Windows folder
            return root.rstrip("\\/") + "\\" + rel_path.replace("/", "\\"), server
        return root.rstrip("/") + "/" + rel_path, server


def _class_names(db: Session, release: Release) -> dict[int, str]:
    space = db.get(LabelSpace, release.label_space_id) if release.label_space_id else None
    return {int(c["id"]): str(c.get("name", c["id"])) for c in (space.classes if space else []) or []}


def _classes_of(db: Session, revision_ids: list[int], names: dict[int, str]) -> dict[int, str]:
    """Each revision's classes and how many of each, as "person 3; car 1"."""
    held: dict[int, list[tuple[int, int]]] = {}
    for start in range(0, len(revision_ids), CHUNK):
        part = revision_ids[start : start + CHUNK]
        rows = db.execute(
            select(RevisionClass.revision_id, RevisionClass.class_id, RevisionClass.count).where(
                RevisionClass.revision_id.in_(part)
            )
        )
        for revision_id, class_id, count in rows:
            held.setdefault(revision_id, []).append((class_id, count))
    return {rid: "; ".join(f"{names.get(cid, cid)} {n}" for cid, n in sorted(pairs)) for rid, pairs in held.items()}


def _row(place: _Places, item: Any, member: Any, dataset: str, version: str, classes: str) -> dict[str, Any]:
    path, server = place.of(item.location_id, item.rel_path or item.file_name)
    return {
        "dataset": dataset,
        "version": version,
        "split": member.split,
        "file_name": item.file_name,
        "path": path,
        "server": server,
        "state": member.state,
        "boxes": member.box_count,
        "classes": classes,
        "width": item.width,
        "height": item.height,
        "item_id": item.id,
        "sha256": item.sha256,
    }


def release_rows(db: Session, settings: HubSettings, release_id: str, filters: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Every image the version screen or 검수 lists under ``filters`` (``catalog.list_items``'), in its order."""
    release = catalog.get_release(db, release_id)
    dataset = db.get(Dataset, release.dataset_id)
    names = _class_names(db, release)
    place = _Places(db)
    query = catalog.list_items(db, settings, release_id, **filters, page=False)["query"]
    columns = query.with_only_columns(
        Item.id, Item.file_name, Item.rel_path, Item.location_id, Item.width, Item.height, Item.sha256,
        ReleaseItem.split, ReleaseItem.state, ReleaseItem.box_count, ReleaseItem.revision_id,
        maintain_column_froms=True,
    )  # fmt: skip
    for part in db.execute(columns.execution_options(yield_per=BATCH)).partitions(BATCH):  # not all held at once
        classes = _classes_of(db, [r.revision_id for r in part], names)
        for r in part:
            yield _row(place, r, r, dataset.name if dataset else "", release.version, classes.get(r.revision_id, ""))


def search_rows(db: Session, found: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """이미지 탐색's results (``embeddings.search``'s items), the nearest first, each in the version it shows."""
    place = _Places(db)
    ids = [f["id"] for f in found]
    items = {item.id: item for start in range(0, len(ids), CHUNK) for item in db.scalars(
        select(Item).where(Item.id.in_(ids[start : start + CHUNK]))
    )}  # fmt: skip
    pairs = [(f["release"]["id"], f["id"]) for f in found if f.get("release")]
    members: dict[tuple[str, int], ReleaseItem] = {}
    for release_id in {rid for rid, _ in pairs}:
        wanted = [item_id for rid, item_id in pairs if rid == release_id]
        for start in range(0, len(wanted), CHUNK):
            for member in db.scalars(
                select(ReleaseItem).where(
                    ReleaseItem.release_id == release_id, ReleaseItem.item_id.in_(wanted[start : start + CHUNK])
                )
            ):
                members[(release_id, member.item_id)] = member
    releases = {r.id: r for r in db.scalars(select(Release).where(Release.id.in_({rid for rid, _ in pairs})))}
    names = {rid: _class_names(db, release) for rid, release in releases.items()}
    revision_of = {key: m.revision_id for key, m in members.items()}
    classes: dict[str, dict[int, str]] = {}
    for rid in releases:
        mine = [revision_of[key] for key in revision_of if key[0] == rid]
        classes[rid] = _classes_of(db, mine, names[rid])
    for rank, f in enumerate(found, start=1):
        item = items.get(f["id"])
        if item is None:
            continue
        where = f.get("release") or {}
        member = members.get((where.get("id", ""), item.id))
        blank = SimpleNamespace(split="", state="", box_count=0)
        row = _row(
            place,
            item,
            member or blank,
            where.get("dataset", ""),
            where.get("version", ""),
            classes.get(where.get("id", ""), {}).get(member.revision_id, "") if member else "",
        )
        yield {"rank": rank, "score": f.get("score"), **row}


def write(rows: Iterable[dict[str, Any]], fmt: str, columns: list[str]) -> tuple[str, int]:
    """The list in a file of its own (removed once sent): CSV with every column (Excel reads its Korean: UTF-8 with
    a BOM), or TXT with the paths only. Returns its path and how many images."""
    handle, path = tempfile.mkstemp(prefix="badamoa-list-", suffix=f".{fmt}")
    count = 0
    try:
        with os.fdopen(handle, "w", encoding="utf-8-sig" if fmt == "csv" else "utf-8", newline="") as out:
            writer = csv.DictWriter(out, columns, extrasaction="ignore") if fmt == "csv" else None
            if writer is not None:
                writer.writeheader()
            for row in rows:
                if writer is not None:
                    writer.writerow(row)
                else:
                    out.write(f"{row['path']}\n")
                count += 1
    except BaseException:
        os.unlink(path)
        raise
    return path, count


def file_name(*parts: str, fmt: str) -> str:
    """A name for the file: the dataset, version and what was asked, without characters a disk refuses."""
    text = "-".join(p for p in parts if p)
    return re.sub(r'[\\/:*?"<>|\s]+', "_", text).strip("_")[:120] + f".{fmt}"
