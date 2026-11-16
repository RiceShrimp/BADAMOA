"""Downloading a release: the node that holds its images writes a zip in a common format, the Hub keeps it a while.

    YOLO       images/<split>/…, labels/<split>/….txt, data.yaml (outlines for segmentation)
    COCO       images/<split>/…, annotations/instances_<split>.json (pixel boxes, polygons)
    폴더       <split>/<class>/… (classification)

With ``images=False`` the zip holds the labels only, and a list matching them to the
registered files. The release pins the labels, so the zip has the labels the release
recorded, not whatever sits next to the images today. A zip is kept ``KEEP_DAYS`` days;
after that the regular storage cleanup removes it and the export can simply be run again.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import catalog, services, training
from .catalog_models import Dataset, Project
from .config import HubSettings
from .db import iso, utcnow
from .models import Job, Node, User
from .services import ServiceError

FORMATS = {"detect": ("yolo", "coco"), "segment": ("yolo", "coco"), "classify": ("imagefolder",)}
FORMAT_LABEL = {"yolo": "YOLO", "coco": "COCO", "imagefolder": "클래스별 폴더"}
KEEP_DAYS = 7
MAX_EXPORT_BYTES = 64 * 1024**3
LISTED = 10


def start_export(db: Session, user: User, release_id: str, fmt: str, images: bool = True, via: str = "api") -> Job:
    release = catalog.get_release(db, release_id)
    if release.status != "ready":
        raise ServiceError(409, "아직 등록 중인 릴리스입니다. 끝난 뒤에 내려받으세요.")
    project = db.get(Project, db.get(Dataset, release.dataset_id).project_id)
    formats = FORMATS.get(project.task, ("yolo",))
    if fmt not in formats:
        offered = ", ".join(FORMAT_LABEL[f] for f in formats)
        raise ServiceError(400, f"이 프로젝트의 릴리스는 {offered} 형식으로 받을 수 있습니다.")
    node = training.data_node(db, release.id)
    params = {"release_id": release.id, "format": fmt, "images": bool(images), "task": project.task}
    job = services.create_job(db, user, "data.export", params, node.id, via=via)
    services.audit(db, user.id, via, "release.export", release.id, format=fmt, images=bool(images))
    return job


def _export_job(db: Session, node: Node, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None or job.node_id != node.id or job.type != "data.export" or job.status != "running":
        raise ServiceError(404, "이 노드에서 실행 중인 내보내기 작업이 아닙니다.")
    return job


def manifest(db: Session, node: Node, job_id: str, after: int = 0, limit: int = 2000) -> dict[str, Any]:
    job = _export_job(db, node, job_id)
    return training.release_manifest(db, node, catalog.get_release(db, job.params["release_id"]), after, limit)


def check_upload(db: Session, node: Node, job_id: str) -> None:
    _export_job(db, node, job_id)


def _view(settings: HubSettings, job: Job) -> dict[str, Any]:
    result = job.result or {}
    expires = job.finished_at + timedelta(days=KEEP_DAYS) if job.finished_at else None
    ready = job.status == "succeeded" and bool(result.get("blob")) and expires is not None and expires > utcnow()
    return {
        "job_id": job.id,
        "status": job.status,
        "format": job.params.get("format"),
        "images": bool(job.params.get("images", True)),
        "created_at": iso(job.created_at),
        "finished_at": iso(job.finished_at),
        "expires_at": iso(expires) if ready else None,
        "file": result.get("file"),
        "bytes": result.get("bytes"),
        "counts": result.get("counts"),
        "skipped": result.get("skipped"),
        "error": job.error,
        "download_url": catalog.media_url(settings, "export", job.id) if ready else None,
    }


def list_exports(db: Session, settings: HubSettings, release_id: str) -> list[dict[str, Any]]:
    catalog.get_release(db, release_id)
    jobs = db.scalars(select(Job).where(Job.type == "data.export").order_by(Job.created_at.desc()).limit(500))
    mine = [j for j in jobs if (j.params or {}).get("release_id") == release_id][:LISTED]
    return [_view(settings, j) for j in mine]


def download(db: Session, job_id: str) -> tuple[str, str]:
    """The blob and file name of a finished export that is still kept."""
    job = db.get(Job, job_id)
    if job is None or job.type != "data.export":
        raise ServiceError(404, "내보내기 작업을 찾을 수 없습니다.")
    result = job.result or {}
    expired = job.finished_at is None or job.finished_at + timedelta(days=KEEP_DAYS) <= utcnow()
    if job.status != "succeeded" or not result.get("blob") or expired:
        raise ServiceError(410, "보관 기간이 지났거나 끝나지 않은 내보내기입니다. 다시 내보내세요.")
    return result["blob"], str(result.get("file") or "release.zip")


def kept_blobs(db: Session) -> set[str]:
    """Zips still on offer (for the storage cleanup)."""
    cutoff = utcnow() - timedelta(days=KEEP_DAYS)
    rows = db.scalars(
        select(Job.result).where(Job.type == "data.export", Job.status == "succeeded", Job.finished_at >= cutoff)
    )
    return {r["blob"] for r in rows if r and r.get("blob")}
