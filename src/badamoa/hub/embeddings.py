"""Image embeddings for 이미지 탐색: one vector per image and model, made on a node that has the images.

An image is known by its content (sha256), so a vector made once serves every version, dataset and project the
same picture is in: a new version of a dataset needs vectors only for the images it added. Vectors are made by a
node with the 임베딩용 role (``registry.EMBEDS``) — usually the server a training received the version on, so
the images are not copied a second time — and kept on the Hub, in part files (``EmbeddingPart``) with an index of
which image is where (``ImageEmbedding``).
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import exists, func, insert, select
from sqlalchemy.orm import Session

from badamoa.core.storage import Storage

from . import catalog, registry, services, training
from .catalog_models import EmbeddingPart, ImageEmbedding, Item, Release, ReleaseItem
from .models import Job, Node, User
from .services import ServiceError

# The models a vector is made with: the script under ``badamoa/embedders`` that makes them, the checkpoint it loads,
# how many numbers a vector has, and the environment it runs in on the node.
MODELS: dict[str, dict[str, Any]] = {
    "siglip2-base": {
        "title": "SigLIP2 base (patch16, 256px)",
        "script": "siglip2.py",
        "checkpoint": "google/siglip2-base-patch16-256",
        "dim": 768,
        "environment": {
            "kind": "auto",
            # torch named itself (transformers does not bring it); ``torch`` picks the build for the node's GPU
            "requirements": ["torch", "transformers>=5,<6", "pillow", "numpy"],
            "python": "3.12",
            "torch": True,
        },
    },
    # tests only: a few numbers from the pixels, with the node's own Python
    "dummy": {
        "title": "test",
        "script": "dummy.py",
        "checkpoint": "",
        "dim": 8,
        "environment": {"kind": "auto", "requirements": [], "python": "3.12", "torch": False},
        "hidden": True,
    },
}
DEFAULT_MODEL = "siglip2-base"
SINCE = (0, 108, 0)  # node versions from here on make embeddings (``model.embed``)
MAX_PART = 64 * 1024 * 1024  # a part file a node sends at once


def model_of(name: str) -> dict[str, Any]:
    found = MODELS.get(name or DEFAULT_MODEL)
    if found is None:
        shown = ", ".join(k for k, v in MODELS.items() if not v.get("hidden"))
        raise ServiceError(400, f"임베딩 모델은 {shown} 중 하나입니다.")
    return found


def _images(release_id: str) -> Any:
    """The distinct images (by content) of a version."""
    return (
        select(Item.sha256)
        .join(ReleaseItem, ReleaseItem.item_id == Item.id)
        .where(ReleaseItem.release_id == release_id)
        .distinct()
    )


def _done(model: str) -> Any:
    return exists().where(ImageEmbedding.model == model, ImageEmbedding.sha256 == Item.sha256)


def coverage(db: Session, release_id: str, model: str = "") -> dict[str, Any]:
    """How many of a version's images have a vector by ``model`` (the default one) already (made for any version or
    project), and how many a job would still make."""
    model = model or DEFAULT_MODEL
    spec = model_of(model)
    total = db.scalar(select(func.count()).select_from(_images(release_id).subquery())) or 0
    done = (
        db.scalar(
            select(func.count()).select_from(
                select(Item.sha256)
                .join(ReleaseItem, ReleaseItem.item_id == Item.id)
                .where(ReleaseItem.release_id == release_id, _done(model))
                .distinct()
                .subquery()
            )
        )
        or 0
    )
    going = db.scalar(
        select(Job.id).where(Job.type == "model.embed", Job.status.in_(("queued", "running"))).where(
            Job.params["release_id"].as_string() == release_id, Job.params["model"].as_string() == model
        )
    )
    return {
        "model": model,
        "title": spec["title"],
        "total": total,
        "done": done,
        "todo": max(0, total - done),
        "going": going or None,
    }


def node_problem(node: Node) -> str:
    """Why vectors cannot be made on ``node`` ("" when they can)."""
    if not node.enabled:
        return f"{node.name}은(는) 작업을 받지 않도록 되어 있습니다."
    if not registry.has_role(node, registry.EMBEDS):
        return f"{node.name}은(는) 임베딩용 서버가 아닙니다. 노드 설정에서 역할 '임베딩용'을 켜세요."
    if not training._version(node.agent_version or "") or training._version(node.agent_version or "") < SINCE:
        need = ".".join(str(n) for n in SINCE)
        return f"{node.name}의 MOA를 {need} 이상으로 업데이트해야 임베딩을 만들 수 있습니다."
    return ""


def start(
    db: Session,
    user: User,
    release: Release,
    model: str = "",
    *,
    node_id: str | None = None,
    nodes: list[str] | None = None,
    after: str | None = None,
    gpus: int | None = None,
    run_id: str = "",
    via: str = "api",
) -> Job | None:
    """Make the vectors a version's images still lack, on ``node_id`` (or any of ``nodes``) once ``after`` is done
    (the copy a training's server receives). None when every image has one already (nothing to do)."""
    model = model or DEFAULT_MODEL
    spec = model_of(model)
    places = [db.get(Node, n) for n in ([node_id] if node_id else nodes or [])]
    if not places or any(p is None for p in places):
        raise ServiceError(400, "임베딩을 만들 서버를 찾을 수 없습니다.")
    for place in places:
        why = node_problem(place)
        if why:
            raise ServiceError(400, why)
    known = coverage(db, release.id, model)
    if known["todo"] == 0 or known["going"]:
        return None  # every image has its vector already, or a job is making them now
    count = (1 if any((p.info or {}).get("gpus") for p in places) else 0) if gpus is None else max(0, min(gpus, 1))
    job = services.create_job(
        db,
        user,
        "model.embed",
        {
            "release_id": release.id,
            "model": model,
            "spec": {k: spec[k] for k in ("script", "checkpoint", "dim")},
            "environment": spec["environment"],
            **({"run_id": run_id} if run_id else {}),
        },
        node_id,
        via=via,
        gpus=count,
        nodes=None if node_id else nodes,
        after=after,
        role=registry.EMBEDS,
    )
    services.audit(db, user.id, via, "embedding.start", release.id, model=model, images=known["todo"], job=job.id)
    return job


def _job(db: Session, node: Node, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None or job.type != "model.embed" or job.node_id != node.id or job.status != "running":
        raise ServiceError(404, "이 노드에서 도는 임베딩 작업이 아닙니다.")
    return job


def manifest(db: Session, node: Node, job_id: str, after: int = 0, limit: int = 2000) -> dict[str, Any]:
    """The version's images still without a vector by the job's model, a page at a time (as a training's)."""
    job = _job(db, node, job_id)
    params = job.params or {}
    release = catalog.get_release(db, params["release_id"])
    model = params.get("model") or DEFAULT_MODEL
    todo = (
        select(ReleaseItem.item_id)
        .join(Item, Item.id == ReleaseItem.item_id)
        .where(ReleaseItem.release_id == release.id, ~_done(model))
    )
    return training.release_manifest(db, node, release, after, limit, only=todo)


def ingest(db: Session, storage: Storage, node: Node, job_id: str, data: bytes) -> dict[str, int]:
    """A part file the job sent: kept in the Hub's storage, and each image in it found by content from now on."""
    job = _job(db, node, job_id)
    model = (job.params or {}).get("model") or DEFAULT_MODEL
    dim = int(model_of(model)["dim"])
    head, _, body = data.partition(b"\n")
    try:
        header = json.loads(head)
        shas = [str(s) for s in header["shas"]]
    except (ValueError, KeyError, TypeError) as e:
        raise ServiceError(400, "임베딩 파일의 머리말을 읽지 못했습니다.") from e
    if header.get("model") != model or int(header.get("dim") or 0) != dim or len(body) != len(shas) * dim * 2:
        raise ServiceError(400, f"임베딩 파일이 {model}({dim}차원) 벡터 {len(shas)}개와 맞지 않습니다.")
    if not shas:
        return {"stored": 0}
    blob = storage.put_blob(data)
    part = EmbeddingPart(
        model=model, dim=dim, count=len(shas), blob=blob, job_id=job.id, release_id=(job.params or {}).get("release_id")
    )
    db.add(part)
    db.flush()
    rows = [{"model": model, "sha256": sha, "part_id": part.id, "row": n} for n, sha in enumerate(shas)]
    before = db.scalar(select(func.count()).select_from(ImageEmbedding).where(ImageEmbedding.model == model)) or 0
    for start_at in range(0, len(rows), 5000):  # an image made twice (two jobs at once) keeps its first vector
        db.execute(insert(ImageEmbedding).prefix_with("OR IGNORE"), rows[start_at : start_at + 5000])
    after = db.scalar(select(func.count()).select_from(ImageEmbedding).where(ImageEmbedding.model == model)) or 0
    return {"stored": after - before}


def models_view() -> list[dict[str, Any]]:
    return [
        {"id": k, "title": v["title"], "dim": v["dim"], "default": k == DEFAULT_MODEL}
        for k, v in MODELS.items()
        if not v.get("hidden")
    ]
