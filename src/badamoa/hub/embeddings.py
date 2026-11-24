"""Image embeddings for 이미지 탐색: one vector per image and model, made on a node that has the images.

An image is known by its content (sha256), so a vector made once serves every version, dataset and project the
same picture is in: a new version of a dataset needs vectors only for the images it added. Vectors are made by a
node with the 임베딩용 role (``registry.EMBEDS``) — usually the server a training received the version on, so
the images are not copied a second time — and kept in the vector DB (``vector_store``, LanceDB, a folder an admin
may move), with which images have one by which model in ``ImageEmbedding`` (how far a version is, at a glance).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from sqlalchemy import exists, func, insert, select
from sqlalchemy.orm import Session

from badamoa.core.storage import Storage

from . import catalog, registry, services, training, vector_store
from .catalog_models import Dataset, EmbeddingPart, ImageEmbedding, Item, Project, Release, ReleaseItem
from .config import HubSettings
from .models import HubSetting, Job, Node, User
from .services import ServiceError

# The models a vector is made with: the script under ``badamoa/embedders`` that makes them, the checkpoint it loads,
# how many numbers a vector has, and the environment it runs in on the node.
MODELS: dict[str, dict[str, Any]] = {
    "siglip2-base": {
        "title": "SigLIP2 base (patch16, 256px)",
        "script": "siglip2.py",
        "checkpoint": "google/siglip2-base-patch16-256",
        "dim": 768,
        "text": True,  # finds images by words too: its text encoder runs on the Hub (``text_vector_on``)
        "environment": {
            "kind": "auto",
            # torch named itself (transformers does not bring it); ``torch`` picks the build for the node's GPU.
            # onnx · onnxruntime: the text encoder the Hub reads queries with, exported (and made smaller) once
            "requirements": ["torch", "transformers>=5,<6", "pillow", "numpy", "onnx", "onnxscript", "onnxruntime"],
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
    text = bool(spec.get("text")) and not text_state(db, model).get("ready")  # 글 검색's encoder still to come
    if (known["todo"] == 0 and not text) or known["going"]:
        return None  # every image has its vector already (and the Hub its text encoder), or a job is on it now
    count = (1 if any((p.info or {}).get("gpus") for p in places) else 0) if gpus is None else max(0, min(gpus, 1))
    job = services.create_job(
        db,
        user,
        "model.embed",
        {
            "release_id": release.id,
            "model": model,
            "images": known["todo"],  # the images it makes (its record says so)
            "spec": {
                **{k: spec[k] for k in ("script", "checkpoint", "dim")},
                # the first job also sends the text encoder the Hub reads 글 검색 queries with
                "export_text": text,
            },
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


def ingest(
    db: Session, settings: HubSettings, storage: Storage, node: Node, job_id: str, data: bytes
) -> dict[str, int]:
    """A part file the job sent: its vectors into the vector DB, a row for each project's image of that picture, and
    each picture found by content from now on."""
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
    import numpy as np  # noqa: PLC0415

    vectors = np.frombuffer(body, dtype="<f2").reshape(len(shas), dim)
    _keep(db, settings, model, dim, shas, vectors)
    part = EmbeddingPart(
        model=model, dim=dim, count=len(shas), blob="", job_id=job.id, release_id=(job.params or {}).get("release_id")
    )
    db.add(part)
    db.flush()
    rows = [{"model": model, "sha256": sha, "part_id": part.id, "row": n} for n, sha in enumerate(shas)]
    before = db.scalar(select(func.count()).select_from(ImageEmbedding).where(ImageEmbedding.model == model)) or 0
    for start_at in range(0, len(rows), 5000):  # an image made twice (two jobs at once) keeps its first vector
        db.execute(insert(ImageEmbedding).prefix_with("OR IGNORE"), rows[start_at : start_at + 5000])
    after = db.scalar(select(func.count()).select_from(ImageEmbedding).where(ImageEmbedding.model == model)) or 0
    return {"stored": after - before}


def _keep(db: Session, settings: HubSettings, model: str, dim: int, shas: list[str], vectors: Any) -> int:
    """Into the vector DB: a row for every image (of any project) that is one of these pictures."""
    at = {sha: n for n, sha in enumerate(shas)}
    rows: list[tuple[str, int, str]] = []
    picked: list[int] = []
    for start_at in range(0, len(shas), 5000):
        chunk = shas[start_at : start_at + 5000]
        for item_id, project_id, sha in db.execute(
            select(Item.id, Item.project_id, Item.sha256).where(Item.sha256.in_(chunk))
        ):
            rows.append((project_id, item_id, sha))
            picked.append(at[sha])
    return vector_store.add(db, settings, model, dim, rows, vectors[picked] if picked else vectors[:0])


def sync_project(db: Session, settings: HubSettings, project_id: str, model: str = "") -> int:
    """A project's images whose picture has a vector (made for another project or before the image came) but which
    the vector DB does not list for this project yet: listed now. Also part files kept before the vector DB, once."""
    import numpy as np  # noqa: PLC0415

    model = model or DEFAULT_MODEL
    dim = int(model_of(model)["dim"])
    _from_parts(db, settings, model, dim)
    has_vector = exists().where(ImageEmbedding.model == model, ImageEmbedding.sha256 == Item.sha256)
    wanted = dict(db.execute(select(Item.id, Item.sha256).where(Item.project_id == project_id, has_vector)).all())
    if not wanted:
        return 0
    known = vector_store.items_of(db, settings, model, dim, project_id)
    missing = [(item_id, sha) for item_id, sha in wanted.items() if item_id not in known]
    if not missing:
        return 0
    found = vector_store.vectors_of(db, settings, model, dim, sorted({sha for _, sha in missing}))
    rows = [(project_id, item_id, sha) for item_id, sha in missing if sha in found]
    if rows:
        vector_store.add(db, settings, model, dim, rows, np.stack([found[sha] for _, _, sha in rows]))
    lost = {sha for _, sha in missing if sha not in found}  # the DB has none for it (moved away, removed): made again
    if lost:
        db.execute(ImageEmbedding.__table__.delete().where(ImageEmbedding.model == model, ImageEmbedding.sha256.in_(lost)))
    return len(rows)


def _from_parts(db: Session, settings: HubSettings, model: str, dim: int) -> None:
    """Part files kept in the Hub's storage before the vector DB (0.108.0 · 0.109.0): into it, then let go."""
    import numpy as np  # noqa: PLC0415

    old = db.scalars(select(EmbeddingPart).where(EmbeddingPart.model == model, EmbeddingPart.blob != "")).all()
    if not old:
        return
    storage = catalog.storage_for(str(settings.storage_dir))
    for part in old:
        try:
            _, _, body = storage.get_blob(part.blob).partition(b"\n")
            shas = [row.sha256 for row in db.scalars(
                select(ImageEmbedding).where(ImageEmbedding.part_id == part.id).order_by(ImageEmbedding.row)
            )]  # fmt: skip
            vectors = np.frombuffer(body, dtype="<f2").reshape(part.count, part.dim)
            rows = {r.sha256: r.row for r in db.scalars(select(ImageEmbedding).where(ImageEmbedding.part_id == part.id))}
            _keep(db, settings, model, dim, shas, vectors[[rows[s] for s in shas]])
            storage.delete(storage.blob_key(part.blob))
        except Exception:  # noqa: BLE001 - a part that cannot be read: its images are made again
            db.execute(ImageEmbedding.__table__.delete().where(ImageEmbedding.part_id == part.id))
        part.blob = ""
    db.flush()


def models_view() -> list[dict[str, Any]]:
    return [
        {"id": k, "title": v["title"], "dim": v["dim"], "default": k == DEFAULT_MODEL}
        for k, v in MODELS.items()
        if not v.get("hidden")
    ]


# ---- 글 검색: the text encoder, on the Hub ------------------------------------------------------------------------
# The first embedding job exports the model's text encoder to ONNX (made smaller to 8-bit weights when it stays as
# good) with its tokenizer, and a few phrases' token ids and vectors as made there. The Hub keeps them, checks it
# reads those phrases the same, and from then on turns a query into a vector on its own CPU (ONNX Runtime, no torch).

TEXT_FILES = ("onnx", "tokenizer", "checks")
_sessions: dict[str, Any] = {}  # model → (tokenizer, session, settings) once loaded
_lock = threading.RLock()


def _text_key(model: str) -> str:
    return f"embedding.text.{model}"


def text_state(db: Session, model: str = "") -> dict[str, Any]:
    row = db.get(HubSetting, _text_key(model or DEFAULT_MODEL))
    return dict(row.value) if row is not None else {}


def _text_dir(settings: HubSettings, model: str) -> Path:
    return settings.home / "embeddings" / model


def text_file(
    db: Session, settings: HubSettings, storage: Storage, node: Node, job_id: str, kind: str, blob: str
) -> dict[str, Any]:
    """A file of the text encoder the job sent (``onnx``, ``tokenizer`` or ``checks``); with all three, checked."""
    job = _job(db, node, job_id)
    model = (job.params or {}).get("model") or DEFAULT_MODEL
    if kind not in TEXT_FILES:
        raise ServiceError(400, f"글 검색 파일은 {', '.join(TEXT_FILES)} 중 하나입니다.")
    state = {k: v for k, v in text_state(db, model).items() if k not in ("ready", "problem")}
    state[kind] = blob
    if all(state.get(k) for k in TEXT_FILES):
        problem = _install_text(settings, storage, model, state)
        state.update(ready=not problem, problem=problem)
    row = db.get(HubSetting, _text_key(model))
    if row is None:
        db.add(HubSetting(key=_text_key(model), value=state))
    else:
        row.value = state  # a new dict: a JSON column changed in place is not saved
    db.flush()
    return {"kind": kind, "ready": bool(state.get("ready")), "problem": state.get("problem", "")}


def _install_text(settings: HubSettings, storage: Storage, model: str, state: dict[str, Any]) -> str:
    """Lay the files out where the Hub loads them from, and check it reads the phrases as the node did ("" when it
    does)."""
    folder = _text_dir(settings, model)
    folder.mkdir(parents=True, exist_ok=True)
    with _lock:
        _sessions.pop(model, None)
    (folder / "text.onnx").write_bytes(storage.get_blob(state["onnx"]))
    (folder / "tokenizer.json").write_bytes(storage.get_blob(state["tokenizer"]))
    (folder / "checks.json").write_bytes(storage.get_blob(state["checks"]))
    try:
        checks = json.loads((folder / "checks.json").read_text(encoding="utf-8"))
        for phrase in checks["phrases"]:
            ids = _tokens(settings, model, phrase["text"])
            if ids != phrase["ids"]:
                return f"'{phrase['text']}'을(를) 노드와 다르게 읽습니다 (토큰이 다름)."
            mine = text_vector_on(settings, model, phrase["text"])
            same = sum(a * b for a, b in zip(mine, phrase["vector"], strict=False))
            if same < 0.98:
                return f"'{phrase['text']}'의 벡터가 노드에서 만든 것과 다릅니다 (코사인 {same:.3f})."
    except Exception as e:  # noqa: BLE001 - say why 글 검색 is not ready, keep the rest working
        return f"글 검색 모델을 읽지 못했습니다: {type(e).__name__}: {e}"
    return ""


def _loaded(settings: HubSettings, model: str) -> tuple[Any, Any, dict[str, Any]]:
    with _lock:
        found = _sessions.get(model)
        if found is None:
            import onnxruntime  # noqa: PLC0415  (only when 글 검색 is used)
            from tokenizers import Tokenizer  # noqa: PLC0415

            folder = _text_dir(settings, model)
            meta = json.loads((folder / "checks.json").read_text(encoding="utf-8"))
            tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
            tokenizer.enable_truncation(max_length=int(meta["max_length"]))
            tokenizer.enable_padding(
                length=int(meta["max_length"]),
                pad_id=int(meta["pad_id"]),
                pad_token=str(meta["pad_token"]),
                direction=str(meta.get("padding_side") or "right"),
            )
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 4
            path = str(folder / "text.onnx")
            session = onnxruntime.InferenceSession(path, options, providers=["CPUExecutionProvider"])
            found = (tokenizer, session, meta)
            _sessions[model] = found
        return found


def _tokens(settings: HubSettings, model: str, text: str) -> list[int]:
    tokenizer, _, meta = _loaded(settings, model)
    return list(tokenizer.encode(text.lower() if meta.get("lowercase") else text).ids)


def text_vector_on(settings: HubSettings, model: str, text: str) -> list[float]:
    """The words' vector, made on the Hub's CPU with the text encoder the first embedding job sent."""
    import numpy as np  # noqa: PLC0415

    _, session, _ = _loaded(settings, model)
    ids = np.array([_tokens(settings, model, text)], dtype=np.int64)
    out = session.run(None, {session.get_inputs()[0].name: ids})[0][0]
    norm = float(np.linalg.norm(out)) or 1.0
    return [float(v) / norm for v in out]


# ---- 이미지 탐색: finding a project's images by words or by another image ----------------------------------------


def explore_status(db: Session, settings: HubSettings, project_id: str, model: str = "") -> dict[str, Any]:
    """How much of a project 이미지 탐색 can find: its images, those with a vector, and whether 글 검색 is ready."""
    model = model or DEFAULT_MODEL
    spec = model_of(model)
    catalog.get_project(db, project_id)
    sync_project(db, settings, project_id, model)
    images = db.scalar(select(func.count()).select_from(Item).where(Item.project_id == project_id)) or 0
    has_vector = exists().where(ImageEmbedding.model == model, ImageEmbedding.sha256 == Item.sha256)
    found = db.scalar(select(func.count()).select_from(Item).where(Item.project_id == project_id, has_vector)) or 0
    text = text_state(db, model)
    return {
        "model": model,
        "title": spec["title"],
        "images": images,
        "embedded": found,
        "text": {"offered": bool(spec.get("text")), "ready": bool(text.get("ready")), "problem": text.get("problem") or ""},
    }


def search(
    db: Session,
    settings: HubSettings,
    storage: Storage,
    project_id: str,
    *,
    text: str = "",
    like: int = 0,
    release_id: str = "",
    limit: int = 60,
    model: str = "",
) -> dict[str, Any]:
    """A project's images (or one version's) nearest to the words ``text`` or to the image ``like`` (an item id),
    the nearest first, each with how near (cosine) and a version it is in."""
    import numpy as np  # noqa: PLC0415

    model = model or DEFAULT_MODEL
    dim = int(model_of(model)["dim"])
    catalog.get_project(db, project_id)
    sync_project(db, settings, project_id, model)
    limit = max(1, min(int(limit), 300))
    if text.strip():
        state = text_state(db, model)
        if not state.get("ready"):
            why = state.get("problem") or "임베딩 작업이 한 번 끝나면 글 검색 모델이 Hub에 들어옵니다."
            raise ServiceError(409, f"글 검색을 아직 쓸 수 없습니다. {why}")
        query = np.array(text_vector_on(settings, model, text.strip()), dtype=np.float32)
    elif like:
        query = vector_store.vector_of_item(db, settings, model, dim, like)
        if query is None:
            raise ServiceError(404, "이 이미지는 아직 벡터가 없어 비슷한 이미지를 찾을 수 없습니다.")
    else:
        raise ServiceError(400, "찾을 말(text)이나 이미지(like)를 주세요.")
    within = (
        list(db.scalars(select(ReleaseItem.item_id).where(ReleaseItem.release_id == release_id))) if release_id else None
    )
    hits = vector_store.search(db, settings, model, dim, project_id, query, limit + 1, within)
    chosen = [(item_id, score) for item_id, score in hits if not (like and item_id == like)][:limit]
    searched = db.scalar(
        select(func.count()).select_from(Item).where(
            Item.project_id == project_id,
            exists().where(ImageEmbedding.model == model, ImageEmbedding.sha256 == Item.sha256),
            *([Item.id.in_(select(ReleaseItem.item_id).where(ReleaseItem.release_id == release_id))] if release_id else []),
        )
    ) or 0
    return {"model": model, "searched": searched, "items": _found(db, settings, chosen, release_id)}



def _found(db: Session, settings: HubSettings, chosen: list[tuple[int, float]], release_id: str) -> list[dict[str, Any]]:
    """The items found, as the grid shows them, each with the newest version it is in (or the one searched)."""
    ids = [item_id for item_id, _ in chosen]
    if not ids:
        return []
    items = {item.id: item for item in db.scalars(select(Item).where(Item.id.in_(ids)))}
    where: dict[int, dict[str, Any]] = {}
    member = (
        select(ReleaseItem.item_id, Release.id, Release.version, Dataset.name)
        .join(Release, Release.id == ReleaseItem.release_id)
        .join(Dataset, Dataset.id == Release.dataset_id)
        .where(ReleaseItem.item_id.in_(ids), Release.status == "ready", Release.deleted_at.is_(None))
    )
    if release_id:
        member = member.where(Release.id == release_id)
    for item_id, rid, version, dataset in db.execute(member.order_by(Release.created_at)):
        where[item_id] = {"id": rid, "version": version, "dataset": dataset}  # the newest last
    out = []
    for item_id, score in chosen:
        item = items.get(item_id)
        if item is None:
            continue
        out.append(
            {
                "id": item.id,
                "file_name": item.file_name,
                "width": item.width,
                "height": item.height,
                "score": round(score, 4),
                "thumb_url": catalog.media_url(settings, "thumb", item.thumb_key) if item.thumb_key else None,
                "image_url": catalog.media_url(settings, "item", str(item.id)) if item.location_id else None,
                "release": where.get(item_id),
            }
        )
    return out


# ---- 임베딩 만들기 without a training: on a server with the role, the images sent there first if need be -------


def _place_why(node: Node) -> list[str]:
    """Why vectors cannot be made on ``node``, in a few words for a list (``node_problem`` says it in full)."""
    if not node.enabled:
        return ["작업을 받지 않음"]
    why = []
    if not registry.has_role(node, registry.EMBEDS):
        why.append("임베딩용 아님 (노드 설정에서 역할 '임베딩용'을 켭니다)")
    version = training._version(node.agent_version or "")
    if not version or version < SINCE:
        why.append(f"MOA {'.'.join(str(n) for n in SINCE)} 이상 필요")
    return why


def places(db: Session, settings: HubSettings, release: Release) -> dict[str, Any]:
    """The servers a version's vectors can be made on, as 오토라벨's are listed: each with whether it can and why not,
    whether the images must be sent there first; the one used unless another is picked (one that has the images
    first). A server not answering now is said so, not refused (the job waits for it)."""
    try:
        holder = training.data_node(db, release.id)
    except ServiceError as e:
        return {"default": "", "holder": None, "nodes": [], "problem": e.message}
    from . import transfers  # noqa: PLC0415  (transfers reads training, which reads this module's callers)

    have = transfers.holders(db, release.id)
    found = []
    for node in db.scalars(select(Node).where(Node.enabled.is_(True)).order_by(Node.name)):
        why = _place_why(node)
        transfer = node.id not in have
        if transfer and not why:
            try:
                transfers.check_route(holder, node)
            except ServiceError as e:
                why.append(e.message)
        found.append(
            {
                "id": node.id,
                "name": node.name,
                "status": services.node_status(node, settings),
                "gpus": len((node.info or {}).get("gpus") or []),
                "fits": not why,
                "why": " · ".join(why),
                "transfer": transfer,  # the version's images are sent there first
                "holds_images": not transfer,
            }
        )
    fitting = [f for f in found if f["fits"]]
    default = next((f["id"] for f in fitting if not f["transfer"]), "") or (fitting[0]["id"] if fitting else "")
    found.sort(key=lambda f: (f["id"] != default, not f["fits"], f["name"]))
    return {"default": default, "holder": {"id": holder.id, "name": holder.name}, "nodes": found, "problem": ""}


def start_alone(
    db: Session,
    settings: HubSettings,
    user: User,
    release_id: str,
    node_id: str = "",
    via: str = "api",
    gpus: int | None = None,
) -> Job:
    """Make a version's missing vectors without a training: on ``node_id`` (else where ``places`` would), which first
    receives the images when it has none (they stay there, as a training's copy)."""
    release = catalog.get_release(db, release_id)
    catalog.require_ready(release)
    known = coverage(db, release.id)
    if known["going"]:
        raise ServiceError(409, "이 버전의 벡터를 지금 만들고 있습니다.")
    listed = places(db, settings, release)
    node_id = node_id or listed["default"]
    if not node_id:
        why = listed.get("problem") or "벡터를 만들 서버가 없습니다. 노드 설정에서 역할 '임베딩용'을 켜세요."
        raise ServiceError(400, why)
    place = next((p for p in listed["nodes"] if p["id"] == node_id), None)
    if place is None:
        raise ServiceError(404, "고른 서버를 찾을 수 없거나 작업을 받지 않습니다.")
    if not place["fits"]:
        raise ServiceError(400, f"{place['name']}에서는 만들 수 없습니다: {place['why']}")
    from . import transfers  # noqa: PLC0415

    node = db.get(Node, node_id)
    after = transfers.sync_job(db, user, release.id, node, via).id if place["transfer"] else None
    job = start(db, user, release, node_id=node.id, after=after, gpus=gpus, via=via)
    if job is None:
        raise ServiceError(409, "이 버전의 이미지는 모두 벡터가 있습니다.")
    return job


def project_versions(db: Session, project_id: str, model: str = "") -> list[dict[str, Any]]:
    """Every version of a project with how far its images have vectors (이미지 탐색's 버전별 임베딩)."""
    catalog.get_project(db, project_id)
    rows = db.execute(
        select(Release, Dataset.name)
        .join(Dataset, Dataset.id == Release.dataset_id)
        .where(Dataset.project_id == project_id, Release.status == "ready", Release.deleted_at.is_(None))
        .order_by(Dataset.name, Release.created_at.desc())
    ).all()
    out = []
    for release, dataset in rows:
        known = coverage(db, release.id, model)
        out.append(
            {
                "release": {"id": release.id, "version": release.version, "dataset": dataset},
                **{k: known[k] for k in ("total", "done", "todo", "going")},
            }
        )
    return out


# ---- 임베딩 작업: each job as a record, as trainings, evaluations and checks are listed --------------------------

RECENT = 3  # a project card's latest jobs
GOING = ("queued", "assigned", "running")


def _job_rows(db: Session, settings: HubSettings, jobs: list[Job]) -> list[dict[str, Any]]:
    """Embedding jobs as records: the version and project, the server and GPU or CPU, the images it had to make and
    how many it sent so far, who started it and from where (a training, a pipeline, here), and the images sent to
    that server first."""
    if not jobs:
        return []
    names = services.node_name_map(db)
    starters = {j.created_by for j in jobs if j.created_by}
    users = dict(db.execute(select(User.id, User.username).where(User.id.in_(starters))).all()) if starters else {}
    wanted = {str((j.params or {}).get("release_id") or "") for j in jobs} - {""}
    releases = {
        r.id: (r, dataset, project_id, project)
        for r, dataset, project_id, project in db.execute(
            select(Release, Dataset.name, Dataset.project_id, Project.name)
            .join(Dataset, Dataset.id == Release.dataset_id)
            .join(Project, Project.id == Dataset.project_id)
            .where(Release.id.in_(wanted))
        )
    }
    ids = [j.id for j in jobs]
    sent_so_far = dict(
        db.execute(
            select(EmbeddingPart.job_id, func.sum(EmbeddingPart.count))
            .where(EmbeddingPart.job_id.in_(ids))
            .group_by(EmbeddingPart.job_id)
        ).all()
    )
    out = []
    for job in jobs:
        params = job.params or {}
        view = services.job_view(db, settings, job, names)
        found = releases.get(str(params.get("release_id") or ""))
        release, dataset, project_id, project = found if found else (None, "", "", "")
        result = job.result or {}
        before_id = (job.requirements or {}).get("after")
        before = db.get(Job, before_id) if before_id else None
        sent = None
        if before is not None and before.type == "data.sync":  # the images went to that server first
            shown = services.job_view(db, settings, before, names)
            sent = {
                "job_id": before.id,
                "status": shown["status"],
                "error": shown["error"],
                "node_name": shown["node_name"] or names.get(before.target_node_id or "", ""),
                "duration_s": shown["duration_s"],
            }
        model = str(params.get("model") or DEFAULT_MODEL)
        out.append(
            {
                "id": job.id,
                "status": view["status"],
                "error": view["error"],
                "wait_reason": view["wait_reason"],
                "model": model,
                "model_title": (MODELS.get(model) or {}).get("title", model),
                "release": {"id": release.id, "version": release.version, "dataset": dataset} if release else None,
                "project": {"id": project_id, "name": project} if project_id else None,
                "node_name": view["node_name"] or names.get(job.target_node_id or "", ""),
                "gpus": view["gpus_requested"],
                "gpu_indices": view["gpu_indices"],
                # the images it had to make (said when it started; older jobs: what it made)
                "images": params.get("images", result.get("embedded")),
                "made": int(sent_so_far.get(job.id) or 0),
                "missing": result.get("missing"),
                "text": bool((params.get("spec") or {}).get("export_text")),
                "duration_s": view["duration_s"],
                "created_at": view["created_at"],
                "finished_at": view["finished_at"],
                "log_count": job.log_count,
                "by": users.get(job.created_by or "", ""),
                "via": "training" if params.get("run_id") else job.created_via,
                "run_id": params.get("run_id") or None,
                "sent": sent,
            }
        )
    return out


def _embed_jobs(db: Session, release_ids: list[str] | None = None, limit: int = 200) -> list[Job]:
    query = select(Job).where(Job.type == "model.embed").order_by(Job.created_at.desc()).limit(limit)
    if release_ids is not None:
        if not release_ids:
            return []
        query = query.where(Job.params["release_id"].as_string().in_(release_ids))
    return list(db.scalars(query))


def history(
    db: Session, settings: HubSettings, *, project_id: str = "", release_id: str = "", limit: int = 100
) -> list[dict[str, Any]]:
    """A project's (or a version's) embedding jobs, newest first: 임베딩 작업 기록."""
    if release_id:
        catalog.get_release(db, release_id)
        wanted: list[str] | None = [release_id]
    elif project_id:
        catalog.get_project(db, project_id)
        wanted = list(
            db.scalars(
                select(Release.id).join(Dataset, Dataset.id == Release.dataset_id).where(Dataset.project_id == project_id)
            )
        )
    else:
        wanted = None
    return _job_rows(db, settings, _embed_jobs(db, wanted, max(1, min(limit, 500))))


def job_detail(db: Session, settings: HubSettings, job_id: str) -> dict[str, Any]:
    """One embedding job as its record (its page: what it made, where, the log)."""
    job = db.get(Job, job_id)
    if job is None or job.type != "model.embed":
        raise ServiceError(404, "임베딩 작업을 찾을 수 없습니다.")
    row = _job_rows(db, settings, [job])[0]
    if row["release"]:
        known = coverage(db, row["release"]["id"], row["model"])
        row["coverage"] = {k: known[k] for k in ("total", "done", "todo")}
    return row


def board(db: Session, settings: HubSettings, model: str = "") -> dict[str, Any]:
    """이미지 탐색's first screen: the embedding jobs going now, and each project with versions: its images, how many
    can be found (have a vector), and its latest jobs."""
    model = model or DEFAULT_MODEL
    spec = model_of(model)
    tasks = catalog.project_tasks(db)
    projects = {p.id: p for p in db.scalars(select(Project).where(Project.deleted_at.is_(None)).order_by(Project.name))}
    images = dict(db.execute(select(Item.project_id, func.count()).group_by(Item.project_id)).all())
    found = dict(db.execute(select(Item.project_id, func.count()).where(_done(model)).group_by(Item.project_id)).all())
    versions = dict(
        db.execute(
            select(Dataset.project_id, func.count(Release.id))
            .join(Dataset, Dataset.id == Release.dataset_id)
            .where(Release.status == "ready", Release.deleted_at.is_(None))
            .group_by(Dataset.project_id)
        ).all()
    )
    rows = _job_rows(db, settings, _embed_jobs(db, None, 500))
    rows = [r for r in rows if r["project"] and r["project"]["id"] in projects]
    cards = []
    for pid, project in projects.items():
        mine = [r for r in rows if r["project"]["id"] == pid]
        if not versions.get(pid) and not mine:
            continue  # no version to make vectors of yet
        cards.append(
            {
                "project": {"id": pid, "name": project.name, "folder": project.folder, "tasks": tasks.get(pid, [])},
                "images": int(images.get(pid) or 0),
                "embedded": int(found.get(pid) or 0),
                "versions": int(versions.get(pid) or 0),
                "jobs": len(mine),
                "going": sum(1 for r in mine if r["status"] in GOING),
                "failed": sum(1 for r in mine if r["status"] == "failed"),
                "recent": mine[:RECENT],
                "last_at": mine[0]["created_at"] if mine else None,
            }
        )
    # the projects worked on most recently first, then the ones never embedded by name
    cards.sort(key=lambda c: c["project"]["name"])
    cards.sort(key=lambda c: c["last_at"] or "", reverse=True)
    text = text_state(db, model)
    return {
        "model": {"name": model, "title": spec["title"]},
        "text": {"offered": bool(spec.get("text")), "ready": bool(text.get("ready")), "problem": text.get("problem") or ""},
        "going": [r for r in rows if r["status"] in GOING],
        "projects": cards,
        "totals": {
            "projects": len(cards),
            "images": sum(c["images"] for c in cards),
            "embedded": sum(c["embedded"] for c in cards),
            "jobs": len(rows),
            "going": sum(c["going"] for c in cards),
            "failed": sum(c["failed"] for c in cards),
        },
    }
