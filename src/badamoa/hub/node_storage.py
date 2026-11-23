"""What BADAMOA keeps on each server, by kind, as the node last measured it (``badamoa.node.storage``).

The node measures now and then and after jobs that add or remove something, and sends the report
with a heartbeat (``Node.storage``). ``view`` adds from the Hub's records what each item is (the
model types an environment serves, the model a weights file holds, the run a work folder was for)
and whether a queued or running job may use it; ``clean`` asks the node to remove items with a
``node.clean`` job, never one a job may use. Received images are listed here only as a total: they
are kept and removed version by version (``node_data``).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import registry, services
from .db import iso, utcnow
from .models import Job, Node, User
from .services import ServiceError
from .train_models import Environment, ModelJob, ModelVersion, Run, Template

CLEAN_SINCE = (0, 50, 0)
ACTIVE = ("queued", "running")
KINDS = ("images", "envs", "containers", "models", "templates", "template_data", "runs", "uv")
REMOVABLE = ("envs", "containers", "models", "templates", "template_data", "runs", "uv")
WORK_KINDS = ("envs", "models", "templates", "template_data", "runs")  # in the node's work folder
STALE_AFTER = timedelta(hours=2)  # the node measures at least every 30 minutes while it is on


def _version(node: Node) -> tuple[int, ...]:
    return tuple(int(p) for p in (node.agent_version or "0").split(".")[:3] if p.isdigit())


def can_clean(node: Node) -> bool:
    return node.enabled and _version(node) >= CLEAN_SINCE


def save(node: Node, report: dict[str, Any]) -> None:
    """A heartbeat brought a new measurement (what the last clean did stays until the next one)."""
    kept = {k: v for k, v in (node.storage or {}).items() if k == "last_clean"}
    node.storage = {**report, **kept}


# ---- what jobs use ----------------------------------------------------------------------------


def _may_run_here(job: Job, node_id: str) -> bool:
    """A job placed here, or a queued one that may still be placed here."""
    if job.node_id:
        return job.node_id == node_id
    if job.target_node_id:
        return job.target_node_id == node_id
    nodes = (job.requirements or {}).get("nodes") or []
    return not nodes or node_id in nodes


def _active_jobs(db: Session, node: Node) -> list[Job]:
    jobs = db.scalars(select(Job).where(Job.status.in_(ACTIVE), Job.type != "node.clean")).all()
    return [j for j in jobs if _may_run_here(j, node.id)]


def _uses(jobs: list[Job]) -> dict[str, Any]:
    """What the given jobs use, kind by kind, and which job (its id) for each."""
    envs: list[tuple[dict[str, Any], str]] = []
    found: dict[str, dict[str, str]] = {k: {} for k in ("containers", "models", "templates", "template_data", "runs")}
    builds = ""
    for job in jobs:
        params = job.params or {}
        env = params.get("environment") if isinstance(params.get("environment"), dict) else {}
        if env.get("kind") == "docker" and env.get("image"):
            found["containers"].setdefault(env["image"], job.id)
        elif job.type == "env.check" and params.get("kind") == "docker" and params.get("image"):
            found["containers"].setdefault(params["image"], job.id)
        elif env and env.get("kind") in (None, "", "auto"):
            envs.append((env, job.id))
            builds = builds or job.id  # it may build a virtual environment, from uv's cache
        template = params.get("template")
        if isinstance(template, dict):
            if template.get("sha256"):
                found["templates"].setdefault(str(template["sha256"]), job.id)
            if template.get("name"):
                found["template_data"].setdefault(str(template["name"]), job.id)
        for key in ("init_weights", "weights"):
            blob = (params.get(key) or {}).get("blob") if isinstance(params.get(key), dict) else None
            if blob:
                found["models"].setdefault(str(blob), job.id)
        found["runs"].setdefault(job.id, job.id)
        resume = params.get("resume") if isinstance(params.get("resume"), dict) else {}
        if resume.get("job_id"):
            found["runs"].setdefault(str(resume["job_id"]), job.id)
    return {"envs": envs, "uv": builds, **found}


def _user_of(kind: str, item: dict[str, Any], uses: dict[str, Any]) -> str:
    """The id of a job that may use the item ("" when none)."""
    if kind == "envs":
        return next((job_id for env, job_id in uses["envs"] if registry._same_env(item, env)), "")
    if kind == "containers":
        return uses["containers"].get(item.get("image") or "", "")
    if kind == "models":
        return uses["models"].get(item.get("blob") or "", "")
    if kind in ("templates", "template_data", "runs"):
        return uses[kind].get(item.get("id") or "", "")
    return ""


# ---- what each item is --------------------------------------------------------------------------


def _templates(db: Session) -> list[tuple[Template, dict[str, Any] | None]]:
    """Every model type kept on the Hub with what it installs on a server."""
    out = []
    for row in db.scalars(select(Template).where(Template.deleted_at.is_(None)).order_by(Template.title)):
        try:
            spec = registry.install_spec(db, registry.manifest_of(row))
        except Exception:  # noqa: BLE001 - a broken declaration only means it installs nothing we can match
            spec = None
        out.append((row, spec))
    return out


# what a built-in kind's library is called on the screen (another goes by the library's own name: torchvision)
LIBRARY_TITLES = {"ultralytics": "Ultralytics YOLO"}


def _newest(versions: list[str]) -> str:
    from packaging.version import InvalidVersion, Version

    def key(text: str) -> Version:
        try:
            return Version(text)
        except InvalidVersion:
            return Version("0")

    found = [v for v in versions if v]
    return max(found, key=key) if found else ""


class _Libraries:
    """The built-in model kinds' libraries and their versions, asked once per view. The built-in kinds on one
    library install the same requirement into one environment, so they share a version, as on Settings → 모델
    종류: the one on this server when any of them ran here, else the newest any server runs, else what installing
    it now gets. Their own numbers (8, 12) only count changes to their code and read like models (YOLO12)."""

    def __init__(self, db: Session, node_id: str) -> None:
        self.db, self.node_id = db, node_id
        self._of: dict[str, dict[str, Any] | None] = {}  # template id → its library and versions
        self._versions: dict[str, str] = {}  # library → its version here

    def of(self, row: Template) -> dict[str, Any] | None:
        if row.id not in self._of:
            try:
                self._of[row.id] = registry.library_versions(self.db, row, registry.manifest_of(row))
            except Exception:  # noqa: BLE001 - a broken declaration only means no version to show
                self._of[row.id] = None
        return self._of[row.id]

    def version(self, name: str) -> str:
        if name not in self._versions:
            built_in = self.db.scalars(select(Template).where(Template.builtin, Template.deleted_at.is_(None)))
            found = [lib for lib in map(self.of, built_in) if lib and lib["name"] == name]
            here = [s["version"] for lib in found for s in lib["installed"] if s.get("node_id") == self.node_id]
            self._versions[name] = (
                _newest(here) or _newest([lib["newest"] for lib in found]) or _newest([lib["available"] for lib in found])
            )
        return self._versions[name]

    def title(self, row: Template) -> str:
        """A model kind as an environment's user: a built-in one by its library and version, for one library
        serves each of its tasks (Ultralytics YOLO 8.4.216 detects, segments and finds keypoints); one an admin
        added by its title and version (Ultralytics YOLO DCL 1.0.5)."""
        library = self.of(row) if row.builtin else None
        if library:
            return f"{LIBRARY_TITLES.get(library['name'], library['name'])} {self.version(library['name'])}".strip()
        return f"{row.title or row.name} {'' if row.builtin else row.version}".strip()

    def code(self, row: Template) -> str:
        """A model kind's code: each task's own (the built-in ones by their task, with their library's version)."""
        library = self.of(row) if row.builtin else None
        version = self.version(library["name"]) if library else ("" if row.builtin else row.version)
        return f"{row.title or row.name} {version}".strip()


def _names(rows: list[Template], libraries: _Libraries) -> list[str]:
    """The model kinds an environment or image serves, each once (a kind's newest version; the built-in ones on
    one library as one)."""
    names: list[str] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda r: r.created_at, reverse=True):
        if row.name in seen:
            continue
        seen.add(row.name)
        name = libraries.title(row)
        if name not in names:
            names.append(name)
    return names


def _describe(db: Session, kind: str, items: list[dict[str, Any]], node_id: str = "") -> None:
    """Add ``label`` (what it is) and ``users`` (the model types it serves) to each item, in place."""
    libraries = _Libraries(db, node_id)
    if kind in ("envs", "containers"):
        templates = _templates(db)
        docker_envs = db.scalars(select(Environment).where(Environment.kind == "docker")).all()
        for item in items:
            rows = [row for row, spec in templates if spec is not None and registry._same_env(item, spec)]
            item["users"] = _names(rows, libraries)
            if kind == "containers":
                item["label"] = item.get("image") or ""
                item["environments"] = [e.name for e in docker_envs if e.image == item.get("image")]
            else:
                parts = [f"Python {item.get('python') or '3.12'}"] + (["PyTorch"] if item.get("torch") else [])
                item["label"] = " · ".join(parts)
    elif kind == "models":
        blobs = [i.get("blob") for i in items if i.get("blob")]
        models = db.scalars(select(ModelVersion).where(ModelVersion.blob.in_(blobs))).all() if blobs else []
        by_blob: dict[str, list[ModelVersion]] = {}
        for model in models:
            by_blob.setdefault(model.blob, []).append(model)
        for item in items:
            found = by_blob.get(item.get("blob") or "", [])
            item["label"] = ", ".join(f"{m.name} v{m.version}" for m in found) or "Hub에 없는 가중치"
            item["model_id"] = found[0].id if found else ""
            item["gone"] = not found or all(m.deleted_at is not None for m in found)
    elif kind == "templates":
        shas = [i.get("id") for i in items]
        rows = db.scalars(select(Template).where(Template.blob.in_(shas))).all() if shas else []
        by_sha = {r.blob: r for r in rows}
        for item in items:
            row = by_sha.get(item.get("id"))
            item["label"] = libraries.code(row) if row else "Hub에 없는 코드"
            item["gone"] = row is None or row.deleted_at is not None
    elif kind == "template_data":
        names = [i.get("id") for i in items]
        rows = db.scalars(select(Template).where(Template.name.in_(names))).all() if names else []
        titles = {r.name: r.title or r.name for r in rows}
        for item in items:
            item["label"] = titles.get(item.get("id") or "", item.get("id") or "")
    elif kind == "runs":
        ids = [i.get("id") for i in items]
        jobs = {j.id: j for j in db.scalars(select(Job).where(Job.id.in_(ids)))} if ids else {}
        runs = {r.job_id: r for r in db.scalars(select(Run).where(Run.job_id.in_(ids)))} if ids else {}
        ops = {o.job_id: o for o in db.scalars(select(ModelJob).where(ModelJob.job_id.in_(ids)))} if ids else {}
        for item in items:
            job = jobs.get(item.get("id") or "")
            run = runs.get(item.get("id"))
            op = ops.get(item.get("id"))
            if run is not None:
                item["label"], item["run_id"] = run.name, run.id
            elif op is not None:
                model = db.get(ModelVersion, op.model_id)
                what = {"eval": "평가", "export": "내보내기", "predict": "테스트"}.get(op.kind, op.kind)
                item["label"] = f"{what}: {model.name} v{model.version}" if model else what
                item["model_id"] = op.model_id
            else:
                item["label"] = job.type if job else "Hub에 없는 작업"
            item["job_status"] = job.status if job else ""


# ---- the view -----------------------------------------------------------------------------------


def _free(report: dict[str, Any], path: str) -> int | None:
    disk = (report.get("disks") or {}).get(path) or {}
    return disk.get("free")


def _mount(report: dict[str, Any], path: str) -> str:
    """The disk (mount point) a folder is on, as the node measured it ("" from nodes before 0.67.0)."""
    disk = (report.get("disks") or {}).get(path) or {}
    return str(disk.get("mount") or "")


def view(db: Session, node: Node) -> dict[str, Any]:
    """The node's last report, kind by kind, with what each item is and whether a job may use it."""
    report = dict(node.storage or {})
    cleaning = db.scalars(
        select(Job).where(Job.type == "node.clean", Job.status.in_(ACTIVE), Job.target_node_id == node.id)
    ).all()
    base = {
        "node_id": node.id,
        "can_clean": can_clean(node),
        "need": ".".join(str(n) for n in CLEAN_SINCE),
        "cleaning": [{"job_id": j.id, "kind": (j.params or {}).get("kind") or ""} for j in cleaning],
        "last_clean": report.get("last_clean"),
    }
    if not report.get("kinds"):
        return {**base, "measured_at": None, "kinds": []}
    uses = _uses(_active_jobs(db, node))
    home = str(report.get("home") or "")
    kinds = []
    for kind in KINDS:
        data = (report.get("kinds") or {}).get(kind)
        if data is None:
            continue
        items = data.get("items")
        entry: dict[str, Any] = {"kind": kind, "path": data.get("path") or "", "removable": kind in REMOVABLE}
        if items is None:  # a total only: received images, uv's cache
            entry.update(bytes=int(data.get("bytes") or 0), files=data.get("files"), shared=data.get("shared"))
            if kind == "uv":
                entry["in_use"] = uses["uv"]
        else:
            items = [dict(i) for i in items]
            _describe(db, kind, items, node.id)
            for item in items:
                item["in_use"] = _user_of(kind, item, uses)
            items.sort(key=lambda i: -(int(i.get("bytes") or 0)))
            entry.update(items=items, bytes=sum(int(i.get("bytes") or 0) for i in items), count=len(items))
        if kind == "containers":  # Docker's own folder (Docker Desktop: inside its virtual disk)
            entry["desktop"] = bool(data.get("desktop"))
            entry["free"] = None if entry["desktop"] else _free(report, entry["path"])
        else:
            entry["free"] = _free(report, entry["path"]) if _free(report, entry["path"]) is not None else _free(report, home)
        entry["mount"] = "" if entry.get("desktop") else (_mount(report, entry["path"]) or _mount(report, home))
        if kind == "images":  # picked by the node (the disk with the most room) or set by an admin
            entry["auto"] = not (node.cache_dir or "").strip()
        # in the node's work folder (0.74.0): picked like the images' folder, or set by an admin
        if report.get("work") and (kind in WORK_KINDS or (kind == "uv" and data.get("work"))):
            entry["work"] = True
            entry["auto"] = not (node.work_dir or "").strip()
        for item in entry.get("items") or []:  # still where the node kept it before: its own disk
            if item.get("path"):
                item["mount"] = _mount(report, item["path"])
        kinds.append(entry)
    measured = report.get("measured_at")
    stale = False
    if measured:
        try:
            stale = utcnow() - datetime.fromisoformat(str(measured).rstrip("Z")) > STALE_AFTER
        except ValueError:
            stale = False
    return {
        **base,
        "measured_at": measured,
        "took_s": report.get("took_s"),
        "stale": stale,
        "home": home,
        "home_mount": _mount(report, home),
        "work": str(report.get("work") or ""),
        "work_mount": _mount(report, str(report.get("work") or "")),
        "work_free": _free(report, str(report.get("work") or "")),
        "free": _free(report, home),
        "bytes": sum(k["bytes"] for k in kinds),
        "kinds": kinds,
    }


# ---- removing -----------------------------------------------------------------------------------


def clean(
    db: Session, user: User, node_id: str, kind: str, items: list[str], everything: bool = False, via: str = "api"
) -> Job:
    """Ask the node to remove items of one kind (or every item no job may use). Refused for items a job uses."""
    services.require_admin(user)
    node = services.get_node(db, node_id)
    if not node.enabled:
        raise ServiceError(400, f"{node.name}은(는) 꺼 둔 서버입니다.")
    if _version(node) < CLEAN_SINCE:
        need = ".".join(str(n) for n in CLEAN_SINCE)
        raise ServiceError(400, f"{node.name}의 MOA는 저장한 것을 지울 수 없습니다. {need} 이상으로 업데이트하세요.")
    if kind == "images":
        raise ServiceError(400, "받아 둔 이미지는 버전별로 지웁니다: 아래의 '받아 둔 버전'에서 지우세요.")
    if kind not in REMOVABLE:
        raise ServiceError(400, f"지울 수 없는 종류입니다: {kind}")
    busy = db.scalar(
        select(Job.id).where(Job.type == "node.clean", Job.status.in_(ACTIVE), Job.target_node_id == node.id).limit(1)
    )
    if busy:
        raise ServiceError(409, f"{node.name}에서 지우는 작업이 이미 있습니다. 끝난 뒤에 다시 하세요.")
    uses = _uses(_active_jobs(db, node))
    if kind == "uv":
        if uses["uv"]:
            raise ServiceError(409, "이 서버에서 가상환경을 만들 수 있는 작업이 있어 uv 캐시를 지금 비울 수 없습니다.")
        names: list[str] = []
    else:
        # only what the node reported last: the screen offers nothing else
        known = list(((node.storage or {}).get("kinds") or {}).get(kind, {}).get("items") or [])
        key = "image" if kind == "containers" else "id"
        by_id = {str(i.get(key)): i for i in known}
        wanted = list(by_id) if everything else list(dict.fromkeys(str(n) for n in items))
        if not wanted:
            raise ServiceError(400, "지울 것이 없습니다." if everything else "지울 것을 고르세요.")
        missing = [n for n in wanted if n not in by_id]
        if missing:
            raise ServiceError(400, f"이 서버가 마지막으로 알려 온 목록에 없습니다: {missing[0]}. 다시 재 본 뒤에 고르세요.")
        in_use = {n: job_id for n in wanted if (job_id := _user_of(kind, by_id[n], uses))}
        if in_use and not everything:
            name, job_id = next(iter(in_use.items()))
            raise ServiceError(409, f"작업이 쓰는 중이라 지울 수 없습니다: {name} (작업 {job_id[:8]})")
        names = [n for n in wanted if n not in in_use]
        if not names:
            raise ServiceError(409, "모두 작업이 쓰는 중이라 지울 것이 없습니다.")
    params = {"kind": kind, "items": names}
    job = services.create_job(db, user, "node.clean", params, node.id, via=via, gpus=0)
    services.audit(db, user.id, via, "node.clean", node.id, kind=kind, items=len(names), everything=everything)
    return job


def measure(db: Session, user: User, node_id: str, via: str = "api") -> Job:
    """Have the node measure again now (it does by itself every 30 minutes and after jobs that change things)."""
    node = services.get_node(db, node_id)
    if not node.enabled:
        raise ServiceError(400, f"{node.name}은(는) 꺼 둔 서버입니다.")
    if _version(node) < CLEAN_SINCE:
        need = ".".join(str(n) for n in CLEAN_SINCE)
        raise ServiceError(400, f"{node.name}의 MOA는 저장한 것을 잴 수 없습니다. {need} 이상으로 업데이트하세요.")
    job = services.create_job(db, user, "node.clean", {"kind": "", "items": []}, node.id, via=via, gpus=0)
    services.audit(db, user.id, via, "node.measure", node.id)
    return job


def on_job_finished(db: Session, job: Job) -> None:
    """Note what a ``node.clean`` job removed (the node measures again and sends a new report)."""
    if job.type != "node.clean" or not job.node_id or not (job.params or {}).get("kind"):
        return
    node = db.get(Node, job.node_id)
    if node is None:
        return
    result = job.result or {}
    note = {
        "job_id": job.id,
        "kind": (job.params or {}).get("kind") or "",
        "ok": job.status == "succeeded",
        "removed": result.get("removed"),
        "bytes": result.get("bytes"),
        "error": "" if job.status == "succeeded" else (job.error or "지우지 못했습니다."),
        "at": iso(utcnow()),
    }
    node.storage = {**(node.storage or {}), "last_clean": note}
