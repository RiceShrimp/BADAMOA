"""Lineage: where a model or a run came from and what came of it, as a small graph.

    release ──▶ run ──▶ model ──▶ evaluation / export / test
       model ──(이어서 학습)──▶ run        interrupted run ──(이어서 실행)──▶ run
       search ──▶ run

Upstream is followed back through the models a run started from and the runs it continued;
downstream through the model's jobs and the runs that started from it. Each node carries the
column it belongs in, so the screen only has to draw.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from badamoa.core.protocol import TERMINAL_STATUSES

from . import training
from .catalog_models import Dataset, Release
from .models import Job
from .train_models import ModelJob, ModelVersion, Run, Sweep

DEPTH = 4  # how far back and forward to follow
MAX_OPS = 12
MAX_CHILDREN = 10
SPLIT = {"val": "val", "test": "test", "train": "train"}


class _Graph:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str], str] = {}

    def node(self, key: str, **data: Any) -> str:
        self.nodes.setdefault(key, {"id": key, **data})
        return key

    def edge(self, source: str, target: str, label: str = "") -> None:
        if source != target:
            self.edges.setdefault((source, target), label)

    # ---- kinds of nodes ----------------------------------------------------------------

    def release(self, release_id: str | None) -> str | None:
        release = self.db.get(Release, release_id) if release_id else None
        if release is None:
            return None
        dataset = self.db.get(Dataset, release.dataset_id)
        items = (release.stats or {}).get("items")
        return self.node(
            f"release:{release.id}",
            kind="release",
            label=f"{dataset.name if dataset else '?'} {release.version}",
            sub=f"{items:,}장" if isinstance(items, int) else "버전",
            href=None if release.deleted_at else f"#/datasets/{release.id}",
        )

    def run(self, run: Run) -> str:
        job = self.db.get(Job, run.job_id) if run.job_id else None
        status = job.status if job else "failed"
        template = run.template or {}
        return self.node(
            f"run:{run.id}",
            kind="run",
            label=run.name,
            sub=f"{template.get('variant') or template.get('title') or run.trainer} · epoch {run.epoch}/{run.epochs}",
            status=status,
            href=None if run.deleted_at else f"#/training/{run.id}",
        )

    def model(self, model: ModelVersion) -> str:
        aliases = training._aliases(model)
        return self.node(
            f"model:{model.id}",
            kind="model",
            label=f"{model.name} v{model.version}",
            sub=" · ".join(aliases) if aliases else (model.arch or model.trainer),
            href=None if model.deleted_at else f"#/models/{model.id}",
        )

    def op(self, op: ModelJob, model: ModelVersion) -> str:
        job = self.db.get(Job, op.job_id) if op.job_id else None
        status = job.status if job else "failed"
        metric = training.PRIMARY_METRIC.get(model.task, "map50_95")
        if op.kind == "eval":
            release = self.db.get(Release, op.release_id) if op.release_id else None
            dataset = self.db.get(Dataset, release.dataset_id) if release else None
            where = f"{dataset.name} {release.version}" if release and dataset else "삭제된 버전"
            value = (op.result or {}).get("metrics", {}).get(metric)
            label = f"{where} {SPLIT.get(str(op.params.get('split')), op.params.get('split', ''))}"
            sub = f"{metric} {value:.3f}" if isinstance(value, int | float) else _status_text(status)
        elif op.kind == "export":
            label, sub = str(op.params.get("format", "")).upper(), _status_text(status)
        else:
            count = len((op.result or {}).get("predictions") or [])
            label, sub = (f"이미지 {count}장", _status_text(status)) if count else ("샘플 이미지", _status_text(status))
        key = f"{op.kind}:{op.id}"
        return self.node(key, kind=op.kind, label=label, sub=sub, status=status, href=f"#/models/{model.id}")

    def sweep(self, sweep_id: str) -> str | None:
        sweep = self.db.get(Sweep, sweep_id)
        if sweep is None:
            return None
        return self.node(
            f"sweep:{sweep.id}",
            kind="sweep",
            label=sweep.name,
            sub=f"탐색 · {', '.join(sweep.space)}",
            href=f"#/training/sweep/{sweep.id}",
        )

    # ---- walking ---------------------------------------------------------------------------

    def upstream(self, run: Run, depth: int = DEPTH) -> str:
        here = self.run(run)
        config = run.config or {}
        if source := self.release(run.release_id):
            self.edge(source, here)  # the data it learned from
        base = (config.get("base_model") or {}).get("id")
        if base and (model := self.db.get(ModelVersion, base)) is not None:
            self.edge(self.model(model), here, "이어서 학습")
            earlier = self.db.get(Run, model.run_id) if model.run_id else None
            if earlier is not None and depth > 0:
                self.edge(self.upstream(earlier, depth - 1), f"model:{model.id}")
        resumed = (config.get("resumed_from") or {}).get("id")
        if resumed and depth > 0 and (interrupted := self.db.get(Run, resumed)) is not None:
            self.edge(self.upstream(interrupted, depth - 1), here, "이어서 실행")
        sweep = (config.get("sweep") or {}).get("id")
        if sweep and (node := self.sweep(sweep)):
            self.edge(node, here)
        return here

    def downstream(self, model: ModelVersion, depth: int = DEPTH) -> None:
        here = self.model(model)
        ops = self.db.scalars(
            select(ModelJob).where(ModelJob.model_id == model.id).order_by(ModelJob.created_at.desc()).limit(MAX_OPS)
        )
        for op in reversed(list(ops)):
            self.edge(here, self.op(op, model))
        if depth <= 0:
            return
        children = [
            r
            for r in self.db.scalars(
                select(Run).where(Run.project_id == model.project_id, Run.deleted_at.is_(None)).order_by(Run.created_at)
            )
            if ((r.config or {}).get("base_model") or {}).get("id") == model.id
        ][:MAX_CHILDREN]
        for child in children:
            self.edge(here, self.run(child), "이어서 학습")
            if source := self.release(child.release_id):
                self.edge(source, f"run:{child.id}")
            grown = self.db.scalar(
                select(ModelVersion).where(ModelVersion.run_id == child.id, ModelVersion.deleted_at.is_(None))
            )
            if grown is not None:
                self.edge(f"run:{child.id}", self.model(grown))
                self.downstream(grown, depth - 1)

    # ---- result ----------------------------------------------------------------------------

    def result(self, current: str) -> dict[str, Any]:
        """Nodes with the column each sits in (the longest path from a source), and the edges."""
        edges = [(s, t) for (s, t) in self.edges if s in self.nodes and t in self.nodes]
        column = {key: 0 for key in self.nodes}
        for _ in range(len(self.nodes)):  # longest path by relaxation: the graph is small and has no cycles
            changed = False
            for source, target in edges:
                if column[target] < column[source] + 1:
                    column[target] = column[source] + 1
                    changed = True
            if not changed:
                break
        nodes = [{**node, "column": column[key], "current": key == current} for key, node in self.nodes.items()]
        return {
            "nodes": nodes,
            "edges": [{"from": s, "to": t, "label": self.edges[(s, t)]} for s, t in edges],
        }


def _status_text(status: str) -> str:
    return {"queued": "대기", "running": "실행 중", "failed": "실패", "canceled": "취소됨"}.get(status, "완료")


def for_model(db: Session, model: ModelVersion) -> dict[str, Any]:
    graph = _Graph(db)
    run = db.get(Run, model.run_id) if model.run_id else None
    here = graph.model(model)
    if run is not None:
        graph.edge(graph.upstream(run), here)
    elif source := graph.release(model.release_id):
        graph.edge(source, here)
    graph.downstream(model)
    return graph.result(here)


def for_run(db: Session, run: Run) -> dict[str, Any]:
    graph = _Graph(db)
    here = graph.upstream(run)
    model = db.scalar(select(ModelVersion).where(ModelVersion.run_id == run.id, ModelVersion.deleted_at.is_(None)))
    if model is not None:
        graph.edge(here, graph.model(model))
        graph.downstream(model)
    job = db.get(Job, run.job_id) if run.job_id else None
    if job is not None and job.status not in TERMINAL_STATUSES:
        graph.nodes[here]["status"] = job.status
    return graph.result(here)
