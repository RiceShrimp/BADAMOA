"""Built-in model kinds are named by the version of their library the servers run (Ultralytics 8.4.216, not the
template's own number): what recent runs installed per server, else what installing it now gets."""

from __future__ import annotations

from badamoa.hub import registry
from badamoa.hub.db import Database
from badamoa.hub.models import Job, Node
from badamoa.hub.train_models import Run


def test_the_library_and_its_version_the_servers_run(client, admin, settings, monkeypatch):
    monkeypatch.setattr(
        registry, "_pypi_newest", lambda requirement: "8.4.216" if "ultralytics" in requirement else "0.24.1"
    )
    listed = {t["name"]: t for t in client.get("/api/templates", headers=admin).json()}
    yolo, tv = listed["yolo"]["library"], listed["torchvision"]["library"]
    assert (yolo["name"], yolo["newest"], yolo["available"]) == ("ultralytics", "", "8.4.216")  # nothing installed yet
    assert tv["name"] == "torchvision" and tv["available"] == "0.24.1"

    # ---- runs say what the servers installed: the newest per server, the newest of all first
    project = client.post("/api/projects", json={"name": "버전"}, headers=admin).json()
    database = Database(settings.database_url)
    with database.session() as db:
        for host, version in (("AI-202", "8.4.200+cu128"), ("AI-201", "8.3.40")):
            node = Node(name=host, token_hash=f"x{host}", roles=["compute"], os="linux", info={})
            db.add(node)
            db.flush()
            job = Job(type="train.run", params={}, status="succeeded", node_id=node.id)
            db.add(job)
            db.flush()
            db.add(
                Run(
                    project_id=project["id"], name=f"run-{host}", release_id="r", trainer="yolo", job_id=job.id,
                    template={"name": "yolo", "version": "12"},
                    info={"packages": [f"ultralytics=={version}", "numpy==2.1.0"]},
                )
            )  # fmt: skip
    database.engine.dispose()
    yolo = next(t for t in client.get("/api/templates", headers=admin).json() if t["name"] == "yolo")["library"]
    assert yolo["newest"] == "8.4.200" and [(s["node"], s["version"]) for s in yolo["installed"]] == [
        ("AI-202", "8.4.200"),
        ("AI-201", "8.3.40"),
    ]
    assert yolo["available"] == "8.4.216"  # a server installing it now gets a newer one


def test_a_servers_environments_name_built_in_kinds_by_their_library(client, admin, settings, monkeypatch):
    """Node → 저장 공간: what an environment serves reads "Ultralytics YOLO 8.4.200", the version this server runs
    (not the template's own number, "Ultralytics YOLO 12", read as YOLO12); one an admin added keeps its own."""
    from datetime import datetime

    from sqlalchemy import select

    from badamoa.hub import node_storage
    from badamoa.hub.train_models import Template

    monkeypatch.setattr(registry, "_pypi_newest", lambda requirement: "8.4.216")
    client.get("/api/templates", headers=admin)  # the built-in kinds are in
    project = client.post("/api/projects", json={"name": "환경 이름"}, headers=admin).json()
    database = Database(settings.database_url)
    with database.session() as db:
        here, there = (Node(name=n, token_hash=f"t{n}", roles=["compute"], os="linux", info={}) for n in ("A", "B"))
        db.add_all([here, there])
        db.flush()
        job = Job(type="train.run", params={}, status="succeeded", node_id=here.id)
        db.add(job)
        db.flush()
        db.add(
            Run(
                project_id=project["id"], name="run-a", release_id="r", trainer="yolo", job_id=job.id,
                template={"name": "yolo", "version": "12"}, info={"packages": ["ultralytics==8.4.200+cu128"]},
            )
        )  # fmt: skip
        db.flush()
        built_in = {r.name: r for r in db.scalars(select(Template).where(Template.builtin, Template.deleted_at.is_(None)))}
        yolo, seg = built_in["yolo"], built_in["yolo-seg"]
        added = Template(name="dcl", title="Ultralytics YOLO DCL", version="1.0.5", builtin=False)
        added.created_at = datetime(2026, 1, 1)

        assert sorted(node_storage._names(db, [yolo, seg, added], here.id)) == sorted(
            [
                f"{yolo.title} 8.4.200",  # what ran on this server
                f"{seg.title} 8.4.216",  # never ran anywhere: what installing it now gets
                "Ultralytics YOLO DCL 1.0.5",  # an admin's kind: its own version
            ]
        )
        assert node_storage._names(db, [yolo], there.id) == [f"{yolo.title} 8.4.200"]  # the newest any server runs
        monkeypatch.setattr(registry, "_pypi_newest", lambda requirement: "")  # no internet, never ran: no number
        assert node_storage._names(db, [seg], there.id) == [seg.title]
    database.engine.dispose()
