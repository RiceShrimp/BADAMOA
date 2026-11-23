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
    """Node → 저장 공간: what an environment serves reads "Ultralytics YOLO 8.4.200" — the built-in kinds on one
    library as one (it detects, segments and finds keypoints alike), with the version this server runs, not their
    own numbers ("Ultralytics YOLO 분할 8, Ultralytics YOLO 12", read as YOLOv8 · YOLO12); an admin's keeps its own."""
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

        def names(rows, node):
            return node_storage._names(rows, node_storage._Libraries(db, node.id))

        # the detection kind ran here on 8.4.200: the segmentation one shares its environment, so its version
        assert names([yolo, seg, added], here) == ["Ultralytics YOLO 8.4.200", "Ultralytics YOLO DCL 1.0.5"]
        assert names([seg], there) == ["Ultralytics YOLO 8.4.200"]  # the newest any server runs
        libraries = node_storage._Libraries(db, here.id)
        assert (libraries.code(seg), libraries.code(added)) == (f"{seg.title} 8.4.200", "Ultralytics YOLO DCL 1.0.5")

        # never ran anywhere: what installing it now gets; nothing known (no internet): the library alone
        tv = built_in["torchvision"]
        assert names([tv], here) == ["torchvision 8.4.216"]  # (the stand-in PyPI answers 8.4.216 to all)
        monkeypatch.setattr(registry, "_pypi_newest", lambda requirement: "")
        assert names([tv], there) == ["torchvision"]
    database.engine.dispose()


def test_an_in_house_build_is_what_only_a_kind_pinning_it_gets(client, admin, settings, monkeypatch):
    """The Hub keeps a fork of the library (DCL's ultralytics 8.4.138.1): uv takes the newest of the Hub's wheels and
    PyPI's releases, so the built-in YOLO still gets 8.4.216 and only a kind pinning the fork gets it."""
    from sqlalchemy import select

    from badamoa.hub.train_models import Package, Template

    monkeypatch.setattr(registry, "_pypi_newest", lambda requirement: "" if "==8.4.138.1" in requirement else "8.4.216")
    client.get("/api/templates", headers=admin)  # the built-in kinds are in
    database = Database(settings.database_url)
    with database.session() as db:
        db.add(Package(name="ultralytics", version="8.4.138.1", filename="ultralytics-8.4.138.1-py3-none-any.whl", blob="b"))
        db.flush()
        yolo = db.scalars(select(Template).where(Template.name == "yolo", Template.deleted_at.is_(None))).one()
        manifest = registry.manifest_of(yolo)
        assert registry.library_versions(db, yolo, manifest)["available"] == "8.4.216"
        manifest.environment.requirements = ["ultralytics==8.4.138.1", "yolo-curriculum==1.0.0"]
        assert registry.library_versions(db, yolo, manifest)["available"] == "8.4.138.1"
        monkeypatch.setattr(registry, "_pypi_newest", lambda requirement: "")  # PyPI out of reach: the Hub's own
        assert registry.library_versions(db, yolo, registry.manifest_of(yolo))["available"] == "8.4.138.1"
    database.engine.dispose()
