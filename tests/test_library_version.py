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
