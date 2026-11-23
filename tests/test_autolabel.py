"""The 오토라벨 screen: runs across projects, what they can run over, starting one from a dataset's images.

Uses the ``dummy`` template (it declares autolabel: a sure box, an unsure one, nothing, in turn) and a real
Node Agent, as ``test_training`` does.
"""

from __future__ import annotations

import shutil

from sample_data import CLASSES, make_yolo
from test_training import _logs, _train, _wait, hub, release  # noqa: F401 (fixtures)
from test_transfer import two_nodes  # noqa: F401 (fixture)


def _raw(client, project_id, tmp_path):
    """A dataset of the project whose images nobody labelled yet (a folder without label files)."""
    root = make_yolo(tmp_path / "raw")
    shutil.rmtree(root / "labels")
    (root / "data.yaml").unlink()
    body = {"node_id": client.node_id, "path": str(root), "dataset": "raw", "version": "v1", "classes": CLASSES}
    imported = _wait(client, client.post(f"/api/projects/{project_id}/imports", json=body).json()["id"])
    assert imported["status"] == "succeeded", _logs(client, imported["id"])
    return imported["result"]["release_id"]


def test_autolabel_screen_runs_targets_and_starting_from_a_dataset(hub, release, tmp_path):  # noqa: F811
    run, job = _train(hub, release)
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    model = hub.get(f"/api/runs/{run['id']}").json()["model"]
    project = release["project_id"]
    raw = _raw(hub, project, tmp_path)

    # ---- before any run: the project is there with a model that can label, nothing run yet
    board = hub.get("/api/autolabel/board").json()
    (card,) = [c for c in board["projects"] if c["project"]["id"] == project]
    assert (card["runs"], card["models"], card["sets_waiting"], card["recent"]) == (0, 1, 0, [])
    assert board["going"] == [] and board["totals"]["runs"] == 0

    # ---- what it can run over: the datasets' images where labels go now, by split and label state
    found = hub.get("/api/autolabel/targets", params={"project_id": project}).json()
    assert found["sets"] == []
    by_name = {d["name"]: d for d in found["datasets"]}
    assert by_name["raw"]["release"] == {"id": raw, "version": "v1", "draft": False}
    assert sum(c.get("unlabelled", 0) for c in by_name["raw"]["counts"].values()) == 8
    assert sum(c.get("unlabelled", 0) for c in by_name["street"]["counts"].values()) == 0

    # ---- the models for a dataset's images (before its set is made) and the server holding them
    offered = hub.get("/api/autolabel/models", params={"release_id": raw}).json()
    assert [(m["id"], m["problem"]) for m in offered["models"]] == [(model["id"], "")]
    servers = offered["servers"]
    assert servers["default"] == hub.node_id and servers["holder"]["id"] == hub.node_id
    (only,) = servers["nodes"]
    assert only["fits"] and only["holds_images"] and not only["transfer"], only["why"]
    assert offered["todo"] is None
    assert hub.get("/api/autolabel/models").status_code == 400

    # ---- nothing to label: refused, and no set is left behind
    street = {"release_id": release["id"], "model_id": model["id"], "gpus": 0}
    refused = hub.post("/api/autolabel", json=street)
    assert refused.status_code == 400 and "아직 라벨링하지 않은 이미지가 없습니다" in refused.json()["detail"]
    assert hub.get("/api/label-sets", params={"project_id": project}).json() == []
    assert hub.post("/api/autolabel", json={"model_id": model["id"]}).status_code == 400
    odd = {**street, "release_id": raw, "split": "dev"}
    assert hub.post("/api/autolabel", json=odd).status_code == 400

    # ---- from the dataset's images: a set of them is made, then the model labels it (sure ones accepted)
    body = {"release_id": raw, "model_id": model["id"], "name": "raw 먼저", "part_size": 4, "accept": 0.9, "gpus": 0}
    started = hub.post("/api/autolabel", json=body)
    assert started.status_code == 200, started.text
    first = started.json()
    assert _wait(hub, first["job_id"])["status"] == "succeeded", _logs(hub, first["job_id"])
    made = hub.get(f"/api/label-sets/{first['set_id']}").json()
    assert (made["name"], made["items"], made["parts"]) == ("raw 먼저", 8, 2)
    assert made["autolabel"]["op_id"] == first["id"] and made["autolabel"]["status"] == "succeeded"

    (row,) = hub.get("/api/autolabel/runs", params={"project_id": project}).json()
    assert row["id"] == first["id"] and row["status"] == "succeeded" and row["current"]
    assert row["set"]["id"] == first["set_id"] and row["dataset"]["name"] == "raw" and row["model"]["id"] == model["id"]
    assert row["result"] == {"images": 8, "labels": 6, "accepted": 3, "empty": 2, "skipped": 0}
    assert (row["items"], row["keep"], row["accept"], row["gpus"]) == (8, 0.25, 0.9, 0)
    assert row["started_by"] == "admin" and row["node_name"] == "train-node"

    shown = hub.get(f"/api/autolabel/runs/{first['id']}").json()
    assert shown["log_count"] > 0 and shown["code"]["name"] == "dummy"
    assert hub.get("/api/autolabel/runs/nope").status_code == 404

    # ---- again over the set: the images still to label; the newest run is the one its editor starts from
    left = hub.get("/api/autolabel/models", params={"set_id": first["set_id"]}).json()
    assert left["todo"] == 5
    again = hub.post("/api/autolabel", json={"set_id": first["set_id"], "model_id": model["id"], "gpus": 0}).json()
    assert _wait(hub, again["job_id"])["status"] == "succeeded", _logs(hub, again["job_id"])
    listed = hub.get("/api/autolabel/runs", params={"set_id": first["set_id"]}).json()
    assert [(r["id"], r["current"]) for r in listed] == [(again["id"], True), (first["id"], False)]
    assert listed[0]["result"]["images"] == 5 and listed[0]["accept"] is None
    assert hub.get("/api/autolabel/runs", params={"project_id": "nope"}).json() == []

    # ---- the board sums it up: runs, what they predicted, the set with images still to label
    board = hub.get("/api/autolabel/board").json()
    (card,) = [c for c in board["projects"] if c["project"]["id"] == project]
    assert (card["runs"], card["succeeded"], card["failed"], card["going"]) == (2, 2, 0, 0)
    assert (card["images"], card["accepted"]) == (13, 3)
    assert [r["id"] for r in card["recent"]] == [again["id"], first["id"]]
    assert card["sets"] == [{"id": first["set_id"], "name": "raw 먼저", "dataset": "raw", "todo": 5, "autolabeled": True}]
    assert board["totals"]["runs"] == 2 and board["totals"]["todo"] == 5
    sets = hub.get("/api/autolabel/targets", params={"project_id": project}).json()["sets"]
    assert [(s["id"], s["todo"], s["items"]) for s in sets] == [(first["set_id"], 5, 8)]


def test_autolabel_runs_on_a_server_picked_which_receives_the_images_first(two_nodes, tmp_path):  # noqa: F811
    """As a training or an evaluation: the images stay where they were registered (a storage server here, not one
    for 오토라벨), and the server picked receives the set's images still to label before it runs."""
    client = two_nodes
    data, worker = client.data.config.node_id, client.train.config.node_id
    client.patch(f"/api/nodes/{data}", json={"roles": ["compute", "storage"]})  # trains, holds images, no 오토라벨
    client.patch(f"/api/nodes/{worker}", json={"roles": ["autolabel"]})
    project = client.post("/api/projects", json={"name": "어디서 오토라벨"}).json()
    body = {"node_id": data, "path": str(make_yolo(tmp_path / "ds")), "dataset": "street", "version": "v1"}
    labelled = _wait(client, client.post(f"/api/projects/{project['id']}/imports", json=body).json()["id"])
    run = client.post(
        "/api/runs", json={"release_id": labelled["result"]["release_id"], "template": "dummy", "gpus": 0}
    ).json()
    assert _wait(client, run["job_id"])["status"] == "succeeded", _logs(client, run["job_id"])
    model = client.get(f"/api/runs/{run['id']}").json()["model"]
    root = make_yolo(tmp_path / "raw")
    shutil.rmtree(root / "labels")
    (root / "data.yaml").unlink()
    body = {"node_id": data, "path": str(root), "dataset": "raw", "version": "v1", "classes": CLASSES}
    imported = _wait(client, client.post(f"/api/projects/{project['id']}/imports", json=body).json()["id"])
    raw = imported["result"]["release_id"]

    # ---- where it can run: not where the images are (no 오토라벨 role), so the other server, which receives them
    servers = client.get("/api/autolabel/models", params={"release_id": raw}).json()["servers"]
    by = {s["id"]: s for s in servers["nodes"]}
    assert servers["default"] == worker and servers["holder"]["id"] == data
    assert by[data]["holds_images"] and not by[data]["fits"] and "오토라벨용" in by[data]["why"]
    assert by[worker]["transfer"] and by[worker]["fits"], by[worker]["why"]
    refused = client.post("/api/autolabel", json={"release_id": raw, "model_id": model["id"], "gpus": 0})
    assert refused.status_code == 400 and "오토라벨용 서버가 아닙니다" in refused.json()["detail"]
    assert client.get("/api/label-sets", params={"project_id": project["id"]}).json() == []
    wrong = {"release_id": raw, "model_id": model["id"], "node_id": "nope"}
    assert client.post("/api/autolabel", json=wrong).status_code == 404

    # ---- on the server picked: the set's images come first (only those), then it labels them
    body = {"release_id": raw, "model_id": model["id"], "accept": 0.9, "gpus": 0, "node_id": worker, "split": "train"}
    started = client.post("/api/autolabel", json=body)
    assert started.status_code == 200, started.text
    first = started.json()
    done = _wait(client, first["job_id"])
    assert done["status"] == "succeeded", _logs(client, first["job_id"])
    assert done["node_id"] == worker
    (sync,) = [j for j in client.get("/api/jobs?limit=30").json() if j["type"] == "data.sync"]
    assert sync["status"] == "succeeded" and sync["node_id"] == worker and sync["params"]["set_id"] == first["set_id"]
    made = client.get(f"/api/label-sets/{first['set_id']}").json()
    assert sync["result"]["items"] == made["items"] < 8  # the train split's only, not the whole version
    (row,) = client.get("/api/autolabel/runs", params={"set_id": first["set_id"]}).json()
    assert row["node_name"] == "train-node" and row["result"]["images"] == made["items"]
    # some of the version's images: the server does not count as holding the version
    again = client.get("/api/autolabel/models", params={"set_id": first["set_id"]}).json()["servers"]
    assert next(s for s in again["nodes"] if s["id"] == worker)["transfer"]
