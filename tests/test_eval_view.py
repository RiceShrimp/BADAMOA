"""An evaluation of its own: detail, running it again, and models compared on the same data."""

from __future__ import annotations

from test_training import _logs, _train, _wait, hub, release  # noqa: F401  (the fixtures)


def _eval(hub, model_id, release_id, **body):  # noqa: F811
    op = hub.post(f"/api/models/{model_id}/eval", json={"release_id": release_id, "gpus": 0, **body}).json()
    assert _wait(hub, op["job_id"])["status"] == "succeeded", _logs(hub, op["job_id"])
    return op


def test_an_evaluation_shows_reruns_and_compares(hub, release):  # noqa: F811
    first, _ = _train(hub, release, {"epochs": 1})
    second, _ = _train(hub, release, {"epochs": 1})
    a = hub.get(f"/api/runs/{first['id']}").json()["model"]
    b = hub.get(f"/api/runs/{second['id']}").json()["model"]
    op = _eval(hub, a["id"], release["id"], split="val")

    # ---- its own screen: where, how long, the numbers per class, the log
    detail = hub.get(f"/api/model-ops/{op['id']}").json()
    assert detail["status"] == "succeeded" and detail["model"]["id"] == a["id"] and detail["splits"] == ["val"]
    assert detail["node_name"] == "train-node" and detail["started_at"] and detail["finished_at"]
    assert detail["result"]["metrics"]["map50_95"] == 0.4 and detail["class_metrics"][0]["map50_95"] == 0.4
    assert detail["log_count"] > 0

    # ---- again, with the same settings, on the server it ran on
    again = hub.post(f"/api/model-ops/{op['id']}/rerun", json={}).json()
    assert again["id"] != op["id"] and again["params"]["split"] == "val"
    assert _wait(hub, again["job_id"])["status"] == "succeeded", _logs(hub, again["job_id"])
    assert hub.get(f"/api/jobs/{again['job_id']}").json()["gpus_requested"] == 0  # as the first one: no GPU
    assert hub.get(f"/api/jobs/{again['job_id']}").json()["node_id"] == hub.node_id

    # ---- several splits in one evaluation, and another model: the board of the same data
    both = _eval(hub, b["id"], release["id"], splits=["train", "val"])
    board = hub.get(f"/api/releases/{release['id']}/evaluations").json()
    assert board["splits"] == ["val", "train"] and board["release"]["version"] == "v1"
    val = [r for r in board["rows"] if r["split"] == "val"]
    assert [(r["model"]["id"], r["latest"]) for r in val] == [(b["id"], True), (a["id"], True), (a["id"], False)]
    assert val[0]["op_id"] == both["id"] and val[0]["class_metrics"] == []  # per class: of its first split (train)
    train = hub.get(f"/api/releases/{release['id']}/evaluations", params={"split": "train"}).json()["rows"]
    assert [(r["model"]["id"], len(r["class_metrics"]) > 0) for r in train] == [(b["id"], True)]

    # ---- a running one cannot be run again; the rest can
    queued = hub.post(f"/api/models/{a['id']}/eval", json={"release_id": release["id"], "split": "val", "gpus": 0}).json()
    if hub.get(f"/api/jobs/{queued['job_id']}").json()["status"] in ("queued", "running"):
        assert hub.post(f"/api/model-ops/{queued['id']}/rerun", json={}).status_code == 409
    _wait(hub, queued["job_id"])
