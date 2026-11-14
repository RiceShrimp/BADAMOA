"""Training: dataset export, the run → metrics → model flow with a real Node Agent, the registry, MCP.

These use the built-in ``dummy`` template, which goes through the same path as a real one
(export, metric reports, weights upload, model registration) without torch.
The real YOLO check lives in ``test_real_data.py`` and is opt-in.
"""

from __future__ import annotations

import hashlib
import io
import threading
import time
import zipfile

import httpx
import pytest
from conftest import ADMIN
from sample_data import CLASSES, make_yolo
from test_e2e import INIT, _rpc

from badamoa.core.jobs import validate_params
from badamoa.node import agent as node_agent
from badamoa.node import exporter

# ---- export (pure, no Hub) ------------------------------------------------------------

SPACE = [{"id": i, "name": name} for i, name in enumerate(CLASSES)]
ALL = [0, 1, 2]


def _sha(name):
    return hashlib.sha256(name.encode()).hexdigest()


def _items(root, specs):
    """specs: (name, split, state, boxes, labelled_class_ids)"""
    items = []
    for n, (name, split, state, boxes, covered) in enumerate(specs):
        path = root / "src" / f"{name}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode() * 10)
        items.append(
            {
                "id": n + 1,
                "sha256": _sha(name),
                "rel_path": f"{name}.jpg",
                "bytes": path.stat().st_size,
                "location_id": "loc",
                "split": split,
                "state": state,
                "boxes": boxes,
                "labelled_class_ids": covered,
            }
        )
    return items, {"loc": str(root / "src")}


def test_export_writes_pinned_labels_and_remaps_a_class_subset(tmp_path):
    items, locations = _items(
        tmp_path,
        [
            ("a", "train", "labelled", [[0, 0.5, 0.5, 0.2, 0.2], [2, 0.3, 0.3, 0.1, 0.1], [1, 0.1, 0.1, 0.1, 0.1]], ALL),
            ("b", "train", "background", [], [0, 1, 2]),
            ("c", "train", "unlabelled", [], []),
            ("d", "val", "labelled", [[2, 0.5, 0.5, 0.4, 0.4]], [0, 1, 2]),
            ("e", "", "labelled", [[0, 0.5, 0.5, 0.4, 0.4]], [0]),  # only "person" was labelled exhaustively
        ],
    )
    out = tmp_path / "out"
    summary = exporter.build(items, SPACE, locations, out, exporter.ExportOptions(classes=[2, 0]))

    assert summary["classes"] == ["person", "dog"] and summary["class_ids"] == [0, 2]  # label-space order, renumbered
    assert summary["items"] == {"train": 2, "val": 1, "test": 0} and summary["val_source"] == "release"
    assert summary["excluded"] == {"unlabelled": 1, "partial": 1, "missing": 0}
    labels = {p.stem: p.read_text() for p in (out / "labels").rglob("*.txt")}
    kept = ["0 0.500000 0.500000 0.200000 0.200000", "1 0.300000 0.300000 0.100000 0.100000"]
    assert labels[_sha("a")].split("\n")[:2] == kept  # the "car" box is dropped, "dog" becomes class 1
    assert labels[_sha("b")] == ""  # background: an empty label file
    assert (out / "images" / "train" / f"{_sha('a')}.jpg").read_bytes() == b"a" * 10
    yaml = (out / "data.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in yaml and "val: images/val" in yaml and '0: "person"' in yaml and '1: "dog"' in yaml


def test_export_unlabelled_as_background_holdout_and_missing_files(tmp_path):
    specs = [(f"n{i}", "train", "labelled", [[1, 0.5, 0.5, 0.2, 0.2]], [0, 1, 2]) for i in range(40)]
    specs.append(("nolabel", "train", "unlabelled", [], []))
    items, locations = _items(tmp_path, specs)
    (tmp_path / "src" / "n0.jpg").unlink()

    kept = exporter.build(items, SPACE, locations, tmp_path / "a", exporter.ExportOptions(unlabelled="background"))
    assert kept["excluded"] == {"unlabelled": 0, "partial": 0, "missing": 1} and kept["as_background"] == 1
    assert kept["val_source"] == "holdout" and 0 < kept["items"]["val"] < kept["items"]["train"]
    again = exporter.build(items, SPACE, locations, tmp_path / "b", exporter.ExportOptions(unlabelled="background"))
    assert again["items"] == kept["items"]  # the held-out set depends only on the image hashes

    no_val = exporter.build(items, SPACE, locations, tmp_path / "c", exporter.ExportOptions(val_fraction=0))
    assert no_val["val_source"] == "train" and "val: images/train" in (tmp_path / "c" / "data.yaml").read_text()

    with pytest.raises(ValueError, match="학습에 쓸 이미지가 없습니다"):
        exporter.build(items[-1:], SPACE, locations, tmp_path / "d", exporter.ExportOptions())


def test_export_never_links_a_file_the_trainer_would_rewrite(tmp_path):
    items, locations = _items(tmp_path, [(name, "train", "labelled", [[0, 0.5, 0.5, 0.2, 0.2]], ALL) for name in "abc"])
    whole, cut = tmp_path / "src" / "a.jpg", tmp_path / "src" / "b.jpg"
    whole.write_bytes(b"\xff\xd8" + b"image" + b"\xff\xd9")
    cut.write_bytes(b"\xff\xd8" + b"truncated jpeg")  # no end marker: Ultralytics would save over it
    for item, path in zip(items, (whole, cut), strict=False):
        item["bytes"] = path.stat().st_size

    out = tmp_path / "out"
    summary = exporter.build(items, SPACE, locations, out, exporter.ExportOptions(val_fraction=0))
    assert summary["files"].get("copy") == 1 and sum(summary["files"].values()) == 3
    exported = out / "images" / "train" / f"{_sha('b')}.jpg"
    exported.write_bytes(b"rewritten by the trainer")
    assert cut.read_bytes() == b"\xff\xd8" + b"truncated jpeg"  # the registered original is untouched


def test_job_params_accept_lists_numbers_and_objects():
    params = validate_params("train.run", {"classes": [0, 2], "val_fraction": "0.1", "params": {"sleep": 0}})
    assert params["classes"] == [0, 2] and params["val_fraction"] == 0.1 and params["params"] == {"sleep": 0}
    with pytest.raises(ValueError):
        validate_params("train.run", {"classes": "0,2"})
    with pytest.raises(ValueError):
        validate_params("train.run", {"val_fraction": 2})


# ---- runs and models with a real node -------------------------------------------------


@pytest.fixture
def hub(live_hub, settings, tmp_path, monkeypatch):
    monkeypatch.setattr(node_agent, "HEARTBEAT_INTERVAL_S", 1.0)
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    with httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=60) as client:
        join = client.post("/api/nodes/join-tokens", json={}).json()
        home = tmp_path / "node"
        home.mkdir()
        runner = node_agent.NodeAgent(node_agent.join(live_hub, join["token"], home, name="train-node"), home)
        threading.Thread(target=runner.run_forever, daemon=True).start()
        client.node_id = runner.config.node_id
        client.node_home = home
        client.hub_home = settings.home
        yield client
        runner.stop()


def _wait(hub, job_id, timeout=120):
    deadline = time.monotonic() + timeout
    job = hub.get(f"/api/jobs/{job_id}").json()
    while time.monotonic() < deadline and job["status"] in ("queued", "running"):
        time.sleep(0.2)
        job = hub.get(f"/api/jobs/{job_id}").json()
    return job


def _logs(hub, job_id):
    return "\n".join(line["text"] for line in hub.get(f"/api/jobs/{job_id}/logs").json())


@pytest.fixture
def release(hub, tmp_path):
    project = hub.post("/api/projects", json={"name": "주차장"}).json()
    body = {"node_id": hub.node_id, "path": str(make_yolo(tmp_path / "ds")), "dataset": "street", "version": "v1"}
    job = _wait(hub, hub.post(f"/api/projects/{project['id']}/imports", json=body).json()["id"])
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    return {"project_id": project["id"], "id": job["result"]["release_id"]}


def _train(hub, release, params=None, **settings):
    body = {"release_id": release["id"], "template": "dummy", "params": params or {}, "gpus": 0, **settings}
    resp = hub.post("/api/runs", json=body)
    assert resp.status_code == 200, resp.text
    run = resp.json()
    job = _wait(hub, run["job_id"])
    return run, job


def test_run_reports_metrics_and_registers_a_model_with_lineage(hub, release):
    run, job = _train(hub, release, {"model": "yolo11n.pt", "ceiling": 0.5})
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    assert run["name"] == "street-v1-yolo11n" and run["status"] in ("queued", "running")

    done = hub.get(f"/api/runs/{run['id']}").json()
    assert done["status"] == "succeeded" and done["epoch"] == 5 and done["epochs"] == 5
    assert done["history"]["steps"] == [1, 2, 3, 4, 5]
    curve = done["history"]["series"]["map50_95"]
    assert curve == sorted(curve) and done["metrics"]["map50_95"] == curve[-1]
    # 8 images: one unlabelled is left out, the background one stays; val comes from the release
    assert done["export"]["items"] == {"train": 5, "val": 2, "test": 0}
    assert done["export"]["excluded"]["unlabelled"] == 1 and done["export"]["val_source"] == "release"
    assert [c["name"] for c in done["class_metrics"]] == CLASSES
    assert not (hub.node_home / "runs" / job["id"]).exists()  # the working folder is cleaned up

    # ---- nothing stays only on the server: what the code left in its output folder is on the Hub,
    # with the folders it was written in (the weights are the model, not listed twice)
    base = str(hub.base_url).rstrip("/")
    files = {f["name"]: f for f in done["files"]}
    assert sorted(files) == ["epoch.txt", "logs/history.csv"], files
    assert "나머지 산출물 2개" in _logs(hub, job["id"])
    table = httpx.get(base + files["logs/history.csv"]["url"])
    assert table.text.startswith("epoch,map50_95\n5,") and "history.csv" in table.headers["content-disposition"]
    assert "logs" not in table.headers["content-disposition"]
    bundle = httpx.get(base + done["files_url"])
    assert bundle.status_code == 200 and "street-v1-yolo11n.zip" in bundle.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
        assert sorted(archive.namelist()) == ["epoch.txt", "logs/history.csv"] and archive.read("epoch.txt") == b"5"
    usage = hub.get("/api/storage").json()
    assert usage["path"] == str(hub.hub_home.resolve()) and usage["free"] > 0
    assert usage["parts"]["models"] > 0 and usage["parts"]["outputs"] == sum(f["bytes"] for f in done["files"])
    assert usage["bytes"] >= usage["parts"]["models"] + usage["parts"]["outputs"]

    # ---- the model, and the way back from it
    model = hub.get(f"/api/models/{done['model']['id']}").json()
    assert (model["name"], model["version"], model["arch"]) == ("street-yolo11n", 1, "yolo11n")
    assert model["classes"] == CLASSES and model["metrics"]["map50_95"] == curve[-1]
    assert model["run"]["id"] == run["id"] and model["release"]["id"] == release["id"]
    assert model["config"]["epochs"] == 5 and model["template"] == {"name": "dummy", "title": "테스트 학습", "version": "2"}
    weights = httpx.get(str(hub.base_url).rstrip("/") + model["download_url"])
    assert weights.status_code == 200 and len(weights.content) == model["bytes"] and b'"dummy": true' in weights.content
    assert "street-yolo11n-v1.pt" in weights.headers["content-disposition"]
    assert httpx.get(str(hub.base_url).rstrip("/") + model["download_url"].replace("sig=", "sig=0")).status_code == 403

    # ---- same settings again → a second version; aliases move within the family
    again = hub.post(f"/api/runs/{run['id']}/rerun").json()
    assert again["name"] == "street-v1-yolo11n-2" and again["source_run_id"] == run["id"]
    assert _wait(hub, again["job_id"])["status"] == "succeeded"
    models = hub.get(f"/api/projects/{release['project_id']}/models").json()
    assert [(m["name"], m["version"]) for m in models] == [("street-yolo11n", 2), ("street-yolo11n", 1)]
    v2, v1 = models
    assert hub.post(f"/api/models/{v1['id']}/alias", json={"alias": "champion"}).json()["alias"] == "champion"
    assert hub.post(f"/api/models/{v2['id']}/alias", json={"alias": "champion"}).json()["alias"] == "champion"
    assert hub.get(f"/api/models/{v1['id']}").json()["alias"] == ""
    for reserved in ("latest", "v3", "two words"):  # version spellings are not aliases
        assert hub.post(f"/api/models/{v1['id']}/alias", json={"alias": reserved}).status_code == 400

    runs = hub.get(f"/api/projects/{release['project_id']}/runs").json()
    assert [r["name"] for r in runs] == ["street-v1-yolo11n-2", "street-v1-yolo11n"]
    assert runs[0]["model"]["alias"] == "champion"
    assert hub.delete(f"/api/runs/{run['id']}").status_code == 200
    assert len(hub.get(f"/api/projects/{release['project_id']}/runs").json()) == 1
    assert hub.get(f"/api/models/{v1['id']}").json()["run"] is None  # the model outlives its run


def test_class_subset_and_failed_run(hub, release):
    run, job = _train(hub, release, classes=[1], model_name="cars", unlabelled="background")
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    done = hub.get(f"/api/runs/{run['id']}").json()
    assert done["export"]["classes"] == ["car"] and done["export"]["items"]["train"] == 6
    assert (done["model"]["name"], done["model"]["version"]) == ("cars", 1)

    failed, job = _train(hub, release, {"fail": True})
    assert job["status"] == "failed" and "의도적 실패" in _logs(hub, job["id"])
    view = hub.get(f"/api/runs/{failed['id']}").json()
    assert view["status"] == "failed" and view["model"] is None and "종료 코드" in view["error"]

    assert hub.post("/api/runs", json={"release_id": release["id"], "classes": [9]}).status_code == 400
    assert hub.post("/api/runs", json={"release_id": "nope"}).status_code == 404
    assert hub.post("/api/runs", json={"release_id": release["id"], "template": "sklearn"}).status_code == 404
    unknown = hub.post("/api/runs", json={"release_id": release["id"], "template": "dummy", "params": {"lr": 1}})
    assert unknown.status_code == 400 and "lr" in unknown.json()["detail"]
    low = hub.post("/api/runs", json={"release_id": release["id"], "template": "dummy", "params": {"epochs": 0}})
    assert low.status_code == 400 and "1 이상" in low.json()["detail"]


def test_training_through_mcp(hub, release):
    token = hub.post("/api/tokens", json={"name": "claude-code"}).json()["token"]
    assert _rpc(hub, "initialize", INIT, token).status_code == 200

    def call(name, arguments):
        res = _rpc(hub, "tools/call", {"name": name, "arguments": arguments}, token).json()["result"]
        assert not res.get("isError"), res
        return res["structuredContent"]

    names = [t["name"] for t in call("list_model_templates", {})["result"]]
    assert names == ["yolo", "yolo-seg", "classify", "torchvision"]  # defaults first; the test helper is not offered
    run = call("start_training", {"release_id": release["id"], "template": "dummy", "params": {"epochs": 3}, "gpus": 0})
    assert run["created_via"] == "mcp"
    assert call("wait_for_job", {"job_id": run["job_id"], "timeout_seconds": 120})["status"] == "succeeded"
    detail = call("get_training_run", {"run_id": run["id"]})
    assert detail["history"]["steps"] == [1, 2, 3] and detail["model"]["version"] == 1
    assert call("list_training_runs", {"project_id": release["project_id"]})["result"][0]["id"] == run["id"]
    model = call("list_models", {"project_id": release["project_id"]})["result"][0]
    assert call("set_model_alias", {"model_id": model["id"], "alias": "challenger"})["alias"] == "challenger"
    fetched = call("get_model", {"model_id": model["id"]})
    assert fetched["alias"] == "challenger" and "download_url" not in fetched and fetched["release"]["version"] == "v1"

    # notes, tags and aliases are there for agents as well
    noted = call("update_training_run", {"run_id": run["id"], "note": "기준선", "tags": ["baseline"]})
    assert (noted["note"], noted["tags"]) == ("기준선", ["baseline"])
    assert [
        r["id"] for r in call("list_training_runs", {"project_id": release["project_id"], "tag": "baseline"})["result"]
    ] == [run["id"]]
    call("set_model_alias", {"model_id": model["id"], "alias": "production"})
    call("update_model", {"model_id": model["id"], "note": "현장 배포용", "tags": ["edge"]})
    found = call("list_models", {"project_id": release["project_id"], "name": model["name"], "alias": "production"})
    assert [(m["id"], m["aliases"], m["note"], m["tags"]) for m in found["result"]] == [
        (model["id"], ["challenger", "production"], "현장 배포용", ["edge"])
    ]
    left = call("set_model_alias", {"model_id": model["id"], "alias": "challenger", "remove": True})
    assert left["aliases"] == ["production"] and left["alias"] == "production"


def test_notes_tags_and_aliases(hub, release):
    project = release["project_id"]
    first, job = _train(hub, release, {"epochs": 2})
    assert job["status"] == "succeeded"
    second = hub.post(
        "/api/runs",
        json={
            "release_id": release["id"],
            "template": "dummy",
            "params": {"epochs": 2},
            "gpus": 0,
            "tags": ["야간", " baseline "],
        },
    ).json()
    assert second["tags"] == ["baseline", "야간"]
    assert _wait(hub, second["job_id"])["status"] == "succeeded"

    # ---- a run's name, note and tags
    edited = hub.patch(
        f"/api/runs/{first['id']}", json={"name": "lr 낮춤", "note": "lr0 0.001로", "tags": ["baseline", "Baseline", "lr"]}
    )
    assert edited.status_code == 200, edited.text
    assert (edited.json()["name"], edited.json()["note"], edited.json()["tags"]) == (
        "lr 낮춤",
        "lr0 0.001로",
        ["baseline", "lr"],
    )
    assert hub.patch(f"/api/runs/{first['id']}", json={"name": second["name"]}).status_code == 409
    assert hub.patch(f"/api/runs/{first['id']}", json={"name": "  "}).status_code == 400
    assert hub.patch(f"/api/runs/{first['id']}", json={"tags": [f"t{i}" for i in range(21)]}).status_code == 400
    assert hub.patch(f"/api/runs/{first['id']}", json={"note": "lr0 0.001로, warmup 없음"}).json()["tags"] == [
        "baseline",
        "lr",
    ]

    runs = lambda **query: [r["id"] for r in hub.get(f"/api/projects/{project}/runs", params=query).json()]  # noqa: E731
    assert runs(tag="baseline") == [second["id"], first["id"]] and runs(tag="lr") == [first["id"]]
    assert runs(q="warmup") == [first["id"]] and runs(q="야간") == [second["id"]] and runs(q="낮춤") == [first["id"]]
    tags = hub.get(f"/api/projects/{project}/tags").json()
    assert tags["runs"] == [{"name": "baseline", "count": 2}, {"name": "lr", "count": 1}, {"name": "야간", "count": 1}]

    # Run again and continue keep the tags: they belong to the experiment.
    again = hub.post(f"/api/runs/{first['id']}/rerun").json()
    assert again["tags"] == ["baseline", "lr"] and again["note"] == ""
    hub.post(f"/api/jobs/{again['job_id']}/cancel")
    _wait(hub, again["job_id"])

    # ---- a model's description, tags and aliases
    model_id = hub.get(f"/api/runs/{first['id']}").json()["model"]["id"]
    other_id = hub.get(f"/api/runs/{second['id']}").json()["model"]["id"]
    described = hub.patch(f"/api/models/{model_id}", json={"note": "야간 데이터 보강 전 기준", "tags": ["기준"]}).json()
    assert (described["note"], described["tags"]) == ("야간 데이터 보강 전 기준", ["기준"])
    for alias in ("champion", "production", "현장-1"):
        assert hub.post(f"/api/models/{model_id}/alias", json={"alias": alias}).status_code == 200
    assert hub.get(f"/api/models/{model_id}").json()["aliases"] == ["champion", "production", "현장-1"]
    moved = hub.post(f"/api/models/{other_id}/alias", json={"alias": "production"}).json()
    assert moved["aliases"] == ["production"]  # one version per alias in a family: it moved
    assert hub.get(f"/api/models/{model_id}").json()["aliases"] == ["champion", "현장-1"]
    removed = hub.delete(f"/api/models/{model_id}/alias/champion").json()
    assert removed["aliases"] == ["현장-1"] and removed["alias"] == "현장-1"
    by_alias = hub.get(f"/api/projects/{project}/models", params={"name": moved["name"], "alias": "production"}).json()
    assert [m["id"] for m in by_alias] == [other_id]
    assert [m["id"] for m in hub.get(f"/api/projects/{project}/models", params={"tag": "기준"}).json()] == [model_id]
    assert [m["id"] for m in hub.get(f"/api/projects/{project}/models", params={"q": "보강"}).json()] == [model_id]
    assert hub.get(f"/api/projects/{project}/tags").json()["models"] == [{"name": "기준", "count": 1}]
    assert "production" in hub.get("/api/model-aliases").json()

    # Deleting takes the tags along.
    assert hub.delete(f"/api/runs/{first['id']}").status_code == 200
    assert hub.get(f"/api/projects/{project}/tags").json()["runs"] == [
        {"name": "baseline", "count": 2},
        {"name": "lr", "count": 1},
        {"name": "야간", "count": 1},
    ]  # the run started again keeps its own copy


def test_evaluate_export_and_test_a_model(hub, release, tmp_path):
    run, job = _train(hub, release, {"model": "yolo11n.pt", "imgsz": 320})
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    model_id = hub.get(f"/api/runs/{run['id']}").json()["model"]["id"]
    assert hub.get(f"/api/models/{model_id}").json()["actions"] == ["eval", "export", "predict"]

    def finished(op):
        job = _wait(hub, op["job_id"])
        assert job["status"] == "succeeded", _logs(hub, job["id"])
        return next(o for o in hub.get(f"/api/models/{model_id}/ops").json() if o["id"] == op["id"])

    # ---- evaluate on a release: the split the user asked for, with the settings the model was trained with
    resp = hub.post(f"/api/models/{model_id}/eval", json={"release_id": release["id"], "split": "val", "gpus": 0})
    assert resp.status_code == 200, resp.text
    evaluation = finished(resp.json())
    assert evaluation["kind"] == "eval" and evaluation["release"]["version"] == "v1"
    assert evaluation["result"]["metrics"] == {"map50_95": 0.4, "map50": 0.6, "images": 2.0}
    assert [c["name"] for c in evaluation["result"]["classes"]] == CLASSES
    assert "'imgsz': 320" in _logs(hub, evaluation["job_id"])
    assert hub.post(f"/api/models/{model_id}/eval", json={"release_id": release["id"], "split": "dev"}).status_code == 400
    assert hub.post(f"/api/models/{model_id}/eval", json={"release_id": "nope"}).status_code == 404

    # ---- export: the file comes back and can be downloaded
    exported = finished(hub.post(f"/api/models/{model_id}/export", json={"format": "onnx"}).json())
    assert exported["result"]["file"] == "model.onnx" and [f["name"] for f in exported["files"]] == ["model.onnx"]
    download = httpx.get(str(hub.base_url).rstrip("/") + exported["files"][0]["url"])
    assert download.content.startswith(b"exported:") and download.headers["content-disposition"].startswith("attachment")
    assert hub.post(f"/api/models/{model_id}/export", json={"format": "exe"}).status_code == 400

    # ---- test on uploaded images
    image = next((tmp_path / "ds" / "images" / "val").iterdir())
    uploads = [
        hub.post("/api/uploads", params={"name": f"sample{i}.jpg"}, content=image.read_bytes()).json() for i in range(2)
    ]
    tested = finished(hub.post(f"/api/models/{model_id}/predict", json={"images": uploads, "conf": 0.5}).json())
    assert [p["image"] for p in tested["result"]["predictions"]] == ["00_sample0.jpg", "01_sample1.jpg"]
    assert tested["result"]["predictions"][0]["boxes"] == [[0, 0.5, 0.5, 0.5, 0.2, 0.2]]
    assert [(f["name"], f["kind"]) for f in tested["files"]] == [("00_sample0.jpg", "image"), ("01_sample1.jpg", "image")]
    shown = httpx.get(str(hub.base_url).rstrip("/") + tested["files"][0]["url"])
    assert shown.content == image.read_bytes() and shown.headers["content-type"] == "image/jpeg"
    assert "images" not in tested["params"] and tested["params"]["conf"] == 0.5
    assert hub.post(f"/api/models/{model_id}/predict", json={"images": []}).status_code == 400
    missing = {"images": [{"name": "x.jpg", "blob": "0" * 64}]}
    assert hub.post(f"/api/models/{model_id}/predict", json=missing).status_code == 400
    assert [o["kind"] for o in hub.get(f"/api/models/{model_id}/ops").json()] == ["predict", "export", "eval"]


def test_continue_from_a_registered_model_and_server_load(hub, release):
    first, job = _train(hub, release, {"model": "yolo11n.pt"})
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    model = hub.get(f"/api/runs/{first['id']}").json()["model"]

    run, job = _train(hub, release, {"model": "yolo11n.pt", "epochs": 2}, base_model_id=model["id"])
    logs = _logs(hub, job["id"])
    assert job["status"] == "succeeded", logs
    assert "등록된 모델의 가중치에서 시작합니다" in logs and '이어서 학습: {"dummy": true' in logs  # the code got the file
    done = hub.get(f"/api/runs/{run['id']}").json()
    assert done["base_model"] == {"id": model["id"], "name": "street-yolo11n", "version": 1}
    assert done["model"]["version"] == 2 and "base_model" not in [s["name"] for s in done["settings"]]
    again = hub.post(f"/api/runs/{run['id']}/rerun").json()  # repeating it starts from the same model
    assert again["base_model"]["id"] == model["id"]
    hub.post(f"/api/jobs/{again['job_id']}/cancel")
    unknown = {"release_id": release["id"], "template": "dummy", "base_model_id": "nope"}
    assert hub.post("/api/runs", json=unknown).status_code == 404

    # ---- the server's load while the run was on it (minute buckets; a heartbeat arrived during the run)
    system = hub.get(f"/api/runs/{run['id']}/system").json()
    assert system["step_s"] == 60 and system["gpu_indices"] == []
    assert len(system["timestamps"]) >= 1 and {"cpu", "mem"} <= set(system["series"])
    assert 0 <= system["series"]["mem"]["avg"][0] <= 100


def test_sdk_callbacks_for_lightning_and_hugging_face(monkeypatch, capsys):
    import importlib
    import json
    import sys
    import types

    from badamoa import templates as shipped

    lightning = types.ModuleType("lightning.pytorch.callbacks")
    lightning.Callback = type("Callback", (), {})
    transformers = types.ModuleType("transformers")
    transformers.TrainerCallback = type("TrainerCallback", (), {})
    for name, module in {
        "lightning": types.ModuleType("lightning"),
        "lightning.pytorch": types.ModuleType("lightning.pytorch"),
        "lightning.pytorch.callbacks": lightning,
        "transformers": transformers,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.syspath_prepend(str(shipped.SDK))
    bm = importlib.import_module("badamoa_sdk")
    bm.load_config([])
    bm.init()

    trainer = types.SimpleNamespace(sanity_checking=True, current_epoch=0, max_epochs=5, callback_metrics={"val_acc": 0.5})
    callback = bm.lightning_callback()
    callback.on_validation_epoch_end(trainer, None)  # Lightning's sanity check is not an epoch
    trainer.sanity_checking = False
    callback.on_validation_epoch_end(trainer, None)
    state = types.SimpleNamespace(global_step=40, max_steps=100)
    bm.hf_callback().on_log(None, state, None, logs={"loss": 1.5, "learning_rate": 0.0001})

    events = [json.loads(line[10:]) for line in capsys.readouterr().out.splitlines() if line.startswith("@@badamoa ")]
    assert events[1] == {"event": "epoch", "epoch": 1, "epochs": 5, "metrics": {"val_acc": 0.5}}
    assert events[2] == {"event": "log", "step": 40, "metrics": {"loss": 1.5, "learning_rate": 0.0001}}
    assert events[3] == {"event": "progress", "current": 40, "total": 100}
    bm.load_config([])


def test_resume_an_interrupted_run_from_its_checkpoint(hub, release):
    started = hub.post(
        "/api/runs",
        json={"release_id": release["id"], "template": "dummy", "params": {"epochs": 40, "sleep": 0.5}, "gpus": 0},
    ).json()
    for _ in range(200):  # let it get a few epochs in, then stop it
        if hub.get(f"/api/runs/{started['id']}").json()["epoch"] >= 3:
            break
        time.sleep(0.1)
    assert hub.get(f"/api/runs/{started['id']}").json()["can_resume"] is False  # still running
    hub.post(f"/api/jobs/{started['job_id']}/cancel")
    assert _wait(hub, started["job_id"])["status"] == "canceled"
    stopped = hub.get(f"/api/runs/{started['id']}").json()
    assert stopped["can_resume"] is True and stopped["epoch"] >= 3
    assert stopped["checkpoint_epoch"] >= 3  # the node handed its last checkpoint to the Hub when it was stopped
    assert "마지막 체크포인트를 Hub에 보관했습니다" in _logs(hub, started["job_id"])
    # What it had written so far is kept with the stopped run as well.
    assert [f["name"] for f in stopped["files"]] == ["epoch.txt"]
    assert "그때까지의 산출물 1개" in _logs(hub, started["job_id"])

    resp = hub.post(f"/api/runs/{started['id']}/resume")
    assert resp.status_code == 200, resp.text
    run = resp.json()
    assert run["resumed_from"] == {"id": started["id"], "name": started["name"]} and run["config"]["epochs"] == 40
    for _ in range(150):  # 40 slow epochs: its first report is enough
        view = hub.get(f"/api/runs/{run['id']}").json()
        if view["history"]["steps"] or view["status"] not in ("queued", "running"):
            break
        time.sleep(0.2)
    hub.post(f"/api/jobs/{run['job_id']}/cancel")
    _wait(hub, run["job_id"])
    logs = _logs(hub, run["job_id"])
    assert "이 서버에 남아 있는 체크포인트" in logs and "체크포인트에서 이어 갑니다: epoch" in logs, logs
    assert view["history"]["steps"] and view["history"]["steps"][0] > 3  # it did not start over

    # The server lost its working folders (or another server takes over): the Hub's copy is used.
    import shutil

    shutil.rmtree(hub.node_home / "runs", ignore_errors=True)
    again = hub.post(f"/api/runs/{run['id']}/resume").json()
    for _ in range(150):
        view = hub.get(f"/api/runs/{again['id']}").json()
        if view["history"]["steps"] or view["status"] not in ("queued", "running"):
            break
        time.sleep(0.2)
    hub.post(f"/api/jobs/{again['job_id']}/cancel")
    _wait(hub, again["job_id"])
    assert "Hub에 보관된 체크포인트" in _logs(hub, again["job_id"])
    assert view["history"]["steps"][0] > run["config"].get("epochs", 0) - 40 and view["history"]["steps"][0] > 3

    done, job = _train(hub, release, {"epochs": 2})
    assert job["status"] == "succeeded"
    assert hub.post(f"/api/runs/{done['id']}/resume").status_code == 400  # a finished run has nothing to resume


def test_a_resumed_run_that_finishes_frees_the_checkpoints_it_came_from(hub, release):
    body = {"release_id": release["id"], "template": "dummy", "params": {"epochs": 14, "sleep": 0.5}, "gpus": 0}
    first = hub.post("/api/runs", json=body).json()
    for _ in range(200):
        if hub.get(f"/api/runs/{first['id']}").json()["epoch"] >= 2:
            break
        time.sleep(0.1)
    hub.post(f"/api/jobs/{first['job_id']}/cancel")
    assert _wait(hub, first["job_id"])["status"] == "canceled"
    assert hub.get(f"/api/runs/{first['id']}").json()["can_resume"] is True

    second = hub.post(f"/api/runs/{first['id']}/resume").json()
    assert _wait(hub, second["job_id"])["status"] == "succeeded", _logs(hub, second["job_id"])
    finished = hub.get(f"/api/runs/{second['id']}").json()
    assert finished["epoch"] == 14 and finished["model"] and finished["checkpoint_epoch"] is None
    assert sorted(f["name"] for f in finished["files"]) == ["epoch.txt", "logs/history.csv"]
    stopped = hub.get(f"/api/runs/{first['id']}").json()
    assert stopped["can_resume"] is False and stopped["checkpoint_epoch"] is None  # continued and done: nothing to keep
    assert [f["name"] for f in stopped["files"]] == ["epoch.txt"]  # what it had produced stays with it
    assert hub.get("/api/storage").json()["parts"]["checkpoints"] == 0
