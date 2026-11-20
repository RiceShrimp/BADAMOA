"""Models of different projects compared on the same data; an evaluation deleted with what it kept."""

from __future__ import annotations

from sample_data import make_yolo
from test_labeling import _login
from test_training import _logs, _train, _wait, hub, release  # noqa: F401  (the fixtures)


def test_models_of_two_projects_compared_and_an_evaluation_deleted(hub, release, tmp_path):  # noqa: F811
    run, _ = _train(hub, release, {"epochs": 1})
    ours = hub.get(f"/api/runs/{run['id']}").json()["model"]
    other = hub.post("/api/projects", json={"name": "다른 현장"}).json()
    body = {"node_id": hub.node_id, "path": str(make_yolo(tmp_path / "other")), "dataset": "road", "version": "v1"}
    job = _wait(hub, hub.post(f"/api/projects/{other['id']}/imports", json=body).json()["id"])
    theirs_release = job["result"]["release_id"]
    their_run, _ = _train(hub, {"id": theirs_release, "project_id": other["id"]}, {"epochs": 1})
    theirs = hub.get(f"/api/runs/{their_run['id']}").json()["model"]

    # ---- a model evaluated on another project's version (same task and classes)
    on_theirs = {"release_id": theirs_release, "split": "val", "gpus": 0}
    op = hub.post(f"/api/models/{ours['id']}/eval", json=on_theirs).json()
    assert _wait(hub, op["job_id"])["status"] == "succeeded", _logs(hub, op["job_id"])
    mine = hub.post(f"/api/models/{theirs['id']}/eval", json=on_theirs).json()
    assert _wait(hub, mine["job_id"])["status"] == "succeeded"

    # ---- found in every project, compared side by side
    found = hub.get("/api/models-find", params={"q": "다른 현장"}).json()
    assert [m["id"] for m in found] == [theirs["id"]] and found[0]["project"]["name"] == "다른 현장"
    both = hub.get("/api/models-compare", params={"ids": f"{ours['id']},{theirs['id']}"}).json()
    assert [m["project"]["name"] for m in both["models"]] == ["주차장", "다른 현장"]
    [shared] = both["evaluations"]
    assert shared["release"]["project"]["name"] == "다른 현장" and shared["split"] == "val"
    assert set(shared["rows"]) == {ours["id"], theirs["id"]}
    board = hub.get(f"/api/releases/{theirs_release}/evaluations").json()["rows"]
    assert {r["model"]["id"] for r in board} == {ours["id"], theirs["id"]}

    # ---- deleted: by whoever ran it or an admin, with its predictions
    hub.post("/api/users", json={"username": "kim", "password": "kim-compare-pw"})
    kim = _login(hub, "kim", "kim-compare-pw")
    assert kim.delete(f"/api/model-ops/{op['id']}").status_code == 403
    assert hub.delete(f"/api/model-ops/{op['id']}").json() == {"ok": True}
    assert hub.get(f"/api/model-ops/{op['id']}").status_code == 404
    again = hub.get("/api/models-compare", params={"ids": f"{ours['id']},{theirs['id']}"}).json()
    assert list(again["evaluations"][0]["rows"]) == [theirs["id"]]
    assert all(p["op_id"] != op["id"] for p in hub.get(f"/api/releases/{theirs_release}/prediction-runs").json())


def test_a_model_of_other_data_checks_labels_with_a_warning_whatever_its_class_order(hub, release, tmp_path):  # noqa: F811
    from sample_data import CLASSES

    run, _ = _train(hub, release, {"epochs": 1})
    model = hub.get(f"/api/runs/{run['id']}").json()["model"]
    # another project whose data names the same classes in another order (the first two swapped)
    other = hub.post("/api/projects", json={"name": "다른 순서"}).json()
    root = make_yolo(tmp_path / "swapped")
    swap = {"0": "1", "1": "0"}
    for label in (root / "labels").rglob("*.txt"):
        lines = [line.split(" ", 1) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        label.write_text("".join(f"{swap.get(c, c)} {rest}\n" for c, rest in lines), encoding="utf-8")
    order = [CLASSES[1], CLASSES[0], *CLASSES[2:]]
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(order))
    (root / "data.yaml").write_text(f"path: .\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    body = {"node_id": hub.node_id, "path": str(root), "dataset": "swapped", "version": "v1"}
    theirs = _wait(hub, hub.post(f"/api/projects/{other['id']}/imports", json=body).json()["id"])["result"]["release_id"]

    # ---- 검수 → 모델로 검사하기 offers it, saying what it learned from and warning about it
    [offered] = [m for m in hub.get(f"/api/releases/{theirs}/eval-models").json()["models"] if m["id"] == model["id"]]
    assert offered["fits"] is True and offered["relation"] == "other_project"
    assert offered["trained"]["project"] == "주차장" and "다른 프로젝트의 데이터" in offered["warning"]
    own = hub.get(f"/api/releases/{release['id']}/eval-models").json()["models"][0]
    assert own["relation"] == "same_version" and own["warning"] == ""

    # ---- evaluated in the model's class order: the predictions come back in this version's class ids
    op = hub.post(f"/api/models/{model['id']}/eval", json={"release_id": theirs, "split": "val", "gpus": 0}).json()
    assert _wait(hub, op["job_id"])["status"] == "succeeded", _logs(hub, op["job_id"])
    space = {c["name"]: c["id"] for c in hub.get(f"/api/releases/{theirs}").json()["label_space"]["classes"]}
    assert [space[n] for n in CLASSES] != sorted(space[n] for n in CLASSES)  # the order really differs


def test_an_evaluation_exports_classes_in_the_models_order():
    from badamoa.node.exporter import ExportOptions, _chosen

    assert _chosen([0, 1, 2], ExportOptions(classes=[1, 0, 2], ordered=True)) == [1, 0, 2]
    assert _chosen([0, 1, 2], ExportOptions(classes=[1, 0])) == [0, 1]  # training: the release's order
    assert _chosen([0, 1, 2], ExportOptions()) == [0, 1, 2]
