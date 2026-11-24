"""Pipelines: steps chained on the Hub (train → evaluate · 검수 → promote through a gate → export), a failed step
stopping the run, stopping and going on from where it stopped, and runs started by a new version of the dataset.

Uses the ``dummy`` template and a real Node Agent, as ``test_training`` does. The dummy's evaluation scores
two thirds of the ``ceiling`` it trained to (0.6 → mAP50-95 0.4), so a gate can be made to pass or hold.
"""

from __future__ import annotations

import time

from test_e2e import INIT, _rpc
from test_training import _logs, _train, _wait, hub, release  # noqa: F401 (fixtures)

from badamoa.hub import pipelines


def _done(client, run_id, timeout=120):
    deadline = time.monotonic() + timeout
    view = client.get(f"/api/pipeline-runs/{run_id}").json()
    while time.monotonic() < deadline and view["status"] == "running":
        time.sleep(0.3)
        view = client.get(f"/api/pipeline-runs/{run_id}").json()
    return view


def _train_step(ceiling=0.6, **params):
    return {"template": "dummy", "params": {"ceiling": ceiling, **params}, "gpus": 0}


EVAL = {"kind": "eval", "config": {"split": "auto", "review": True}}
PROMOTE = {"kind": "promote", "config": {"alias": "SOTA"}}
EXPORT = {"kind": "export", "config": {"format": "onnx"}}


def test_a_new_training_goes_on_to_evaluate_review_promote_and_export(hub, release):  # noqa: F811
    body = {"release_id": release["id"], "train": _train_step(), "after": [EVAL, PROMOTE, EXPORT]}
    started = hub.post("/api/pipeline-runs", json=body)
    assert started.status_code == 200, started.text
    run = started.json()
    assert [s["kind"] for s in run["steps"]] == ["train", "eval", "promote", "export"]
    assert run["steps"][0]["status"] == "running" and run["steps"][1]["status"] == "pending"
    assert run["trigger"] == "training" and run["pipeline"] is None and run["name"] == run["steps"][0]["run"]["name"]
    assert run["steps"][1]["label"] == "평가 · 검수" and run["steps"][2]["summary"] == "더 나으면 SOTA"
    # the training run's screen knows what follows it
    follow = hub.get(f"/api/runs/{run['train_run_id']}/follow").json()
    assert follow["pipeline"]["id"] == run["id"]

    done = _done(hub, run["id"])
    assert done["status"] == "succeeded", [(s["status"], s["message"], s.get("job")) for s in done["steps"]]
    assert [s["status"] for s in done["steps"]] == ["succeeded"] * 4
    train, evaluated, promoted, exported = done["steps"]
    model = train["model"]
    assert train["message"] == f"{model['name']} v1 등록"
    # evaluated on the version's test split, or val when it has none; with 검수: every split in one evaluation
    assert evaluated["split"] == "val" and evaluated["metric"]["value"] == 0.4
    assert evaluated["message"] == "mAP50-95 0.400 (v1 val)" and evaluated["review_release_id"] == release["id"]
    op = hub.get(f"/api/model-ops/{evaluated['op_id']}").json()
    assert op["splits"] == ["val", "train"] and op["predicted"] == 8
    checks = hub.get(f"/api/releases/{release['id']}/checks").json()
    assert checks[0]["id"] == evaluated["op_id"]  # 검수's newest model check is this model's
    # no SOTA yet: the first good model takes the alias
    assert promoted["outcome"] == "promoted" and "SOTA이(가) 없어" in promoted["message"]
    assert hub.get(f"/api/models/{model['id']}").json()["aliases"] == ["SOTA"]
    assert exported["message"] == "model.onnx 만듦"
    follow = hub.get(f"/api/runs/{run['train_run_id']}/follow").json()
    assert follow["evaluations"] == 1

    # ---- the same steps after a training started on its own (the nudge on a finished run)
    plain, job = _train(hub, release, {"ceiling": 0.3})
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    assert hub.get(f"/api/runs/{plain['id']}/follow").json() == {"pipeline": None, "evaluations": 0}
    after = {"run_id": plain["id"], "after": [{"kind": "eval", "config": {"split": "val"}}, PROMOTE]}
    late = hub.post("/api/pipeline-runs", json=after).json()
    assert late["trigger"] == "training" and late["steps"][0]["summary"] == "따로 시작한 학습"
    late = _done(hub, late["id"])
    # the gate: 0.2 does not beat the SOTA's 0.4 on the same data, so the alias stays and the run is held
    assert late["status"] == "held" and late["steps"][2]["outcome"] == "held"
    assert late["steps"][2]["message"] == "mAP50-95 0.200 ≤ v1 0.400: SOTA은(는) v1 그대로"
    assert late["steps"][2]["holder"]["version"] == 1
    assert hub.get(f"/api/models/{model['id']}").json()["aliases"] == ["SOTA"]
    # one going pipeline per training: asking twice while the first goes is refused
    slow, _ = _train_slowly(hub, release)
    first = hub.post("/api/pipeline-runs", json={"run_id": slow["id"], "after": [EVAL]})
    assert first.status_code == 200, first.text
    again = hub.post("/api/pipeline-runs", json={"run_id": slow["id"], "after": [EVAL]})
    assert again.status_code == 409
    assert _done(hub, first.json()["id"])["status"] == "succeeded"


def _train_slowly(hub, release, epochs=15):  # noqa: F811
    body = {"release_id": release["id"], "template": "dummy", "params": {"epochs": epochs, "sleep": 0.2}, "gpus": 0}
    resp = hub.post("/api/runs", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json(), None


def test_steps_are_checked_before_anything_starts(hub, release):  # noqa: F811
    def refused(steps, **extra):
        resp = hub.post("/api/pipelines", json={"release_id": release["id"], "steps": steps, **extra})
        assert resp.status_code == 400, resp.text
        return resp.json()["detail"]

    train = {"kind": "train", "config": _train_step()}
    # a step that uses a model comes after one that makes it, or names one by its alias
    assert "1번째 단계(평가): 앞에 학습이나 탐색이 없으면 쓸 모델을 고르세요" in refused([EVAL, train])
    assert "쓸 모델을 고르세요" in refused([EXPORT])
    assert "쓸 모델을 고르세요" in refused([{"kind": "autolabel", "config": {}}])
    assert "모델 이름과 alias" in refused([{"kind": "eval", "config": {"model": {"alias": "SOTA"}}}])
    # a promotion decides on the evaluation right before it, of the model made before that
    assert "2번째 단계(승격)" in refused([train, PROMOTE])
    assert "바로 앞의 평가" in refused([train, EVAL, train, PROMOTE])
    assert "모르는 단계" in refused([train, {"kind": "deploy"}])
    assert "latest" in refused([train, EVAL, {"kind": "promote", "config": {"alias": "latest"}}])
    assert "split" in refused([train, {"kind": "eval", "config": {"split": "dev"}}])
    assert "서버를 고르세요" in refused([{"kind": "sync", "config": {}}, train])
    assert "시험할 설정" in refused([{"kind": "sweep", "config": {"template": "dummy"}}])
    assert "trigger" in refused([train], trigger="hourly")
    bad = {"kind": "train", "config": {"template": "dummy", "params": {"epochs": "many"}}}
    refused([bad, EVAL])  # the training settings are checked against the newest version when saved
    assert hub.get(f"/api/projects/{release['project_id']}/pipelines").json()["pipelines"] == []
    # a new training whose follow-up cannot run is not started either
    nothing = {"release_id": release["id"], "train": _train_step(), "after": [{"kind": "promote", "config": {}}]}
    assert hub.post("/api/pipeline-runs", json=nothing).status_code == 400
    assert hub.get(f"/api/projects/{release['project_id']}/runs").json() == []


def test_a_failed_step_stops_the_run_and_a_stopped_one_goes_on_from_there(hub, release):  # noqa: F811
    body = {"release_id": release["id"], "train": _train_step(fail=True), "after": [EVAL, PROMOTE]}
    failed = _done(hub, hub.post("/api/pipeline-runs", json=body).json()["id"])
    assert failed["status"] == "failed" and failed["error"].startswith("학습 단계: ")
    assert [s["status"] for s in failed["steps"]] == ["failed", "skipped", "skipped"]

    # ---- stopped while training: the training is canceled, the rest skipped
    body = {"release_id": release["id"], "train": _train_step(epochs=20, sleep=0.2), "after": [EVAL, PROMOTE]}
    run = hub.post("/api/pipeline-runs", json=body).json()
    job_id = hub.get(f"/api/runs/{run['train_run_id']}").json()["job_id"]
    deadline = time.monotonic() + 30
    while hub.get(f"/api/jobs/{job_id}").json()["status"] != "running" and time.monotonic() < deadline:
        time.sleep(0.2)
    stopped = hub.post(f"/api/pipeline-runs/{run['id']}/stop").json()
    assert stopped["status"] == "stopped" and [s["status"] for s in stopped["steps"]] == ["stopped", "skipped", "skipped"]
    assert "멈췄습니다" in stopped["error"]
    assert _wait(hub, job_id)["status"] == "canceled"
    assert hub.post(f"/api/pipeline-runs/{run['id']}/stop").status_code == 409

    # ---- go on from the step that stopped: the training again with the same settings, then the rest
    again = hub.post(f"/api/pipeline-runs/{run['id']}/retry")
    assert again.status_code == 200, again.text
    retried = again.json()
    assert retried["status"] == "running" and retried["train_run_id"] != run["train_run_id"]
    done = _done(hub, run["id"])
    assert done["status"] == "succeeded" and done["steps"][2]["outcome"] == "promoted"
    assert hub.get(f"/api/runs/{done['train_run_id']}").json()["source_run_id"] == run["train_run_id"]
    assert hub.post(f"/api/pipeline-runs/{run['id']}/retry").status_code == 409


def test_saved_pipelines_run_on_demand_and_on_each_new_version(hub, release):  # noqa: F811
    project = release["project_id"]
    steps = [{"kind": "train", "config": _train_step()}, EVAL, PROMOTE]
    made = hub.post("/api/pipelines", json={"release_id": release["id"], "name": "매일 학습", "steps": steps, "run": True})
    assert made.status_code == 200, made.text
    saved = made.json()
    assert saved["name"] == "매일 학습" and saved["trigger"] == "manual" and saved["dataset"]["name"] == "street"
    assert [s["summary"] for s in saved["steps"]] == [
        "테스트 학습 · dummy · epoch 5",
        "학습한 버전 · test (없으면 val) · 검수에도",
        "더 나으면 SOTA",
    ]
    first = _done(hub, saved["started"]["id"])
    assert first["status"] == "succeeded" and first["pipeline"]["id"] == saved["id"]
    sota = first["steps"][0]["model"]

    # ---- a better model takes the alias; a worse one leaves it (the gate, both ways)
    def run_with(ceiling):
        hub.patch(
            f"/api/pipelines/{saved['id']}",
            json={"steps": [{"kind": "train", "config": _train_step(ceiling)}, EVAL, PROMOTE]},
        )
        return _done(hub, hub.post(f"/api/pipelines/{saved['id']}/run", json={}).json()["id"])

    worse = run_with(0.3)
    assert worse["status"] == "held" and hub.get(f"/api/models/{sota['id']}").json()["aliases"] == ["SOTA"]
    better = run_with(0.9)
    assert better["status"] == "succeeded" and better["steps"][2]["outcome"] == "promoted"
    assert better["steps"][2]["message"] == "mAP50-95 0.600 > v1 0.400: SOTA을(를) v3(으)로 옮김"
    assert hub.get(f"/api/models/{sota['id']}").json()["aliases"] == []
    view = hub.get(f"/api/pipelines/{saved['id']}").json()
    assert view["runs"] == 3 and [r["status"] for r in view["history"]] == ["succeeded", "held", "succeeded"]

    # ---- on each new version: retrain (the SOTA is measured on the new version too, to compare on the same
    # data), and the SOTA alone measured on it (how it holds up; 검수's model check for that version)
    hub.patch(f"/api/pipelines/{saved['id']}", json={"trigger": "new_version"})
    check = {"kind": "eval", "config": {"model": {"name": sota["name"], "alias": "SOTA"}, "review": True, "gpus": 0}}
    watcher = hub.post("/api/pipelines", json={"release_id": release["id"], "steps": [check], "trigger": "new_version"})
    assert watcher.status_code == 200, watcher.text
    assert watcher.json()["steps"][0]["summary"] == f"{sota['name']} SOTA → 그 버전 · test (없으면 val) · 검수에도"
    v2 = hub.post("/api/releases", json={"sources": [release["id"]], "version": "v2"}).json()["release_id"]
    pipelines.wake()
    deadline = time.monotonic() + 30
    runs = []
    while time.monotonic() < deadline and len(runs) < 2:
        time.sleep(0.3)
        runs = [r for r in hub.get(f"/api/projects/{project}/pipelines").json()["runs"] if r["release"]["id"] == v2]
    assert len(runs) == 2 and {r["trigger"] for r in runs} == {"new_version"}
    by_pipeline = {r["pipeline"]["id"]: _done(hub, r["id"]) for r in runs}
    retrained, measured = by_pipeline[saved["id"]], by_pipeline[watcher.json()["id"]]
    # trained as the SOTA was (ceiling 0.9): as good is not better, so the SOTA stays
    assert retrained["status"] == "held", retrained
    gate = retrained["steps"][2]
    assert gate["message"] == "mAP50-95 0.600 ≤ v3 0.600: SOTA은(는) v3 그대로"
    holder_op = hub.get(f"/api/model-ops/{gate['holder']['op_id']}").json()
    assert holder_op["release"]["version"] == "v2" and gate["holder"]["version"] == 3  # measured on v2 for the gate
    assert measured["status"] == "succeeded", measured
    sota_check = measured["steps"][0]
    assert sota_check["model"]["version"] == 3 and sota_check["before"] == {
        "release_id": release["id"], "version": "v1", "value": 0.6,
    }  # fmt: skip
    assert sota_check["message"] == "mAP50-95 0.600 (v2 val) · v1보다 +0.000"
    # a version is taken once: the next passes start nothing more
    pipelines.wake()
    time.sleep(2)
    assert len([r for r in hub.get(f"/api/projects/{project}/pipelines").json()["runs"] if r["release"]["id"] == v2]) == 2
    board = hub.get(f"/api/projects/{project}/pipelines").json()
    assert board["totals"]["watching"] == 2 and board["totals"]["held"] == 2
    assert hub.delete(f"/api/pipelines/{watcher.json()['id']}").status_code == 200
    assert len(hub.get(f"/api/projects/{project}/pipelines").json()["pipelines"]) == 1


def test_agents_run_steps_after_a_training_through_mcp(hub, release):  # noqa: F811
    token = hub.post("/api/tokens", json={"name": "claude-code"}).json()["token"]
    assert _rpc(hub, "initialize", INIT, token).status_code == 200

    def call(name, arguments):
        res = _rpc(hub, "tools/call", {"name": name, "arguments": arguments}, token).json()["result"]
        assert not res.get("isError"), res
        return res["structuredContent"]

    run, job = _train(hub, release)
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    started = call("run_pipeline", {"follow_run_id": run["id"], "steps": [EVAL, PROMOTE]})
    assert started["trigger"] == "training" and [s["kind"] for s in started["steps"]] == ["train", "eval", "promote"]
    done = _done(hub, started["id"])
    assert done["status"] == "succeeded"
    assert call("get_pipeline_run", {"run_id": done["id"]})["steps"][2]["outcome"] == "promoted"
    assert call("list_pipelines", {"project_id": release["project_id"]})["runs"][0]["id"] == done["id"]
    assert hub.get(f"/api/runs/{run['id']}").json()["created_via"] == "ui"  # the training as it was started


def _run(client, on, steps, **extra):
    resp = client.post("/api/pipeline-runs", json={"release_id": on["id"], "steps": steps, **extra})
    assert resp.status_code == 200, resp.text
    return _done(client, resp.json()["id"])


def test_composed_steps_go_on_by_success_or_through_their_gates(hub, release):  # noqa: F811
    train = {"kind": "train", "config": _train_step()}
    # 검수 확인: a gate on the rule checks' flagged images
    lenient = _run(hub, release, [{"kind": "check", "config": {"max_flagged": 100000}}, train])
    assert lenient["status"] == "succeeded", lenient
    flagged = int(lenient["steps"][0]["message"].split("이미지 ")[1].split("장")[0].replace(",", ""))
    assert lenient["steps"][0]["gate"] and lenient["steps"][0]["release"]["id"] == release["id"]
    strict = _run(hub, release, [{"kind": "check", "config": {"max_flagged": 0}}, train])
    assert strict["status"] == ("held" if flagged else "succeeded")

    # an evaluation that must reach a number: 0.4 < 0.5 holds the run, the export after it is skipped
    gated = {"kind": "eval", "config": {"split": "val", "at_least": 0.5}}
    low = _run(hub, release, [train, gated, EXPORT])
    assert low["status"] == "held" and [s["status"] for s in low["steps"]] == ["succeeded", "held", "skipped"]
    assert low["steps"][1]["message"] == "mAP50-95 0.400 (v1 val) · 기준 0.500에 못 미쳐 멈춤" and low["steps"][1]["gate"]
    high = _run(hub, release, [{"kind": "train", "config": _train_step(0.9)}, gated, EXPORT])
    assert high["status"] == "succeeded" and high["steps"][1]["message"].endswith("기준 0.500 넘음")
    # a training that must reach a number on its own validation
    weak = _run(hub, release, [{"kind": "train", "config": {**_train_step(0.3), "at_least": 0.5}}, EVAL])
    assert weak["status"] == "held" and weak["steps"][0]["message"].endswith("기준 0.500에 못 미쳐 멈춤")

    # any order that can run: the second training's model is the one evaluated and promoted
    val = {"kind": "eval", "config": {"split": "val"}}
    two = _run(hub, release, [train, val, {"kind": "train", "config": _train_step(0.9)}, val, PROMOTE])
    assert two["status"] == "succeeded", two
    assert two["steps"][4]["model"]["id"] == two["steps"][2]["model"]["id"] != two["steps"][0]["model"]["id"]


def test_a_search_then_its_best_model_evaluated_and_exported(hub, release):  # noqa: F811
    space = {"ceiling": {"values": [0.3, 0.9]}}
    config = {"template": "dummy", "params": {"epochs": 2}, "space": space, "method": "grid", "max_runs": 2, "gpus": 0}
    done = _run(hub, release, [{"kind": "sweep", "config": config}, {"kind": "eval", "config": {"split": "val"}}, EXPORT])
    assert done["status"] == "succeeded", done
    searched, evaluated, exported = done["steps"]
    assert searched["sweep"]["total"] == 2 and searched["message"].startswith("2번 중 최고 mAP50-95 ")
    assert searched["message"].endswith("(ceiling=0.9)")
    assert evaluated["model"]["id"] == searched["model"]["id"] and evaluated["metric"]["value"] == 0.6
    assert exported["model"]["id"] == searched["model"]["id"]


def test_import_then_send_and_the_labeling_loop_on_its_own_versions(hub, release, tmp_path, settings):  # noqa: F811
    from sample_data import make_yolo
    from sqlalchemy import update
    from test_autolabel import _raw

    from badamoa.hub.catalog_models import LabelSetItem
    from badamoa.hub.db import Database

    # ---- 가져오기 → 서버로 보내기: the folder again as the next version, then on the server (it holds it already)
    folder = make_yolo(tmp_path / "again", train=8)
    imported = _run(
        hub,
        release,
        [{"kind": "import", "config": {"node_id": hub.node_id, "path": str(folder)}},
         {"kind": "sync", "config": {"node_id": hub.node_id}}],
    )  # fmt: skip
    assert imported["status"] == "succeeded", imported
    assert imported["steps"][0]["message"] == "v2 · 이미지 10장" and imported["steps"][0]["made_release"]["version"] == "v2"
    assert imported["steps"][1]["message"] == "이미 그 서버에 있음"

    # ---- 오토라벨 → 라벨링 → 버전 확정 on a dataset nobody labelled, by the model holding SOTA
    made, job = _train(hub, release)
    assert job["status"] == "succeeded", _logs(hub, job["id"])
    model = hub.get(f"/api/runs/{made['id']}").json()["model"]
    hub.post(f"/api/models/{model['id']}/alias", json={"alias": "SOTA"}).raise_for_status()
    raw = _raw(hub, release["project_id"], tmp_path)
    # the set the 오토라벨 makes, as 새 오토라벨 makes it: its name, all of it to one person, approving asked for
    by_sota = {"model": {"name": model["name"], "alias": "SOTA"}, "accept": 0.9, "gpus": 0}
    by_sota.update(name="루프 묶음", divide="one", review=True)
    confirm = {"kind": "confirm", "config": {"version": "loop-{n}", "note": "루프에서"}}
    loop = [{"kind": "autolabel", "config": by_sota}, {"kind": "label", "config": {}}, confirm]
    saved = hub.post("/api/pipelines", json={"release_id": raw, "steps": loop, "trigger": "new_version", "run": True})
    assert saved.status_code == 200, saved.text
    run_id = saved.json()["started"]["id"]
    deadline = time.monotonic() + 60
    view = hub.get(f"/api/pipeline-runs/{run_id}").json()
    while time.monotonic() < deadline and view["steps"][1]["status"] != "running":
        time.sleep(0.3)
        view = hub.get(f"/api/pipeline-runs/{run_id}").json()
    labelled, waiting = view["steps"][0], view["steps"][1]
    assert labelled["status"] == "succeeded" and "자동 승인" in labelled["message"], [
        (s["kind"], s["status"], s["message"], s.get("job")) for s in view["steps"]
    ]
    assert waiting["set"]["id"] == labelled["set"]["id"] and waiting["set"]["done"] < waiting["set"]["total"]
    made_set = hub.get(f"/api/label-sets/{waiting['set']['id']}").json()
    assert made_set["name"] == "루프 묶음" and made_set["review"] and len(made_set["part_list"]) == 1
    time.sleep(1)
    assert hub.get(f"/api/pipeline-runs/{run_id}").json()["status"] == "running"  # it waits for people
    with Database(settings.database_url).session() as db:  # people finish the set
        db.execute(update(LabelSetItem).where(LabelSetItem.set_id == waiting["set"]["id"]).values(state="approved"))
    pipelines.wake()
    done = _done(hub, run_id)
    assert done["status"] == "succeeded", [(s["kind"], s["status"], s["message"], s.get("set")) for s in done["steps"]]
    confirmed = done["steps"][2]
    assert confirmed["made_release"]["version"] == "loop-1", confirmed
    assert confirmed["message"].startswith("loop-1 확정 · 고친 이미지")
    # the version the pipeline made starts no run of it
    pipelines.wake()
    time.sleep(2)
    assert len(hub.get(f"/api/pipelines/{saved.json()['id']}").json()["history"]) == 1


def test_review_step_checks_every_image_where_asked_and_versions_take_the_names_given(hub, release, settings):  # noqa: F811
    train = {"kind": "train", "config": _train_step()}
    # 검수 (모델로 검사하기): the model made before predicts every split of the version, on the server picked
    review = {"kind": "review", "config": {"node_id": hub.node_id}}
    done = _run(hub, release, [train, review, {"kind": "export", "config": {"format": "onnx", "node_id": hub.node_id}}])
    assert done["status"] == "succeeded", [(s["kind"], s["status"], s["message"]) for s in done["steps"]]
    checked = done["steps"][1]
    assert checked["label"] == "검수" and not checked["gate"] and checked["release"]["id"] == release["id"]
    assert checked["model"]["id"] == done["steps"][0]["model"]["id"]
    assert "이미지 8장 검사" in checked["message"] and checked["flagged"] is not None
    op = hub.get(f"/api/model-ops/{checked['op_id']}").json()
    assert sorted(op["splits"]) == ["train", "val"] and op["node_id"] == hub.node_id
    assert hub.get(f"/api/releases/{release['id']}/checks").json()[0]["id"] == checked["op_id"]  # 검수 offers it

    def refused(steps, status=400):
        resp = hub.post("/api/pipelines", json={"release_id": release["id"], "steps": steps})
        assert resp.status_code == status, resp.text
        return resp.json()["detail"]

    # it needs a model: one made before, or one named by its alias
    assert "쓸 모델을 고르세요" in refused([{"kind": "review", "config": {}}])
    assert "서버를 찾을 수 없습니다" in refused([train, {"kind": "eval", "config": {"node_id": "nope"}}], 404)
    # a version's name: as typed, but never one the dataset has; {n} the next free number, {date} the day
    assert "v1 버전이 이미 있습니다" in refused([{"kind": "confirm", "config": {"version": "v1"}}], 409)
    assert "한 번씩만" in refused([{"kind": "confirm", "config": {"version": "x-{n}-{n}"}}])
    assert "한 번씩만" in refused([{"kind": "confirm", "config": {"version": "x-{week}"}}])
    fixed = [{"kind": "confirm", "config": {"version": "fixed-{n}"}}]
    saved = hub.post("/api/pipelines", json={"release_id": release["id"], "steps": fixed})
    assert saved.status_code == 200, saved.text
    assert saved.json()["steps"][0]["summary"] == "작업 중을 새 버전으로 (fixed-{n})"
    from badamoa.hub.db import Database

    dataset_id = hub.get(f"/api/releases/{release['id']}").json()["dataset"]["id"]
    with Database(settings.database_url).session() as db:
        assert pipelines._version_name(db, dataset_id, "") == "v2"
        assert pipelines._version_name(db, dataset_id, "v{n}") == "v2"
        assert pipelines._version_name(db, dataset_id, "fixed-{n}") == "fixed-1"
        assert pipelines._version_name(db, dataset_id, "d{date}").startswith("d20")


def test_a_step_takes_only_the_servers_its_own_screen_offers(hub, release, tmp_path):  # noqa: F811
    from sample_data import make_yolo

    dataset_id = hub.get(f"/api/releases/{release['id']}").json()["dataset"]["id"]
    node = hub.node_id

    def offered(kind, **model):
        query = {"dataset_id": dataset_id, "kind": kind, **model}
        listed = hub.get("/api/pipeline-places", params=query)
        assert listed.status_code == 200, listed.text
        return next(n for n in listed.json()["nodes"] if n["id"] == node)

    def refused(steps):
        resp = hub.post("/api/pipelines", json={"release_id": release["id"], "steps": steps})
        assert resp.status_code == 400, resp.text
        return resp.json()["detail"]

    train = {"kind": "train", "config": _train_step()}
    assert all(offered(kind)["fits"] for kind in ("import", "sync", "autolabel", "review", "eval", "export"))
    try:
        # a data server only: it imports, but evaluates, auto-labels and receives nothing
        assert hub.patch(f"/api/nodes/{node}", json={"roles": ["storage"]}).status_code == 200
        assert offered("import")["fits"]
        assert "학습용 서버가 아님" in offered("eval")["why"] and not offered("review")["fits"]
        assert "오토라벨용 서버가 아님" in offered("autolabel")["why"]
        assert "학습용 서버가 아님" in refused([train, {"kind": "eval", "config": {"node_id": node}}])
        assert "오토라벨용" in refused([train, {"kind": "autolabel", "config": {"node_id": node}}])
        # a training server only: no 가져오기 there, in a pipeline as in 데이터셋 가져오기
        assert hub.patch(f"/api/nodes/{node}", json={"roles": ["compute"]}).status_code == 200
        assert "데이터 서버용 서버가 아님" in offered("import")["why"] and offered("eval")["fits"]
        folder = make_yolo(tmp_path / "more", train=4)
        assert "데이터 서버용" in refused([{"kind": "import", "config": {"node_id": node, "path": str(folder)}}])
        body = {"node_id": node, "path": str(folder), "dataset": "street", "version": "v9"}
        direct = hub.post(f"/api/projects/{release['project_id']}/imports", json=body)
        assert direct.status_code == 400 and "데이터 서버용" in direct.json()["detail"]
        # the new-run dialog's servers say which train
        choices = hub.get(f"/api/releases/{release['id']}/nodes").json()["nodes"]
        assert next(n for n in choices if n["id"] == node)["roles"] == ["compute"]
    finally:
        hub.patch(f"/api/nodes/{node}", json={"roles": ["compute", "autolabel", "inference", "storage"]})


def test_templates_are_kept_and_offered(hub, release):  # noqa: F811
    listed = hub.get("/api/pipeline-templates").json()
    builtin = [t for t in listed if t["builtin"]]
    assert [t["id"] for t in builtin][:2] == ["builtin-train", "builtin-retrain"] and len(builtin) == 6
    assert builtin[0]["starts_with_train"]
    assert not next(t for t in builtin if t["id"] == "builtin-sota")["starts_with_train"]
    assert builtin[0]["steps"][2]["summary"] == "더 나으면 SOTA" and builtin[0]["steps"][2]["gate"]
    steps = [{"kind": "train", "config": {**_train_step(), "classes": [0], "name": "keep-me-not"}}, EVAL, PROMOTE]
    made = hub.post("/api/pipeline-templates", json={"name": "우리 팀 방식", "description": "검수까지", "steps": steps})
    assert made.status_code == 200, made.text
    mine = next(t for t in made.json() if t["name"] == "우리 팀 방식")
    assert not mine["builtin"] and mine["created_by"] == "admin" and mine["starts_with_train"]
    assert "classes" not in mine["steps"][0]["config"] and "name" not in mine["steps"][0]["config"]  # the dataset's own
    assert hub.post("/api/pipeline-templates", json={"name": "우리 팀 방식", "steps": steps}).status_code == 409
    assert hub.post("/api/pipeline-templates", json={"name": "빈 것", "steps": [PROMOTE]}).status_code == 400
    # a pipeline made from it runs as any other
    started = hub.post("/api/pipelines", json={"release_id": release["id"], "steps": mine["steps"], "run": True}).json()
    assert _done(hub, started["started"]["id"])["status"] == "succeeded"
    assert hub.delete(f"/api/pipeline-templates/{mine['id']}").status_code == 200
    assert all(t["name"] != "우리 팀 방식" for t in hub.get("/api/pipeline-templates").json())
