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
    assert "순서" in refused([EVAL, train])
    assert "평가 단계를 앞에" in refused([train, PROMOTE])
    assert "한 번만" in refused([train, EVAL, EVAL])
    assert "모르는 단계" in refused([train, {"kind": "deploy"}])
    assert "alias가 붙은 모델" in refused([EXPORT])
    assert "모델 이름과 alias" in refused([{"kind": "eval", "config": {}}])
    assert "latest" in refused([train, EVAL, {"kind": "promote", "config": {"alias": "latest"}}])
    assert "split" in refused([train, {"kind": "eval", "config": {"split": "dev"}}])
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
