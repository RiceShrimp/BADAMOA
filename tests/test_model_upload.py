"""Uploading a model trained elsewhere: an Ultralytics ``.pt`` with its class yaml, the class counts checked first.

The checkpoints here are tiny fakes laid out as ``torch.save`` writes them (a zip with ``data.pkl``, the model an
object of ``ultralytics.nn.tasks``), made without torch or Ultralytics: the Hub reads them the same way.
"""

from __future__ import annotations

import io
import pickle
import sys
import types
import zipfile
from pathlib import Path
from unittest import mock

import pytest
from test_training import hub, release  # noqa: F401 (fixtures)

from badamoa.hub import model_upload

KINDS = {"detect": "DetectionModel", "segment": "SegmentationModel", "pose": "PoseModel", "classify": "ClassificationModel"}


class _Tensor:
    """Stands for a tensor: pickled as a reference to storage kept apart, as torch does."""


class _Pickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, _Tensor):
            return ("storage", "FloatStorage", "0", "cpu", 4)
        return None


def fake_pt(
    names: list[str], task: str = "detect", *, say_task: bool = True, imgsz: int = 640, kind: str = ""
) -> bytes:
    """An Ultralytics-like checkpoint: ``{"model": <DetectionModel with names, nc, yaml, args>, "train_args": …}``."""
    modules = {n: types.ModuleType(n) for n in ("ultralytics", "ultralytics.nn", "ultralytics.nn.tasks")}
    cls = type(kind or KINDS[task], (), {"__module__": "ultralytics.nn.tasks"})
    modules["ultralytics.nn.tasks"].__dict__[cls.__name__] = cls
    model = cls()
    model.__dict__.update(
        names=dict(enumerate(names)),
        nc=len(names),
        yaml={"nc": len(names), "yaml_file": "yolo11n.yaml", "scale": "n"},
        args={"imgsz": imgsz, **({"task": task} if say_task else {})},
        weight=_Tensor(),
    )
    top = {"date": "2026-11-23", "version": "8.3.100", "model": model, "ema": None, "train_args": {"imgsz": imgsz}}
    buffer = io.BytesIO()
    with mock.patch.dict(sys.modules, modules):
        _Pickler(buffer, protocol=2).dump(top)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("best/data.pkl", buffer.getvalue())
        archive.writestr("best/data/0", b"\0" * 16)
        archive.writestr("best/version", "3\n")
    return out.getvalue()


def yaml_of(names: list[str]) -> bytes:
    return ("path: .\ntrain: images/train\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names))).encode()


def _blobs(client) -> int:
    return sum(1 for f in (Path(client.hub_home) / "storage" / "blobs").rglob("*") if f.is_file())


def _upload(client, project_id, pt: bytes | None, data: bytes | None, **fields):
    files = {}
    if pt is not None:
        files["weights"] = ("yolo11n-street.pt", pt, "application/octet-stream")
    if data is not None:
        files["data"] = ("data.yaml", data, "application/x-yaml")
    return client.post(f"/api/projects/{project_id}/models/upload", files=files, data=fields)


def test_reading_checkpoints_without_torch():
    info = model_upload.read_checkpoint(io.BytesIO(fake_pt(["a", "b"], "segment", imgsz=1280)))
    assert info["classes"] == ["a", "b"] and info["task"] == "segment" and info["imgsz"] == 1280
    assert info["arch"] == "yolo11n" and info["ultralytics"] == "8.3.100"
    # no task written: the model's class says it
    assert model_upload.read_checkpoint(io.BytesIO(fake_pt(["a"], "pose", say_task=False)))["task"] == "pose"
    with pytest.raises(model_upload.ServiceError, match="읽을 수 없는"):
        model_upload.read_checkpoint(io.BytesIO(b"not a checkpoint at all"))
    plain = io.BytesIO()
    with zipfile.ZipFile(plain, "w") as archive:  # a state_dict alone (an OrderedDict of tensors)
        archive.writestr("x/data.pkl", pickle.dumps({"conv.weight": 1}, protocol=2))
    with pytest.raises(model_upload.ServiceError, match="Ultralytics YOLO 가중치가 아닙니다"):
        model_upload.read_checkpoint(plain)

    # the yaml: names as a list or numbered, nc agreeing, no blanks or twins
    assert model_upload.read_yaml("names: [a, b]\n")["names"] == ["a", "b"]
    for text, why in (("train: x\n", "names"), ("nc: 3\nnames: [a, b]\n", "nc는 3"), ("names: [a, a]\n", "같은 이름")):
        with pytest.raises(model_upload.ServiceError, match=why):
            model_upload.read_yaml(text)


def test_upload_is_registered_and_a_count_mismatch_is_refused(hub, release):  # noqa: F811
    project = release["project_id"]
    names = ["person", "car", "dog"]  # the classes of the release fixture's dataset

    # ---- the yaml is required, and the counts must agree: nothing is kept otherwise
    before = _blobs(hub)
    missing = _upload(hub, project, fake_pt(names), None)
    assert missing.status_code == 400 and "yaml" in missing.json()["detail"]
    refused = _upload(hub, project, fake_pt([f"c{i}" for i in range(80)]), yaml_of(names))
    assert refused.status_code == 400
    assert "pt는 클래스 80개, yaml은 3개" in refused.json()["detail"]
    assert _blobs(hub) == before
    assert hub.get(f"/api/projects/{project}/models").json() == []
    wrong_task = _upload(hub, project, fake_pt(names, "segment"), yaml_of(names), task="detect")
    assert wrong_task.status_code == 400 and "Seg 모델" in wrong_task.json()["detail"]
    unknown = _upload(hub, project, fake_pt(names, say_task=False, kind="SomeModel"), yaml_of(names))
    assert unknown.status_code == 400 and "태스크를 고르세요" in unknown.json()["detail"]

    # ---- accepted: a version of the project's models, run by the built-in Ultralytics YOLO kind
    pt = fake_pt(["0", "1", "2"])  # weights whose own names are numbers: the yaml names the classes
    done = _upload(hub, project, pt, yaml_of(names), note="밖에서 학습")
    assert done.status_code == 200, done.text
    model = done.json()
    assert (model["name"], model["version"], model["task"]) == ("yolo11n-street", 1, "detect")
    assert model["source"] == "upload" and model["classes"] == names and model["imgsz"] == 640
    assert model["upload"]["file_name"] == "yolo11n-street.pt" and model["upload"]["yaml"] == "data.yaml"
    assert model["upload"]["weights_classes"] == ["0", "1", "2"] and model["upload"]["renamed"] == 3
    assert model["template"]["name"] == "yolo" and model["run"] is None and model["release"] is None
    assert model["note"] == "밖에서 학습" and model["bytes"] == len(pt)
    assert sorted(model["actions"]) == ["eval", "export", "predict"]
    assert _blobs(hub) == before + 1
    again = _upload(hub, project, pt, yaml_of(names), name="yolo11n street")
    assert (again.json()["name"], again.json()["version"]) == ("yolo11n-street", 2)  # the same family

    # ---- the screens that pick a model take it: 검수's model check, 오토라벨, lineage
    offered = hub.get(f"/api/releases/{release['id']}/eval-models").json()["models"]
    (row,) = [m for m in offered if m["id"] == model["id"]]
    assert row["fits"] and row["relation"] == "uploaded" and "밖에서 학습해 올린" in row["warning"]
    labels = hub.get("/api/autolabel/models", params={"release_id": release["id"]}).json()["models"]
    assert any(m["id"] == model["id"] and m["problem"] == "" for m in labels)
    graph = hub.get(f"/api/models/{model['id']}/lineage").json()
    assert {n["kind"] for n in graph["nodes"]} == {"upload", "model"}

    # ---- an evaluation's per-class rows read as the yaml names them, not as the weights did
    from badamoa.hub.modelops import _named_as_model

    result = {"classes": [{"id": 1, "name": "1", "map50": 0.5}], "splits": {"val": {"classes": [{"id": 2, "name": "2"}]}}}
    named = _named_as_model(result, names)
    assert named["classes"][0]["name"] == "car" and named["splits"]["val"]["classes"][0]["name"] == "dog"


def test_sample_yaml_downloads(client):
    for kind, first in (("example", "person"), ("coco", "person"), ("coco-pose", "person")):
        resp = client.get("/api/model-upload/sample", params={"kind": kind})
        assert resp.status_code == 200 and "attachment" in resp.headers["content-disposition"]
        parsed = model_upload.read_yaml(resp.text)
        assert parsed["names"][0] == first
    assert len(model_upload.read_yaml(client.get("/api/model-upload/sample?kind=coco").text)["names"]) == 80
    assert client.get("/api/model-upload/sample", params={"kind": "nope"}).status_code == 404
