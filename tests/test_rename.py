"""Renaming a dataset (every version of it) and a version."""

from __future__ import annotations

from test_labeling import _login
from test_training import hub, release  # noqa: F401  (the fixtures)


def test_a_dataset_and_a_version_are_renamed(hub, release):  # noqa: F811
    before = hub.get(f"/api/releases/{release['id']}").json()
    dataset_id = before["dataset"]["id"]

    assert hub.patch(f"/api/datasets/{dataset_id}", json={"name": "  도로  "}).json()["name"] == "도로"
    assert hub.patch(f"/api/releases/{release['id']}", json={"version": "2026-11"}).json()["version"] == "2026-11"
    after = hub.get(f"/api/releases/{release['id']}").json()
    assert (after["dataset"]["name"], after["version"]) == ("도로", "2026-11")
    tree = hub.get(f"/api/projects/{release['project_id']}/datasets").json()
    assert [(d["name"], [r["version"] for r in d["releases"]]) for d in tree] == [("도로", ["2026-11"])]

    # what a version may not be called
    assert hub.patch(f"/api/releases/{release['id']}", json={"version": "작업 중"}).status_code == 400
    assert hub.patch(f"/api/releases/{release['id']}", json={"version": "  "}).status_code == 400
    assert hub.patch(f"/api/datasets/{dataset_id}", json={"name": ""}).status_code == 400

    # another dataset's name in the same project is taken
    made = hub.post("/api/releases", json={"sources": [release["id"]], "version": "v1", "dataset": "다른"})
    assert made.status_code == 200, made.text  # a second dataset, made from this version
    taken = hub.patch(f"/api/datasets/{dataset_id}", json={"name": "다른"})
    assert taken.status_code == 409 and "이미 있습니다" in taken.json()["detail"]

    # only the project's maker or an admin
    hub.post("/api/users", json={"username": "kim", "password": "kim-rename-pw"})
    kim = _login(hub, "kim", "kim-rename-pw")
    assert kim.patch(f"/api/datasets/{dataset_id}", json={"name": "김"}).status_code == 403
    assert kim.patch(f"/api/releases/{release['id']}", json={"version": "v9"}).status_code == 403
