"""Folders: made empty or with projects, kept when they empty, renamed and removed (projects stay)."""

from __future__ import annotations


def _folders(client, auth):
    return {f["name"]: f["projects"] for f in client.get("/api/folders", headers=auth).json()}


def test_folders_live_on_their_own(client, admin):
    a = client.post("/api/projects", json={"name": "주차장", "folder": "공장"}, headers=admin).json()
    b = client.post("/api/projects", json={"name": "정문"}, headers=admin).json()
    assert _folders(client, admin) == {"공장": 1}

    # ---- an empty folder, then one that takes a project in
    assert client.post("/api/folders", json={"name": "현장A"}, headers=admin).json() == {"name": "현장A", "projects": 0}
    assert client.post("/api/folders", json={"name": "현장B", "projects": [b["id"]]}, headers=admin).status_code == 200
    assert _folders(client, admin) == {"공장": 1, "현장A": 0, "현장B": 1}
    assert client.post("/api/folders", json={"name": "공장"}, headers=admin).status_code == 409
    assert client.post("/api/folders", json={"name": "  "}, headers=admin).status_code == 400

    # ---- a folder that empties stays; one picked for a new project is kept too
    client.patch(f"/api/projects/{a['id']}", json={"folder": ""}, headers=admin)
    client.post("/api/projects", json={"name": "후문", "folder": "새 현장"}, headers=admin)
    assert _folders(client, admin) == {"공장": 0, "새 현장": 1, "현장A": 0, "현장B": 1}

    # ---- renamed: the projects move along; the name must be free
    assert client.patch("/api/folders", json={"name": "현장B", "new_name": "현장A"}, headers=admin).status_code == 409
    renamed = client.patch("/api/folders", json={"name": "현장B", "new_name": "서문 쪽"}, headers=admin).json()
    assert renamed == {"name": "서문 쪽", "projects": 1}
    assert next(p for p in client.get("/api/projects", headers=admin).json() if p["id"] == b["id"])["folder"] == "서문 쪽"

    # ---- removed: the projects stay, outside any folder; a name with odd characters works as well
    assert client.delete("/api/folders", params={"name": "서문 쪽"}, headers=admin).json()["projects"] == 1
    assert next(p for p in client.get("/api/projects", headers=admin).json() if p["id"] == b["id"])["folder"] == ""
    client.post("/api/folders", json={"name": "a/b?c&d"}, headers=admin)
    assert client.delete("/api/folders", params={"name": "a/b?c&d"}, headers=admin).status_code == 200
    assert client.delete("/api/folders", params={"name": "없는 폴더"}, headers=admin).status_code == 404
    assert _folders(client, admin) == {"공장": 0, "새 현장": 1, "현장A": 0}

    # ---- a member makes folders, but moves only the projects they may change
    client.post("/api/users", json={"username": "kim", "password": "kim-first-pw"}, headers=admin)
    token = client.post("/api/auth/login", json={"username": "kim", "password": "kim-first-pw"}).json()["token"]
    kim = {"Authorization": f"Bearer {token}"}
    assert client.post("/api/folders", json={"name": "김의 폴더"}, headers=kim).status_code == 200
    assert client.post("/api/folders", json={"name": "남의 것", "projects": [b["id"]]}, headers=kim).status_code == 403
    assert client.delete("/api/folders", params={"name": "새 현장"}, headers=kim).status_code == 403
    assert client.delete("/api/folders", params={"name": "김의 폴더"}, headers=kim).status_code == 200

    actions = {a["action"] for a in client.get("/api/activity?limit=30", headers=admin).json()}
    assert {"folder.create", "folder.rename", "folder.delete"} <= actions
