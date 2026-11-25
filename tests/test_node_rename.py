"""An admin renames a server (노드 → 설정 → 이름); the name stays until someone changes it again."""

from __future__ import annotations


def test_an_admin_renames_a_node(client, admin):
    token = client.post("/api/nodes/join-tokens", json={}, headers=admin).json()["token"]
    info = {"hostname": "gpu-01", "os": "linux"}
    reg = client.post("/api/agent/register", json={"join_token": token, "agent_version": "0.121.0", "info": info}).json()
    view = lambda: client.get(f"/api/nodes/{reg['node_id']}", headers=admin).json()  # noqa: E731
    assert view()["name"] == "gpu-01"  # its hostname until renamed

    done = client.patch(f"/api/nodes/{reg['node_id']}", json={"name": "  학습 서버 A  "}, headers=admin)
    assert done.status_code == 200 and done.json()["name"] == "학습 서버 A"
    assert client.patch(f"/api/nodes/{reg['node_id']}", json={"name": "   "}, headers=admin).status_code == 200
    assert view()["name"] == "학습 서버 A"  # a blank name keeps the one it has

    client.post("/api/users", json={"username": "member", "password": "member-test-pw", "role": "member"}, headers=admin)
    login = client.post("/api/auth/login", json={"username": "member", "password": "member-test-pw"}).json()
    member = {"Authorization": f"Bearer {login['token']}"}
    assert client.patch(f"/api/nodes/{reg['node_id']}", json={"name": "x"}, headers=member).status_code == 403
    assert view()["name"] == "학습 서버 A"


def test_the_name_given_with_the_install_command_is_the_nodes_name(client, admin):
    """노드 추가 → 이름 (선택): the server joins under that name; left empty, under its hostname. The name rides with
    the token, not in the command line, so any name needs no quoting."""

    def joined(name: str, hostname: str) -> tuple[dict, dict]:
        made = client.post("/api/nodes/join-tokens", json={"name": name}, headers=admin).json()
        body = {"join_token": made["token"], "agent_version": "0.121.0", "info": {"hostname": hostname}}
        reg = client.post("/api/agent/register", json=body).json()
        return made, client.get(f"/api/nodes/{reg['node_id']}", headers=admin).json()

    made, node = joined("  학습  서버 B ", "gpu-02")
    assert made["name"] == node["name"] == "학습 서버 B"
    assert all("학습" not in command for command in made["commands"].values())
    made, node = joined("", "gpu-03")
    assert made["name"] == "" and node["name"] == "gpu-03"
