"""Admins see who uses the Hub now: a token used in the last three minutes, with where it came from."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import update

from badamoa.hub.db import utcnow
from badamoa.hub.models import ApiToken


def test_admins_see_who_is_here_now(client, admin):
    client.post("/api/users", json={"username": "kim", "password": "kim-online-pw"}, headers=admin)
    kim = {
        "Authorization": "Bearer "
        + client.post("/api/auth/login", json={"username": "kim", "password": "kim-online-pw"}).json()["token"]
    }
    client.get("/api/auth/me", headers=kim)

    here = client.get("/api/users/online", headers=admin).json()
    assert sorted(p["username"] for p in here) == ["admin", "kim"]
    who = next(p for p in here if p["username"] == "kim")
    assert who["web"] == 1 and who["api"] == 0 and who["ips"] == ["testclient"]
    listed = {u["username"]: u for u in client.get("/api/users", headers=admin).json()}
    assert listed["kim"]["online"] is True and listed["kim"]["ips"] == ["testclient"]
    assert client.get("/api/users/online", headers=kim).status_code == 403  # admins only

    # ---- gone for more than three minutes: no longer here (seen at stays)
    db = client.app.state.hub.db
    with db.session() as s:
        s.execute(update(ApiToken).values(last_used_at=utcnow() - timedelta(minutes=10)))
    here = client.get("/api/users/online", headers=admin).json()  # this call brings the admin back
    assert [p["username"] for p in here] == ["admin"]
    listed = {u["username"]: u for u in client.get("/api/users", headers=admin).json()}
    assert listed["kim"]["online"] is False and listed["kim"]["last_seen_at"]
