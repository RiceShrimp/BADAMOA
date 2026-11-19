"""The guides the app shows: the user manual and the template guide, read by name."""

from __future__ import annotations

import httpx
from conftest import ADMIN


def test_guides_are_listed_and_read(live_hub):
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    hub = httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    listed = {g["name"]: g for g in hub.get("/api/guides").json()}
    assert set(listed) == {"manual", "templates"} and all(g["ready"] for g in listed.values())
    manual = hub.get("/api/guides/manual").json()
    assert manual["title"] == "사용 설명서" and manual["markdown"].startswith("# BADAMOA 사용 설명서")
    guide = hub.get("/api/guides/templates").json()
    assert "badamoa.toml" in guide["markdown"] and guide["file"] == "TEMPLATE_GUIDE.md"
    assert hub.get("/api/guides/nope").status_code == 404
    assert httpx.get(f"{live_hub}/api/guides").status_code == 401  # signed in only, like the release notes
