"""The guides the app shows: the user manual and the template guide, read by name."""

from __future__ import annotations

import re

import httpx
from conftest import ADMIN


def test_guides_are_listed_and_read(live_hub):
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    hub = httpx.Client(base_url=live_hub, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    listed = {g["name"]: g for g in hub.get("/api/guides").json()}
    assert set(listed) == {"manual", "templates"} and all(g["ready"] for g in listed.values())
    manual = hub.get("/api/guides/manual").json()
    assert manual["title"] == "사용 설명서" and manual["markdown"].startswith("# MLOps Orchestration 사용 설명서")
    assert "github" not in manual["markdown"].lower() and "README.md" not in manual["markdown"]  # all of it inside the Hub
    guide = hub.get("/api/guides/templates").json()
    assert "badamoa.toml" in guide["markdown"] and guide["file"] == "TEMPLATE_GUIDE.md"
    assert "github" not in guide["markdown"].lower()
    assert hub.get("/api/guides/nope").status_code == 404
    assert httpx.get(f"{live_hub}/api/guides").status_code == 401  # signed in only, like the release notes


def test_the_manuals_pictures_come_with_their_sizes_and_load_by_name(live_hub):
    token = httpx.post(f"{live_hub}/api/setup", json=ADMIN).json()["token"]
    manual = httpx.get(f"{live_hub}/api/guides/manual", headers={"Authorization": f"Bearer {token}"}).json()
    sizes = manual["images"]
    referenced = set(re.findall(r"\]\(img/manual/([a-z0-9-]+\.(?:png|jpg))\)", manual["markdown"]))
    assert referenced and set(sizes) == referenced  # every picture the manual shows is there, with its size
    assert all(w > 0 and h > 0 for w, h in sizes.values())
    png = next(f for f in referenced if f.endswith(".png"))
    shown = httpx.get(f"{live_hub}/api/guides/images/{png}")  # an <img> fetches it: no sign-in
    assert shown.status_code == 200 and shown.headers["content-type"] == "image/png"
    assert "max-age" in shown.headers["cache-control"]
    jpg = next((f for f in referenced if f.endswith(".jpg")), None)
    if jpg:
        assert httpx.get(f"{live_hub}/api/guides/images/{jpg}").headers["content-type"] == "image/jpeg"
    for bad in ("MANUAL.md", "..%2FMANUAL.md", "nope.png", "Layout.PNG"):
        assert httpx.get(f"{live_hub}/api/guides/images/{bad}").status_code == 404, bad
