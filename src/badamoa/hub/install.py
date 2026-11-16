"""Remote node installation: one-line scripts + a PEP 503 package index served by the Hub.

Remote machines only need to reach the Hub. BADAMOA's own packages come from the
Hub's index (``--index`` takes priority over PyPI under uv's default
``first-index`` strategy, which also prevents dependency-confusion attacks);
third-party dependencies still come from PyPI until the offline bundle lands.
"""

from __future__ import annotations

import html
import logging
import re
import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

import badamoa

from . import services
from .config import HubSettings
from .deps import Hub, get_hub, public_base_url

router = APIRouter(prefix="/install")
log = logging.getLogger("badamoa.hub.install")

_NORMALIZE = re.compile(r"[-_.]+")
_SCRIPTS_DIR = Path(__file__).parent / "scripts"


def _normalize(name: str) -> str:
    return _NORMALIZE.sub("-", name).lower()


def _wheel_project(filename: str) -> str:
    return _normalize(filename.split("-", 1)[0])


# ---- the node package for this Hub's version -----------------------------------------
#
# The install scripts ask the Hub's index for ``badamoa==<the Hub's version>``. A Hub run from
# a source checkout builds that wheel itself at start-up; an installed Hub ships with it.


def node_wheel(settings: HubSettings, version: str = services.HUB_VERSION) -> Path | None:
    assert settings.wheels_dir is not None
    found = sorted(settings.wheels_dir.glob(f"badamoa-{version}-*.whl"))
    return found[-1] if found else None


def _source_root() -> Path | None:
    """The checkout this Hub runs from (an editable install), if it is one."""
    root = Path(badamoa.__file__).resolve().parents[2]
    project = root / "pyproject.toml"
    if project.is_file() and 'name = "badamoa"' in project.read_text(encoding="utf-8"):
        return root
    return None


def ensure_node_wheel(settings: HubSettings) -> str:
    """Have the wheel nodes install: "ready" (there), "built" (just made), or "missing" (with a reason logged)."""
    if node_wheel(settings):
        return "ready"
    root, uv = _source_root(), shutil.which("uv")
    if root is None or uv is None:
        hint = "`badamoa hub publish-wheel <wheel>`로 올리세요"
        log.warning("노드 설치 파일(badamoa %s wheel)이 없습니다. %s.", services.HUB_VERSION, hint)
        return "missing"
    done = subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(settings.wheels_dir), str(root)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if done.returncode != 0 or node_wheel(settings) is None:
        log.warning("노드 설치 파일을 만들지 못했습니다: %s", (done.stderr or done.stdout)[-500:])
        return "missing"
    log.info("노드 설치 파일을 만들었습니다: %s", node_wheel(settings).name)
    return "built"


def join_commands(hub_url: str, token: str) -> dict[str, str]:
    return {
        "linux": f"curl -fsSL {hub_url}/install/node.sh | sudo bash -s -- --token {token}",
        "windows": f"& ([scriptblock]::Create((irm {hub_url}/install/node.ps1))) -Token {token}",
        "dev": f"uv run badamoa node join --hub {hub_url} --token {token}",
    }


def _render(name: str, hub_url: str) -> str:
    # Always serve LF: a CRLF checkout on a Windows Hub would break `curl … | bash` on Linux nodes.
    text = "\n".join((_SCRIPTS_DIR / name).read_text(encoding="utf-8").splitlines()) + "\n"
    return text.replace("{{HUB_URL}}", hub_url).replace("{{VERSION}}", services.HUB_VERSION)


@router.get("/node.sh", response_class=PlainTextResponse)
def node_sh(request: Request, hub: Hub = Depends(get_hub)) -> str:
    return _render("node.sh", public_base_url(request, hub))


@router.get("/node.ps1", response_class=PlainTextResponse)
def node_ps1(request: Request, hub: Hub = Depends(get_hub)) -> str:
    return _render("node.ps1", public_base_url(request, hub))


@router.get("/node-update.sh", response_class=PlainTextResponse)
def node_update_sh(request: Request, hub: Hub = Depends(get_hub)) -> str:
    return _render("node-update.sh", public_base_url(request, hub))


@router.get("/node-update.ps1", response_class=PlainTextResponse)
def node_update_ps1(request: Request, hub: Hub = Depends(get_hub)) -> str:
    return _render("node-update.ps1", public_base_url(request, hub))


@router.get("/node-remove.sh", response_class=PlainTextResponse)
def node_remove_sh(request: Request, hub: Hub = Depends(get_hub)) -> str:
    return _render("node-remove.sh", public_base_url(request, hub))


@router.get("/node-remove.ps1", response_class=PlainTextResponse)
def node_remove_ps1(request: Request, hub: Hub = Depends(get_hub)) -> str:
    return _render("node-remove.ps1", public_base_url(request, hub))


def remove_commands(hub_url: str) -> dict[str, str]:
    """Taking BADAMOA off a machine by hand (the node was off when it was deleted)."""
    return {
        "linux": f"curl -fsSL {hub_url}/install/node-remove.sh | sudo bash",
        "windows": f"irm {hub_url}/install/node-remove.ps1 | iex",
    }


def update_commands(hub_url: str) -> dict[str, str]:
    """For a node installed before it could update itself: the program only, the registration stays."""
    return {
        "linux": f"curl -fsSL {hub_url}/install/node-update.sh | sudo bash",
        "windows": f"irm {hub_url}/install/node-update.ps1 | iex",
    }


def _wheels(hub: Hub) -> list[Path]:
    assert hub.settings.wheels_dir is not None
    return sorted(hub.settings.wheels_dir.glob("*.whl"))


@router.get("/simple/", response_class=HTMLResponse)
def simple_index(hub: Hub = Depends(get_hub)) -> str:
    projects = sorted({_wheel_project(w.name) for w in _wheels(hub)})
    links = "".join(f'<a href="{p}/">{p}</a>\n' for p in projects)
    return f"<!DOCTYPE html><html><body>\n{links}</body></html>"


@router.get("/simple/{project}/", response_class=HTMLResponse)
def simple_project(project: str, hub: Hub = Depends(get_hub)) -> str:
    wanted = _normalize(project)
    files = [w for w in _wheels(hub) if _wheel_project(w.name) == wanted]
    if not files:
        raise HTTPException(status_code=404, detail="not found")
    links = "".join(
        f'<a href="../../wheels/{html.escape(w.name)}">{html.escape(w.name)}</a>\n' for w in files
    )
    return f"<!DOCTYPE html><html><body>\n{links}</body></html>"


@router.get("/wheels/{filename}")
def wheel_file(filename: str, hub: Hub = Depends(get_hub)) -> FileResponse:
    assert hub.settings.wheels_dir is not None
    path = (hub.settings.wheels_dir / filename).resolve()
    if path.parent != hub.settings.wheels_dir.resolve() or not path.is_file() or path.suffix != ".whl":
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="application/octet-stream", filename=filename)
