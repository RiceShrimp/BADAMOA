"""Remote node installation: one-line scripts + a PEP 503 package index served by the Hub.

Remote machines only need to reach the Hub. BADAMOA's own packages come from the
Hub's index (``--index`` takes priority over PyPI under uv's default
``first-index`` strategy, which also prevents dependency-confusion attacks);
third-party dependencies still come from PyPI until the offline bundle lands.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

from . import services
from .deps import Hub, get_hub, public_base_url

router = APIRouter(prefix="/install")

_NORMALIZE = re.compile(r"[-_.]+")
_SCRIPTS_DIR = Path(__file__).parent / "scripts"


def _normalize(name: str) -> str:
    return _NORMALIZE.sub("-", name).lower()


def _wheel_project(filename: str) -> str:
    return _normalize(filename.split("-", 1)[0])


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
