"""`badamoa` command line."""

from __future__ import annotations

import logging
import os
import shutil
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Annotated

import typer

from badamoa import __version__
from badamoa.core.paths import component_home

app = typer.Typer(help="BADAMOA — 설치형 비전 MLOps 플랫폼", no_args_is_help=True, add_completion=False)
hub_app = typer.Typer(help="Hub(제어 서버) 실행·관리", no_args_is_help=True)
node_app = typer.Typer(help="Node Agent(학습/저장/추론 서버) 실행·관리", no_args_is_help=True)
template_app = typer.Typer(help="모델 템플릿(사내 학습 코드) 만들기·확인·올리기", no_args_is_help=True)
app.add_typer(hub_app, name="hub")
app.add_typer(node_app, name="node")
app.add_typer(template_app, name="template")

HomeOpt = Annotated[Path | None, typer.Option("--home", help="데이터 디렉터리 (기본: OS별 표준 위치)")]


def _logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def version() -> None:
    """버전을 출력합니다."""
    typer.echo(__version__)


# ---- hub ---------------------------------------------------------------------


def _require_hub():
    try:
        import uvicorn  # noqa: F401

        from badamoa.hub import app as hub_module  # noqa: F401
    except ImportError as e:
        typer.secho(f"Hub 구성요소가 설치되지 않았습니다 ({e.name}). `badamoa[hub]`로 설치하세요.", fg="red")
        raise typer.Exit(1) from e


def _has_screen() -> bool:
    """A desktop to open a browser on: Windows and macOS, or Linux with a display (not a server or a service)."""
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _open_when_up(port: int, url: str) -> None:
    """Open the web UI once the Hub answers (at most a minute after start-up)."""
    import httpx

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/api/health", timeout=2).status_code == 200:
                webbrowser.open(url)
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)


def _lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


@hub_app.command("run")
def hub_run(
    host: Annotated[str, typer.Option(help="바인드 주소")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="포트")] = 8470,
    home: HomeOpt = None,
    public_url: Annotated[str | None, typer.Option(help="원격 노드·에이전트가 접속할 URL")] = None,
    local_node: Annotated[bool, typer.Option(help="이 머신도 노드로 사용 (All-in-one)")] = True,
    gpu_busy_memory_pct: Annotated[
        float,
        typer.Option(
            envvar="BADAMOA_GPU_BUSY_MEMORY_PCT",
            help="다른 프로세스가 GPU 메모리를 이만큼(%) 넘게 쓰면 사용 중으로 보고 배정하지 않음. 100보다 크면 항상 배정",
        ),
    ] = 10.0,
    gpu_busy_util_pct: Annotated[
        float,
        typer.Option(envvar="BADAMOA_GPU_BUSY_UTIL_PCT", help="GPU 사용률이 이만큼(%) 넘으면 사용 중으로 봄"),
    ] = 50.0,
    open_browser: Annotated[
        bool | None,
        typer.Option(
            "--open/--no-open",
            envvar="BADAMOA_OPEN_BROWSER",
            help="시작하면 웹 UI를 이 PC의 네트워크 주소로 엽니다. 기본: 화면이 있는 PC에서만",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Hub를 실행합니다. 기본값은 All-in-one(이 머신도 노드로 동작)입니다."""
    _require_hub()
    _logging(verbose)
    import uvicorn

    from badamoa.hub.app import create_app
    from badamoa.hub.config import HubSettings

    settings = HubSettings(
        home=component_home("hub", home),
        host=host,
        port=port,
        public_url=public_url,
        local_node=local_node,
        gpu_busy_memory_pct=gpu_busy_memory_pct,
        gpu_busy_util_pct=gpu_busy_util_pct,
    )
    application = create_app(settings)
    if local_node:
        _start_local_node(application, settings)

    # The address people and servers reach this Hub at: the one set on the screen or at launch, else
    # this PC's network address. localhost only when the Hub takes no connections from outside.
    local_only = host in ("127.0.0.1", "localhost", "::1")
    lan = None if local_only else _lan_ip()
    saved = settings.extra.get("hub_address")
    url = saved or public_url or (f"http://{lan}:{port}" if lan and lan != "127.0.0.1" else f"http://localhost:{port}")
    typer.secho(f"\n  BADAMOA Hub {__version__}", bold=True)
    where = f"   (이 PC에서는 http://localhost:{port} 도 됨)"
    if local_only:
        where = "   (이 PC 안에서만 접속됨: --host 127.0.0.1)"
    typer.echo(f"  웹 UI   : {url}{where}")
    typer.echo(f"  MCP     : {url}/mcp")
    typer.echo(f"  데이터  : {settings.home}\n")
    from badamoa.hub.install import ensure_node_wheel

    # The package remote nodes install: made in the background from this checkout when it is missing.
    threading.Thread(target=ensure_node_wheel, args=(settings,), name="node-wheel", daemon=True).start()
    if open_browser if open_browser is not None else _has_screen():
        threading.Thread(target=_open_when_up, args=(port, url), name="open-browser", daemon=True).start()
    uvicorn.run(application, host=host, port=port, log_level="debug" if verbose else "warning")


def _start_local_node(application, settings) -> None:
    from badamoa.core import sysinfo
    from badamoa.hub import migration, services
    from badamoa.node.agent import NodeAgent, NodeConfig

    hub = application.state.hub
    home = settings.local_node_home
    home.mkdir(parents=True, exist_ok=True)
    if migration.read_only(hub) and NodeConfig.path(home).is_file():
        # Being moved (or moved): this PC's node keeps the identity the export carried, and once handed over it
        # is an ordinary node of the new Hub (node.json holds that address)
        config = NodeConfig.load(home)
    else:
        with hub.db.session() as db:
            node, token = services.ensure_local_node(db, sysinfo.static_info(), __version__)
            if token is None:  # existing local node: issue a fresh token for this process
                token = services.rotate_node_token(db, node)
            node_id = node.id
        config = NodeConfig(hub_url=f"http://127.0.0.1:{settings.port}", node_id=node_id, node_token=token)
    agent = NodeAgent(config, home)
    application.state.local_agent = agent
    threading.Thread(target=agent.run_forever, name="local-node", daemon=True).start()


@hub_app.command("export")
def hub_export(
    out: Annotated[Path, typer.Argument(help="내보낸 파일을 둘 폴더")],
    home: HomeOpt = None,
) -> None:
    """이 Hub를 파일 하나로 내보냅니다 (새 PC로 옮길 때). Hub를 끈 채로 하세요. 켜져 있으면 화면의 설정 → Hub 옮기기."""
    from datetime import datetime

    from badamoa.hub import migration
    from badamoa.hub.config import HubSettings

    settings = HubSettings(home=component_home("hub", home))
    folder = out.expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"badamoa-hub-{datetime.now().strftime('%Y%m%d-%H%M')}.tar"
    typer.echo(f"내보냅니다: {settings.home} → {path}")
    done = migration.export(settings, path)
    typer.echo(f"다 됐습니다: {path} ({done['bytes'] / 1024**3:.1f} GB, 파일 {done['files']:,}개)")
    typer.echo(f"새 PC에서: badamoa hub import {path.name}  (그 PC의 Hub를 끈 채로)")
    typer.echo("이 Hub를 다시 켜서 바꾼 것은 새 Hub에 가지 않습니다. 켜 둔 채 옮기려면 화면의 설정 → Hub 옮기기를 쓰세요.")


@hub_app.command("import")
def hub_import(
    file: Annotated[Path, typer.Argument(help="다른 PC의 Hub에서 내보낸 파일 (badamoa-hub-….tar)")],
    home: HomeOpt = None,
    force: Annotated[bool, typer.Option(help="이미 쓰던 Hub가 있어도 (지금 것은 옆 폴더로 옮겨 둠)")] = False,
) -> None:
    """다른 PC에서 내보낸 Hub를 이 PC로 가져옵니다. 이 PC의 Hub를 끈 채로 하고, 다 되면 Hub를 켭니다."""
    from badamoa.hub import migration
    from badamoa.hub.services import ServiceError

    target = component_home("hub", home)
    try:
        done = migration.import_file(file.expanduser().resolve(), target, force=force, log_line=typer.echo)
    except ServiceError as e:
        typer.secho(e.message, fg="red")
        raise typer.Exit(1) from e
    made = done["manifest"]
    typer.echo(f"가져왔습니다: {made.get('exported_at')}에 BADAMOA {made.get('version')}에서 내보낸 Hub → {target}")
    typer.echo("이제 Hub를 켜고(badamoa hub run), 예전 Hub의 설정 → Hub 옮기기에서 이 Hub 주소를 적으면 노드가 넘어옵니다.")


@hub_app.command("publish-wheel")
def hub_publish_wheel(wheel: Path, home: HomeOpt = None) -> None:
    """원격 노드 설치용 wheel을 Hub에 등록합니다 (Hub가 /install/simple/ 로 서빙)."""
    target = component_home("hub", home) / "wheels"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wheel, target / wheel.name)
    typer.echo(f"등록됨: {target / wheel.name}")


@hub_app.command("install-service")
def hub_install_service(
    home: HomeOpt = None,
    port: Annotated[int, typer.Option(help="포트")] = 8470,
    public_url: Annotated[str | None, typer.Option(help="원격 노드·에이전트가 접속할 URL")] = None,
) -> None:
    """이 데이터 폴더 · 포트의 Hub가 PC가 켜질 때 자동으로 실행되게 합니다 (화면의 설정 → Hub 실행과 같음).

    관리자 권한(root)이면 로그인 없이 부팅 때, 아니면 Windows는 이 계정이 로그인할 때 · Linux는 이 계정의 crontab으로.
    """
    from badamoa.hub import autostart
    from badamoa.hub.config import HubSettings

    settings = HubSettings(home=component_home("hub", home), port=port, public_url=public_url, local_node=True)
    autostart.save_launch(settings)
    why = autostart.turn_on(settings.home)
    if why:
        typer.secho(f"등록하지 못했습니다: {why}", fg="red")
        raise typer.Exit(1)
    view = autostart.status(settings.home, port)
    when = "로그인 없이 부팅 때" if view["when"] == "boot" else "이 계정이 로그인할 때"
    typer.echo(f"{view['name']}: {when} 실행합니다. 로그: {view['log']}")


@hub_app.command("uninstall-service")
def hub_uninstall_service(home: HomeOpt = None, port: Annotated[int, typer.Option(help="포트")] = 8470) -> None:
    """Hub 자동 실행을 끕니다."""
    from badamoa.hub import autostart

    why = autostart.turn_off(component_home("hub", home), port)
    if why:
        typer.secho(why, fg="red")
        raise typer.Exit(1)
    typer.echo("자동 실행을 껐습니다.")


# ---- node --------------------------------------------------------------------


@node_app.command("join")
def node_join(
    hub: Annotated[str, typer.Option(help="Hub URL, 예: http://192.168.0.10:8470")],
    token: Annotated[str, typer.Option(help="Hub UI의 '노드 추가'에서 받은 join token")],
    name: Annotated[str | None, typer.Option(help="노드 표시 이름 (기본: 호스트명)")] = None,
    home: HomeOpt = None,
    rejoin: Annotated[bool, typer.Option(help="이미 등록되어 있어도 새 노드로 다시 등록")] = False,
) -> None:
    """이 머신을 Hub에 노드로 등록합니다. 이미 그 Hub에 등록된 머신이면 등록은 그대로 둡니다."""
    from badamoa.node import agent, service

    node_dir = agent.node_home(home)
    if not rejoin:
        try:  # installed with sudo where the node ran without it: the same node, not a second one
            took = service.take_over_user_node(node_dir)
        except RuntimeError as e:
            typer.secho(str(e), fg="red")
            raise typer.Exit(1) from e
        if took:
            typer.echo(took)
    existing = None if rejoin else agent.already_joined(hub, node_dir)
    if existing is not None:
        typer.secho(f"이미 이 Hub에 등록된 노드입니다: node_id={existing.node_id}. 등록은 그대로 둡니다.", fg="green")
        return
    try:
        config = agent.join(hub, token, node_dir, name)
    except Exception as e:  # noqa: BLE001
        typer.secho(str(e), fg="red")
        raise typer.Exit(1) from e
    typer.secho(f"등록 완료: node_id={config.node_id}", fg="green")
    typer.echo(f"설정 파일: {agent.NodeConfig.path(node_dir)}")
    typer.echo("이제 `badamoa node run` 또는 `badamoa node install-service`를 실행하세요.")


@node_app.command("run")
def node_run(
    home: HomeOpt = None,
    max_jobs: Annotated[int | None, typer.Option(help="동시에 실행할 작업 수 (기본: 4와 GPU 수 중 큰 값)")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Node Agent를 실행합니다 (Ctrl+C로 종료)."""
    _logging(verbose)
    from badamoa.node import agent, service

    node_dir = agent.node_home(home)
    try:
        config = agent.NodeConfig.load(node_dir)
    except FileNotFoundError as e:
        typer.secho(str(e), fg="red")
        raise typer.Exit(1) from e
    lock = service.hold_lock(node_dir)  # held until this process ends
    if lock is None:
        typer.echo(f"이 노드({node_dir})의 Node Agent가 이미 실행 중입니다.")
        return
    runner = agent.NodeAgent(config, node_dir, max_jobs=max_jobs)
    try:
        runner.run_forever()
    except KeyboardInterrupt:
        runner.stop()


@node_app.command("status")
def node_status(home: HomeOpt = None) -> None:
    """이 머신의 노드 설정과 하드웨어 정보를 보여줍니다."""
    from badamoa.core import sysinfo
    from badamoa.node import agent

    node_dir = agent.node_home(home)
    try:
        config = agent.NodeConfig.load(node_dir)
        typer.echo(f"hub     : {config.hub_url}\nnode_id : {config.node_id}")
    except FileNotFoundError:
        typer.echo("아직 Hub에 등록되지 않았습니다.")
    info = sysinfo.static_info()
    typer.echo(f"host    : {info['hostname']} ({info['os_release']})")
    for gpu in info["gpus"]:
        typer.echo(f"gpu {gpu['index']}   : {gpu['name']} {gpu['memory_total_mb']} MB (driver {gpu['driver_version']})")


@node_app.command("start")
def node_start(home: HomeOpt = None) -> None:
    """Node Agent를 백그라운드로 켭니다. 이미 켜져 있으면 그대로 둡니다 (sudo 없이 설치한 노드가 부팅 시 씀)."""
    from badamoa.node import agent, service

    node_dir = agent.node_home(home)
    if not agent.NodeConfig.path(node_dir).is_file():
        typer.secho(f"노드 설정이 없습니다: {node_dir}. 먼저 `badamoa node join`을 실행하세요.", fg="red")
        raise typer.Exit(1)
    started = service.start(node_dir)
    typer.echo(f"Node Agent를 켰습니다. 로그: {node_dir / service.LOG}" if started else "Node Agent가 이미 실행 중입니다.")


@node_app.command("install-service")
def node_install_service() -> None:
    """부팅 시 Node Agent가 자동 실행되도록 등록하고 켭니다. root면 systemd, sudo가 없는 계정이면 그 계정의 crontab."""
    from badamoa.node import service

    typer.echo(service.install("node", "node run"))


@node_app.command("trust-registries")
def node_trust_registries(
    hub: Annotated[str, typer.Option(help="Hub 주소 (기본: 이 노드가 등록된 Hub)")] = "",
    home: HomeOpt = None,
) -> None:
    """Hub의 모델 종류 · 환경이 쓰는 사내 Docker 레지스트리 중 HTTPS가 없는 것을 이 서버의 Docker에 허용합니다 (root).

    /etc/docker/daemon.json의 insecure-registries에 넣고 Docker가 다시 읽게 합니다(돌고 있는 컨테이너는 그대로).
    노드 설치 · 업데이트(sudo)가 부르고, sudo 없이 설치한 노드에는 관리자가 <hub>/install/docker-registries.sh로 한 번."""
    import httpx

    from badamoa.node import agent, registries

    url = hub.rstrip("/")
    if not url:
        try:
            url = agent.NodeConfig.load(agent.node_home(home)).hub_url
        except FileNotFoundError as e:
            raise _fail(str(e)) from e
    if not shutil.which("docker"):
        typer.echo("이 서버에 Docker가 없어 할 일이 없습니다.")
        return
    try:
        names = httpx.get(f"{url}/install/registries", timeout=15).raise_for_status().json().get("registries", [])
    except (httpx.HTTPError, ValueError) as e:
        raise _fail(f"Hub에서 레지스트리 목록을 받지 못했습니다 ({url}): {e}") from e
    have = set(registries.listed())
    wanted = [r for r in names if r not in have and registries.plain(r)]
    if not wanted:
        already = f" (이미 허용: {', '.join(sorted(have))})" if have else ""
        typer.echo(f"Docker에 더 허용할 레지스트리가 없습니다{already}.")
        return
    if not registries.can_write():
        typer.echo(
            f"HTTPS 없는 레지스트리 {', '.join(wanted)}: root만 Docker 설정을 바꿀 수 있습니다. "
            f"관리자가 한 번 실행하세요: {registries.command(url)}"
            if sys.platform.startswith("linux")
            else f'HTTPS 없는 레지스트리 {", ".join(wanted)}: Docker Desktop의 Settings → Docker Engine에 '
            f'"insecure-registries": {list(wanted)}를 넣고 Apply & restart.'
        )
        return
    registries.trust(wanted, typer.echo)


@node_app.command("uninstall-service")
def node_uninstall_service() -> None:
    from badamoa.node import service

    typer.echo(service.uninstall("node"))


if __name__ == "__main__":
    app()


# ---- template ----------------------------------------------------------------


def _fail(message: str) -> typer.Exit:
    typer.secho(message, fg="red", err=True)
    return typer.Exit(1)


@template_app.command("init")
def template_init(
    folder: Annotated[Path, typer.Argument(help="만들 폴더")],
    name: Annotated[str | None, typer.Option(help="템플릿 이름 (기본: 폴더 이름)")] = None,
    task: Annotated[str, typer.Option(help="detect | segment | classify")] = "detect",
) -> None:
    """새 모델 템플릿의 뼈대를 만듭니다: 선언 파일, 학습 코드 예시, SDK(badamoa_sdk.py)."""
    import re

    from badamoa import templates
    from badamoa.core import tasks as task_names
    from badamoa.core.templates import MANIFEST

    try:
        task = task_names.normalize(task) or "detect"
    except ValueError as e:
        raise _fail(str(e)) from e
    if folder.exists() and any(folder.iterdir()):
        raise _fail(f"{folder} 폴더가 비어 있지 않습니다.")
    title = name or folder.resolve().name
    slug = re.sub(r"[^a-z0-9._-]+", "-", title.lower()).strip("-._") or "my-model"
    folder.mkdir(parents=True, exist_ok=True)
    for source in templates.SCAFFOLD.iterdir():
        if not source.is_file():
            continue
        text = source.read_text(encoding="utf-8")
        if source.name == MANIFEST:
            text = text.replace("__NAME__", slug).replace("__TITLE__", title).replace("__TASK__", task)
        (folder / source.name).write_text(text, encoding="utf-8", newline="\n")
    sdk = templates.SDK / "badamoa_sdk.py"  # a copy, so the code also runs outside BADAMOA
    (folder / sdk.name).write_text(sdk.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    typer.echo(f"템플릿 뼈대를 만들었습니다: {folder}")
    typer.echo("  1. train.py의 표시한 부분을 학습 코드로 바꿉니다")
    typer.echo(f"  2. badamoa template check {folder}")
    typer.echo(f"  3. badamoa template push {folder} --hub http://<hub>:8470 --token <API 토큰>")


STEP_LABEL = {
    "data": "시험 데이터",
    "env": "환경",
    "train": "학습",
    "predict": "테스트",
    "autolabel": "오토라벨",
    "eval": "평가",
    "export": "내보내기",
}


def _print_trial(result: dict) -> None:
    """A trial's verdict, step by step, as the Hub's admin screen shows it."""
    from badamoa.core import tasks as task_names

    for step in result.get("steps") or []:
        label = STEP_LABEL.get(step["step"], step["step"])
        if step.get("task"):  # a kind of several tasks: each task's pass
            label = f"{task_names.short(step['task'])} {label}"
        if step.get("skipped"):
            typer.echo(f"  건너뜀 {label}: {step['detail']}")
        else:
            typer.echo(f"  통과 {label} ({step['took_s']:.0f}초): {step['detail']}")
    failed = result.get("failed")
    if failed:
        typer.echo(f"  실패 {STEP_LABEL.get(failed, failed)}: {result.get('reason')}")
    elif result.get("reason"):
        typer.echo(f"  {result['reason']}")
    for note in result.get("notes") or []:
        typer.echo(f"  알림: {note}")


def _write_report(folder: Path, manifest, result: dict) -> Path:
    """What ``check --run`` found, for ``push`` to send with the bundle: the steps, this PC and the bundle's hash."""
    import hashlib
    import json
    import platform
    from datetime import UTC, datetime

    from badamoa import __version__
    from badamoa.core import sysinfo
    from badamoa.core.templates import CONTRACT, REPORT, bundle

    try:
        gpus = [g.get("name") for g in sysinfo.static_info().get("gpus") or []]
    except Exception:  # noqa: BLE001 - only a description of this PC
        gpus = []
    report = {
        "ok": bool(result.get("ok")),
        "steps": result.get("steps") or [],
        "failed": result.get("failed") or "",
        "reason": result.get("reason") or "",
        "notes": result.get("notes") or [],
        "took_s": result.get("took_s"),
        "sdk": __version__,
        "contract": CONTRACT,
        "python": platform.python_version(),
        "os": platform.platform(),
        "gpus": gpus,
        "device": result.get("device") or "",
        "packages": result.get("packages") or [],
        "sources": result.get("sources") or {},
        "bundle_sha256": hashlib.sha256(bundle(folder)).hexdigest(),
        "name": manifest.name,
        "version": manifest.version,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = folder / REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _read_report(folder: Path) -> tuple[dict | None, str]:
    """The last ``check --run`` report if it is for the bundle as it is now and it passed; else why not."""
    import hashlib
    import json

    from badamoa.core.templates import REPORT, bundle

    try:
        report = json.loads((folder / REPORT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "이 폴더에서 badamoa template check . --run을 한 기록이 없습니다."
    if report.get("bundle_sha256") != hashlib.sha256(bundle(folder)).hexdigest():
        return None, "마지막 check --run 뒤에 코드가 바뀌었습니다. 다시 badamoa template check . --run을 하세요."
    if not report.get("ok"):
        return None, "마지막 check --run이 통과하지 않았습니다. 고친 뒤 다시 하세요."
    return report, ""


def _task_pass(manifest, task: str) -> dict:
    """One task's pass of a trial of a kind of several tasks: its data format and settings."""
    from badamoa.core.templates import check_params

    mine = manifest.for_task(task)
    return {"task": task, "data_format": mine.data_format, "params": check_params(mine)}


def _local_trial(folder: Path, manifest, home: Path | None) -> dict:
    """The Hub's trial run, here: the same sample, environment, steps and pass rules, nothing sent anywhere."""
    import time

    from badamoa.core.templates import check_params
    from badamoa.node import checker

    env = manifest.environment
    if env.image and not env.requirements:
        spec = {"kind": "docker", "name": "Docker 이미지", "image": env.image}
    else:
        spec = {"kind": "auto", "name": "자동 설치", "requirements": env.requirements, "python": env.python}
        spec["torch"] = env.torch
    if env.default:
        typer.echo(f"  Hub에 등록된 환경 '{env.default}'은 이 PC에서 알 수 없어 requirements로 설치합니다.")
    params = {
        "template": {
            "name": manifest.name,
            "version": manifest.version,
            "builtin": False,
            "sha256": "",
            "task": manifest.task,
            "data_format": manifest.data_format,
            "resume": manifest.resume,
            "autolabel": manifest.autolabel,
            "entrypoints": manifest.entrypoints,
        },
        "environment": spec,
        "params": check_params(manifest.for_task(manifest.task)),
        # a kind of several tasks: each in turn, as the server's trial does
        **({"tasks": [_task_pass(manifest, t) for t in manifest.trains()]} if len(manifest.trains()) > 1 else {}),
        "check": manifest.check.model_dump(exclude={"params"}),
        "data": manifest.data.model_dump(),
        "code": str(folder),
    }
    where = (home or Path.home() / ".badamoa" / "template-trial").expanduser().resolve()
    typer.echo(f"이 PC에서 시험 실행합니다 (환경 · 캐시: {where})")
    return checker.attempt(params, where, f"local-{int(time.time())}")


@template_app.command("check")
def template_check(
    folder: Annotated[Path, typer.Argument(help="템플릿 폴더")] = Path("."),
    run: Annotated[
        bool, typer.Option("--run", help="올린 뒤 서버가 하는 시험 실행(합성 데이터로 학습 → 테스트 · 평가)을 이 PC에서")
    ] = False,
    home: Annotated[Path | None, typer.Option(help="--run의 환경 · 캐시를 둘 폴더 (기본 ~/.badamoa/template-trial)")] = None,
) -> None:
    """선언 파일(badamoa.toml)이 올바른지 확인합니다. --run이면 서버처럼 짧게 돌려 봅니다."""
    from badamoa.core.templates import TemplateError, bundle, load_manifest

    try:
        manifest = load_manifest(folder)
        size = len(bundle(folder))
    except TemplateError as e:
        raise _fail(str(e)) from e
    typer.echo(f"{manifest.title} ({manifest.name} {manifest.version})")
    if len(manifest.trains()) > 1:
        typer.echo("  태스크 : " + " · ".join(f"{t} ({manifest.for_task(t).data_format})" for t in manifest.trains()))
    else:
        typer.echo(f"  태스크 : {manifest.task} · 데이터 형식 {manifest.data_format}")
    typer.echo(f"  진입점 : {', '.join(manifest.entrypoints)}{' · 오토라벨' if manifest.autolabel else ''}")
    if manifest.reads_weights_from:
        typer.echo(f"  이전 버전 : {manifest.reads_weights_from}로 학습한 모델에도 이 버전의 코드를 씀")
    basic = [name for name, spec in manifest.params.items() if not spec.advanced]
    detailed = [name for name, spec in manifest.params.items() if spec.advanced]
    typer.echo(f"  설정   : {', '.join(basic) or '(없음)'}")
    if detailed:
        typer.echo(f"  세부 설정 : {', '.join(detailed)}")
    environment = manifest.environment.default or ("자동 설치" if manifest.environment.requirements else "노드의 Python")
    typer.echo(f"  환경   : {environment}")
    typer.echo(f"  묶음   : {size / 1024:.0f} KB")
    for note in manifest.notes:
        typer.echo(f"  알림   : {note}")
    if not run:
        return
    result = _local_trial(folder.resolve(), manifest, home)
    typer.echo("시험 실행 결과:")
    _print_trial(result)
    written = _write_report(folder.resolve(), manifest, result)
    if not result["ok"]:
        raise typer.Exit(1)
    typer.echo(f"통과 ({result['took_s']:.0f}초). 보고서: {written}")
    typer.echo("badamoa template push가 이 보고서를 함께 보냅니다. Hub에서는 관리자가 다시 시험 실행하고 허용합니다.")


@template_app.command("push")
def template_push(
    folder: Annotated[Path, typer.Argument(help="템플릿 폴더")] = Path("."),
    hub: Annotated[str, typer.Option(envvar="BADAMOA_HUB_URL", help="Hub 주소")] = "http://localhost:8470",
    token: Annotated[str, typer.Option(envvar="BADAMOA_TOKEN", help="관리자의 API 토큰 (웹 UI → 에이전트 연결)")] = "",
    trial: Annotated[
        bool, typer.Option("--trial", help="올린 뒤 학습 서버에서 시험 실행을 시작하고 끝날 때까지 기다립니다")
    ] = False,
    wait_min: Annotated[int, typer.Option(help="--trial로 기다릴 최대 시간(분)")] = 60,
) -> None:
    """템플릿을 Hub에 올립니다. 코드를 바꿔 다시 올릴 때는 badamoa.toml의 version을 올리세요.

    먼저 `badamoa template check . --run`을 통과하세요: 그 보고서를 함께 보내고, 관리자가 아닌 계정은 보고서
    없이 올릴 수 없습니다(올린 것은 초안). 올린 버전은 "시험 전"입니다. 관리자가 시험 실행하고(--trial이면
    여기서) 통과한 것을 설정 → 모델 종류에서 "허용"하면 모두의 학습 화면에 나옵니다."""
    import httpx

    from badamoa.core.templates import TemplateError, bundle

    if not token:
        raise _fail("API 토큰이 필요합니다: --token 또는 환경 변수 BADAMOA_TOKEN")
    try:
        data = bundle(folder)
    except TemplateError as e:
        raise _fail(str(e)) from e
    base = hub.rstrip("/")
    headers = {"Authorization": f"Bearer {token}"}
    report, why = _read_report(folder.resolve())
    upload = {**headers, "Content-Type": "application/zip"}
    if report is not None:  # the proof that it ran here: required from members, kept with the version
        import base64
        import gzip
        import json

        upload["X-Badamoa-Check"] = base64.b64encode(gzip.compress(json.dumps(report).encode("utf-8"))).decode()
        typer.echo(f"이 PC의 시험 실행 보고서를 함께 보냅니다 ({report.get('at', '')}).")
    else:
        typer.echo(f"알림: {why} (관리자가 아니면 Hub가 받지 않습니다)")
    try:
        resp = httpx.post(f"{base}/api/templates", content=data, headers=upload, timeout=120)
    except httpx.HTTPError as e:
        raise _fail(f"Hub에 연결하지 못했습니다 ({hub}): {e}") from e
    if resp.status_code != 200:
        raise _fail(f"등록 실패 ({resp.status_code}): {_detail(resp)}")
    registered = resp.json()
    typer.echo(f"등록됨: {registered['title']} ({registered['name']} {registered['version']})")
    check = registered.get("check") or {}
    if not trial:
        state = check.get("state")
        if state in ("pending", "unchecked"):
            typer.echo('시험 전입니다. 설정 → 모델 종류에서 "시험 실행" 후 통과하면 "허용"하세요 (--trial로 여기서 시작).')
        else:
            typer.echo(f"같은 내용이 이미 있어 앞 상태를 그대로 씁니다: {state}")
        return
    started = httpx.post(f"{base}/api/templates/{registered['id']}/check", headers=headers, timeout=60)
    if started.status_code != 200:
        raise _fail(f"시험 실행을 시작하지 못했습니다: {_detail(started)}")
    check = started.json().get("check") or {}
    import time

    typer.echo(f"{check.get('node') or '학습 서버'}에서 시험 실행 중… (환경을 처음 만들면 몇 분 걸립니다)")
    deadline = time.monotonic() + wait_min * 60
    while check.get("state") == "checking" and time.monotonic() < deadline:
        time.sleep(3)
        try:
            view = httpx.get(f"{base}/api/templates/{registered['id']}", headers=headers, timeout=30)
            check = (view.json().get("check") or check) if view.status_code == 200 else check
        except (httpx.HTTPError, ValueError):
            continue
    if check.get("state") == "checking":
        raise _fail(f"{wait_min}분 안에 끝나지 않았습니다. 설정 → 모델 종류에서 결과를 보세요.")
    _print_trial(check)
    if check.get("state") != "passed":
        raise _fail("시험 실행을 통과하지 못했습니다 (설정 → 모델 종류에서 로그를 보세요).")
    typer.echo('통과: 설정 → 모델 종류에서 "허용"하면 모두의 학습 화면에 나옵니다.')


def _detail(resp) -> str:
    try:
        return str(resp.json().get("detail", resp.text))
    except ValueError:
        return resp.text
