"""`badamoa` command line."""

from __future__ import annotations

import logging
import shutil
import socket
import threading
from pathlib import Path
from typing import Annotated

import typer

from badamoa import __version__
from badamoa.core.paths import component_home

app = typer.Typer(help="BADAMOA — 설치형 비전 MLOps 플랫폼", no_args_is_help=True, add_completion=False)
hub_app = typer.Typer(help="Hub(제어 서버) 실행·관리", no_args_is_help=True)
node_app = typer.Typer(help="Node Agent(학습/저장/추론 서버) 실행·관리", no_args_is_help=True)
app.add_typer(hub_app, name="hub")
app.add_typer(node_app, name="node")

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
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Hub를 실행합니다. 기본값은 All-in-one(이 머신도 노드로 동작)입니다."""
    _require_hub()
    _logging(verbose)
    import uvicorn

    from badamoa.hub.app import create_app
    from badamoa.hub.config import HubSettings

    settings = HubSettings(
        home=component_home("hub", home), host=host, port=port, public_url=public_url, local_node=local_node
    )
    application = create_app(settings)
    if local_node:
        _start_local_node(application, settings)

    url = public_url or f"http://{_lan_ip()}:{port}"
    typer.secho(f"\n  BADAMOA Hub {__version__}", bold=True)
    typer.echo(f"  웹 UI   : http://localhost:{port}   (네트워크: {url})")
    typer.echo(f"  MCP     : {url}/mcp")
    typer.echo(f"  데이터  : {settings.home}\n")
    uvicorn.run(application, host=host, port=port, log_level="debug" if verbose else "warning")


def _start_local_node(application, settings) -> None:
    from badamoa.core import sysinfo
    from badamoa.hub import services
    from badamoa.node.agent import NodeAgent, NodeConfig

    hub = application.state.hub
    with hub.db.session() as db:
        node, token = services.ensure_local_node(db, sysinfo.static_info(), __version__)
        if token is None:  # existing local node: issue a fresh token for this process
            token = services.rotate_node_token(db, node)
        node_id = node.id
    home = settings.local_node_home
    home.mkdir(parents=True, exist_ok=True)
    config = NodeConfig(hub_url=f"http://127.0.0.1:{settings.port}", node_id=node_id, node_token=token)
    agent = NodeAgent(config, home)
    threading.Thread(target=agent.run_forever, name="local-node", daemon=True).start()


@hub_app.command("publish-wheel")
def hub_publish_wheel(wheel: Path, home: HomeOpt = None) -> None:
    """원격 노드 설치용 wheel을 Hub에 등록합니다 (Hub가 /install/simple/ 로 서빙)."""
    target = component_home("hub", home) / "wheels"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wheel, target / wheel.name)
    typer.echo(f"등록됨: {target / wheel.name}")


@hub_app.command("install-service")
def hub_install_service(port: int = 8470) -> None:
    """부팅 시 Hub가 자동 실행되도록 등록합니다 (관리자 권한 필요)."""
    from badamoa.node import service

    typer.echo(service.install("hub", f"hub run --port {port}"))


@hub_app.command("uninstall-service")
def hub_uninstall_service() -> None:
    from badamoa.node import service

    typer.echo(service.uninstall("hub"))


# ---- node --------------------------------------------------------------------


@node_app.command("join")
def node_join(
    hub: Annotated[str, typer.Option(help="Hub URL, 예: http://192.168.0.10:8470")],
    token: Annotated[str, typer.Option(help="Hub UI의 '노드 추가'에서 받은 join token")],
    name: Annotated[str | None, typer.Option(help="노드 표시 이름 (기본: 호스트명)")] = None,
    home: HomeOpt = None,
) -> None:
    """이 머신을 Hub에 노드로 등록합니다."""
    from badamoa.node import agent

    node_dir = agent.node_home(home)
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
    from badamoa.node import agent

    node_dir = agent.node_home(home)
    try:
        config = agent.NodeConfig.load(node_dir)
    except FileNotFoundError as e:
        typer.secho(str(e), fg="red")
        raise typer.Exit(1) from e
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


@node_app.command("install-service")
def node_install_service() -> None:
    """부팅 시 Node Agent가 자동 실행되도록 등록합니다 (관리자 권한 필요)."""
    from badamoa.node import service

    typer.echo(service.install("node", "node run"))


@node_app.command("uninstall-service")
def node_uninstall_service() -> None:
    from badamoa.node import service

    typer.echo(service.uninstall("node"))


if __name__ == "__main__":
    app()
