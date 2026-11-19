"""MCP server (``/mcp``, Streamable HTTP).

The Hub *is* the MCP server: the built-in agent runtime and external agents
(Claude Code, …) are all clients of this one surface, and every tool goes
through the same ``services`` layer and RBAC as the REST API.

Auth: ``Authorization: Bearer <API token>``. An ASGI guard rejects anonymous
requests; each tool re-authenticates from the request headers so calls are
attributed to the right user.
"""

from __future__ import annotations

import functools
import json
import time
from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp.server.streamable_http_manager import StreamableHTTPASGIApp
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from badamoa.core.protocol import TERMINAL_STATUSES

from . import (
    catalog,
    checks,
    drafts,
    gpus,
    labeling,
    lineage,
    node_data,
    qa,
    registry,
    resources,
    review,
    services,
    sweeps,
    training,
)
from .deps import Hub
from .install import join_commands
from .models import User

INSTRUCTIONS = """\
BADAMOA는 설치형 비전 MLOps 플랫폼입니다. 이 서버로 Hub에 연결된 노드(학습/저장/추론 서버)를
조회하고 작업(Job)을 실행·추적할 수 있습니다.

- 먼저 list_nodes로 온라인 노드를 확인하세요.
- run_job은 작업을 큐에 넣고 즉시 반환합니다. wait_for_job으로 완료를 기다린 뒤 get_job_logs로 로그를 봅니다.
- 실패한 작업은 get_job의 error와 로그 끝부분을 함께 확인하세요.
- 학습: list_projects → list_datasets로 릴리스를 고르고 start_training으로 시작합니다.
  어떤 모델로 학습할지는 list_model_templates의 템플릿 이름과 설정 항목에서 고릅니다. 반환된 job_id를
  wait_for_job으로 기다린 뒤 get_training_run(지표 이력)과 list_models(등록된 모델)로 결과를 봅니다.
"""

READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)
DESTRUCTIVE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False)


class BearerGuard:
    """Reject requests without a valid user token before they reach the MCP transport."""

    def __init__(self, app: ASGIApp, hub: Hub) -> None:
        self.app = app
        self.hub = hub

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            if _user_from_header(self.hub, headers.get("authorization")) is None:
                response = JSONResponse(
                    {"error": "unauthorized", "detail": "Authorization: Bearer <BADAMOA API token> 헤더가 필요합니다."},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _user_from_header(hub: Hub, header: str | None) -> User | None:
    scheme, _, token = (header or "").partition(" ")
    if scheme.lower() != "bearer":
        return None
    with hub.db.session() as db:
        user = services.authenticate(db, token.strip())
        if user is not None:
            db.expunge(user)
        return user


def build_mcp(hub: Hub) -> tuple[MCPServer, ASGIApp]:
    mcp = MCPServer(
        name="badamoa",
        title="BADAMOA",
        instructions=INSTRUCTIONS,
        version=services.HUB_VERSION,
        log_level="WARNING",
    )

    def user_of(ctx: Context) -> User:
        headers = ctx.headers or {}
        user = _user_from_header(hub, headers.get("authorization"))
        if user is None:
            raise ToolError("인증되지 않은 요청입니다.")
        return user

    def guarded(fn):
        """Surface ServiceError to the model as an anticipated tool failure (its message is shown)."""

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except services.ServiceError as e:
                raise ToolError(e.message) from e

        return wrapper

    # ---- tools: nodes --------------------------------------------------------

    @mcp.tool(title="노드 목록", annotations=READ_ONLY)
    @guarded
    def list_nodes(ctx: Context) -> list[dict[str, Any]]:
        """Hub에 연결된 모든 노드와 상태(online/offline), OS, 역할, GPU, 현재 사용률을 반환합니다."""
        user_of(ctx)
        with hub.db.session() as db:
            return services.list_nodes(db, hub.settings)

    @mcp.tool(title="노드 상세", annotations=READ_ONLY)
    @guarded
    def get_node(node_id: str, ctx: Context) -> dict[str, Any]:
        """노드 하나의 상세 정보(하드웨어, 드라이버, 최근 지표)를 반환합니다."""
        user_of(ctx)
        with hub.db.session() as db:
            return services.node_view(db, hub.settings, services.get_node(db, node_id))

    @mcp.tool(title="GPU 자리표", annotations=READ_ONLY)
    @guarded
    def get_gpu_board(ctx: Context) -> dict[str, Any]:
        """모든 서버의 GPU를 한 번에: GPU마다 state(free · job · busy · reserved · offline)와 주인
        (BADAMOA 실행의 이름 · epoch · 남은 초 eta_s / 사람의 사용자 · 프로그램), 메모리 · 사용률.
        대기 중인 작업과 그 이유, 가장 먼저 끝날 학습(next_free)도 함께. "지금 어디서 돌릴 수 있나"를 볼 때 씁니다.
        """
        user_of(ctx)
        with hub.db.session() as db:
            return gpus.board(db, hub.settings)

    @mcp.tool(title="노드 자원 이력", annotations=READ_ONLY)
    @guarded
    def get_node_metrics(node_id: str, ctx: Context, range: str = "1h") -> dict[str, Any]:
        """노드의 자원 사용 이력을 요약합니다(시리즈별 평균·최대·최근값).

        range: 1h | 6h | 24h | 7d | 30d. 시리즈 예: cpu, mem, gpu0_util, gpu0_mem, gpu0_temp, power,
        disk_pct:<마운트>. 디스크 증가 추세와 가득 차는 예상 시점도 함께 반환합니다.
        """
        user_of(ctx)
        with hub.db.session() as db:
            node = services.get_node(db, node_id)
            try:
                data = resources.summarize(db, node_id, range)
            except ValueError as e:
                raise ToolError(str(e)) from e
            data["disk_forecast"] = resources.disk_forecast(db, node)
            return data

    @mcp.tool(title="노드 추가 명령 생성", annotations=WRITE)
    @guarded
    def create_node_join_command(ctx: Context, roles: list[str] | None = None, ttl_minutes: int = 60) -> dict[str, Any]:
        """원격 머신을 노드로 추가하는 1회용 설치 명령(Linux/Windows)을 만듭니다. 관리자 전용.

        roles: compute | storage | inference 중 선택 (기본: 전부).
        반환된 명령을 원격 머신의 터미널에서 실행하면 노드가 Hub에 등록됩니다.
        """
        user = user_of(ctx)
        with hub.db.session() as db:
            token, row = services.create_join_token(db, user, roles, ttl_minutes)
            services.audit(db, user.id, "mcp", "node.join_token", row.id, roles=row.roles)
            hub_url = (hub.settings.public_url or _origin(ctx) or "").rstrip("/")
            return {
                "roles": row.roles,
                "expires_at": row.expires_at.isoformat() + "Z",
                "commands": join_commands(hub_url, token),
            }

    # ---- tools: jobs ---------------------------------------------------------

    @mcp.tool(title="작업 유형 목록", annotations=READ_ONLY)
    @guarded
    def list_job_types(ctx: Context) -> list[dict[str, Any]]:
        """실행할 수 있는 작업 유형과 각 유형의 파라미터 스키마를 반환합니다."""
        user_of(ctx)
        return services.list_job_types()

    @mcp.tool(title="작업 실행", annotations=WRITE)
    @guarded
    def run_job(
        type: str,
        ctx: Context,
        params: dict[str, Any] | None = None,
        node_id: str | None = None,
        gpus: int | None = None,
        min_vram_mb: int = 0,
    ) -> dict[str, Any]:
        """작업을 큐에 넣습니다. node_id를 생략하면 조건에 맞는 아무 온라인 노드가 실행합니다.

        gpus: 필요한 GPU 수(생략하면 작업 유형의 기본값). 다른 사람이 쓰고 있는 GPU에는 배정되지 않으며,
        빈 GPU가 없으면 대기합니다. 이때 반환값의 wait_reason에 이유가 나옵니다.
        즉시 반환하므로 wait_for_job으로 완료를 기다리세요.
        """
        user = user_of(ctx)
        with hub.db.session() as db:
            job = services.create_job(db, user, type, params, node_id, via="mcp", gpus=gpus, min_vram_mb=min_vram_mb)
            return services.job_view(db, hub.settings, job)

    @mcp.tool(title="작업 조회", annotations=READ_ONLY)
    @guarded
    def get_job(job_id: str, ctx: Context) -> dict[str, Any]:
        """작업의 상태, 실행 노드, 결과 또는 오류를 반환합니다."""
        user_of(ctx)
        with hub.db.session() as db:
            return services.job_view(db, hub.settings, services.get_job(db, job_id))

    @mcp.tool(title="작업 완료 대기", annotations=READ_ONLY)
    @guarded
    def wait_for_job(job_id: str, ctx: Context, timeout_seconds: int = 30) -> dict[str, Any]:
        """작업이 끝날 때까지(최대 timeout_seconds, 상한 120초) 기다린 뒤 상태를 반환합니다.

        시간 안에 끝나지 않으면 현재 상태(running 등)를 그대로 반환하니 다시 호출하세요.
        """
        user_of(ctx)
        deadline = time.monotonic() + max(0, min(timeout_seconds, 120))
        while True:
            with hub.db.session() as db:
                job = services.get_job(db, job_id)
                data = services.job_view(db, hub.settings, job)
            if data["status"] in TERMINAL_STATUSES or time.monotonic() >= deadline:
                return data
            time.sleep(0.5)

    @mcp.tool(title="작업 목록", annotations=READ_ONLY)
    @guarded
    def list_jobs(
        ctx: Context, status: str | None = None, node_id: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """최근 작업 목록. status(queued|running|succeeded|failed|canceled)와 node_id로 거를 수 있습니다."""
        user_of(ctx)
        with hub.db.session() as db:
            names = services.node_name_map(db)
            return [services.job_view(db, hub.settings, j, names) for j in services.list_jobs(db, status, node_id, limit)]

    @mcp.tool(title="작업 로그", annotations=READ_ONLY)
    @guarded
    def get_job_logs(job_id: str, ctx: Context, tail: int = 200) -> str:
        """작업 로그의 마지막 tail줄(상한 2000)을 텍스트로 반환합니다."""
        user_of(ctx)
        tail = max(1, min(tail, 2000))
        with hub.db.session() as db:
            job = services.get_job(db, job_id)
            lines = services.get_logs(db, job_id, after=max(0, job.log_count - tail), limit=tail)
        return "\n".join(f"[{line['stream']}] {line['text']}" for line in lines) or "(로그 없음)"

    @mcp.tool(title="작업 취소", annotations=DESTRUCTIVE)
    @guarded
    def cancel_job(job_id: str, ctx: Context) -> dict[str, Any]:
        """대기 중이거나 실행 중인 작업을 취소합니다."""
        user = user_of(ctx)
        with hub.db.session() as db:
            job = services.cancel_job(db, user, job_id, via="mcp")
            return services.job_view(db, hub.settings, job)

    # ---- tools: datasets -----------------------------------------------------

    @mcp.tool(title="프로젝트 목록", annotations=READ_ONLY)
    @guarded
    def list_projects(ctx: Context) -> list[dict[str, Any]]:
        """프로젝트 목록과 각 프로젝트의 릴리스 수·항목 수를 반환합니다."""
        user_of(ctx)
        with hub.db.session() as db:
            return catalog.list_projects(db)

    @mcp.tool(title="데이터셋 목록", annotations=READ_ONLY)
    @guarded
    def list_datasets(project_id: str, ctx: Context) -> list[dict[str, Any]]:
        """프로젝트의 데이터셋 계열과 각 계열의 릴리스(버전, 항목 수, 라벨 공간)를 반환합니다."""
        user_of(ctx)
        with hub.db.session() as db:
            return catalog.tree(db, project_id)

    @mcp.tool(title="릴리스 조회", annotations=READ_ONLY)
    @guarded
    def get_release(release_id: str, ctx: Context) -> dict[str, Any]:
        """릴리스의 통계(항목 수, split, 라벨 상태, 클래스별 박스 수), 라벨 공간, base 릴리스, 지문(digest)을 반환합니다.

        release_id 자리에 id 대신 지문의 앞부분(7자 이상, 커밋 해시처럼)이나 "데이터셋@버전"(예: "street@v3")을 줘도 됩니다.
        same_as: 내용이 똑같은(지문이 같은) 같은 프로젝트의 다른 버전.
        """
        user_of(ctx)
        with hub.db.session() as db:
            return catalog.release_view(db, catalog.find_release(db, release_id).id)

    @mcp.tool(title="버전이 있는 서버", annotations=READ_ONLY)
    @guarded
    def list_version_copies(release_id: str, ctx: Context) -> dict[str, Any]:
        """버전의 이미지가 있는 서버: 등록한 곳(origin)과, 학습하려고 받아 둔 서버들(copies).

        copies마다 크기 · 받은 때 · 그 서버에서 한 학습 수(runs) · 마지막으로 쓴 때 · 지금 쓰는 작업(in_use) ·
        상태(kept · remove · held · removing)와 지울 수 있는지(removable, 아니면 why_not).
        release_id는 지문이나 "데이터셋@버전"도 됩니다.
        """
        user_of(ctx)
        with hub.db.session() as db:
            return node_data.release_copies(db, hub.settings, catalog.find_release(db, release_id))

    @mcp.tool(title="받아 둔 버전 지우기", annotations=DESTRUCTIVE)
    @guarded
    def remove_version_copy(release_id: str, node_id: str, ctx: Context) -> dict[str, Any]:
        """서버가 학습하려고 받아 둔 버전의 사본을 지웁니다. 원본(등록한 폴더)은 건드리지 않고,
        그 서버의 다른 버전과 함께 쓰는 이미지는 남깁니다. 그 서버에서 이 버전을 쓰는 작업이 있으면 거부됩니다.
        다시 학습하면 다시 받습니다."""
        user = user_of(ctx)
        with hub.db.session() as db:
            release = catalog.find_release(db, release_id)
            node_data.remove_copy(db, user, release.id, node_id, via="mcp")
            db.flush()
            return node_data.release_copies(db, hub.settings, release)

    @mcp.tool(title="서버에 받아 둔 버전", annotations=READ_ONLY)
    @guarded
    def list_node_copies(node_id: str, ctx: Context) -> dict[str, Any]:
        """서버가 다른 서버에서 받아 둔 버전들과 마지막 확인 결과(check: unknown = Hub 기록에 없는 파일)."""
        user_of(ctx)
        with hub.db.session() as db:
            return node_data.node_copies(db, hub.settings, services.get_node(db, node_id))

    @mcp.tool(title="받아 둔 이미지 맞춰 보기", annotations=WRITE)
    @guarded
    def check_node_copies(node_id: str, ctx: Context, clean: bool = False) -> dict[str, Any]:
        """서버가 받아 둔 이미지를 Hub의 기록과 맞춰 봅니다(버전마다 실제로 있는 수). clean=True면
        남길 버전 어느 것도 쓰지 않는 파일을 지웁니다. 작업을 반환하므로 wait_for_job 뒤 list_node_copies로 확인하세요."""
        user = user_of(ctx)
        with hub.db.session() as db:
            job = node_data.check_node(db, user, node_id, clean=clean, via="mcp")
            db.flush()
            return services.job_view(db, hub.settings, job)

    @mcp.tool(title="릴리스 비교", annotations=READ_ONLY)
    @guarded
    def compare_releases(release_id: str, other_release_id: str, ctx: Context) -> dict[str, Any]:
        """other_release_id에서 release_id로 가면서 바뀐 것: 추가·제거된 항목, 라벨이 바뀐 항목, 클래스별 박스 수 증감."""
        user_of(ctx)
        with hub.db.session() as db:
            return catalog.compare(db, release_id, other_release_id)

    @mcp.tool(title="릴리스 만들기", annotations=WRITE)
    @guarded
    def create_release(
        sources: list[str],
        version: str,
        ctx: Context,
        dataset: str = "",
        note: str = "",
        filter: dict[str, Any] | None = None,
        exclude: bool = False,
        split: dict[str, float] | None = None,
        group_by: str = "",
        classes: dict[str, str] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """기존 릴리스의 항목으로 새 릴리스를 만듭니다. 이미지는 옮기지 않습니다. 화면에서는 릴리스를 "버전"이라고 부릅니다.

        sources: 담을 릴리스 id들(여러 개면 합침, 같은 항목은 앞의 것의 split · 라벨). 클래스 목록이 같아야 합니다.
        dataset: 비우면 첫 릴리스의 데이터셋에 새 버전으로. filter: {"split", "state", "class_id", "site", "camera",
        "q"} 중 일부 — 맞는 항목만(exclude=true면 맞는 항목을 뺀 나머지). split: {"train": 80, "val": 10, "test": 10}처럼
        주면 다시 나눔(해시로 정해 같은 항목은 늘 같은 쪽). group_by: "camera" 또는 "site"면 같은 카메라 · 사이트는 같은
        split으로. classes: {"원래 클래스 번호": "새 이름"} — 이름 바꾸기, 같은 새 이름이면 합치기, ""이면 빼기(분류는 그
        클래스의 이미지를 뺌). 라벨은 새 개정으로 기록되고 원래 릴리스는 그대로입니다. dry_run=true면 만들지 않고 split별
        항목 수와 클래스 수만 반환합니다.
        """
        user = user_of(ctx)
        with hub.db.session() as db:
            return catalog.new_release(
                db,
                user,
                sources=sources,
                version=version,
                dataset=dataset,
                note=note,
                match=filter,
                exclude=exclude,
                split=split,
                group_by=group_by,
                classes=classes,
                dry_run=dry_run,
                via="mcp",
            )

    @mcp.tool(title="데이터셋 가져오기", annotations=WRITE)
    @guarded
    def import_dataset(
        project_id: str,
        node_id: str,
        path: str,
        dataset: str,
        version: str,
        ctx: Context,
        format: str = "auto",
        note: str = "",
        task: str = "",
        unlabelled: str = "auto",
        classes: list[str] | None = None,
    ) -> dict[str, Any]:
        """노드에 있는 YOLO/COCO 형식 폴더를 복사 없이 새 릴리스로 등록합니다.

        node_id는 데이터가 있는 노드, path는 그 노드에서의 폴더 경로입니다. format: auto | yolo | coco.
        task: 데이터셋의 태스크 — Det · Seg · Kpt · Cls 또는 팀이 정한 이름(예: OCR). 비우면 폴더를 보고 정합니다.
        이미 있는 데이터셋의 새 버전이면 그 데이터셋의 태스크를 따릅니다.
        unlabelled: 라벨 파일이 없는 이미지 — "background"(배경), "unlabelled"(아직 라벨 안 함), "auto"(폴더에 라벨
        파일이 하나도 없으면 아직 안 함, 아니면 배경). classes: 클래스 이름이 없는 폴더(라벨할 이미지)에 쓸 클래스 목록.
        작업을 반환하므로 wait_for_job으로 완료를 기다린 뒤 result.release_id를 get_release로 확인하세요.
        """
        user = user_of(ctx)
        with hub.db.session() as db:
            job = catalog.start_import(
                db,
                user,
                project_id=project_id,
                node_id=node_id,
                path=path,
                dataset=dataset,
                version=version,
                fmt=format,
                note=note,
                task=task,
                unlabelled=unlabelled,
                classes=classes,
                via="mcp",
            )
            return services.job_view(db, hub.settings, job)

    @mcp.tool(title="릴리스 이미지 목록", annotations=READ_ONLY)
    @guarded
    def list_release_items(
        release_id: str,
        ctx: Context,
        split: str | None = None,
        state: str = "",
        class_id: int | None = None,
        q: str = "",
        edited: bool = False,
        suspect: str = "",
        label_set: str = "",
        part: int | None = None,
        check: str = "",
        mistake: str = "",
        let_be: bool = False,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """릴리스의 이미지와 라벨(박스는 [클래스, cx, cy, w, h] — 이미지 크기에 대한 비율)을 페이지로 반환합니다.

        state: labelled(라벨 있음) | background(배경) | unlabelled(아직 라벨 안 함). edited=true면 데이터셋의
        "작업 중"에서 고친 이미지만. suspect: 평가 id(list_release_checks) — 그 평가가 잰 이미지만, 모델 예측과 라벨이
        어긋난 것(놓친 라벨 + 라벨에 없는 예측)이 많은 순서로, 이미지마다 check {found, missed, extra}. 이미지 하나의 전체
        라벨과 고칠 수 있는지는 get_release_item으로 봅니다. check(또는 suspect + mistake)로 볼 때 "문제 없음"으로 둔
        이미지는 빠지고, let_be=true면 그것만 나옵니다.
        """
        user_of(ctx)
        with hub.db.session() as db:
            return catalog.list_items(
                db, hub.settings, release_id, split=split, state=state, class_id=class_id, q=q, edited=edited,
                suspect=suspect, label_set=label_set, part=part, check=check, mistake=mistake, let_be=let_be,
                offset=offset, limit=limit,
            )

    @mcp.tool(title="검수: 미라벨 · 오라벨 찾기", annotations=READ_ONLY)
    @guarded
    def get_label_checks(release_id: str, ctx: Context) -> dict[str, Any]:
        """릴리스의 라벨 검사 결과: 규칙 검사(unlabelled 라벨 안 한 이미지, duplicate 같은 물체에 박스 둘, class_overlap 한
        물체에 클래스 둘, rare_class 드문 클래스, tiny 아주 작은 박스, huge 이미지를 거의 덮는 박스)마다 이미지 수, 그리고
        평가한 모델마다 extra(라벨에 없는 물체) · missed(모델이 못 찾은 라벨) 이미지 수. 이미지는 list_release_items의
        check=<검사 id>, 또는 suspect=<평가 id> + mistake=extra|missed로 봅니다. "문제 없음"으로 둔 이미지는 세지 않고
        따로 let_be(규칙) · extra_let_be · missed_let_be(모델)로 셉니다."""
        user_of(ctx)
        with hub.db.session() as db:
            return qa.summary(db, catalog.get_release(db, release_id))

    @mcp.tool(title="검수: 문제 없음으로 두기", annotations=WRITE)
    @guarded
    def let_be_label_check(
        release_id: str, item_id: int, check: str, ctx: Context, note: str = "", undo: bool = False
    ) -> dict[str, Any]:
        """검사에 걸린 이미지의 라벨이 맞다고(문제 없음) 남깁니다. check: 규칙 검사 id 또는 extra | missed(모델).
        그 이미지의 라벨이 그대로인 동안 다음 버전에서도 세지 않고, 라벨이 바뀌면 다시 셉니다. undo=true면 되돌립니다."""
        user = user_of(ctx)
        with hub.db.session() as db:
            return qa.let_be(db, user, catalog.get_release(db, release_id), item_id, check, note, undo=undo)

    @mcp.tool(title="검수 기록", annotations=READ_ONLY)
    @guarded
    def get_release_review(release_id: str, ctx: Context) -> dict[str, Any]:
        """버전의 검수 기록: 작업 중을 확정할 때 남은 것 — 바탕 버전과 이 버전의 걸린 이미지 수(미라벨 · 오라벨 ·
        박스 오류, 검사별), 고친 이미지 수, 문제 없음으로 둔 수, 모델 기준으로 걸린 이미지의 결과, 검수한 사람.
        작업 중이면 지금 확정할 때 남을 기록(preview), 가져온 버전이면 지금 걸린 수(now)."""
        user_of(ctx)
        with hub.db.session() as db:
            return review.view(db, catalog.get_release(db, release_id))

    @mcp.tool(title="라벨링 묶음 목록", annotations=READ_ONLY)
    @guarded
    def list_label_sets(ctx: Context, project_id: str = "") -> list[dict[str, Any]]:
        """라벨링 묶음들: 이미지 수, 상태별 수(todo · labelled · approved · rejected), 일감 수와 보드 칸별 수."""
        user = user_of(ctx)
        with hub.db.session() as db:
            return labeling.list_sets(db, user, project_id)

    @mcp.tool(title="라벨링 묶음 조회", annotations=READ_ONLY)
    @guarded
    def get_label_set(set_id: str, ctx: Context) -> dict[str, Any]:
        """라벨링 묶음의 일감마다 맡은 사람 · 검수자 · 상태별 수 · 보드 칸(unassigned · labeling · review · done)과
        라벨링할 곳(open_release_id). 일감의 이미지는 list_release_items(open_release_id, label_set=…)로 봅니다."""
        user = user_of(ctx)
        with hub.db.session() as db:
            return labeling.set_view(db, user, set_id)

    @mcp.tool(title="라벨링 묶음 만들기", annotations=WRITE)
    @guarded
    def create_label_set(
        release_id: str,
        name: str,
        ctx: Context,
        filter: dict[str, Any] | None = None,
        part_size: int = 50,
        review: bool = True,
        instructions: str = "",
    ) -> dict[str, Any]:
        """최신 버전(또는 작업 중)의 이미지로 라벨링 묶음을 만들어 part_size장씩 일감으로 나눕니다. filter: 버전 화면의
        필터 {"split", "state"(예: "unlabelled"), "class_id", "site", "camera", "q"}와 "edited"(작업 중에서 고친 것),
        "suspect"(평가 id: 그 평가에서 어긋난 이미지, 많이 어긋난 순). 라벨은 데이터셋의 작업 중에 모입니다."""
        user = user_of(ctx)
        with hub.db.session() as db:
            found = labeling.create_set(
                db, user, release_id=release_id, name=name, instructions=instructions, match=filter,
                part_size=part_size, review=review, via="mcp",
            )
            return labeling.set_view(db, user, found.id)

    @mcp.tool(title="라벨 의심 평가 목록", annotations=READ_ONLY)
    @guarded
    def list_release_checks(release_id: str, ctx: Context) -> list[dict[str, Any]]:
        """이 릴리스의 이미지를 "라벨 의심 순"으로 볼 수 있는 모델 평가들: 모델, 평가한 릴리스 · split, 잰 이미지 수, 어긋난
        이미지 수. id를 list_release_items의 suspect로 주면 틀렸을 법한 라벨부터 보고, get_release_item_predictions로
        그 이미지의 예측을 봅니다."""
        user_of(ctx)
        with hub.db.session() as db:
            return checks.release_checks(db, catalog.get_release(db, release_id))

    @mcp.tool(title="이미지의 모델 예측", annotations=READ_ONLY)
    @guarded
    def get_release_item_predictions(release_id: str, item_id: int, check_id: str, ctx: Context) -> dict[str, Any]:
        """평가(check_id)의 모델이 이 이미지에 예측한 라벨(릴리스의 라벨 형식, 확신도와 함께)과 평가할 때의 맞춤 · 놓침 ·
        더 찾음. 예측을 라벨로 넣으려면 fix_labels에 지금 라벨과 합쳐서 줍니다."""
        user_of(ctx)
        with hub.db.session() as db:
            return checks.item_predictions(db, catalog.get_release(db, release_id), item_id, check_id)

    @mcp.tool(title="릴리스 이미지 조회", annotations=READ_ONLY)
    @guarded
    def get_release_item(release_id: str, item_id: int, ctx: Context) -> dict[str, Any]:
        """이미지 하나의 라벨 개정 전부(박스 포함), 들어 있는 버전들, 그리고 edit — 여기서 고칠 수 있는지(allowed ·
        reason)와 고칠 때 바탕이 되는 라벨(revision_id)을 반환합니다."""
        user_of(ctx)
        with hub.db.session() as db:
            return catalog.item_detail(db, hub.settings, release_id, item_id)

    @mcp.tool(title="라벨 고치기", annotations=WRITE)
    @guarded
    def fix_labels(
        release_id: str,
        item_id: int,
        boxes: list[list[float]],
        ctx: Context,
        parent_revision_id: int | None = None,
    ) -> dict[str, Any]:
        """이미지 하나의 라벨을 고쳐 데이터셋의 "작업 중"에 저장합니다. 버전(릴리스)은 바뀌지 않습니다.

        최신 버전이나 작업 중에서만 고칠 수 있습니다. boxes: 이 이미지의 라벨 전부 — [[클래스, cx, cy, w, h], …]
        (비율 0~1), 분류는 [[클래스]], 빈 목록이면 배경(찾을 물체 없음). parent_revision_id: 고치기 시작한 라벨
        (get_release_item의 edit.revision_id) — 그 사이 다른 사람이 고쳤으면 거부됩니다. 반환의 release_id가 작업 중이고,
        confirm_draft로 새 버전이 됩니다.
        """
        user = user_of(ctx)
        with hub.db.session() as db:
            return drafts.save_labels(db, user, release_id, item_id, boxes, parent_id=parent_revision_id, via="mcp")

    @mcp.tool(title="작업 중을 버전으로 확정", annotations=WRITE)
    @guarded
    def confirm_draft(draft_id: str, version: str, ctx: Context, note: str = "") -> dict[str, Any]:
        """데이터셋의 "작업 중"(고친 라벨)을 새 버전으로 확정합니다. 확정한 버전은 바뀌지 않고 학습에 쓸 수 있습니다.

        draft_id: 작업 중의 id(fix_labels가 반환한 release_id, 또는 get_release의 draft.id). note를 비우면 "이미지 N장의
        라벨을 고침"이 들어갑니다.
        """
        user = user_of(ctx)
        with hub.db.session() as db:
            return drafts.confirm(db, user, draft_id, version, note, via="mcp")

    # ---- tools: training and models --------------------------------------------

    @mcp.tool(title="학습 시작", annotations=WRITE)
    @guarded
    def start_training(
        release_id: str,
        ctx: Context,
        template: str = "",
        params: dict[str, Any] | None = None,
        environment: str = "",
        name: str = "",
        model_name: str = "",
        classes: list[int] | None = None,
        unlabelled: str = "exclude",
        gpus: int | None = None,
        base_model_id: str = "",
        tags: list[str] | None = None,
        note: str = "",
        nodes: list[str] | None = None,
        data_after: str = "",
    ) -> dict[str, Any]:
        """릴리스로 학습을 시작합니다. 학습은 그 릴리스의 이미지가 있는 노드에서 실행됩니다.

        release_id 자리에 지문의 앞부분이나 "데이터셋@버전"을 줘도 됩니다.
        nodes: 학습할 수 있는 노드 id들(비우면 이미지가 있는 노드). 이미지가 없는 노드면 먼저 이미지를 받은 뒤 학습합니다.
        data_after: 그렇게 받은 이미지를 학습이 끝난 뒤 keep(남겨 두기, 다음 학습이 바로 시작) 또는
        delete(학습이 성공하면 지움). 남긴 것은 list_version_copies · remove_version_copy로 보고 지웁니다.

        tags · note: 나중에 찾고 묶어 보기 위한 태그와 메모 (예: tags=["야간", "baseline"]).

        base_model_id: 등록된 모델에서 이어서 학습할 때 그 모델의 id (비우면 사전학습 가중치에서 시작).

        template: 모델 템플릿 이름(list_model_templates로 확인, 비우면 프로젝트 기본값 — 보통 yolo).
        params: 그 템플릿의 설정. yolo라면 {"model": "yolo11n.pt", "epochs": 50, "imgsz": 640, "batch": 16}.
        environment: 등록된 환경 이름(비우면 템플릿의 기본 환경, 없으면 자동 설치).
        classes: 학습할 클래스 id(비우면 전체). unlabelled: 라벨 안 된 이미지를 exclude(제외) 또는 background(배경).
        gpus: 쓸 GPU 수(비우면 노드에 GPU가 있을 때 1, 0이면 CPU). GPU가 모두 사용 중이면 대기합니다.
        반환값의 job_id로 wait_for_job / get_job_logs를 쓰고, get_training_run으로 지표를 확인하세요.
        """
        user = user_of(ctx)
        with hub.db.session() as db:
            run = training.start_run(
                db,
                user,
                release_id=catalog.find_release(db, release_id).id,
                template=template,
                environment=environment,
                params=params,
                name=name,
                model_name=model_name,
                classes=classes,
                unlabelled=unlabelled,
                gpus=gpus,
                base_model_id=base_model_id,
                tags=tags,
                note=note,
                nodes=nodes,
                data_after=data_after,
                via="mcp",
            )
            db.flush()
            return training.run_view(db, hub.settings, run)

    @mcp.tool(title="모델 템플릿 목록", annotations=READ_ONLY)
    @guarded
    def list_model_templates(ctx: Context, task: str = "") -> list[dict[str, Any]]:
        """학습에 쓸 수 있는 모델 템플릿과 각 템플릿의 설정 항목(params)을 반환합니다.

        기본 제공: yolo · torchvision(검출), classify(분류: ResNet 등). 사내에서 등록한 것도 함께 나옵니다.
        task로 detect 또는 classify만 볼 수 있습니다. 프로젝트의 task와 같은 템플릿만 그 프로젝트에서 쓸 수 있습니다.
        """
        user = user_of(ctx)
        with hub.db.session() as db:
            return registry.list_templates(db, task=task, admin=user.role == "admin")

    @mcp.tool(title="모델 종류의 환경 미리 준비", annotations=WRITE)
    @guarded
    def prepare_environment(template: str, ctx: Context, node_ids: list[str] | None = None) -> dict[str, Any]:
        """모델 종류(템플릿)의 학습 환경을 첫 학습 전에 서버에 미리 설치합니다 (관리자).

        node_ids를 비우면 켜져 있는 모든 서버에. 준비된 서버는 list_model_templates의 environment.ready에 나옵니다.
        """
        user = user_of(ctx)
        with hub.db.session() as db:
            jobs = registry.prepare_environment(db, user, template, node_ids, via="mcp")
            return {"jobs": [services.job_view(db, hub.settings, j) for j in jobs]}

    @mcp.tool(title="학습 실행 목록", annotations=READ_ONLY)
    @guarded
    def list_training_runs(
        project_id: str, ctx: Context, limit: int = 30, tag: str = "", query: str = ""
    ) -> list[dict[str, Any]]:
        """프로젝트의 학습 실행을 최신순으로 반환합니다: 상태, epoch, 지표(mAP 등), 릴리스, 만들어진 모델, 메모, 태그.

        tag: 이 태그가 붙은 실행만. query: 이름 · 메모 · 태그에서 찾을 글자.
        """
        user_of(ctx)
        with hub.db.session() as db:
            return training.list_runs(db, hub.settings, project_id, limit=limit, tag=tag, q=query)

    @mcp.tool(title="하이퍼파라미터 탐색 시작", annotations=WRITE)
    @guarded
    def start_sweep(
        release_id: str,
        space: dict[str, Any],
        ctx: Context,
        template: str = "",
        params: dict[str, Any] | None = None,
        method: str = "grid",
        metric: str = "",
        goal: str = "",
        max_runs: int = 10,
        parallel: int = 1,
        name: str = "",
        gpus: int | None = None,
        tags: list[str] | None = None,
        nodes: list[str] | None = None,
        data_after: str = "",
    ) -> dict[str, Any]:
        """여러 설정으로 학습을 돌려 가장 좋은 설정을 찾습니다 (W&B Sweeps).

        nodes · data_after: start_training과 같음. data_after=delete면 탐색이 다 끝난 뒤 받은 이미지를 지웁니다.

        space: 시험할 설정. 값 목록 {"lr0": {"values": [0.01, 0.001]}}
        또는 범위 {"lr0": {"min": 0.0001, "max": 0.01, "log": true}}.
        method: grid(값 목록의 모든 조합), random(max_runs번 무작위로 뽑음) 또는 bayes(처음 몇 번은 무작위,
        그다음은 앞선 결과를 보고 고름 — 같은 횟수로 더 좋은 설정을 찾기 쉬움). 범위는 random · bayes에서만.
        params: 모든 실행이 같이 쓰는 설정 (예: {"model": "yolo11n.pt", "epochs": 30}).
        metric · goal: 무엇이 좋은지 (비우면 과제의 주 지표, 손실이면 minimize). parallel: 동시에 돌릴 실행 수.
        실행은 Hub가 차례로 시작합니다. get_sweep으로 진행과 최고 설정을 확인하세요.
        """
        user = user_of(ctx)
        with sweeps.lock, hub.db.session() as db:
            sweep = sweeps.start_sweep(
                db,
                user,
                release_id=catalog.find_release(db, release_id).id,
                space=space,
                template=template,
                params=params,
                method=method,
                metric=metric,
                goal=goal,
                max_runs=max_runs,
                parallel=parallel,
                name=name,
                gpus=gpus,
                tags=tags,
                nodes=nodes,
                data_after=data_after,
                via="mcp",
            )
            return sweeps.sweep_view(db, hub.settings, sweep)

    @mcp.tool(title="하이퍼파라미터 탐색 조회", annotations=READ_ONLY)
    @guarded
    def get_sweep(sweep_id: str, ctx: Context) -> dict[str, Any]:
        """탐색의 진행(시작 · 완료 · 실패 수), 가장 좋은 실행과 그 설정, 실행마다의 설정과 결과(points)를 반환합니다."""
        user_of(ctx)
        with hub.db.session() as db:
            return sweeps.sweep_view(db, hub.settings, sweeps.get_sweep(db, sweep_id), detail=True)

    @mcp.tool(title="하이퍼파라미터 탐색 중단", annotations=WRITE)
    @guarded
    def stop_sweep(sweep_id: str, ctx: Context) -> dict[str, Any]:
        """탐색을 멈춥니다: 더 시작하지 않고, 도는 실행은 중단합니다. 끝난 실행과 모델은 남습니다."""
        user = user_of(ctx)
        with sweeps.lock, hub.db.session() as db:
            sweep = sweeps.stop_sweep(db, user, sweep_id, via="mcp")
            return sweeps.sweep_view(db, hub.settings, sweep)

    @mcp.tool(title="학습 실행 메모 · 태그", annotations=WRITE)
    @guarded
    def update_training_run(
        run_id: str, ctx: Context, name: str | None = None, note: str | None = None, tags: list[str] | None = None
    ) -> dict[str, Any]:
        """학습 실행의 이름을 바꾸거나 메모를 쓰거나 태그를 정합니다. 넘긴 것만 바뀌고, tags는 목록 전체를 바꿉니다."""
        user = user_of(ctx)
        with hub.db.session() as db:
            run = training.update_run(db, user, run_id, name=name, note=note, tags=tags, via="mcp")
            return training.run_view(db, hub.settings, run)

    @mcp.tool(title="학습 실행 조회", annotations=READ_ONLY)
    @guarded
    def get_training_run(run_id: str, ctx: Context) -> dict[str, Any]:
        """학습 실행 하나의 설정, epoch별 지표 이력(history), 클래스별 지표를 반환합니다.

        export에는 학습에 쓴 이미지 수와 제외된 이미지 수가 들어 있습니다.
        진행 중이면 그때까지의 이력을 반환하므로 학습 곡선을 중간에 확인할 수 있습니다.
        """
        user_of(ctx)
        with hub.db.session() as db:
            return training.run_view(db, hub.settings, training.get_run(db, run_id), detail=True)

    @mcp.tool(title="모델 목록", annotations=READ_ONLY)
    @guarded
    def list_models(
        project_id: str, ctx: Context, name: str = "", alias: str = "", tag: str = "", query: str = ""
    ) -> list[dict[str, Any]]:
        """프로젝트에 등록된 모델 버전과 지표, alias, 태그, 설명, 계보(학습 실행·릴리스)를 반환합니다.

        name과 alias를 함께 주면 그 alias가 가리키는 버전만 돌려줍니다 (예: name="street-yolo11n", alias="production").
        tag: 이 태그가 붙은 버전만. query: 이름 · 설명 · 태그에서 찾을 글자.
        """
        user_of(ctx)
        with hub.db.session() as db:
            return training.list_models(db, hub.settings, project_id, tag=tag, q=query, name=name, alias=alias)

    @mcp.tool(title="모델 설명 · 태그", annotations=WRITE)
    @guarded
    def update_model(model_id: str, ctx: Context, note: str | None = None, tags: list[str] | None = None) -> dict[str, Any]:
        """모델 버전의 설명을 쓰거나 태그를 정합니다. 넘긴 것만 바뀌고, tags는 목록 전체를 바꿉니다."""
        user = user_of(ctx)
        with hub.db.session() as db:
            model = training.update_model(db, user, model_id, note=note, tags=tags, via="mcp")
            return training.model_view(db, hub.settings, model)

    @mcp.tool(title="모델 조회", annotations=READ_ONLY)
    @guarded
    def get_model(model_id: str, ctx: Context) -> dict[str, Any]:
        """모델 버전 하나의 지표, 클래스별 지표, 학습 설정, 계보를 반환합니다. 두 모델을 비교하려면 각각 조회하세요."""
        user_of(ctx)
        with hub.db.session() as db:
            view = training.model_view(db, hub.settings, training.get_model(db, model_id), detail=True)
            view.pop("download_url", None)  # a signed browser link, not useful to an agent
            return view

    @mcp.tool(title="모델 계보", annotations=READ_ONLY)
    @guarded
    def get_model_lineage(model_id: str, ctx: Context) -> dict[str, Any]:
        """모델이 어디서 왔고 무엇이 되었는지 (계보).

        릴리스 → 학습 실행(이어서 학습한 모델, 이어서 실행한 실행, 탐색) → 모델 → 평가 · 내보내기 · 테스트,
        그리고 이 모델에서 이어서 학습한 실행과 모델. nodes(kind, label, column)와 edges(from, to)로 반환합니다.
        """
        user_of(ctx)
        with hub.db.session() as db:
            return lineage.for_model(db, training.get_model(db, model_id))

    @mcp.tool(title="모델 alias 지정", annotations=WRITE)
    @guarded
    def set_model_alias(model_id: str, alias: str, ctx: Context, remove: bool = False) -> dict[str, Any]:
        """모델 버전에 alias를 붙입니다 (예: champion, challenger, production, staging — 다른 이름도 됨).

        같은 이름의 다른 버전에 있던 같은 alias는 이 버전으로 옮겨집니다. 한 버전에 여러 alias를 붙일 수 있습니다.
        remove=True면 그 alias만 떼고, alias=""이면 이 버전의 alias를 모두 뗍니다.
        """
        user = user_of(ctx)
        with hub.db.session() as db:
            model = training.set_alias(db, user, model_id, alias, via="mcp", remove=remove)
            return training.model_view(db, hub.settings, model)

    # ---- resources -----------------------------------------------------------

    @mcp.resource("badamoa://nodes", name="nodes", title="노드 현황", mime_type="application/json")
    def nodes_resource() -> str:
        """연결된 노드와 현재 상태."""
        with hub.db.session() as db:
            return json.dumps(services.list_nodes(db, hub.settings), ensure_ascii=False, indent=2)

    @mcp.resource("badamoa://jobs/{job_id}", name="job", title="작업", mime_type="application/json")
    def job_resource(job_id: str) -> str:
        """작업 하나의 상태와 결과."""
        with hub.db.session() as db:
            try:
                job = services.get_job(db, job_id)
            except services.ServiceError as e:
                raise ResourceNotFoundError(e.message) from e
            data = services.job_view(db, hub.settings, job)
        return json.dumps(data, ensure_ascii=False, indent=2)

    @mcp.resource("badamoa://jobs/{job_id}/logs", name="job-logs", title="작업 로그", mime_type="text/plain")
    def job_logs_resource(job_id: str) -> str:
        """작업의 전체 로그(최대 5000줄)."""
        with hub.db.session() as db:
            try:
                lines = services.get_logs(db, job_id, limit=5000)
            except services.ServiceError as e:
                raise ResourceNotFoundError(e.message) from e
        return "\n".join(line["text"] for line in lines)

    # ---- prompts -------------------------------------------------------------

    @mcp.prompt(name="diagnose-failed-job", title="실패한 작업 진단")
    def diagnose_failed_job(job_id: str) -> str:
        """실패한 작업의 원인을 찾습니다."""
        return (
            f"BADAMOA 작업 {job_id}가 실패했습니다. get_job으로 상태와 error를 확인하고, "
            "get_job_logs로 로그 끝부분을 읽은 뒤, 실행된 노드를 get_node로 조회해 "
            "원인(환경·드라이버·메모리·파라미터 등)과 해결 방법을 한국어로 정리해 주세요."
        )

    @mcp.prompt(name="node-health-report", title="노드 상태 점검")
    def node_health_report() -> str:
        """모든 노드의 상태를 점검합니다."""
        return (
            "list_nodes로 BADAMOA의 모든 노드를 조회하고, 오프라인 노드, 디스크 여유가 10% 미만인 노드, "
            "GPU 메모리가 거의 찬 노드를 찾아 표로 정리해 주세요. 온라인 노드마다 system.info 작업을 "
            "실행해(run_job → wait_for_job) 드라이버·GPU 정보도 함께 확인해 주세요."
        )

    # Building the app creates the session manager; we mount only the transport endpoint.
    mcp.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        # The Hub is reached by LAN address and every request needs a bearer token,
        # so Host-header pinning (meant for unauthenticated localhost servers) is off.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    return mcp, BearerGuard(StreamableHTTPASGIApp(mcp.session_manager), hub)


def _origin(ctx: Context) -> str | None:
    headers = ctx.headers or {}
    host = headers.get("host")
    if not host:
        return None
    scheme = headers.get("x-forwarded-proto", "http")
    return f"{scheme}://{host}"
