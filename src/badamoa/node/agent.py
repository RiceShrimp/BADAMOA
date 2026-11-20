"""Node Agent: joins a Hub, reports health, and runs the jobs it is given.

All traffic is outbound HTTP(S) to the Hub, so a node behind NAT or a firewall
needs no open inbound port.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil

from badamoa import __version__
from badamoa.core import sysinfo
from badamoa.core.paths import component_home, is_elevated
from badamoa.node import browse, cache, containers, dataserver, envs, places, remover, storage, updater, workdirs

log = logging.getLogger("badamoa.node")

HEARTBEAT_INTERVAL_S = 10.0
CLAIM_WAIT_S = 20.0
LOG_FLUSH_INTERVAL_S = 1.0
LOG_BATCH_MAX = 500
CACHE_USAGE_EVERY_S = 1800.0  # how often what BADAMOA keeps here is measured again (and after jobs that change it)
WORK_CHECK_EVERY_S = 3600.0  # how often the work folder is checked again (an admin's folder may become writable)
MOVE_RETRY_S = 3600.0  # after a move into the work folder that left something behind


def data_port() -> int:
    """The port other nodes take images from (``BADAMOA_NODE_DATA_PORT``; 0 lets the system pick)."""
    try:
        return int(os.environ.get("BADAMOA_NODE_DATA_PORT", dataserver.DEFAULT_PORT))
    except ValueError:
        return dataserver.DEFAULT_PORT


@dataclass
class NodeConfig:
    hub_url: str
    node_id: str
    node_token: str
    name: str = ""

    @staticmethod
    def path(home: Path) -> Path:
        return home / "node.json"

    @classmethod
    def load(cls, home: Path) -> NodeConfig:
        path = cls.path(home)
        if not path.is_file():
            raise FileNotFoundError(f"노드 설정이 없습니다: {path}. 먼저 `badamoa node join`을 실행하세요.")
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def save(self, home: Path) -> Path:
        path = self.path(home)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        if os.name != "nt":
            path.chmod(0o600)
        return path


def node_home(explicit: str | os.PathLike[str] | None = None) -> Path:
    return component_home("node", explicit)


def account() -> dict[str, Any]:
    """Who the agent runs as: root (installed with sudo: a systemd service that can open the firewall and let Docker
    use in-house registries) or an account of its own (installed without sudo: starts from that account's crontab)."""
    try:
        name = getpass.getuser()
    except (KeyError, OSError):
        name = ""
    return {"name": name, "admin": is_elevated()}


def already_joined(hub_url: str, home: Path) -> NodeConfig | None:
    """This machine's registration, when the Hub at ``hub_url`` still accepts it (running the install again)."""
    try:
        config = NodeConfig.load(home)
    except (FileNotFoundError, ValueError, TypeError):
        return None
    hub_url = hub_url.rstrip("/")
    try:
        resp = httpx.post(
            f"{hub_url}/api/agent/heartbeat",
            json={"agent_version": __version__, "metrics": {}, "running_job_ids": []},
            headers={"Authorization": f"Bearer {config.node_token}"},
            timeout=15,
        )
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None  # deleted on the Hub, or another Hub: join anew
    if config.hub_url != hub_url:  # the same Hub reached at another address (localhost → its IP)
        config.hub_url = hub_url
        config.save(home)
    return config


def join(hub_url: str, token: str, home: Path, name: str | None = None) -> NodeConfig:
    hub_url = hub_url.rstrip("/")
    resp = httpx.post(
        f"{hub_url}/api/agent/register",
        json={"join_token": token, "agent_version": __version__, "info": sysinfo.static_info(), "name": name},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Hub 등록 실패 ({resp.status_code}): {_detail(resp)}")
    data = resp.json()
    config = NodeConfig(hub_url=hub_url, node_id=data["node_id"], node_token=data["node_token"], name=name or "")
    config.save(home)
    return config


def _detail(resp: httpx.Response) -> str:
    try:
        return str(resp.json().get("detail", resp.text))
    except ValueError:
        return resp.text


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _terminate_tree(proc: subprocess.Popen) -> None:
    """Stop a job and everything it started (a trainer and its dataloader workers).

    Terminating only the job process would orphan its children, which keep the GPU busy.
    """
    try:
        family = psutil.Process(proc.pid).children(recursive=True)
    except psutil.Error:
        family = []
    proc.terminate()
    for child in family:
        try:
            child.terminate()
        except psutil.Error:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    for child in psutil.wait_procs(family, timeout=5)[1]:
        try:
            child.kill()
        except psutil.Error:
            pass


class NodeAgent:
    def __init__(self, config: NodeConfig, home: Path, max_jobs: int | None = None) -> None:
        self.config = config
        self.home = Path(home).resolve()  # jobs run in their own working directory
        # GPU jobs are limited by the GPUs the Hub hands out; this caps everything else.
        self.max_jobs = max_jobs or max(4, len(sysinfo.gpus()))
        self.stop_event = threading.Event()
        self._envs_changed = threading.Event()
        self._lock = threading.Lock()
        self._running: dict[str, threading.Event] = {}  # job_id -> cancel event
        self._last_sweep = float("-inf")  # when old work folders were last looked at (monotonic)
        self._updating = False  # moving to the Hub's version: no new jobs meanwhile
        self._update_tried: dict[str, float] = {}  # version -> when (monotonic), to retry a failed one later
        self._told_manual = False
        self._rejected = False  # the Hub no longer knows this node (deleted): said once, then quiet
        self._sampler = sysinfo.MetricsSampler(str(home))
        # Images other nodes take from here, and where images taken from other nodes are kept.
        self._data = dataserver.DataServer(config.hub_url, config.node_token, config.node_id, data_port())
        self._cache_setting = ""  # the admin's choice, from the Hub ("" = the node picks)
        # Where environments, work folders and fetched files go (``places``): resolved once the Hub said
        # whether an admin set a folder
        self._work_setting = ""
        self._work: Path | None = None
        self._work_error = ""  # the admin's folder could not be used: why
        self._work_checked = float("-inf")
        self._moving = False  # moving what was kept elsewhere into the work folder: no new jobs meanwhile
        self._move_tried = float("-inf")
        self._docker_root: dict[str, Any] | None = None  # Docker's data folder (measured with the storage report)
        self._cache_usage: tuple[int, int] | None = None
        self._storage_report: dict[str, Any] | None = None  # measured, not yet delivered to the Hub
        self._recount = threading.Event()
        self._first_beat = threading.Event()  # the Hub's settings (where to keep images) are known
        self._client = httpx.Client(
            base_url=config.hub_url,
            headers={"Authorization": f"Bearer {config.node_token}"},
            timeout=httpx.Timeout(30.0, read=CLAIM_WAIT_S + 15.0),
        )

    # ---- lifecycle -------------------------------------------------------------

    def stop(self) -> None:
        self.stop_event.set()
        self._recount.set()
        self._data.stop()

    def cache_root(self) -> Path:
        return cache.root(self.home, self._cache_setting)

    def work_root(self) -> Path:
        """The work folder (``places``): environments, work folders of jobs, fetched weights and code."""
        return self._work or places.of(self.home).work

    def _resolve_work(self, setting: str) -> None:
        """The admin's folder (the Hub's ``work_dir``), else the node's own pick; things kept elsewhere move later."""
        work, error = places.root(self.home, setting)
        changed = self._work is None or not places.same(work, self._work)
        self._work, self._work_setting, self._work_error = work, setting, error
        self._work_checked = time.monotonic()
        places.use(self.home, work)
        if error:
            log.warning(error)
        if changed:
            log.info("작업 폴더(가상환경 · 작업 폴더 · 받아 둔 가중치와 코드): %s", work)
            self._move_tried = float("-inf")
            self._envs_changed.set()  # environments are looked up there now: tell the Hub again
            self._recount.set()

    def _places(self, disks: list[dict[str, Any]], images: Path, work: Path) -> list[dict[str, Any]]:
        """The disks BADAMOA writes to here and their room: the Hub warns when one is nearly full."""
        found = [
            {"kind": "home", **places.disk_of(self.home, disks)},
            {"kind": "work", **places.disk_of(work, disks)},
            {"kind": "images", **places.disk_of(images, disks)},
        ]
        docker = self._docker_root
        if docker and docker.get("desktop"):  # inside Docker Desktop's virtual disk: not one of this machine's
            found.append({"kind": "docker", "path": docker["path"], "desktop": True})
        elif docker:
            found.append({"kind": "docker", **places.disk_of(Path(docker["path"]), disks)})
        return found

    def _maybe_move(self) -> None:
        """At a moment no job runs, move what is kept outside the work folder (work folders, weights, code,
        template caches) into it. Environments stay where they are and are used there."""
        if self._moving or time.monotonic() - self._move_tried < MOVE_RETRY_S:
            return
        where = places.Places(self.home, self.work_root())
        if not places.pending(where):
            return
        with self._lock:
            if self._running or self._updating:
                return
            self._moving = True
        self._move_tried = time.monotonic()
        threading.Thread(target=self._move, args=(where,), name="move", daemon=True).start()

    def _move(self, where: places.Places) -> None:
        try:
            log.info("작업 폴더 밖에 둔 것을 %s(으)로 옮깁니다. 그동안 새 작업은 기다립니다.", where.work)
            places.move_old(where, log.info)
        except Exception:  # noqa: BLE001 - what stayed behind is still found where it is
            log.exception("작업 폴더로 옮기지 못했습니다")
        finally:
            self._moving = False
            self._recount.set()

    def run_forever(self) -> None:
        log.info("node %s → %s", self.config.node_id, self.config.hub_url)
        self._data.start()
        threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True).start()
        threading.Thread(target=self._count_cache, name="cache-usage", daemon=True).start()
        self._first_beat.wait(15)  # a job taken before would not know the admin's settings yet
        threading.Thread(target=self._requests_loop, name="requests", daemon=True).start()
        backoff = 1.0
        while not self.stop_event.is_set():
            if self._slots_free() <= 0:
                self.stop_event.wait(0.5)
                continue
            try:
                resp = self._client.post("/api/agent/claim", json={"wait_seconds": CLAIM_WAIT_S})
                if resp.status_code == 401:
                    if not self._rejected:
                        self._rejected = True
                        log.error(
                            "Hub가 이 노드를 모릅니다(삭제되었거나 토큰이 바뀜). 다시 쓰려면 노드 추가로 다시 붙이고, "
                            "이 머신에서 지우려면 Hub의 /install/node-remove.sh를 실행하세요. 5분마다 다시 확인합니다."
                        )
                    self.stop_event.wait(300)
                    continue
                self._rejected = False
                resp.raise_for_status()
                backoff = 1.0
                job = resp.json().get("job")
                if job:
                    self._start_job(job)
            except httpx.HTTPError as e:
                log.warning("Hub 연결 실패: %s (%.0fs 후 재시도)", e, backoff)
                self.stop_event.wait(backoff)
                backoff = min(backoff * 2, 30.0)

    def _slots_free(self) -> int:
        with self._lock:
            return 0 if self._updating or self._moving else self.max_jobs - len(self._running)

    # ---- heartbeat -------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        sent_info = False
        while not self.stop_event.is_set():
            if self._envs_changed.is_set():  # a job may have built an environment: tell the Hub what is ready now
                self._envs_changed.clear()
                sent_info = False
            try:
                with self._lock:
                    running = list(self._running)
                root = self.cache_root()
                work = self.work_root()
                usage = self._cache_usage
                sample = self._sampler.sample()
                metrics = {
                    **sample,
                    "serve": self._data.info(),
                    "cache": {
                        "dir": str(root),
                        "auto": not self._cache_setting,
                        "free_gb": cache.free_gb(root),
                        "bytes": usage[0] if usage else None,
                        "files": usage[1] if usage else None,
                    },
                    "work": {
                        "dir": str(work),
                        "auto": not self._work_setting,
                        "error": self._work_error,
                        "free_gb": cache.free_gb(work),
                        "moving": self._moving,
                    },
                    "places": self._places(sample.get("disks") or [], root, work),
                }
                body: dict[str, Any] = {"agent_version": __version__, "metrics": metrics, "running_job_ids": running}
                report = self._storage_report
                if report is not None:
                    body["storage"] = report
                if not sent_info:
                    body["info"] = {
                        **sysinfo.static_info(),
                        # what environments are ready here, and whether this node can run containers
                        "envs": envs.listing(self.home) + containers.listing(self.home),
                        "docker": containers.capability(),
                        "account": account(),
                    }
                resp = self._client.post("/api/agent/heartbeat", json=body, timeout=15)
                if resp.status_code == 200:
                    sent_info = True
                    if report is not None and self._storage_report is report:
                        self._storage_report = None  # delivered: the next one after the next measurement
                    reply = resp.json()
                    self._request_cancel(reply.get("cancel_job_ids", []))
                    setting = str(reply.get("cache_dir") or "")
                    if setting != self._cache_setting:
                        self._cache_setting = setting
                        self._recount.set()
                    setting = str(reply.get("work_dir") or "")
                    stale = time.monotonic() - self._work_checked > WORK_CHECK_EVERY_S
                    if self._work is None or setting != self._work_setting or stale:
                        self._resolve_work(setting)
                    self._maybe_move()  # before the first job is taken
                    self._first_beat.set()
                    self._maybe_sweep(int(reply.get("workdir_days") or 0))
                    self._maybe_update(str(reply.get("hub_version") or ""))
            except httpx.HTTPError as e:
                log.debug("heartbeat failed: %s", e)
            self.stop_event.wait(HEARTBEAT_INTERVAL_S)

    def _count_cache(self) -> None:
        """Now and then (and after a job that adds or removes something), what BADAMOA keeps here and how much."""
        self._first_beat.wait(15)  # the work folder is known then
        while not self.stop_event.is_set():
            try:
                self._docker_root = containers.storage_root()
            except Exception:  # noqa: BLE001 - only the Docker disk warning goes without it
                self._docker_root = None
            try:
                report = storage.measure(self.home, self.cache_root())
                images = report["kinds"]["images"]
                self._cache_usage = (images["bytes"], images["files"])
                self._storage_report = report  # the next heartbeat takes it to the Hub
            except Exception:  # noqa: BLE001 - a measurement must never stop the node
                log.exception("measuring what is kept here failed")
                try:
                    self._cache_usage = cache.usage(self.cache_root())
                except OSError:
                    pass
            self._recount.wait(CACHE_USAGE_EVERY_S)
            self._recount.clear()

    # ---- keeping up with the Hub's version -------------------------------------

    def _maybe_update(self, version: str) -> None:
        """When the Hub is newer and nothing runs here, install its version and restart (script installs only)."""
        if not version or not updater.newer(version, __version__):
            return
        if time.monotonic() - self._update_tried.get(version, float("-inf")) < 3600:
            return
        if updater.layout() is None:
            if not self._told_manual:
                self._told_manual = True
                log.warning("Hub는 %s, 이 노드는 %s입니다. 이 노드는 스스로 업데이트할 수 없습니다.", version, __version__)
            return
        with self._lock:
            if self._running or self._updating:
                return
            self._updating = True
        self._update_tried[version] = time.monotonic()
        threading.Thread(target=self._update, args=(version,), name="update", daemon=True).start()

    def _update(self, version: str) -> None:
        log.info("Hub가 %s입니다. 이 노드(%s)를 업데이트합니다.", version, __version__)
        moved = updater.update(self.config.hub_url, version)  # Linux: does not return when it worked
        if moved and updater.WINDOWS:
            self.stop()
            os._exit(0)  # the helper installs once this process is gone, then starts the node again
        with self._lock:
            self._updating = False

    # ---- quick questions from the Hub (a folder listing) ----------------------

    def _requests_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                resp = self._client.post("/api/agent/requests", json={"wait_seconds": CLAIM_WAIT_S})
                if resp.status_code in (401, 404):  # deleted on the Hub, or an older Hub: nothing to ask
                    self.stop_event.wait(300)
                    continue
                if resp.status_code != 200:
                    self.stop_event.wait(5)
                    continue
                for request in resp.json().get("requests", []):
                    threading.Thread(target=self._answer, args=(request,), name="request", daemon=True).start()
            except httpx.HTTPError:
                self.stop_event.wait(5)

    def _answer(self, request: dict[str, Any]) -> None:
        params = request.get("params") or {}
        kind = request.get("kind")
        leaving = hard = False
        try:
            if kind == "browse":
                body: dict[str, Any] = {"result": browse.listing(str(params.get("path") or ""), bool(params.get("hidden")))}
            elif kind == "file":  # an original image the Hub cannot read itself
                body = {"result": self._send_file(params)}
            elif kind == "remove":  # deleted on the Hub: take BADAMOA off this machine
                self._stop_jobs()
                body = {"result": {"removing": remover.remove(self.home)}}
                leaving = True
                hard = updater.layout() is not None  # a helper deletes the files once this process is gone
            else:
                raise ValueError(f"모르는 요청입니다: {kind}")
        except Exception as e:  # noqa: BLE001 - the person asking gets the reason
            body = {"error": str(e) or e.__class__.__name__}
        try:
            self._client.post(f"/api/agent/requests/{request['id']}", json=body, timeout=15)
        except httpx.HTTPError as e:
            log.debug("answer %s failed: %s", request.get("id"), e)
        if leaving:
            log.warning("Hub에서 이 노드를 지웠습니다. BADAMOA를 정리하고 끝냅니다: %s", body["result"]["removing"])
            self.stop()
            if hard:
                os._exit(0)  # the cleanup helper takes it from here

    def _send_file(self, params: dict[str, Any]) -> dict[str, Any]:
        """Upload a registered file, if it is still the one registered (same size)."""
        path = Path(str(params.get("path") or ""))
        if not path.is_file():
            raise ValueError(f"파일이 없습니다: {path}")
        expected = int(params.get("bytes") or 0)
        if expected and path.stat().st_size != expected:
            raise ValueError("등록한 뒤 파일이 바뀌었습니다(크기가 다름).")
        content = path.read_bytes()
        resp = self._client.put(f"/api/agent/content/{params['key']}", content=content, timeout=120)
        resp.raise_for_status()
        return {"bytes": len(content)}

    def _stop_jobs(self, wait_s: float = 10.0) -> None:
        """Cancel what runs here (the Hub fails those jobs anyway) so no process is left behind."""
        with self._lock:
            events = list(self._running.values())
        for event in events:
            event.set()
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            with self._lock:
                if not self._running:
                    return
            time.sleep(0.2)

    def _maybe_sweep(self, days: int) -> None:
        """Once an hour, with no job running, remove work folders older than the Hub's retention."""
        if days <= 0 or time.monotonic() - self._last_sweep < workdirs.SWEEP_EVERY_S:
            return
        with self._lock:
            if self._running:
                return  # a resumed run may be reading an old folder; wait for a quiet moment
        self._last_sweep = time.monotonic()
        threading.Thread(target=workdirs.sweep, args=(self.home, days, set()), name="workdirs", daemon=True).start()

    def _request_cancel(self, job_ids: list[str]) -> None:
        with self._lock:
            for job_id in job_ids:
                event = self._running.get(job_id)
                if event:
                    event.set()

    # ---- job execution ---------------------------------------------------------

    def _start_job(self, job: dict[str, Any]) -> None:
        cancel = threading.Event()
        with self._lock:
            self._running[job["id"]] = cancel
        threading.Thread(target=self._run_job, args=(job, cancel), name=f"job-{job['id'][:8]}", daemon=True).start()

    def _run_job(self, job: dict[str, Any], cancel: threading.Event) -> None:
        job_id = job["id"]
        log.info("job %s (%s) started", job_id, job["type"])
        status, result, error = "failed", None, None
        buffer: list[dict[str, str]] = []
        buffer_lock = threading.Lock()

        def emit(text: str, stream: str = "stdout") -> None:
            with buffer_lock:
                buffer.append({"ts": _now(), "stream": stream, "text": text})

        def flush() -> None:
            with buffer_lock:
                batch = buffer[:LOG_BATCH_MAX]
                del buffer[:LOG_BATCH_MAX]
            if not batch:
                return
            try:
                resp = self._client.post(f"/api/agent/jobs/{job_id}/logs", json={"lines": batch}, timeout=15)
                if resp.status_code == 200 and resp.json().get("cancel_requested"):
                    cancel.set()
            except httpx.HTTPError:
                with buffer_lock:  # keep the lines and retry on the next flush
                    buffer[:0] = batch

        if self._moving:
            emit("작업 폴더로 옮기는 중입니다. 끝나면 시작합니다.", "system")
        while self._moving and not cancel.is_set() and not self.stop_event.is_set():
            flush()
            time.sleep(1)
        try:
            with tempfile.TemporaryDirectory(prefix="badamoa-job-") as tmp:
                params_path = Path(tmp) / "params.json"
                result_path = Path(tmp) / "result.json"
                params_path.write_text(json.dumps(job.get("params") or {}), encoding="utf-8")
                env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
                # environments, the job's work folder, fetched weights and code: the node's work folder
                env[places.ENV] = str(self.work_root())
                if job.get("gpu_indices"):
                    # The job sees only the GPUs the Hub reserved for it.
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in job["gpu_indices"])
                    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # match nvidia-smi numbering
                if job["type"].startswith(("data.", "train.", "model.", "env.", "template.")):
                    # These jobs exchange data with the Hub themselves (import chunks, metrics, weights).
                    env["BADAMOA_HUB_URL"] = self.config.hub_url
                    env["BADAMOA_NODE_TOKEN"] = self.config.node_token
                    env["BADAMOA_JOB_ID"] = job_id
                    env["BADAMOA_NODE_HOME"] = str(self.home)
                    env["BADAMOA_CACHE_DIR"] = str(self.cache_root())
                if job["type"] == "node.clean":  # removes what BADAMOA keeps here, never what runs now
                    env["BADAMOA_NODE_HOME"] = str(self.home)
                    with self._lock:
                        env["BADAMOA_RUNNING"] = ",".join(j for j in self._running if j != job_id)
                proc = subprocess.Popen(
                    [sys.executable, "-m", "badamoa.node.builtin", job["type"], str(params_path), str(result_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    cwd=tmp,
                )

                def pump() -> None:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        emit(line.rstrip("\r\n"))

                reader = threading.Thread(target=pump, daemon=True)
                reader.start()
                canceled = False
                while proc.poll() is None:
                    if cancel.is_set() or self.stop_event.is_set():
                        canceled = True
                        _terminate_tree(proc)
                        break
                    flush()
                    time.sleep(LOG_FLUSH_INTERVAL_S if buffer else 0.2)
                reader.join(timeout=5)

                if canceled:
                    status = "canceled"
                    emit("작업이 취소되었습니다.", "system")
                elif proc.returncode == 0:
                    status = "succeeded"
                    if result_path.is_file():
                        result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    error = f"프로세스가 종료 코드 {proc.returncode}로 끝났습니다."
        except Exception as e:  # noqa: BLE001 - report anything to the Hub
            error = f"{type(e).__name__}: {e}"
            emit(error, "system")
        finally:
            if status != "succeeded" and job["type"].startswith(("train.", "model.", "template.")):
                containers.kill(job_id)  # a container its stopped process left behind
            if status != "succeeded" and job["type"].startswith("train."):
                self._keep_checkpoint(job_id, emit)
            for _ in range(20):
                flush()
                if not buffer:
                    break
                time.sleep(0.5)
            self._complete(job_id, status, result, error)
            changes = job["type"].startswith(("env.", "train.", "model.", "template.")) or job["type"] == "node.clean"
            if changes:
                self._envs_changed.set()  # tell the Hub which environments are here now
            if changes or job["type"] in ("data.sync", "data.check"):
                self._recount.set()  # measure again what is kept here
            with self._lock:
                self._running.pop(job_id, None)
            log.info("job %s %s", job_id, status)

    def _keep_checkpoint(self, job_id: str, emit: Any) -> None:
        """A training that was stopped or failed: put its last checkpoint and what it had produced on the Hub.

        With the checkpoint on the Hub the run can be continued later from any server,
        whether or not this one still has the working folder. The plots, tables and weights
        written so far are kept with the run like those of a finished one.
        """
        from badamoa.node import trainer

        base = f"/api/agent/jobs/{job_id}/train"
        job_dir = places.Places(self.home, self.work_root()).find("runs", job_id)
        try:
            if trainer.upload_checkpoint(self._client, base, job_dir):
                emit("마지막 체크포인트를 Hub에 보관했습니다. 이어서 실행할 수 있습니다.", "system")
        except (httpx.HTTPError, OSError) as e:
            log.warning("job %s: 체크포인트를 Hub에 보관하지 못했습니다: %s", job_id, e)
        kept, size = trainer.keep_outputs(job_dir, lambda path, name: trainer.put_run_file(self._client, base, path, name))
        if kept:
            emit(f"그때까지의 산출물 {kept:,}개 ({size / 1024**2:.1f} MB)를 Hub에 보관했습니다.", "system")

    def _complete(self, job_id: str, status: str, result: Any, error: str | None) -> None:
        body = {"status": status, "result": result, "error": error}
        for attempt in range(5):
            try:
                resp = self._client.post(f"/api/agent/jobs/{job_id}/complete", json=body, timeout=15)
                if resp.status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2**attempt)
        log.error("job %s: 완료 보고 실패", job_id)
