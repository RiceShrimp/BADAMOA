"""Node Agent: joins a Hub, reports health, and runs the jobs it is given.

All traffic is outbound HTTP(S) to the Hub, so a node behind NAT or a firewall
needs no open inbound port.
"""

from __future__ import annotations

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
from badamoa.core.paths import component_home
from badamoa.node import browse, envs, updater, workdirs

log = logging.getLogger("badamoa.node")

HEARTBEAT_INTERVAL_S = 10.0
CLAIM_WAIT_S = 20.0
LOG_FLUSH_INTERVAL_S = 1.0
LOG_BATCH_MAX = 500


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
        self._sampler = sysinfo.MetricsSampler(str(home))
        self._client = httpx.Client(
            base_url=config.hub_url,
            headers={"Authorization": f"Bearer {config.node_token}"},
            timeout=httpx.Timeout(30.0, read=CLAIM_WAIT_S + 15.0),
        )

    # ---- lifecycle -------------------------------------------------------------

    def stop(self) -> None:
        self.stop_event.set()

    def run_forever(self) -> None:
        log.info("node %s → %s", self.config.node_id, self.config.hub_url)
        threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True).start()
        threading.Thread(target=self._requests_loop, name="requests", daemon=True).start()
        backoff = 1.0
        while not self.stop_event.is_set():
            if self._slots_free() <= 0:
                self.stop_event.wait(0.5)
                continue
            try:
                resp = self._client.post("/api/agent/claim", json={"wait_seconds": CLAIM_WAIT_S})
                if resp.status_code == 401:
                    log.error("Hub가 이 노드를 거부했습니다(삭제되었거나 토큰이 바뀜). 다시 join 하세요.")
                    self.stop_event.wait(30)
                    continue
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
            return 0 if self._updating else self.max_jobs - len(self._running)

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
                body: dict[str, Any] = {
                    "agent_version": __version__,
                    "metrics": self._sampler.sample(),
                    "running_job_ids": running,
                }
                if not sent_info:
                    body["info"] = {**sysinfo.static_info(), "envs": envs.listing(self.home)}
                resp = self._client.post("/api/agent/heartbeat", json=body, timeout=15)
                if resp.status_code == 200:
                    sent_info = True
                    reply = resp.json()
                    self._request_cancel(reply.get("cancel_job_ids", []))
                    self._maybe_sweep(int(reply.get("workdir_days") or 0))
                    self._maybe_update(str(reply.get("hub_version") or ""))
            except httpx.HTTPError as e:
                log.debug("heartbeat failed: %s", e)
            self.stop_event.wait(HEARTBEAT_INTERVAL_S)

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
                if resp.status_code == 404:  # an older Hub: nothing to ask
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
        try:
            if request.get("kind") != "browse":
                raise ValueError(f"모르는 요청입니다: {request.get('kind')}")
            body: dict[str, Any] = {"result": browse.listing(str(params.get("path") or ""), bool(params.get("hidden")))}
        except Exception as e:  # noqa: BLE001 - the person asking gets the reason
            body = {"error": str(e) or e.__class__.__name__}
        try:
            self._client.post(f"/api/agent/requests/{request['id']}", json=body, timeout=15)
        except httpx.HTTPError as e:
            log.debug("answer %s failed: %s", request.get("id"), e)

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

        try:
            with tempfile.TemporaryDirectory(prefix="badamoa-job-") as tmp:
                params_path = Path(tmp) / "params.json"
                result_path = Path(tmp) / "result.json"
                params_path.write_text(json.dumps(job.get("params") or {}), encoding="utf-8")
                env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
                if job.get("gpu_indices"):
                    # The job sees only the GPUs the Hub reserved for it.
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in job["gpu_indices"])
                    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # match nvidia-smi numbering
                if job["type"].startswith(("data.", "train.", "model.", "env.prepare")):
                    # These jobs exchange data with the Hub themselves (import chunks, metrics, weights).
                    env["BADAMOA_HUB_URL"] = self.config.hub_url
                    env["BADAMOA_NODE_TOKEN"] = self.config.node_token
                    env["BADAMOA_JOB_ID"] = job_id
                    env["BADAMOA_NODE_HOME"] = str(self.home)
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
            if status != "succeeded" and job["type"].startswith("train."):
                self._keep_checkpoint(job_id, emit)
            for _ in range(20):
                flush()
                if not buffer:
                    break
                time.sleep(0.5)
            self._complete(job_id, status, result, error)
            if job["type"].startswith(("env.", "train.", "model.")):
                self._envs_changed.set()
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

        base, job_dir = f"/api/agent/jobs/{job_id}/train", self.home / "runs" / job_id
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
