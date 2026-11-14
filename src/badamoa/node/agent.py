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
        self._lock = threading.Lock()
        self._running: dict[str, threading.Event] = {}  # job_id -> cancel event
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
            return self.max_jobs - len(self._running)

    # ---- heartbeat -------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        sent_info = False
        while not self.stop_event.is_set():
            try:
                with self._lock:
                    running = list(self._running)
                body: dict[str, Any] = {
                    "agent_version": __version__,
                    "metrics": self._sampler.sample(),
                    "running_job_ids": running,
                }
                if not sent_info:
                    body["info"] = sysinfo.static_info()
                resp = self._client.post("/api/agent/heartbeat", json=body, timeout=15)
                if resp.status_code == 200:
                    sent_info = True
                    self._request_cancel(resp.json().get("cancel_job_ids", []))
            except httpx.HTTPError as e:
                log.debug("heartbeat failed: %s", e)
            self.stop_event.wait(HEARTBEAT_INTERVAL_S)

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
                if job["type"].startswith(("data.", "train.")):
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
            for _ in range(20):
                flush()
                if not buffer:
                    break
                time.sleep(0.5)
            self._complete(job_id, status, result, error)
            with self._lock:
                self._running.pop(job_id, None)
            log.info("job %s %s", job_id, status)

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
