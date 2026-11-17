"""Receiving a version's images from the node that holds them (``data.sync`` jobs, and before a training).

The Hub hands out the list of images and a grant (``/api/agent/jobs/<job>/sync/plan``); the
images come straight from the holding node's own server (``dataserver``), in tar batches over a
few connections at once, and go into this node's cache by content (``cache``). Every file is
checked against its registered size and SHA-256 before it is kept. What is in the cache already
is not fetched again, so an interrupted transfer, or the next version, picks up where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from badamoa.node import cache

STREAMS = 4  # batches fetched at once
BATCH_FILES = 1000
BATCH_BYTES = 1024**3
RETRIES = 3
MARGIN = 1024**3  # room left over on the disk after receiving
CONNECT_S = 4.0
READ_S = 120.0
REPORT_EVERY_S = 2.0
GB = 1024**3


def _size(n: float) -> str:
    """ "320.4 GB", or "7.2 MB" below a gigabyte."""
    return f"{n / GB:,.1f} GB" if n >= GB else f"{n / 1024**2:,.1f} MB"


class Refused(RuntimeError):
    """The holding node (or the Hub behind it) said no: retrying will not help."""


def _hub(client: httpx.Client, job_id: str, after: int) -> dict[str, Any]:
    resp = client.get(f"/api/agent/jobs/{job_id}/sync/plan", params={"after": after}, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Hub가 이미지 목록을 주지 않았습니다 ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


def _reach(source: dict[str, Any]) -> str:
    """The first address of the holding node that answers."""
    port = int(source.get("port") or 0)
    name = source.get("name") or "데이터 노드"
    if not port:
        raise RuntimeError(
            f"{name}에서 이미지를 받을 수 없습니다. "
            f"{name}가 0.37.0 이상이어야 하고, 이미지 전송용 포트가 열려 있어야 합니다."
        )
    tried = []
    for address in source.get("addresses") or []:
        base = f"http://{address}:{port}"
        tried.append(f"{address}:{port}")
        try:
            with urllib.request.urlopen(f"{base}/badamoa/ping", timeout=CONNECT_S) as resp:  # noqa: S310 - LAN node
                if json.loads(resp.read() or b"{}").get("node_id") == source.get("id"):
                    return base
        except (OSError, ValueError):
            continue
    where = ", ".join(tried) or "알려진 주소 없음"
    raise RuntimeError(
        f"{name}({where})에 연결할 수 없습니다. 두 서버가 서로 닿는 네트워크에 있는지, "
        f"{name}의 방화벽에서 TCP {port} 포트가 열려 있는지 확인하세요."
    )


def _batches(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for item in items:
        if current and (len(current) >= BATCH_FILES or size + item["bytes"] > BATCH_BYTES):
            batches.append(current)
            current, size = [], 0
        current.append(item)
        size += item["bytes"]
    if current:
        batches.append(current)
    return batches


def _keep(root: Path, item: dict[str, Any], stream: Any) -> str | None:
    """Write one image into the cache if it is the registered one. Returns why not, else None."""
    final = cache.path_of(root, item["sha256"], item["ext"])
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(f"{final.name}.part-{os.getpid()}-{threading.get_ident()}")
    digest = hashlib.sha256()
    size = 0
    try:
        with part.open("wb") as out:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
                out.write(chunk)
        if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
            part.unlink(missing_ok=True)
            return "내용이 등록할 때와 다릅니다(파일이 바뀜)."
        os.replace(part, final)
        return None
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def _fetch(base: str, grant: str, root: Path, batch: list[dict[str, Any]], got: Callable[[int], None]) -> dict[int, str]:
    """One batch from the holding node. Returns the images it could not give, with the reason."""
    by_id = {item["id"]: item for item in batch}
    body = json.dumps({"grant": grant, "ids": list(by_id)}).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - the holding node on the LAN
        f"{base}/badamoa/files", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    bad: dict[int, str] = {}
    seen: set[int] = set()
    try:
        with urllib.request.urlopen(request, timeout=READ_S) as resp, tarfile.open(fileobj=resp, mode="r|") as tar:
            for member in tar:
                stream = tar.extractfile(member)
                if stream is None:
                    continue
                if member.name == "errors.json":
                    for e in json.loads(stream.read() or b"{}").get("errors") or []:
                        bad[int(e["id"])] = str(e.get("reason") or "")
                    continue
                item = by_id.get(int(member.name))
                if item is None:
                    continue
                reason = _keep(root, item, stream)
                seen.add(item["id"])
                if reason:
                    bad[item["id"]] = reason
                else:
                    got(item["bytes"])
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read() or b"{}").get("detail") or ""
        except ValueError:
            pass
        if e.code == 403:
            raise Refused(detail or "허가되지 않은 전송입니다.") from e
        raise ConnectionError(f"{e.code} {detail}") from e
    missing = [i for i in by_id if i not in seen and i not in bad]
    if missing:
        raise ConnectionError(f"전송이 중간에 끊겼습니다 ({len(missing)}장 남음)")
    return bad


def receive(client: httpx.Client, job_id: str, root: Path, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Bring every image of the job's version into ``root`` that is not there yet."""
    started = time.monotonic()
    first = _hub(client, job_id, 0)
    total = first.get("total") or {}
    if first.get("local"):
        return {"items": total.get("items", 0), "bytes": total.get("bytes", 0), "fetched": 0, "local": True}
    items: list[dict[str, Any]] = []
    page = first
    while True:
        items.extend(page["items"])
        if page.get("next") is None:
            break
        page = _hub(client, job_id, page["next"])
    missing = [it for it in items if not cache.has(root, it["sha256"], it["ext"], it["bytes"])]
    need = sum(it["bytes"] for it in missing)
    all_bytes = sum(it["bytes"] for it in items)
    source = first.get("source") or {}
    log(
        f"{source.get('name', '데이터 노드')}에서 이미지를 직접 받습니다: "
        f"전체 {len(items):,}장 ({_size(all_bytes)}) 중 {len(missing):,}장 ({_size(need)}), "
        f"이미 있는 것 {len(items) - len(missing):,}장 · 보관 위치 {root}"
    )
    result = {
        "items": len(items),
        "bytes": all_bytes,
        "fetched": 0,
        "fetched_bytes": 0,
        "cached": len(items) - len(missing),
        "changed": [],
        "changed_count": 0,
        "cache_dir": str(root),
    }
    if not missing:
        return result
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < need + MARGIN:
        raise RuntimeError(
            f"디스크 공간이 모자랍니다: 받을 양 {_size(need)}, 남은 공간 {_size(free)} ({root}). "
            "노드 설정에서 받은 이미지 보관 위치를 여유 있는 디스크로 바꾸세요."
        )
    base = _reach(source)
    grant = first["grant"]

    lock = threading.Lock()
    done = {"items": 0, "bytes": 0}
    stop = threading.Event()

    def got(size: int) -> None:
        with lock:
            done["items"] += 1
            done["bytes"] += size

    def report() -> None:
        last_bytes, last_t = 0, time.monotonic()
        rate = 0.0
        while not stop.wait(REPORT_EVERY_S):
            with lock:
                now_bytes, now_items = done["bytes"], done["items"]
            now = time.monotonic()
            if now > last_t:
                rate = 0.6 * rate + 0.4 * ((now_bytes - last_bytes) / (now - last_t))
            last_bytes, last_t = now_bytes, now
            progress = {
                "items": len(items),
                "bytes": all_bytes,
                "done_items": result["cached"] + now_items,
                "done_bytes": (all_bytes - need) + now_bytes,
                "cached_items": result["cached"],
                "rate": round(rate),
                "eta_s": round((need - now_bytes) / rate) if rate > 0 else None,
            }
            try:
                client.post(f"/api/agent/jobs/{job_id}/sync/progress", json={"progress": progress}, timeout=15)
            except httpx.HTTPError:
                pass

    reporter = threading.Thread(target=report, daemon=True)
    reporter.start()
    changed: dict[int, str] = {}  # gone or changed on the holding node: left out, not retried
    failed: dict[int, str] = {}  # still missing after the last try
    try:
        pending = _batches(missing)
        for attempt in range(RETRIES + 1):
            failed, again = {}, []
            with ThreadPoolExecutor(max_workers=STREAMS) as pool:
                futures = {pool.submit(_fetch, base, grant, root, batch, got): batch for batch in pending}
                for future in as_completed(futures):
                    batch = futures[future]
                    try:
                        changed.update(future.result())
                    except Refused:
                        raise
                    except (OSError, ConnectionError, tarfile.TarError, ValueError) as e:
                        left = [
                            it
                            for it in batch
                            if it["id"] not in changed and not cache.has(root, it["sha256"], it["ext"], it["bytes"])
                        ]
                        if left:
                            again.append(left)
                            failed.update({it["id"]: str(e) for it in left})
            if not again or attempt == RETRIES:
                break
            log(f"받지 못한 묶음 {len(again)}개를 다시 받습니다 ({attempt + 1}/{RETRIES}).")
            time.sleep(2 * (attempt + 1))
            pending = again
    finally:
        stop.set()
        reporter.join(timeout=5)
    took = time.monotonic() - started
    result.update(
        fetched=done["items"],
        fetched_bytes=done["bytes"],
        changed=[{"id": i, "reason": r} for i, r in list(changed.items())[:20]],
        changed_count=len(changed),
        seconds=round(took, 1),
    )
    rate = done["bytes"] / took / 1024**2 if took > 0 else 0
    log(f"{done['items']:,}장 ({_size(done['bytes'])})을 받았습니다 · {took:,.1f}초 · 평균 {rate:,.0f} MB/s")
    if changed:
        example = next(iter(changed.values()))
        log(f"등록한 뒤 바뀌었거나 없어진 이미지 {len(changed):,}장은 받지 않았습니다(학습에서 빠짐). 예: {example}")
    if failed:
        raise RuntimeError(f"이미지 {len(failed):,}장을 받지 못했습니다: {next(iter(failed.values()))}")
    return result


def run(params: dict[str, Any]) -> dict[str, Any]:
    """The ``data.sync`` job: bring a version's images here ahead of the training that waits for them."""
    hub, token, job_id, folder = (
        os.environ.get(k) for k in ("BADAMOA_HUB_URL", "BADAMOA_NODE_TOKEN", "BADAMOA_JOB_ID", "BADAMOA_CACHE_DIR")
    )
    if not (hub and token and job_id and folder):
        raise RuntimeError("이 작업은 Node Agent가 실행해야 합니다.")
    with httpx.Client(base_url=hub, headers={"Authorization": f"Bearer {token}"}, timeout=120) as client:
        result = receive(client, job_id, Path(folder))
    if result.get("local"):
        print("이 노드에 등록된 이미지입니다. 받을 것이 없습니다.")
    return result
