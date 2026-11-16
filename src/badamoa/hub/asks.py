"""Quick questions the Hub puts to a node and waits for, outside the job queue (a folder listing).

The node keeps one long-poll open for them (``/api/agent/requests``) next to the one for jobs, so
an answer comes back in well under a second without a job record for every folder clicked.
Questions live in memory: a Hub restart simply drops the few in flight.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

from .services import ServiceError

WAIT_S = 15.0
SINCE = (0, 31, 0)  # node versions from here on answer (a folder listing)
REMOVE_SINCE = (0, 32, 0)  # … and take themselves off their machine when deleted

_lock = threading.Lock()
_pending: dict[str, list[dict[str, Any]]] = {}  # node id -> questions not yet taken
_events: dict[str, threading.Event] = {}
_answers: dict[str, dict[str, Any]] = {}


def answers_questions(agent_version: str, since: tuple[int, int, int] = SINCE) -> bool:
    try:
        return tuple(int(p) for p in agent_version.split(".")) >= since
    except ValueError:
        return True  # a development build: let it try


def ask(node_id: str, kind: str, params: dict[str, Any], wait: float = WAIT_S) -> Any:
    """Put a question to a node and wait for its answer (or say why there is none)."""
    question = {"id": uuid.uuid4().hex, "kind": kind, "params": params}
    done = threading.Event()
    with _lock:
        _pending.setdefault(node_id, []).append(question)
        _events[question["id"]] = done
    answered = done.wait(wait)
    with _lock:
        _events.pop(question["id"], None)
        answer = _answers.pop(question["id"], None)
        waiting = _pending.get(node_id, [])
        if question in waiting:
            waiting.remove(question)
    if not answered or answer is None:
        raise ServiceError(504, "노드가 답하지 않습니다. 노드가 켜져 있는지, 노드 프로그램이 0.31.0 이상인지 확인하세요.")
    if answer.get("error"):
        raise ServiceError(400, str(answer["error"]))
    return answer.get("result")


def take(node_id: str) -> list[dict[str, Any]]:
    with _lock:
        return _pending.pop(node_id, [])


def answer(question_id: str, payload: dict[str, Any]) -> bool:
    with _lock:
        done = _events.get(question_id)
        if done is None:
            return False  # nobody waits any more
        _answers[question_id] = payload
        done.set()
        return True
