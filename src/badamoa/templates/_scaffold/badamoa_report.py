"""BADAMOA에 학습 진행을 알리는 도우미. 이 파일 하나면 되고 다른 의존성이 없습니다.

W&B나 MLflow를 써 봤다면 익숙한 모양입니다. 호출하는 즉시 BADAMOA 화면의 그래프·진행률에 반영됩니다.

    import badamoa_report as report

    run = report.init()                       # 설정을 읽고 실행을 시작 (wandb.init / mlflow.start_run)
    lr = run.config["lr"]                     # badamoa.toml의 [params]에서 사용자가 고른 값
    for epoch in range(1, epochs + 1):
        ...
        report.log({"train_loss": 1.2, "map50_95": 0.41}, step=epoch)   # 지표 (wandb.log / mlflow.log_metrics)
        report.progress(epoch, epochs)                                   # 진행률 막대
    report.log_model("out/best.pt")           # 모델로 등록할 가중치 (mlflow.log_model)
    report.finish({"map50_95": 0.52})         # 최종 지표와 함께 끝냄 (wandb.finish)

- `report.epoch(n, total, metrics)` 는 `log` + `progress` 를 한 번에 합니다.
- `map50_95 · map50 · precision · recall` 이름을 쓰면 실행 목록의 표와 타일에 바로 나옵니다. 다른 이름은 그래프에 나옵니다.
- `print` 로 찍은 것은 모두 작업 로그가 됩니다.

동작 방식: 표준 출력에 `@@badamoa {JSON}` 한 줄을 씁니다. 네트워크도 토큰도 쓰지 않으므로
가상환경·Docker·기존 서버 환경 어디서나 같은 코드가 돕니다. BADAMOA 밖에서 그냥 실행해도 줄이 찍힐 뿐 문제없습니다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class Run:
    """지금 실행. `config` 는 사용자가 고른 설정, 나머지는 BADAMOA가 정해 준 경로와 장치입니다."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.config: dict[str, Any] = cfg.get("params", {})
        self.data: str = cfg.get("data", "")  # 학습 데이터 (YOLO 형식이면 data.yaml 경로)
        self.data_dir = Path(cfg.get("data_dir", "."))  # images/train, labels/train, …
        self.out = Path(cfg.get("out", "out"))  # 결과를 쓸 폴더
        self.cache = Path(cfg.get("cache", "."))  # 실행 사이에 유지되는 폴더 (사전학습 가중치 등)
        self.device: str = cfg.get("device", "cpu")  # "cpu" 또는 "0", "0,1" (배정받은 GPU는 0번부터)
        self.classes: list[str] = cfg.get("classes", [])
        self.summary: dict[str, float] = {}
        self._step = 0
        self._weights = ""
        self._class_metrics: list[dict[str, Any]] = []
        self._finished = False
        self.out.mkdir(parents=True, exist_ok=True)


_run: Run | None = None


def _emit(event: str, **data: Any) -> None:
    print("@@badamoa " + json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def _numbers(metrics: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in metrics.items():
        try:
            out[str(key).replace("/", "_")] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def load_config(argv: list[str] | None = None) -> dict[str, Any]:
    """`--config <파일>` 로 받은 설정 JSON. 인자가 없으면(BADAMOA 밖에서 실행) 빈 설정입니다."""
    argv = sys.argv if argv is None else argv
    if "--config" not in argv:
        return {}
    return json.loads(Path(argv[argv.index("--config") + 1]).read_text(encoding="utf-8"))


def init(**info_fields: Any) -> Run:
    """실행을 시작합니다. 넘긴 값(라이브러리 버전 등)은 실행 기록의 환경 정보에 남습니다."""
    global _run
    _run = Run(load_config())
    info(device=_run.device, **info_fields)
    return _run


def _current() -> Run:
    return _run if _run is not None else init()


def info(**data: Any) -> None:
    """라이브러리 버전, 장치 등 실행 기록에 남길 정보."""
    _emit("info", **data)


def log(metrics: dict[str, Any], step: int | None = None) -> None:
    """지표를 기록합니다. step을 생략하면 1씩 늘어납니다. 같은 step에 여러 번 부르면 합쳐집니다."""
    run = _current()
    run._step = run._step + 1 if step is None else int(step)
    values = _numbers(metrics)
    run.summary.update(values)
    _emit("log", step=run._step, metrics=values)


def progress(current: int, total: int) -> None:
    """진행률 막대. 보통 (지금 epoch, 전체 epoch)."""
    _emit("progress", current=int(current), total=int(total))


def epoch(epoch: int, epochs: int, metrics: dict[str, Any]) -> None:
    """epoch 하나가 끝날 때: 그 epoch의 지표와 진행률을 함께 알립니다."""
    run = _current()
    run._step = int(epoch)
    values = _numbers(metrics)
    run.summary.update(values)
    _emit("epoch", epoch=int(epoch), epochs=int(epochs), metrics=values)


def log_model(weights: str | Path, classes: list[dict[str, Any]] | None = None) -> None:
    """모델로 등록할 가중치 파일을 정합니다. 여러 번 부르면 마지막 것이 등록됩니다.

    classes(선택)는 클래스별 지표입니다:
    {"id": 0, "name": "person", "precision": .., "recall": .., "map50": .., "map50_95": ..}
    """
    run = _current()
    run._weights = str(weights)
    run._class_metrics = classes or []


def finish(metrics: dict[str, Any] | None = None) -> None:
    """실행을 끝냅니다. metrics를 주면 그것이 최종 지표가 되고, 없으면 마지막으로 기록한 값이 됩니다."""
    run = _current()
    if run._finished:
        return
    run._finished = True
    final = _numbers(metrics) if metrics else run.summary
    _emit("result", weights=run._weights, metrics=final, classes=run._class_metrics)


def result(
    weights: str | Path | None, metrics: dict[str, Any] | None = None, classes: list[dict[str, Any]] | None = None
) -> None:
    """`log_model` + `finish` 를 한 번에."""
    if weights:
        log_model(weights, classes)
    finish(metrics)
