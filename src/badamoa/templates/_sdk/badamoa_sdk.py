"""BADAMOA SDK — 학습 코드를 BADAMOA에 붙이는 한 파일짜리 모듈. 다른 의존성이 없습니다.

BADAMOA가 코드를 실행할 때는 이 모듈을 자동으로 import할 수 있게 해 주므로 따로 설치하지 않아도 됩니다.
(`badamoa template init`이 폴더에 복사해 두는 사본은 BADAMOA 밖에서 실행하거나 편집기의 자동 완성을 위한 것입니다.)

    import badamoa_sdk as bm

    # 1. 설정 선언 — 학습 대화상자의 입력 칸이 됩니다. advanced=True 는 "세부 설정"에 접힙니다.
    hp = bm.params(
        model=bm.choice(["resnet18", "resnet50"], label="모델"),
        epochs=bm.integer(30, min=1, label="epoch"),
        lr=bm.number(0.001, label="학습률", group="최적화", advanced=True),
        augment=bm.boolean(True, label="증강 사용", group="데이터", advanced=True),
    )

    # 2. 실행 시작 — BADAMOA가 정해 준 데이터·출력 폴더·장치
    run = bm.init()

    for epoch in range(1, hp.epochs + 1):
        ...
        bm.log({"train_loss": loss, "map50_95": score}, step=epoch)   # 3. 지표 → 화면의 그래프
        bm.progress(epoch, hp.epochs)                                  #    진행률

    bm.log_artifact("confusion.png")      # (선택) 그림·파일 → 실행 화면의 "파일"
    bm.log_model(run.out / "best.pt")     # 4. 모델로 등록할 가중치
    bm.finish({"map50_95": best})         # 5. 최종 지표와 함께 끝냄

W&B(`init · log · finish`)나 MLflow(`start_run · log_metrics · log_model`)와 같은 흐름입니다.

- 설정 선언은 값을 직접 적어야 합니다(변수·계산식 불가). BADAMOA는 코드를 실행하지 않고 읽기만 해서 입력 칸을 만듭니다.
- `map50_95 · map50 · precision · recall` 이름의 지표는 실행 목록의 표와 타일에 나오고, 다른 이름은 그래프로 나옵니다.
- `print` 한 것은 모두 작업 로그가 됩니다.
- 동작 방식: 표준 출력에 `@@badamoa {JSON}` 한 줄을 씁니다. 네트워크도 토큰도 쓰지 않으므로 가상환경·Docker·기존 환경
  어디서나 같은 코드가 돌고, BADAMOA 밖에서 그냥 실행하면 선언한 기본값으로 돕니다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

__version__ = "2"


# ---- 설정 선언 ---------------------------------------------------------------------------


class _Spec:
    def __init__(self, default: Any) -> None:
        self.default = default


def choice(
    options: list[Any],
    default: Any = None,
    *,
    label: str = "",
    help: str = "",
    group: str = "",
    advanced: bool = False,
    custom: bool = False,
) -> Any:
    """목록에서 하나를 고르는 설정. custom=True 면 목록에 없는 값도 직접 입력할 수 있습니다."""
    return _Spec(options[0] if default is None else default)


def integer(
    default: int,
    *,
    min: int | None = None,
    max: int | None = None,
    label: str = "",
    help: str = "",
    group: str = "",
    advanced: bool = False,
) -> Any:
    """정수 설정 (epoch, 배치 크기 …)."""
    return _Spec(default)


def number(
    default: float,
    *,
    min: float | None = None,
    max: float | None = None,
    label: str = "",
    help: str = "",
    group: str = "",
    advanced: bool = False,
) -> Any:
    """실수 설정 (학습률, weight decay …)."""
    return _Spec(default)


def boolean(default: bool, *, label: str = "", help: str = "", group: str = "", advanced: bool = False) -> Any:
    """켜고 끄는 설정."""
    return _Spec(default)


def text(default: str = "", *, label: str = "", help: str = "", group: str = "", advanced: bool = False) -> Any:
    """글자로 적는 설정 (가중치 경로 등)."""
    return _Spec(default)


class Params(dict):
    """선언한 설정의 값. `hp.lr` 처럼 속성으로도, `hp["lr"]` 처럼 키로도 읽습니다."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"선언하지 않은 설정입니다: {name}") from None


def params(**declared: Any) -> Params:
    """설정을 선언하고, 이번 실행에서 사용자가 고른 값(없으면 기본값)을 돌려줍니다."""
    chosen = load_config().get("params", {})
    values = Params()
    for name, spec in declared.items():
        default = spec.default if isinstance(spec, _Spec) else spec
        value = chosen.get(name, default)
        if isinstance(default, bool):
            value = bool(value)
        elif isinstance(default, int) and value is not None:
            value = int(value)
        elif isinstance(default, float) and value is not None:
            value = float(value)
        values[name] = value
    return values


# ---- 실행 --------------------------------------------------------------------------------


class Run:
    """지금 실행. `config` 는 사용자가 고른 설정 전체, 나머지는 BADAMOA가 정해 준 경로와 장치입니다."""

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
_config: dict[str, Any] | None = None


def _emit(event: str, **data: Any) -> None:
    print("@@badamoa " + json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def _numbers(metrics: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key, value in metrics.items():
        try:
            number_value = float(value)
        except (TypeError, ValueError):
            continue
        if number_value == number_value and abs(number_value) != float("inf"):  # NaN과 무한대는 뺍니다
            out[str(key).replace("/", "_")] = number_value
    return out


def load_config(argv: list[str] | None = None) -> dict[str, Any]:
    """`--config <파일>` 로 받은 설정 JSON. 인자가 없으면(BADAMOA 밖에서 실행) 빈 설정입니다."""
    global _config
    if _config is None or argv is not None:
        argv = sys.argv if argv is None else argv
        _config = {}
        if "--config" in argv:
            _config = json.loads(Path(argv[argv.index("--config") + 1]).read_text(encoding="utf-8"))
    return _config


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


def log_artifact(path: str | Path, name: str | None = None) -> None:
    """파일을 실행에 기록합니다 (혼동 행렬 그림, 예측 예시, 보고서 …). 실행 화면의 "파일"에 바로 나옵니다.

    이미지(png · jpg · gif · webp)는 화면에서 미리보기로 보이고, 그 밖의 파일은 내려받을 수 있습니다.
    같은 이름으로 다시 기록하면 새 파일로 바뀝니다 (epoch마다 갱신하는 그림에 씁니다).
    """
    _current()
    _emit("artifact", path=str(Path(path).resolve()), name=name or Path(path).name)


log_image = log_artifact  # 이미지도 같은 방식으로 기록합니다


def finish(metrics: dict[str, Any] | None = None) -> None:
    """실행을 끝냅니다. metrics를 주면 그것이 최종 지표가 되고, 없으면 마지막으로 기록한 값이 됩니다."""
    run = _current()
    if run._finished:
        return
    run._finished = True
    final = _numbers(metrics) if metrics else run.summary
    _emit("result", weights=run._weights, metrics=final, classes=run._class_metrics)
