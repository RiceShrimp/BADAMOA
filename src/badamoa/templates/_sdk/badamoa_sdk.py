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

    bm.log_artifact("confusion.png")      # (선택) 그림·파일 → 실행 화면의 "산출물"
    bm.log_image("pred.jpg", key="예측", step=epoch)   # (선택) 단계별 그림 → step을 넘기며 봄
    bm.log_table("val", ["image", "pred"], rows)       # (선택) 표 → 정렬·검색되는 표
    bm.overlay("a.jpg", boxes=[[0, 0.9, 0.5, 0.5, 0.2, 0.3]])  # (선택) 표 칸: 이미지 위에 박스를 겹쳐 그림
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

__version__ = "7"


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
    unused_when: dict[str, list[Any]] | None = None,
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
    unused_when: dict[str, list[Any]] | None = None,
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
    unused_when: dict[str, list[Any]] | None = None,
) -> Any:
    """실수 설정 (학습률, weight decay …).

    unused_when: 다른 설정이 이 값들일 때 코드가 이 설정을 쓰지 않음 (예: {"optimizer": ["AdamW"]}).
    하이퍼파라미터 탐색이 아무 차이 없는 실행을 만들지 않게 합니다.
    """
    return _Spec(default)


def boolean(
    default: bool,
    *,
    label: str = "",
    help: str = "",
    group: str = "",
    advanced: bool = False,
    unused_when: dict[str, list[Any]] | None = None,
) -> Any:
    """켜고 끄는 설정."""
    return _Spec(default)


def text(
    default: str = "",
    *,
    label: str = "",
    help: str = "",
    group: str = "",
    advanced: bool = False,
    unused_when: dict[str, list[Any]] | None = None,
) -> Any:
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
        # 등록된 모델에서 이어서 학습할 때 그 모델의 가중치 파일 (아니면 None)
        self.init_weights = Path(cfg["init_weights"]) if cfg.get("init_weights") else None
        # 멈춘 실행을 이어 갈 때 그 실행이 마지막으로 남긴 체크포인트 파일 (아니면 None)
        self.resume_checkpoint = Path(cfg["resume_checkpoint"]) if cfg.get("resume_checkpoint") else None
        # 평가·내보내기·테스트(eval · export · predict 진입점)에서만 채워지는 값
        self.weights = Path(cfg["weights"]) if cfg.get("weights") else None  # 다룰 모델의 가중치 파일
        self.train_params: dict[str, Any] = cfg.get("train_params", {})  # 그 모델을 학습할 때의 설정
        self.split: str = cfg.get("split", "val")  # 평가할 split
        self.source = Path(cfg.get("source", "."))  # 테스트할 이미지가 있는 폴더
        # 등록 전 시험 실행이면 True: 작은 합성 데이터로 돌므로 몇 step만 하고 끝내도 됩니다
        self.check: bool = bool(cfg.get("check", False))
        self.summary: dict[str, float] = {}
        self._step = 0
        self._weights = ""
        self._class_metrics: list[dict[str, Any]] = []
        self._table_images: dict[tuple[str, int, int], str] = {}
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


def lightning_callback() -> Any:
    """PyTorch Lightning용 콜백: `Trainer(callbacks=[bm.lightning_callback()])`.

    검증이 끝날 때마다 Lightning이 모은 지표(`self.log(...)`로 기록한 것)와 진행률을 BADAMOA에 알립니다.
    """
    try:
        from lightning.pytorch.callbacks import Callback
    except ImportError:
        from pytorch_lightning.callbacks import Callback

    class BadamoaCallback(Callback):
        def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
            if getattr(trainer, "sanity_checking", False):
                return
            epoch(trainer.current_epoch + 1, trainer.max_epochs or 0, dict(trainer.callback_metrics))

    return BadamoaCallback()


def hf_callback() -> Any:
    """Hugging Face Trainer용 콜백: `Trainer(..., callbacks=[bm.hf_callback()])`.

    Trainer가 로그를 남길 때마다(loss, eval 지표) step과 함께 BADAMOA에 알립니다.
    """
    from transformers import TrainerCallback

    class BadamoaCallback(TrainerCallback):
        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            if logs:
                log(logs, step=state.global_step)
                progress(state.global_step, state.max_steps or 0)

    return BadamoaCallback()


def save_checkpoint(path: str | Path, epoch: int | None = None) -> None:
    """이어서 실행할 수 있게 지금 상태를 저장한 파일을 알립니다 (보통 epoch마다 덮어쓰는 last.pt).

    학습이 중단되거나 실패하면 BADAMOA가 이 파일을 보관하고, "이어서 실행"할 때
    `run.resume_checkpoint`로 돌려줍니다. 선언 파일에 `resume = true`를 적어야 버튼이 나옵니다.
    """
    run = _current()
    _emit("checkpoint", path=str(Path(path).resolve()), epoch=int(run._step if epoch is None else epoch))


def log_artifact(path: str | Path, name: str | None = None) -> None:
    """파일을 실행에 기록합니다 (혼동 행렬 그림, 예측 예시, 보고서 …). 실행 화면의 "파일"에 바로 나옵니다.

    이미지(png · jpg · gif · webp)는 화면에서 미리보기로 보이고, 그 밖의 파일은 내려받을 수 있습니다.
    같은 이름으로 다시 기록하면 새 파일로 바뀝니다 (epoch마다 갱신하는 그림에 씁니다).
    """
    _current()
    _emit("artifact", path=str(Path(path).resolve()), name=name or Path(path).name)


def log_predictions(images: list[dict[str, Any]]) -> None:
    """평가한 이미지마다 모델의 예측을 기록합니다. 데이터셋 화면의 "라벨 의심 순"(틀린 라벨을 찾기)과
    이미지 보기의 예측 겹쳐 보기 · "라벨로 넣기"에 쓰입니다 (평가 진입점에서만).

    images: 이미지마다 {"image": 평가용으로 내보낸 파일 이름에서 확장자를 뺀 것(이미지의 sha256),
    "labels": [[클래스 번호, cx, cy, w, h, …]] (학습 클래스 순서의 번호, 0~1 비율. 분할은 뒤에 윤곽 x, y …,
    키포인트는 뒤에 x, y, 보임(0 · 2) …, 분류는 [[클래스 번호]]), "scores": [확신도, …],
    "found": 맞춘 라벨 수, "missed": 모델이 못 찾은 라벨 수, "extra": 라벨에 없는 예측 수}
    """
    run = _current()
    folder = run.out / ".badamoa"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "predictions.json"
    path.write_text(json.dumps({"images": images[:MAX_PREDICTED]}, ensure_ascii=False), encoding="utf-8")
    _emit("artifact", path=str(path.resolve()), name="predictions.json", media="predictions")


MAX_PREDICTED = 500_000  # images in one log_predictions
MAX_TABLE_ROWS = 10000
MAX_SHAPES = 300  # boxes or outlines per kind in one overlay cell
MAX_POINTS = 400  # points of one outline


def _key(key: str) -> str:
    """A key as a folder name: no separators, no surprises."""
    cleaned = "".join(c if c.isalnum() or c in "-_ ." else "-" for c in str(key)).strip(" .")
    return cleaned[:60] or "media"


def log_image(path: str | Path, key: str | None = None, step: int | None = None, caption: str = "") -> str:
    """이미지를 기록합니다. 기록된 이름을 돌려줍니다 (log_table의 칸에 넣으면 표에서 그림으로 보입니다).

    key를 주면 단계별 기록이 됩니다: 같은 key의 이미지를 실행 화면에서 step을 넘기며 봅니다
    (W&B의 `wandb.log({key: wandb.Image(...)})`, MLflow의 `log_image(key=, step=)`).
    step을 비우면 마지막으로 log한 step입니다. key 없이 부르면 log_artifact와 같습니다.
    """
    run = _current()
    source = Path(path)
    if key is None:
        log_artifact(source)
        return source.name
    at = run._step if step is None else int(step)
    name = f"{_key(key)}/{at:06d}-{source.name}"
    _emit("artifact", path=str(source.resolve()), name=name, key=_key(key), step=at, caption=str(caption)[:200])
    return name


def _cell(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    try:
        number_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return round(number_value, 6) if number_value == number_value and abs(number_value) != float("inf") else None


def _shapes(rows: Any, scored: bool, outline: bool) -> list[list[Any]]:
    """Boxes ``[class, (score,) cx, cy, w, h]`` or outlines ``[class, (score,) [x1, y1, …]]``, cleaned; bad rows dropped."""
    out: list[list[Any]] = []
    for row in list(rows or [])[:MAX_SHAPES]:
        try:
            row = list(row)
            head: list[Any] = [int(row[0])] + ([round(float(row[1]), 4)] if scored else [])
            if outline:
                flat = list(row[len(head)])
                pairs = [flat[i : i + 2] for i in range(0, len(flat) - 1, 2)]
                pairs = pairs[:: -(-len(pairs) // MAX_POINTS)] if pairs else []  # a long outline thinned evenly
                points = [round(float(v), 5) for pair in pairs for v in pair]
                if len(points) >= 6:
                    out.append([*head, points])
            else:
                coords = [round(float(v), 5) for v in row[len(head) : len(head) + 4]]
                if len(coords) == 4:
                    out.append([*head, *coords])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def overlay(
    image: str | Path,
    boxes: list[list[Any]] | None = None,
    truth: list[list[Any]] | None = None,
    outlines: list[list[Any]] | None = None,
    truth_outlines: list[list[Any]] | None = None,
) -> dict[str, Any]:
    """표 칸 하나: 이미지 위에 예측(과 정답)을 겹쳐 그립니다 (W&B의 `wandb.Image(boxes=…, masks=…)`).

        bm.log_table("val", ["image", "missed"], [[bm.overlay("a.jpg", boxes=[[0, 0.91, 0.5, 0.5, 0.2, 0.3]]), 0]])

    image: 이미지 파일 경로(표와 함께 기록됨) 또는 log_image가 돌려준 이름
    boxes: 예측 박스 [[클래스 번호, 확신도, cx, cy, w, h], …] — 좌표는 이미지 크기에 대한 비율(0~1), YOLO 라벨과 같음
    truth: 정답 박스 [[클래스 번호, cx, cy, w, h], …]
    outlines: 분할 예측 [[클래스 번호, 확신도, [x1, y1, x2, y2, …]], …] (비율 좌표)
    truth_outlines: 분할 정답 [[클래스 번호, [x1, y1, …]], …]

    화면에서 정답을 함께 보거나 확신도로 거를 수 있고, 클래스 이름은 log_table의 classes(기본: 학습 클래스)에서 옵니다.
    """
    cell: dict[str, Any] = {"image": str(image)}
    for field, rows, scored, outline in (
        ("boxes", boxes, True, False),
        ("truth", truth, False, False),
        ("outlines", outlines, True, True),
        ("truth_outlines", truth_outlines, False, True),
    ):
        if rows is not None:
            cell[field] = _shapes(rows, scored, outline)
    return cell


def _table_image(run: Run, cell: dict[str, Any]) -> dict[str, Any]:
    """An overlay whose image is a file: record the file with the table (once, however many steps show it)."""
    source = Path(str(cell["image"]))
    if not source.is_file():
        return cell  # a name log_image returned
    resolved = source.resolve()
    stamp = (str(resolved), resolved.stat().st_size, resolved.stat().st_mtime_ns)
    name = run._table_images.get(stamp)
    if name is None:
        # The suffix tells the Hub it is a picture: keep it even when the name is long (hash-named exports).
        name = f"tables/{len(run._table_images) + 1:05d}-{_key(source.stem)[:40]}{source.suffix.lower()}"
        run._table_images[stamp] = name
        _emit("artifact", path=str(resolved), name=name, media="table-image")
    return {**cell, "image": name}


def log_table(
    key: str, columns: list[str], rows: list[list[Any]], step: int | None = None, classes: list[str] | None = None
) -> None:
    """표를 기록합니다 — 샘플별 예측, 클래스별 결과처럼 행이 많은 결과 (W&B Tables, MLflow log_table).

    칸에는 숫자·글자를 넣고, 이 실행에 log_image로 기록한 이미지의 이름을 넣으면 화면에서 그림으로 보입니다.
    bm.overlay(…)를 넣으면 이미지 위에 박스·윤곽을 겹쳐 그립니다.
    같은 key로 step마다 기록하면 실행 화면에서 step을 넘기며 봅니다. 행은 최대 10,000개까지 남습니다.
    classes: 박스의 클래스 번호 → 이름 (생략하면 학습 클래스)
    """
    run = _current()
    at = run._step if step is None else int(step)
    data: dict[str, Any] = {
        "columns": [str(c) for c in columns],
        "rows": [
            [_table_image(run, v) if isinstance(v, dict) and "image" in v else _cell(v) for v in row]
            for row in rows[:MAX_TABLE_ROWS]
        ],
    }
    names = run.classes if classes is None else classes
    if names:
        data["classes"] = [str(c) for c in names]
    folder = run.out / ".badamoa"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_key(key)}-{at:06d}.table.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    name = f"{_key(key)}/{at:06d}.table.json"
    _emit("artifact", path=str(path.resolve()), name=name, key=_key(key), step=at, media="table")


def finish(metrics: dict[str, Any] | None = None, **extra: Any) -> None:
    """실행을 끝냅니다. metrics를 주면 그것이 최종 지표가 되고, 없으면 마지막으로 기록한 값이 됩니다.

    평가는 `finish(metrics, classes=[…])`, 테스트는 `finish(predictions=[…])` 처럼 결과를 함께 넘깁니다.
    predictions 항목: {"image": 파일 이름, "boxes": [[클래스 번호, 확신도, cx, cy, w, h], …]} (탐지)
                     {"image": 파일 이름, "top": [[클래스 번호, 확률], …]} (분류)
    """
    run = _current()
    if run._finished:
        return
    run._finished = True
    final = _numbers(metrics) if metrics else run.summary
    extra.setdefault("classes", run._class_metrics)
    _emit("result", weights=run._weights, metrics=final, **extra)
