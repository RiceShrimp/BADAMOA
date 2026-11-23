# 템플릿 가이드 — 팀의 학습 코드를 모델 종류로

팀이 쓰던 학습 코드를 MOA의 **모델 종류**(템플릿)로 만들어 올리는 방법입니다. 올린 모델 종류는 학습 화면에서 기본 모델(YOLO 등)과 똑같이 고르고, 학습 · 평가 · 테스트 · 내보내기 · 오토라벨을 합니다.
화면 사용법은 [사용 설명서](MANUAL.md)를 보세요. 이 가이드는 0.57 기준입니다.

## 차례

1. [한눈에](#1-한눈에)
2. [준비: 도구 설치와 토큰](#2-준비-도구-설치와-토큰)
3. [폴더 구성](#3-폴더-구성)
4. [선언 파일 badamoa.toml](#4-선언-파일-badamoatoml)
5. [학습 코드](#5-학습-코드)
6. [평가 · 테스트 · 오토라벨 · 내보내기](#6-평가--테스트--오토라벨--내보내기)
7. [SDK 함수](#7-sdk-함수)
8. [확인하기](#8-확인하기)
9. [올리기와 허용](#9-올리기와-허용)
10. [환경](#10-환경)
11. [버전](#11-버전)
12. [자주 나는 실패](#12-자주-나는-실패)

---

## 1. 한눈에

```
도구 설치 → template init → 코드 채우기 → template check → template check --run (보고서) → template push
        → (관리자) 설정 → 모델 종류에서 "시험 실행" → 통과 → "허용" → 모두의 학습 화면에
```

- 꼭 있어야 하는 것은 **학습(train)** 하나입니다. 평가 · 테스트 · 내보내기는 있으면 그 기능이 모델 화면에 생기고, 오토라벨은 `autolabel = true`로 할 수 있다고 적었을 때만 합니다.
- 코드는 MOA SDK(`badamoa_sdk`)로 설정을 선언하고 진행 · 결과를 알립니다. W&B나 MLflow와 비슷한 모양입니다.
- 올리면 바로 쓰이지 않습니다. 관리자가 서버에서 시험 실행하고 허용해야 모두에게 보입니다.
- 관리자 말고도, 관리자가 **모델 종류 올리기**를 켜 준 멤버가 올릴 수 있습니다. 멤버가 올린 것은 초안이고, 올리기 전에 자기 PC에서 `check --run`을 통과해야 합니다.

## 2. 준비: 도구 설치와 토큰

**도구(`badamoa`) 설치**: Hub가 설치 파일을 내줍니다(로그인 필요 없음).

```bash
uv tool install --python 3.12 --index http://<Hub 주소>/install/simple/ badamoa
```

pip을 쓰면(Hub가 http라 `--trusted-host`가 필요합니다):

```bash
pip install --extra-index-url http://<Hub 주소>/install/simple/ --trusted-host <Hub 호스트> badamoa
```

wheel 파일을 직접 받으려면 `http://<Hub 주소>/install/wheels/badamoa-<버전>-py3-none-any.whl`. badamoa가 쓰는 라이브러리(pydantic · httpx · typer 등)는 PyPI(또는 사내 미러)에서 받습니다.

**올릴 수 있는 계정**: 관리자, 그리고 관리자가 **설정 → 사용자**에서 "모델 종류 올리기"를 켜 준 멤버. 그런 계정에는 **설정 → 모델 종류**에 "내 PC에서 만들어 올리기"(관리자는 "만드는 방법")가 보이고, 이 Hub의 주소가 채워진 설치 · 뼈대 · 시험 실행 · 올리기 명령과 **토큰 만들기**(한 번만 보임)가 있습니다.

**API 토큰**: 위의 "토큰 만들기", 또는 Hub의 **에이전트** 화면. 토큰은 환경 변수 `BADAMOA_TOKEN`, Hub 주소는 `BADAMOA_HUB_URL`에 넣어 두면 명령마다 적지 않아도 됩니다.

**SDK는 설치하지 않습니다.** `template init`이 `badamoa_sdk.py` 한 파일(표준 라이브러리만 씀)을 폴더에 넣어 주고, MOA가 실행할 때는 노드가 넣어 줍니다.

## 3. 폴더 구성

```bash
badamoa template init my-model
```

```
my-model/
  badamoa.toml     선언 파일 (이름 · 버전 · 태스크 · 진입점 · 환경 …)
  train.py         학습 코드 (뼈대: 표시한 곳을 자신의 코드로)
  ops.py           (선택) 평가 · 테스트 · 오토라벨 · 내보내기 코드
  badamoa_sdk.py   SDK 사본 (BADAMOA 밖에서 python train.py로 돌려 볼 때)
  README.md        이 폴더의 할 일 목록
```

가중치나 데이터는 넣지 않습니다. 올리는 묶음은 64 MB까지입니다.

## 4. 선언 파일 badamoa.toml

```toml
name = "my-model"              # 모델 종류 이름 (같은 이름 = 같은 모델 종류의 다른 버전)
title = "우리 검출기"           # 화면에 보이는 이름
version = "1.0.0"              # 코드를 바꾸면 올림
description = "사내 검출 모델"
task = "detect"                # detect · segment · pose · classify 또는 팀이 정한 태스크 이름
# tasks = ["detect", "segment"]  # (선택) 여러 태스크를 학습하는 코드라면 대신 이렇게 (아래 "여러 태스크")
data_format = "yolo"           # 학습 데이터를 받을 형식 (아래 표)
variant = "model"              # (선택) 모델 크기를 고르는 설정 이름 — 모델 이름에 붙음 (예: yolo11n)
resume = true                  # (선택) 멈춘 학습을 이어서 할 수 있음 (save_checkpoint 필요)
autolabel = true               # (선택) predict 코드가 오토라벨도 함 (6장, predict 필요)
reads_weights_from = "1.*"     # (선택) 이전 버전으로 학습한 모델에도 이 버전의 코드를 씀 (11장)
license = "AGPL-3.0"           # (선택) 학습 창에 보임

[entrypoints]
train   = "python train.py --config {config}"   # 필수
eval    = "python ops.py --config {config}"     # 평가
predict = "python ops.py --config {config}"     # 테스트 · 오토라벨
export  = "python ops.py --config {config}"     # 내보내기 (ONNX · TorchScript)

[environment]
requirements = ["ultralytics>=8.3,<9"]   # 서버가 가상환경에 설치할 패키지
python = "3.12"
torch = true                             # 서버의 GPU 드라이버에 맞는 PyTorch를 골라 설치
# image = "registry.local/team/trainer:1.2"   # Docker 이미지로 학습하려면
# default = "사내-cuda12"                      # Hub에 등록한 환경 이름을 쓰려면

[data]                         # 이 모델이 받는 데이터 — 학습 시작 전에 고른 버전과 맞춰 봄
classes = { min = 2, max = 0 } # 클래스 수 범위 (0 = 제한 없음, 분류는 기본 min 2)
min_images = 8                 # 라벨 있는 이미지 최소
keypoints = 0                  # 키포인트 모델이면 점 개수 (0 = 아무 개수)

[check]                        # 시험 실행 설정 (9장)
params = { epochs = 1, imgsz = 64 }  # 시험 실행에서만 쓰는 설정 값
timeout_min = 15               # 환경 준비를 뺀 시간
gpu = false                    # CPU로 돌 수 없는 모델만 true
skip = []                      # 시험하지 않을 진입점, 예: ["export"]
```

**태스크와 데이터 형식의 짝**: 노드가 버전을 그 형식의 폴더로 풀어 학습 코드에 넘깁니다.

| 태스크 | data_format | 넘기는 것 |
|---|---|---|
| 검출 `detect` | `yolo` | 박스 그대로 |
| 분할 `segment` | `yolo-seg` | 윤곽 그대로 |
| 분할 `segment` | `yolo` | 박스만 (윤곽은 버림) |
| 키포인트 `pose` | `yolo-pose` | 박스와 점 그대로 (data.yaml에 `kpt_shape`, 점 이름이 left · right로 짝지어지면 `flip_idx`) |
| 키포인트 `pose` | `yolo` | 박스만 (점은 버림) |
| 분류 `classify` | `imagefolder` | 클래스별 폴더 |
| 팀이 정한 태스크 | `yolo` | 박스만 (박스 뒤의 값은 버림) |

맞지 않는 짝은 올릴 때 거부됩니다. 팀 고유의 데이터 형식(manifest · 팀 형식 등록)은 계획에 있습니다.

**여러 태스크를 한 모델 종류로.** 라이브러리 하나가 여러 태스크를 하면(Ultralytics YOLO, torchvision처럼) 모델 종류를 나누지 말고 `tasks`에 모두 적습니다. 학습 창은 고른 데이터셋의 태스크로 이 모델 종류를 보여 주고, 코드는 `run.task`(설정 파일의 `task`)를 보고 무엇을 할지 고릅니다. 환경은 하나라서 서버에 한 번만 설치됩니다.

```toml
tasks = ["detect", "segment", "classify", "pose"]   # 첫 것이 대표 태스크
# 태스크마다 그 태스크의 형식으로 받음: detect yolo · segment yolo-seg · classify imagefolder · pose yolo-pose
# data_formats = { segment = "yolo" }               # (선택) 다르게 받으려면

[params.model]
type = "string"
default = "yolo11n.pt"
choices = ["yolo11n.pt", "yolo11s.pt"]
by_task.segment = { default = "yolo11n-seg.pt", choices = ["yolo11n-seg.pt", "yolo11s-seg.pt"] }  # 태스크마다 다른 목록 · 기본값

[params.mosaic]
type = "number"
default = 1.0
tasks = ["detect", "segment", "pose"]               # 이 태스크들에서만 보이는 설정
```

- `by_task.<태스크>`에는 `default` · `choices` · `label` · `help`를 바꿔 적을 수 있습니다.
- 시험 실행(8 · 9장)은 적은 태스크마다 그 형식의 합성 데이터로 학습부터 내보내기까지 한 번씩 합니다.
- 모델은 학습한 태스크를 기억합니다. 평가 · 내보내기 · 테스트 · 오토라벨도 그 태스크로 코드에 넘어갑니다.
- 태스크마다 나뉘어 있던 모델 종류를 합칠 때는, 예전 것을 `hidden = true`와 `replaced_by = "<합친 이름>"`으로 남겨 두세요. 새 학습은 합친 것으로 가고, 예전 것으로 학습한 모델은 그 코드로 계속 평가 · 내보내기 · 테스트됩니다.

**설정(하이퍼파라미터)은 코드에서 선언합니다.** `bm.params(...)`(5장)에 적은 값이 학습 창의 입력 칸이 됩니다. MOA는 코드를 실행하지 않고 읽어서 칸을 만들므로, 기본값 · 범위는 숫자나 글자로 직접 적어야 합니다(변수 · 계산식 불가).

## 5. 학습 코드

```python
import badamoa_sdk as bm

hp = bm.params(                                    # 설정 선언 → 학습 창의 입력 칸
    model=bm.choice(["resnet18", "resnet50"], label="모델"),
    epochs=bm.integer(30, min=1, label="epoch"),
    lr=bm.number(0.001, label="학습률", group="최적화", advanced=True),  # "세부 설정"의 "최적화" 묶음
)
run = bm.init()                                    # BADAMOA가 정해 준 데이터 · 출력 폴더 · 장치

for epoch in range(1, hp.epochs + 1):
    ...
    bm.log({"train_loss": loss, "map50_95": score}, step=epoch)   # 화면의 그래프
    bm.progress(epoch, hp.epochs)                                  # 진행률
    bm.save_checkpoint(run.out / "last.pt", epoch=epoch)           # (선택) 이어서 하기용

bm.log_model(run.out / "best.pt")                  # 모델로 등록할 가중치 (필수)
bm.finish({"map50_95": best})                      # 최종 지표와 함께 끝냄 (필수)
```

`run`이 주는 것:

| 이름 | 뜻 |
|---|---|
| `run.config` | 사용자가 고른 설정 전체 |
| `run.data` | 학습 데이터 (YOLO 형식이면 `data.yaml` 경로) |
| `run.data_dir` | 데이터 폴더 (`images/train`, `labels/train` …) |
| `run.out` | 결과를 쓸 폴더 |
| `run.cache` | 실행 사이에 남는 폴더 (사전학습 가중치 등) |
| `run.device` | `"cpu"` 또는 `"0"`, `"0,1"` (배정받은 GPU는 0번부터) |
| `run.classes` | 클래스 이름 목록 (학습 순서) |
| `run.task` | 이번 실행의 태스크 (`detect` · `segment` · `classify` · `pose` …). `tasks`로 여러 태스크를 학습하는 코드가 봄 |
| `run.init_weights` | 등록된 모델에서 이어서 학습할 때 그 가중치 (아니면 None) |
| `run.resume_checkpoint` | 멈춘 학습을 이어 갈 때 마지막 체크포인트 (아니면 None) |
| `run.check` | 시험 실행이면 True — 몇 step만 하고 끝내도 됨 |

- `map50_95 · map50 · precision · recall`(분할은 `mask_map50_95`, 키포인트는 `pose_map50_95`, 분류는 `top1` 등) 이름의 지표는 학습 표의 열과 타일에, 다른 이름은 그래프로 나옵니다. 학습끼리는 태스크의 첫 지표로 순위를 매깁니다.
- `print`한 것은 모두 작업 로그가 됩니다.
- PyTorch Lightning이나 Hugging Face Trainer를 쓰면 `bm.lightning_callback()` · `bm.hf_callback()`을 콜백으로 넣으면 지표와 진행이 알아서 갑니다.

## 6. 평가 · 테스트 · 오토라벨 · 내보내기

진입점 하나의 파일(`ops.py`)에서 `run.cfg["action"]`으로 나누면 됩니다. 모두 `run.weights`(학습한 가중치)와 `run.train_params`(그 모델을 학습할 때의 설정), `run.classes`를 받습니다.

```python
import badamoa_sdk as bm

run = bm.init()
action = run.cfg["action"]
model = load(run.weights, run.train_params)

if action == "eval":                     # 평가
    metrics, per_image = evaluate(model, run.data_dir, run.split)
    bm.log_predictions(per_image)        # (선택) 이미지마다 예측 → 검수 · "라벨 의심 순"
    bm.finish(metrics)
elif action == "export":                 # 내보내기
    path = export(model, run.config.get("format", "onnx"), run.out)
    bm.log_artifact(path)
    bm.finish(file=path.name)
elif action == "autolabel":              # 오토라벨 (predict 진입점)
    bm.log_predictions(label_every_image(model, run.source, run.config.get("conf", 0.25)))
    bm.finish()
else:                                    # predict: 테스트
    bm.finish(predictions=predict_images(model, run.source, run.config.get("conf", 0.25)))
```

| 진입점 | 받는 것 | 알려야 하는 것 |
|---|---|---|
| **평가** `eval` | `run.data_dir`(평가할 데이터, 학습과 같은 형식) · `run.split` | `bm.finish(지표)`. 이미지마다 예측을 `bm.log_predictions`로 남기면 검수의 "모델 기준"과 버전 화면의 "라벨 의심 순"에 쓰임 |
| **테스트** `predict` | `run.source`(올린 이미지 폴더) · `run.config["conf"]` | `bm.finish(predictions=[{"image": 파일 이름, "boxes": [[클래스, 확신도, cx, cy, w, h], …]}, …])` (분류는 `"top": [[클래스, 확률], …]`) |
| **오토라벨** (`predict`, `action = "autolabel"`) | `run.source`(파일 이름 = 이미지의 sha256) · `run.config["conf"]` | `bm.log_predictions([{"image": 파일 이름(확장자 뺀 것), "labels": [[클래스, cx, cy, w, h, …]], "scores": [확신도, …]}, …])` — 데이터셋의 라벨 형식 그대로(분할은 뒤에 윤곽 x, y …, 키포인트는 x, y, 보임 …, 분류는 `[[클래스]]`) |
| **내보내기** `export` | `run.config["format"]`(`onnx` · `torchscript`) | `bm.log_artifact(파일)`, `bm.finish(file=...)` |

좌표는 모두 0~1 비율이고, 클래스 번호는 `run.classes`의 순서입니다.

**오토라벨은 선언합니다.** predict 코드가 `action = "autolabel"`을 처리한다면 선언 파일 맨 위에 `autolabel = true`를 적습니다(`predict`가 있어야 함). 적지 않은 모델 종류의 모델은 오토라벨 창에서 "이 모델 종류는 오토라벨을 하지 않습니다"로 막힙니다. 예측이 라벨 모양이 아닌 태스크처럼 오토라벨을 할 수 없으면 적지 않습니다. 시험 실행은 적었을 때만 오토라벨 모드를 돌려 모든 이미지에 예측 줄이 있는지 봅니다.

## 7. SDK 함수

| 함수 | 하는 일 |
|---|---|
| `bm.params(**설정)` | 설정 선언. 값은 `bm.choice(목록, label=)` · `bm.integer(기본, min=, max=)` · `bm.number(기본)` · `bm.boolean(기본)` · `bm.text(기본)`. 공통 인자: `label`, `help`, `group`(세부 설정의 묶음), `advanced`(세부 설정에 접기) |
| `bm.init()` | 실행 시작, `run` 돌려줌 |
| `bm.info(**값)` | 라이브러리 버전 · 장치 같은 정보 기록 |
| `bm.log(지표, step=)` | 지표 기록 → 그래프 |
| `bm.epoch(epoch, epochs, 지표)` | epoch 단위로 지표와 진행을 한 번에 |
| `bm.progress(현재, 전체)` | 진행률 |
| `bm.save_checkpoint(경로, epoch=)` | 이어서 하기용 체크포인트 (`resume = true`일 때) |
| `bm.log_model(가중치, classes=)` | 모델로 등록할 가중치 (클래스별 지표를 함께 넘길 수 있음) |
| `bm.log_artifact(경로, name=)` | 파일 → 실행 화면의 "산출물" |
| `bm.log_image(경로, key=, step=, caption=)` | 단계별 그림 → 단계를 넘기며 봄 |
| `bm.log_table(이름, 열, 행)` | 표 → 정렬 · 검색되는 표 |
| `bm.overlay(그림, boxes=, truth=, …)` | 표 칸: 이미지 위에 박스 · 윤곽을 겹쳐 그림 |
| `bm.log_predictions(이미지들)` | 이미지마다 예측 (평가 · 오토라벨) |
| `bm.finish(지표, **결과)` | 끝 (모든 진입점에서 필수) |
| `bm.lightning_callback()` · `bm.hf_callback()` | Lightning · HF Trainer용 콜백 |

동작 방식: SDK는 표준 출력에 `@@badamoa {JSON}` 한 줄을 씁니다. 네트워크도 토큰도 쓰지 않으므로 가상환경 · Docker · 서버의 기존 환경 어디서나 같은 코드가 돌고, MOA 밖에서 그냥 실행하면 선언한 기본값으로 돕니다.

## 8. 확인하기

```bash
badamoa template check my-model
```

선언 · 진입점 파일 · 문법 · 태스크와 형식의 짝 · 패키지 표기를 봅니다(실행하지 않음).

```bash
badamoa template check my-model --run
```

**이 PC에서 서버와 같은 시험 실행**을 합니다. 작은 합성 데이터(색 사각형과 그 라벨)를 만들어 학습 → 테스트 · 오토라벨(선언했을 때) · 평가 · 내보내기를 `[check]` 설정으로 돌리고, 실패하면 단계와 이유를 보여 줍니다(종료 코드 1). 환경과 캐시는 `~/.badamoa/template-trial`에 둡니다(`--home`으로 바꿈).

끝나면 **보고서**를 폴더의 `.badamoa/check-report.json`에 남깁니다: 단계별 결과와 시간, MOA SDK 버전, Python · OS · GPU, 설치된 패키지, 묶음의 sha256. 이 폴더는 올리는 묶음에 들어가지 않습니다. `template push`가 보고서를 함께 보내고, 코드를 바꾼 뒤에는 다시 `--run` 해야 합니다(보고서의 해시가 맞지 않음).

## 9. 올리기와 허용

```bash
badamoa template push my-model --hub http://<Hub 주소> --token <관리자 API 토큰>
```

- 8장의 보고서를 함께 보냅니다. **관리자가 아닌 계정은 같은 묶음의 통과한 보고서 없이는 올릴 수 없습니다** ("먼저 badamoa template check . --run을 통과하세요").
- 멤버가 올린 것은 **초안**입니다. 올린 사람만 모델 종류 목록에서 "내 초안"으로 보고 자기 학습에 쓸 수 있고, 다른 사람에게는 관리자가 허용한 뒤에 보입니다.
- `--trial`(관리자)을 붙이면 올린 뒤 서버의 시험 실행까지 시작하고 결과를 기다립니다.
- 관리자는 폴더를 zip으로 묶어 Hub의 **설정 → 모델 종류 → 올리기 (.zip)**로 올려도 같습니다(보고서 없이).
- 보고서는 실수를 막는 장치이지 보안 장치가 아닙니다(손으로 만들 수도 있음). 진짜 확인은 관리자가 거는 서버의 시험 실행이고, 관리자는 그 옆에서 "올린 사람 PC에서 통과"(단계 수 · Python · OS · GPU · SDK)를 봅니다.

**올린 뒤 (관리자)**: 올린 버전은 **시험 전**이고, 아무에게도 보이지 않습니다.

1. 설정 → 모델 종류에서 그 버전의 **시험 실행**. 학습용 서버 한 대가 합성 데이터로 짧게 돌려 봅니다(빈 GPU가 있으면 GPU, 없으면 CPU. `[check] gpu = true`면 GPU를 기다림).
2. 결과: **통과 · 허용 대기** 또는 **실패 — 단계: 이유**(로그 보기, 다시 시험).
3. **허용**을 누르면 그 버전이 모두의 학습 화면에 나옵니다.
4. 급할 때는 이유를 적고 "통과 없이 허용"할 수 있습니다(기록에 남음).

0.53.0 전에 올린 버전은 그대로 쓰이고, 관리자가 원할 때 손으로 시험해 볼 수 있습니다. 내장 모델 종류는 시험 없이 늘 쓸 수 있습니다.

## 10. 환경

| 방식 | 선언 | 언제 |
|---|---|---|
| 자동 가상환경 (기본) | `[environment] requirements`, `python`, `torch` | 대부분. 서버가 처음 학습할 때 만들고 다음부터 다시 씀 |
| Docker 이미지 | `[environment] image` 또는 설정 → 환경에 등록 | 이미 만든 이미지가 있을 때. Docker가 있는 서버에서만 |
| 서버에 있는 Python | 설정 → 환경에 등록하고 `default`에 이름 | 서버에 이미 갖춰진 환경을 쓸 때 |

- **사내 패키지**: PyPI에 없는 패키지는 wheel(.whl)로 만들어 설정 → 환경 → "사내 패키지"에 올리고 `requirements`에 이름을 적습니다. 서버가 Hub에서 받아 설치합니다.
- 시험 실행을 통과한 버전은 그때 설치된 패키지 버전을 고정해, 다른 서버도 같은 버전으로 설치합니다(PyTorch는 버전만 고정하고 CUDA 빌드는 서버에 맞춤).
- 환경을 서버에 미리 깔아 두려면 설정 → 모델 종류 → **환경 관리**.
- **HTTPS 없는 사내 레지스트리**(예: `192.168.0.10:5000`): Docker는 HTTPS가 아닌 레지스트리에서 받지 않습니다. sudo로 설치한 노드는 설치 · 업데이트할 때와 받다 막혔을 때 그 서버 Docker의 `insecure-registries`에 넣고 다시 받습니다. sudo 없이 설치한 노드는 서버 관리자가 한 번 `curl -fsSL <hub>/install/docker-registries.sh | sudo bash`를 실행합니다(학습 로그에 그대로 나옴). Docker Desktop은 Settings → Docker Engine에 같은 값을 넣습니다.
- **Hub의 레지스트리**: Hub는 자기 PC에 레지스트리를 마련합니다(있던 것을 쓰거나 `badamoa-registry`를 만듦, 설정 → 환경). 이미지를 만들었으면 그 레지스트리 이름으로 태그를 붙여 `docker push` 하세요. 시험 실행은 그 PC의 이미지로 통과해도 다른 서버는 레지스트리에서 받습니다. 설정 → 환경에서 쓰는 이미지가 레지스트리에 있는지 보이고, 없으면 Hub PC에 있는 이미지는 **레지스트리에 올리기**로 올립니다.

## 11. 버전

- 코드를 바꾸면 `version`을 올려 다시 올립니다. 같은 버전 번호에 다른 내용을 올리면 거부되고, 똑같은 내용을 다시 올리면 아무 일도 없습니다.
- 예전 버전은 지워지지 않습니다. 각 학습은 자기가 쓴 버전을 기억하므로, "다시 실행"은 같은 코드로 돕니다.
- 새 버전이 허용되기 전까지 다른 사람은 그 전 버전을 그대로 씁니다.
- 모델의 평가 · 테스트 · 내보내기 · 오토라벨은 **그 모델을 학습한 버전의 코드**로 돕니다(가중치를 쓴 코드가 읽음).
- 그 버전에 없는 기능을 새 버전에서 더했다면, 새 버전의 선언에 **`reads_weights_from`**으로 어느 이전 버전의 가중치를 읽을 수 있는지 적습니다. 예: 1.1.0에 `reads_weights_from = "1.*"` → 1.0.0으로 학습한 모델에도 1.1.0의 내보내기가 생깁니다. 모두에게 허용된 버전 중 그렇게 적은 가장 새 버전의 코드로, 학습 때 설정을 그대로 넘겨 돕니다. 모델 화면의 기록에 "1.1.0 코드로"가 붙습니다.
- 범위는 pip의 버전 범위입니다(`">=1.0,<2"`). 숫자로 시작하면 그 버전입니다(`"1.*"` = 1점대 전부, `"1.2"` = 1.2만). 적지 않으면 이전 버전의 모델에는 새 기능이 생기지 않습니다.
- **환경은 가상환경 먼저**: 모델 작업은 그 코드가 설치할 패키지(`requirements`)를 적었으면 Docker 이미지도 적혀 있고 모델이 이미지에서 학습했어도 그 가상환경에서 돕니다(학습 창의 기본과 같음). 패키지 없이 이미지만 적은 버전으로 학습한 모델은, 가상환경을 설치하고 `reads_weights_from`으로 그 버전을 읽는다고 적은 새 버전이 허용되어 있으면 그 버전의 코드로 가상환경에서 돌고, 없으면 학습한 이미지에서 돕니다.

## 12. 자주 나는 실패

| 보이는 말 | 고칠 점 |
|---|---|
| `[entrypoints]에 train 명령이 필요합니다` | `train = "python train.py --config {config}"`를 적으세요 |
| `… 데이터는 data_format …으로 풀 수 없습니다` | 4장의 짝 표에 있는 형식으로 |
| `가중치 파일을 알리지 않았습니다. 학습한 가중치를 log_model로 알리세요.` | 학습 끝에 `bm.log_model(경로)` |
| `지표를 하나도 알리지 않았습니다. …` | `bm.log` · `bm.epoch`로 지표를 하나 이상 |
| `결과를 알리지 않았습니다. 끝날 때 finish(또는 result)를 호출해야 합니다.` | 모든 진입점의 끝에 `bm.finish(...)` |
| `예측을 알리지 않았습니다. finish(predictions=[…])로 …` | 테스트(predict)는 이미지마다 결과를 `finish(predictions=…)`로 |
| `이미지 N장 중 M장에 예측이 없습니다` | 예측이 없는 이미지도 빈 목록으로 한 줄씩 |
| `평가 지표를 하나도 알리지 않았습니다. …` | 평가는 `bm.finish(지표)` |
| `정해진 시간 안에 끝나지 않았습니다 ([check] timeout_min).` | `run.check`면 몇 step만 하거나 `[check] params`로 줄이기 |
| `템플릿 파일이 너무 큽니다.` | 가중치 · 데이터를 빼고 64 MB 아래로 |
| `… 이미 다른 내용으로 등록되어 있습니다. badamoa.toml의 version을 올리세요.` | `version`을 올리기 |
| `모델 종류를 올릴 권한이 없습니다. …` | 관리자에게 설정 → 사용자에서 "모델 종류 올리기"를 켜 달라고 하기 |
| `먼저 badamoa template check . --run을 통과하세요. …` | 그 폴더에서 `check --run`을 통과한 뒤 다시 push |
| `시험 실행 보고서가 이 묶음의 것이 아닙니다. …` · `마지막 check --run 뒤에 코드가 바뀌었습니다` | 코드를 바꿨으면 `check --run`을 다시 |
| `autolabel = true에는 predict 진입점이 필요합니다` | predict를 적거나 `autolabel`을 빼기 |
| `예측을 남기지 않았습니다. 오토라벨에서는 bm.log_predictions(...)로 …` | 오토라벨 모드(`action == "autolabel"`)에서 이미지마다 `bm.log_predictions` |
| `이미지 N장 중 M장에 예측 줄이 없습니다 (이미지는 sha256 이름)` | 오토라벨은 `run.source`의 모든 파일(확장자 뺀 이름)마다 한 줄씩, 예측이 없으면 빈 `labels` |
| `reads_weights_from '…'은 버전 범위가 아닙니다` | `"1.*"`, `">=1.0,<2"`처럼 |
| `이 모델 종류는 오토라벨을 하지 않습니다` (오토라벨 창) | 그 모델 종류가 오토라벨을 하면 `autolabel = true`를 적은 새 버전을 올리고, `reads_weights_from`으로 이전 모델에도 쓰게 함 |
