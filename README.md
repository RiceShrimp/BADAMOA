<p align="center">
  <img src="docs/img/banner.svg" alt="BADAMOA 바다모아 — 데이터의 바다를 모아, 모델로" width="880">
</p>

<p align="center">
  <a href="https://github.com/RiceShrimp/BADAMOA/releases"><img src="https://img.shields.io/github/v/release/RiceShrimp/BADAMOA?label=release&color=0a7c8c" alt="최신 릴리즈"></a>
  <a href="https://github.com/RiceShrimp/BADAMOA/actions/workflows/ci.yml"><img src="https://github.com/RiceShrimp/BADAMOA/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776ab" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/platform-Ubuntu%20%7C%20Windows-4b5563" alt="Ubuntu | Windows">
</p>

<p align="center">
  <b>한국어</b> · <a href="README.en.md">English</a>
</p>

<br>

**BADAMOA**는 사내 서버에 설치해 쓰는 비전 MLOps 플랫폼입니다. 이미지 데이터셋을 등록하고 라벨을 다듬고, 팀의 GPU 서버에서 학습하고, 나온 모델을 평가하고 관리하는 일을 웹 화면 하나에서 합니다. 설치에는 Docker나 별도 서버 프로그램이 필요 없고, Python 패키지 하나로 돌아갑니다.

이름은 "바다를 모아"에서 왔습니다. 흩어진 데이터의 바다를 모아 모델로 만든다는 뜻이고, 영문으로는 **B**uild · **A**nnotate · **D**evelop · **A**utomate와 **M**LOps **O**rchestration & **A**gents의 머리글자입니다.

<p align="center">
  <img src="docs/img/shot-project.png" alt="프로젝트 화면: COCO128 검출 · 분할 데이터셋 카드와 이 프로젝트에서 학습한 모델 순위" width="880">
</p>

- **데이터는 제자리에 둡니다.** 학습 서버나 NAS에 있는 폴더를 고르면 목록과 라벨만 Hub에 모입니다. 수백 GB를 옮기지 않습니다.

- **라벨은 모델과 함께 다듬습니다.** 학습한 모델의 예측으로 빠지거나 틀린 라벨을 찾고, 그 자리에서 고치거나 일감으로 나눠 팀원에게 맡깁니다. 고친 라벨은 새 버전으로 확정됩니다.

- **GPU 서버는 명령 한 줄로 붙습니다.** 서버는 Hub로 나가는 연결만 쓰므로 방화벽에 구멍을 낼 필요가 없고, 남이 쓰고 있는 GPU는 피해서 배정합니다.

- **결과는 Hub에 남습니다.** 지표, 그림, 체크포인트, 가중치가 모두 Hub에 보관되어 학습 서버가 바뀌어도 이어서 학습하고 비교할 수 있습니다.

- **에이전트가 직접 다룹니다.** Hub가 MCP 서버를 겸하므로 Claude Code 같은 에이전트에게 학습을 맡기고 결과를 물어볼 수 있습니다.

<br>

## 구성

<p align="center">
  <img src="docs/img/architecture.svg" alt="구성: 브라우저와 에이전트가 Hub에 붙고, 학습·데이터 서버(노드)는 Hub로 나가는 연결만 씁니다" width="880">
</p>

Hub는 웹 화면, API, MCP 서버, 카탈로그와 모델 레지스트리를 한 프로세스에서 맡습니다. 실제 학습과 데이터 읽기는 노드가 합니다. Hub를 띄운 PC도 처음부터 노드 하나로 동작하므로, 서버가 한 대뿐이어도 바로 쓸 수 있습니다.

<br>

## 설치

### Hub 띄우기

[uv](https://docs.astral.sh/uv/)와 Node.js가 필요합니다. 저장소를 받은 폴더에서 차례로 실행합니다.

1. 패키지를 설치하고 웹 화면을 빌드합니다.

   ```bash
   uv sync --all-extras
   ```

   ```bash
   npm --prefix web install
   ```

   ```bash
   npm --prefix web run build
   ```

2. Hub를 띄웁니다.

   ```bash
   uv run badamoa hub run --home .badamoa/hub
   ```

3. 브라우저가 이 PC의 네트워크 주소(예: `http://192.168.0.10:8470`)로 열리면 관리자 계정을 만듭니다.

   - 다른 서버는 이 주소로 Hub를 찾아오므로 `localhost` 대신 이 주소로 접속하세요.
   - 계정을 만들면 "샘플 (COCO8)" 프로젝트에 검출 · 분할 · 키포인트 데이터셋과, 일부러 틀린 라벨을 넣은 검수 예시가 들어 있습니다. 필요 없으면 지우면 됩니다.

4. (선택) PC를 다시 켰을 때 Hub도 저절로 켜지게 하려면 **설정 → Hub 실행**에서 "재부팅 시 자동 실행"을 켭니다. 지금과 같은 데이터 폴더 · 포트 · 주소로 실행되고, 노드는 Hub가 켜지면 알아서 다시 붙습니다.

   | Hub를 띄운 곳 | 켜지는 때 |
   |---|---|
   | Ubuntu (root) | 부팅 때 (systemd) |
   | Ubuntu (일반 계정) | 부팅 때 (그 계정의 crontab) |
   | Windows (관리자) | 로그인 없이 부팅 때 |
   | Windows (일반) | 그 계정이 로그인할 때. 화면에 나오는 명령을 관리자 PowerShell에서 한 번 실행하면 로그인 없이 부팅 때 |

> [!TIP]
> 시험해 볼 데이터는 `uv run python scripts/fetch_coco128.py`(COCO128, 약 7 MB)와 `uv run python scripts/fetch_mnist160.py`(MNIST160, 약 70 KB)로 받습니다. `.badamoa/data/` 아래에 풀립니다.

### Hub를 다른 PC로 옮기기

1. 예전 Hub의 **설정 → Hub 옮기기**(관리자)에서 이 Hub를 파일 하나(`badamoa-hub-<날짜>.tar`)로 내보냅니다.

   - 들어가는 것: 계정 · 프로젝트 · 데이터셋 기록 · 학습 · 모델 · 평가 · 썸네일 · 결과 파일
   - 그대로 두는 것: 데이터셋 원본과 받아 둔 이미지(노드에 있음)
   - 내보내기를 누른 때부터 예전 Hub는 읽기만 됩니다(취소하면 다시 씀).

2. 새 PC에서는 위처럼 Hub를 띄울 준비만 하고, 켜기 전에 파일을 가져옵니다(`--home`은 Hub를 띄울 때와 같게).

   ```bash
   uv run badamoa hub import badamoa-hub-20261120-1500.tar --home .badamoa/hub
   ```

3. 새 Hub를 켠 뒤 예전 Hub의 같은 화면에 새 Hub 주소를 적습니다. 노드는 새 Hub가 자기를 아는지 확인하고 넘어가고, 예전 주소로 접속한 사람은 새 Hub로 안내됩니다.

예전 Hub를 끈 채로 내보내려면 `uv run badamoa hub export <폴더> --home .badamoa/hub`를 씁니다.

### 학습 · 데이터 서버 붙이기

1. 웹 화면의 **노드 → 노드 추가**에서 서버가 맡을 역할(학습용 · 오토라벨용 · 추론용 · 데이터 서버용)을 고릅니다. Ubuntu용과 Windows용 명령이 한 줄씩 나옵니다.

2. 붙일 서버에서 그 명령을 실행합니다(Ubuntu는 `sudo`, Windows는 관리자 PowerShell).

3. 자동 시작으로 등록되고, 잠시 뒤 노드 화면에 나타납니다. 재부팅해도 다시 켜져 Hub에 붙고, 이후 업데이트는 Hub 버전을 따라 노드가 스스로 합니다.

알아 둘 것:

- **sudo 권한이 없는 서버**(예: 데이터셋만 읽어 올 서버)에는 같은 화면의 "sudo 권한이 없을 때" 명령을 씁니다. 그 계정의 홈(`~/.badamoa`)에 설치되고, 그 계정의 crontab으로 부팅 때와 5분마다(꺼져 있으면) 켜집니다. 데이터셋 등록은 이미지를 읽기만 하므로 그 계정에 읽기 권한만 있으면 됩니다.

- **어디서 무엇이 도는지**: 학습은 학습용 서버에서, 오토라벨은 이미지가 있는 서버가 오토라벨용일 때 그 서버에서 돕니다. 모델 환경은 학습용 · 오토라벨용 서버에만 설치됩니다.

- **포트**: 노드끼리 이미지를 주고받을 때만 TCP 8475를 씁니다. 설치 명령이 방화벽에서 열어 둡니다. sudo 없이 설치하면 방화벽은 그대로이니, 켜져 있다면 서버 관리자에게 열어 달라고 하세요.

<details>
<summary>Hub를 저장소가 아닌 곳에서 띄웠다면</summary>

<br>

저장소에서 `uv`로 Hub를 띄우면 노드가 받아 갈 설치 파일(wheel)을 Hub가 알아서 만듭니다. 그렇지 않으면 직접 만들어 올립니다.

1. 설치 파일을 만듭니다.

   ```bash
   uv build --wheel -o .badamoa/dist
   ```

2. Hub에 올립니다.

   ```bash
   uv run badamoa hub publish-wheel <wheel 파일> --home .badamoa/hub
   ```

</details>

### 에이전트 연결 (MCP)

1. **에이전트** 화면에서 토큰을 발급하면 연결 명령이 나옵니다.

   ```bash
   claude mcp add --transport http badamoa http://192.168.0.10:8470/mcp --header "Authorization: Bearer <토큰>"
   ```

2. 연결한 뒤 말로 시킵니다. 예: "노드 상태 점검해 줘", "coco128 v1으로 yolo11n 학습하고 끝나면 지표 알려 줘".

에이전트는 노드와 GPU 현황, 데이터셋과 버전, 검수와 라벨링 묶음, 라벨 고치기와 버전 확정, 학습 시작과 하이퍼파라미터 탐색, 모델 조회와 계보, 작업 로그와 실패 원인 진단까지 다룹니다.

<br>

## 둘러보기

아래는 모두 실제 화면입니다. COCO128과 MNIST160으로 학습해 둔 Hub에서 찍었습니다. 화면마다 자세한 사용법은 [사용 설명서](docs/MANUAL.md)에 있습니다. 앱 안에서는 위쪽 줄의 **설명서**로 열며, 읽는 곳을 따라가는 차례와 찾기가 함께 나옵니다.

### 데이터셋

<p align="center">
  <img src="docs/img/feat-images.png" alt="버전 화면의 이미지 그리드: COCO 이미지마다 클래스별 색의 박스" width="880">
</p>

- YOLO, COCO, 클래스별 폴더 형식을 그대로 읽어 검출 · 분할 · 키포인트 · 분류 데이터셋으로 등록합니다.

- 가져올 때마다 버전이 생기고, 버전마다 커밋 해시 같은 지문이 붙어 버전끼리의 차이를 바로 비교합니다.

- 이미지는 split, 클래스, 라벨 상태로 걸러 보고, 학습하려고 다른 서버에 보낸 사본이 어디에 남아 있는지도 확인합니다.

### 라벨 다듬기

<p align="center">
  <img src="docs/img/feat-review.png" alt="검수 화면: 기준 모델과 미라벨 · 오라벨 · 박스 오류로 나뉜 검사 결과" width="880">
</p>

**검수**는 라벨이 빠졌거나(미라벨) 틀렸을(오라벨) 이미지를 찾습니다.

- **규칙 검사**: 겹친 박스나 지나치게 작은 박스 같은 것은 모델 없이 바로 찾습니다.

- **모델 기준**: 학습한 모델로 평가해 두면 모델이 찾았는데 라벨이 없는 물체, 라벨은 있는데 모델이 못 찾은 것까지 짚습니다.

- **문제 없음**: 확인해 보니 맞는 라벨은 "문제 없음"으로 두면, 라벨이 그대로인 동안 다음 버전에서도 다시 걸리지 않습니다.

<p align="center">
  <img src="docs/img/feat-labeling.png" alt="라벨링 묶음 보드: 일감 카드가 미배정 · 라벨링 · 검수 · 완료 칸에 놓여 있다" width="49%">
  <img src="docs/img/feat-label-edit.png" alt="라벨 고치기 화면: 이미지 위의 박스와 클래스 목록, 묶음의 지시문과 완료하고 다음 버튼" width="49%">
</p>

찾은 이미지는 그 자리에서 고치거나 **라벨링 묶음**으로 만들어 팀원에게 나눠 맡깁니다.

- **보드**: 묶음은 일감 단위로 라벨링 → 검수 → 완료를 보드에서 따라가고, 라벨링하는 사람은 지시문을 보며 박스 · 윤곽 · 키포인트 · 클래스를 고칩니다.

- **오토라벨**: 등록한 모델이 묶음의 이미지에 라벨을 먼저 그려 두면, 사람은 처음부터 그리지 않고 고치기만 합니다.

- **버전으로 확정**: 고친 라벨은 데이터셋의 "작업 중"에 모였다가, 무엇이 바뀌었는지 훑어본 뒤 확정하면 새 버전이 됩니다. 이미 있는 버전은 바뀌지 않고, 그때의 검수 기록이 새 버전에 함께 남습니다.

### 학습과 하이퍼파라미터 탐색

<p align="center">
  <img src="docs/img/feat-training.png" alt="학습 실행 화면: 정확도 · 손실 곡선과 지표 카드" width="49%">
  <img src="docs/img/feat-sweep.png" alt="하이퍼파라미터 탐색 화면: 설정별 지표와 가장 좋은 실행" width="49%">
</p>

- 데이터셋 버전과 모델을 고르면 빈 GPU가 있는 노드에서 학습이 돌고, 지표 곡선과 epoch별 예측 그림이 실시간으로 올라옵니다.

- 실행끼리 곡선을 겹쳐 비교하고, 멈춘 학습은 체크포인트에서 이어 갑니다.

- 탐색은 모든 조합 · 무작위 · 베이지안(Optuna TPE)을 지원합니다.

### 모델

<p align="center">
  <img src="docs/img/feat-model.png" alt="모델 화면: 평가 기록, 내보낸 ONNX 파일, 테스트 이미지" width="880">
</p>

- 학습이 끝나면 가중치가 모델 버전으로 등록됩니다.

- champion · production 같은 alias를 붙이고, 같은 데이터로 평가해 비교하고, ONNX · TorchScript로 내보내고, 이미지를 올려 바로 테스트합니다.

- 계보를 따라가면 그 모델이 어느 데이터와 설정에서 나왔는지 거꾸로 확인할 수 있습니다.

<p align="center">
  <img src="docs/img/feat-lineage.png" alt="계보: 데이터셋 버전에서 학습, 모델, 평가 · 내보내기 · 테스트로 이어지는 그림" width="880">
</p>

### 서버와 Hub 상태

<p align="center">
  <img src="docs/img/feat-nodes.png" alt="노드 화면: 서버별 CPU · 메모리 · 디스크와 GPU 칸" width="49%">
  <img src="docs/img/feat-home.png" alt="홈 화면: 데이터 · 모델 · 노드 · 저장 공간 숫자와 시작하기 목록" width="49%">
</p>

- 노드 화면은 서버마다 GPU 칸과 CPU · 메모리 · 디스크를 한 판에 보여 줍니다.

- 사람이 직접 쓰고 있는 GPU는 사용 중으로 보고 작업을 보내지 않습니다.

- 홈에서는 데이터, 모델, 노드, 저장 공간을 숫자로 훑어봅니다.

<br>

## 지원하는 모델

| 작업 | 기본 제공 | 평가 · 내보내기 · 테스트 |
|---|---|---|
| 검출 | Ultralytics YOLO11 · YOLO26<br>torchvision Faster R-CNN · RetinaNet · FCOS · SSDlite | mAP, ONNX · TorchScript, 박스를 그린 이미지 |
| 분할 | Ultralytics YOLO11-seg · YOLO26-seg<br>torchvision Mask R-CNN (v1 · v2) | mask mAP, ONNX, 마스크를 그린 이미지 |
| 분류 | Ultralytics YOLO11-cls · YOLO26-cls<br>torchvision ResNet · EfficientNet · MobileNetV3 · ConvNeXt · RegNet · ViT · Swin | Top-1 · Top-5, 혼동 행렬, ONNX |
| 키포인트 | Ultralytics YOLO11-pose · YOLO26-pose<br>torchvision Keypoint R-CNN | pose mAP(OKS), ONNX, 점을 그린 이미지 |

- 모델 종류는 라이브러리마다 하나입니다: "Ultralytics YOLO"와 "torchvision"이 네 작업을 모두 하고, 고른 데이터셋의 작업에 맞는 모델 목록이 나옵니다.

- 노드가 처음 학습할 때 PyTorch와 학습 라이브러리를 노드 안의 별도 가상환경에 설치합니다(몇 분 걸림). GPU 드라이버에 맞는 빌드를 자동으로 고르고, 노드 환경 변수 `BADAMOA_TORCH_BACKEND`(예: `cpu`, `cu128`)로 지정할 수도 있습니다.

<br>

## 사내 학습 코드 등록하기

기본 제공 모델 말고 팀의 학습 코드를 쓰려면, 코드를 폴더 하나로 묶고 선언 파일(`badamoa.toml`)을 더해 **모델 템플릿**으로 올립니다. 그러면 학습 화면의 "모델 종류"에 나타나고, 기본 모델과 똑같이 학습 · 평가 · 내보내기를 할 수 있습니다. 자세한 방법은 [템플릿 가이드](docs/TEMPLATE_GUIDE.md)에 있습니다.

<p align="center">
  <img src="docs/img/feat-templates.png" alt="설정의 모델 종류: 내장 YOLO · torchvision 템플릿 목록" width="880">
</p>

1. 뼈대를 만듭니다.

   ```bash
   badamoa template init my-model
   ```

2. `my-model/train.py`의 표시한 부분을 자신의 코드로 바꿉니다. 설정 선언과 진행 보고는 SDK(`badamoa_sdk`)로 하며, W&B나 MLflow와 비슷하게 생겼습니다. BADAMOA가 코드를 실행할 때 SDK를 넣어 주므로 따로 설치하지 않습니다.

   ```python
   import badamoa_sdk as bm

   hp = bm.params(                                  # 설정 선언 → 학습 화면의 입력 칸
       model=bm.choice(["resnet18", "resnet50"], label="모델"),
       epochs=bm.integer(30, min=1, label="epoch"),
       lr=bm.number(0.001, label="학습률", group="최적화", advanced=True),   # "세부 설정"의 "최적화" 묶음
   )
   run = bm.init()                                  # run.data, run.out, run.device, run.classes …
   for epoch in range(1, hp.epochs + 1):
       ...
       bm.log({"train_loss": loss, "map50_95": score}, step=epoch)       # 화면의 그래프에 바로 반영
       bm.progress(epoch, hp.epochs)
   bm.log_artifact("confusion.png")                 # (선택) 그림·파일 → 실행 화면의 "파일"
   bm.log_model(run.out / "best.pt")                # 모델로 등록할 가중치
   bm.finish({"map50_95": best})
   ```

3. 이 PC에서 확인합니다.

   ```bash
   badamoa template check my-model
   ```

4. Hub에 올립니다.

   ```bash
   badamoa template push my-model --hub http://<hub>:8470 --token <관리자 API 토큰>
   ```

알아 둘 것:

- **학습 환경**: `badamoa.toml`의 `[environment] requirements`에 패키지를 적으면 서버가 만들어 주는 가상환경이 기본입니다. 이미 만든 Docker 이미지가 있으면 `[environment] image`에 적거나 **설정 → 환경**에 등록해 컨테이너로 학습합니다.

- **미리 설치**: 어느 서버에 환경을 미리 둘지는 **설정 → 모델 종류 → 환경 관리**에서 고릅니다.

- **다시 올리기**: 코드를 바꿔 다시 올릴 때는 `version`을 올리면 됩니다. 각 학습은 자기가 쓴 버전을 기억하므로 "다시 실행"은 같은 코드로 돕니다.

<br>

## 현재 상태

0.x 개발 버전입니다. 무엇이 되고 무엇이 남았는지는 아래와 같고, 자세한 순서는 [로드맵](docs/ROADMAP.md)에 있습니다.

| 영역 | 되는 것 | 남은 것 |
|---|---|---|
| **Build** · 데이터 | ✅ 복사 없는 등록, 버전과 지문, 버전 비교, 노드 간 직접 전송과 서버별 사본 관리 | 🚧 이미지 탐색(모든 데이터셋의 이미지를 훑고, 글 · 비슷한 이미지로 찾고, 골라 버전으로), 큐레이션 |
| **Annotate** · 라벨 | ✅ 박스 · 윤곽 · 키포인트 · 클래스 고치기, 작업 중 → 버전 확정, 라벨링 묶음과 보드, 검수(규칙 · 모델 기준), 오토라벨 | 🚧 CVAT 연동 |
| **Develop** · 학습 | ✅ 학습, 탐색, 평가, 내보내기, 모델 레지스트리, 사내 템플릿, 가상환경 · Docker 환경 | 🚧 TensorRT |
| **Automate** · 배포 | ✅ 파이프라인(학습 → 평가 · 검수 → 승격 게이트 → 내보내기, 새 버전마다 자동) | 🚧 추론 서비스, 배포, 사람 차례가 있는 파이프라인 |
| **MOA** · 운영 | ✅ 노드와 GPU 배정, 자동 업데이트, MCP 에이전트 | 🚧 승인 게이트, 내장 에이전트 |

> [!NOTE]
> 지금은 HTTP로만 동작하니(TLS는 예정) 사내망 안에서 쓰세요. 노드마다 PyTorch가 든 가상환경을 만들어 수 GB를 차지합니다. 확인은 COCO128 · MNIST160 같은 작은 데이터로 했고, 큰 데이터에서의 속도는 실제로 올릴 때 잽니다.

<br>

## 개발

- **테스트와 린트**

  ```bash
  uv run pytest
  ```

  ```bash
  uv run ruff check .
  ```

  COCO128을 받아 두면 `uv run pytest`가 실데이터 확인도 함께 하고, `BADAMOA_TEST_YOLO=1`을 붙이면 실제 YOLO 학습(1 epoch, CPU)까지 확인합니다.

- **웹 화면 고치기**: Hub를 띄운 채로 `npm --prefix web run dev`를 실행하면 `/api` 요청이 Hub로 넘어갑니다.

- **새 버전 내기**: `src/badamoa/__init__.py`의 버전과 [CHANGELOG.md](CHANGELOG.md)를 함께 올려 main에 푸시하면 됩니다. 같은 내용의 릴리즈가 [GitHub Releases](https://github.com/RiceShrimp/BADAMOA/releases)에 자동으로 만들어집니다.

**폴더 구성**

```
src/badamoa/core       공용 스키마 · storage · 시스템 정보
src/badamoa/hub        Hub: REST · 노드 프로토콜 · MCP · 웹 화면 제공
src/badamoa/node       Node Agent · 데이터 가져오기 · 학습용 변환 · 환경 · 노드 간 전송
src/badamoa/templates  기본 모델 템플릿 · SDK(badamoa_sdk) · 새 템플릿의 뼈대
web/                   웹 화면 (React + Vite)
tests/                 단위 · API · E2E
```

### 문서

- [docs/MANUAL.md](docs/MANUAL.md) — 사용 설명서: 화면마다 할 수 있는 것과 순서, 관리자 설정, 단축키 (앱의 **설명서**에서도 봄)
- [docs/TEMPLATE_GUIDE.md](docs/TEMPLATE_GUIDE.md) — 템플릿 가이드: 팀의 학습 코드를 모델 종류로 만들어 확인하고 올리는 방법
- [CHANGELOG.md](CHANGELOG.md) — 버전별 변경 (앱 안의 릴리즈 노트와 같은 내용)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 목표, 설계 원칙, 구조
- [docs/ROADMAP.md](docs/ROADMAP.md) — 단계별 계획과 체크리스트
- [docs/EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md) — 사내 학습 코드 · 환경 · 저장 위치, MLflow · W&B 대조표
- [docs/LABELING_PLAN.md](docs/LABELING_PLAN.md) — 라벨 고치기, 라벨링 묶음, 검수, CVAT 연동 계획
- [docs/EXTENSION_PLAN.md](docs/EXTENSION_PLAN.md) — 태스크 · 데이터 형식 · 모델 · 지표 · 환경을 넓히는 계획
- [docs/GENERATION_PLAN.md](docs/GENERATION_PLAN.md) — 데이터 생성 · 증강 계획 (이미지 속성 증강부터)
- [docs/SCREEN_SPECS.md](docs/SCREEN_SPECS.md) — 화면 구성
- [docs/REFERENCE_STUDY.md](docs/REFERENCE_STUDY.md) · [docs/PLANS.md](docs/PLANS.md) — 참고 제품 분석, 최초 구상
