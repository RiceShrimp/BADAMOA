# BADAMOA (바다모아)

데이터의 바다를 모아, 모델로 — 설치형 비전 MLOps 플랫폼.

- **BADA** — **B**uild · **A**nnotate · **D**evelop · **A**utomate
- **MOA** — **M**LOps **O**rchestration & **A**gents

데이터 관리, 오토라벨링, 학습(일반 학습 + Teacher→Student 증류), 평가, 모델 관리, 배포를 한 곳에서 하고,
원격 학습·데이터 서버를 한 줄 명령으로 붙이며, Claude Code 같은 에이전트가 MCP로 전부 조작할 수 있게 만드는 것이 목표입니다.

지금 되는 것: Hub와 노드 연결, 작업 실행·로그, 노드 자원 관리(GPU 단위 배정), 데이터셋 카탈로그(복사 없이 등록, 릴리스, 라벨 개정, 비교), YOLO 학습(지표 곡선, 실행 비교)과 모델 등록, MCP 서버, 웹 UI.
바뀐 내용은 [CHANGELOG.md](CHANGELOG.md), 설계는 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), 일정은 [docs/ROADMAP.md](docs/ROADMAP.md), 다음 단계 계획(사내 학습 코드 등록 · 환경 선택 · 저장 위치)은 [docs/EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md), 화면 구성과 레퍼런스 분석은 [docs/REFERENCE_STUDY.md](docs/REFERENCE_STUDY.md), 최초 구상은 [docs/PLANS.md](docs/PLANS.md).

## 실행 (개발 환경)

[uv](https://docs.astral.sh/uv/)와 Node.js가 필요합니다.

```bash
uv sync --all-extras
```

```bash
npm --prefix web install
```

```bash
npm --prefix web run build
```

```bash
uv run badamoa hub run --home .badamoa/hub
```

브라우저에서 <http://localhost:8470> 을 열어 관리자 계정을 만듭니다. Hub를 실행한 머신이 첫 번째 노드로 바로 등록됩니다(All-in-one).

## 원격 서버를 노드로 추가

1. 원격 서버가 받아 갈 패키지를 Hub에 올립니다.

   ```bash
   uv build --wheel -o .badamoa/dist
   ```

   ```bash
   uv run badamoa hub publish-wheel .badamoa/dist/badamoa-0.5.0-py3-none-any.whl --home .badamoa/hub
   ```

2. 웹 UI → **노드** → **노드 추가** 에서 나오는 한 줄 명령을 원격 서버(Ubuntu 또는 Windows)에서 실행합니다.

노드는 Hub로 나가는 연결만 사용하므로 원격 서버 쪽 포트를 열 필요가 없습니다.

## 에이전트 연결 (MCP)

웹 UI → **에이전트 연결** 에서 토큰을 발급하면 연결 명령이 나옵니다.

```bash
claude mcp add --transport http badamoa http://localhost:8470/mcp --header "Authorization: Bearer <토큰>"
```

연결한 뒤 "BADAMOA 노드 상태 점검해줘"처럼 요청하면 됩니다.

## 데이터셋 등록해 보기

웹 UI → **데이터셋** 에서 프로젝트를 만들고 **데이터셋 가져오기** 를 누른 뒤, 노드에 있는 YOLO 또는 COCO 형식 폴더의 경로를 넣습니다.
이미지는 복사하지 않고 제자리에 둡니다.

시험해 볼 데이터가 필요하면 COCO128(COCO에서 뽑은 128장, 약 7 MB)을 받아 쓰면 됩니다. `.badamoa/data/coco128` 에 풀립니다.

```bash
uv run python scripts/fetch_coco128.py
```

이 데이터가 있으면 `uv run pytest` 가 실데이터 확인 테스트(`tests/test_real_data.py`)도 함께 실행합니다.

## 학습해 보기

웹 UI → **학습** → **새 학습** 에서 릴리스와 모델(예: yolo11n)을 고르면 됩니다. 릴리스 화면의 **이 릴리스로 학습** 버튼으로도 시작할 수 있습니다.

- 학습은 그 릴리스의 이미지가 있는 노드에서 실행됩니다.
- 노드가 처음 학습할 때 PyTorch와 Ultralytics를 별도 가상환경에 설치합니다(몇 분, 노드에 uv 필요). GPU 드라이버에 맞는 PyTorch 빌드를 자동으로 고르며, 노드의 환경 변수 `BADAMOA_TORCH_BACKEND`(예: `cpu`, `cu128`)로 지정할 수도 있습니다.
- 끝나면 가중치가 **모델** 화면에 버전으로 등록됩니다.

COCO128이 있으면 실제 YOLO 학습(1 epoch, CPU)까지 확인하는 테스트를 돌릴 수 있습니다.

```bash
BADAMOA_TEST_YOLO=1 uv run pytest tests/test_real_data.py -k yolo
```

## 개발

```bash
uv run pytest
```

```bash
uv run ruff check .
```

웹 UI를 고칠 때는 Hub를 띄운 채로 `npm --prefix web run dev` 를 실행하면 `/api` 요청이 Hub로 프록시됩니다.

```
src/badamoa/core   공용 스키마 · storage · 시스템 정보
src/badamoa/hub    Hub: REST · 노드 프로토콜 · MCP · 웹 UI 호스팅
src/badamoa/node   Node Agent · 데이터 가져오기 · 학습용 변환 · 플러그인 환경
src/badamoa/plugins  플러그인 환경에서 도는 학습 스크립트
web/               웹 UI (React + Vite)
tests/             단위 · API · E2E
```
