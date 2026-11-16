# BADAMOA 아키텍처

> **BADAMOA (바다모아)** — 데이터의 바다를 모아, 모델로.
>
> - **BADA** — **B**uild · **A**nnotate · **D**evelop · **A**utomate: 기능 파이프라인
> - **MOA** — **M**LOps **O**rchestration & **A**gents: 제어·에이전트 계층

```
BADAMOA
├─ BADA
│   ├─ Build     수집 · 검색 · 큐레이션 · 데이터 생성 · 데이터셋 버전
│   ├─ Annotate  OVD/VLM 오토라벨 · 검수
│   ├─ Develop   학습(일반 지도학습 + 증류) · 평가 · 레지스트리 · export
│   └─ Automate  자동 재학습 · 배포 · OTA
└─ MOA
    └─ Hub · Job 큐 · 원격 노드 · MCP 서버 · 서브에이전트
```

최초 구상은 [PLANS.md](PLANS.md), 단계별 일정은 [ROADMAP.md](ROADMAP.md),
화면 구성과 레퍼런스(Ultralytics Platform · Roboflow) 분석은 [REFERENCE_STUDY.md](REFERENCE_STUDY.md),
사내 화면을 바탕으로 한 데이터셋·노드 자원 화면 설계는 [SCREEN_SPECS.md](SCREEN_SPECS.md).

## 1. 목표와 전제

데이터 관리·학습·평가·배포를 아우르는 **설치형 비전 MLOps 플랫폼**.
일반 학습(YOLO 학습 등)이 기본이고, Foundation 모델(OVD · Grounding VLM · SAM · SigLIP2)을
Teacher로 써서 데이터를 찾고·만들고·자동 라벨링해 경량 Student로 증류하는 경로까지 지원한다.

| 항목 | 결정 |
|---|---|
| 태스크 | Detection, Segmentation, Classification (VLM 파인튜닝은 후순위) |
| 학습 | 일반 지도학습이 기본, Teacher→Student 증류는 선택 모드 |
| 형태 | 설치형 — Ubuntu + Windows. UI에서 원격 학습/데이터 서버를 붙였다 뗀다 |
| 사용 범위 | 사내용 |
| 배포 타겟 | Edge(Jetson/TensorRT, 기종 미정) + 서버 추론 API |
| 에이전트 | MCP 중심 — 로컬 LLM 내장 에이전트 + Claude Code 등 외부 에이전트 |

## 2. 설계 원칙

1. **Hub는 얇게.** Hub에는 torch도 ML 라이브러리도 없다. 모든 ML(임베딩·오토라벨·학습·평가·큐레이션)은 노드의 **플러그인 가상환경**에서 Job으로 돈다. Hub 설치가 가볍고, 모델마다 다른 의존성이 서로 충돌하지 않는다.
2. **배포 단위는 하나.** 파이썬 패키지 `badamoa` 하나가 CLI·Node Agent를 담고, `badamoa[hub]`가 Hub를 더한다. Hub와 노드의 버전이 어긋날 일이 없고, Hub가 노드에게 서빙할 wheel도 하나다.
3. **API-first.** 모든 기능은 Hub의 `services` 계층에 있고 REST(웹 UI)와 MCP(에이전트)가 같은 함수를 부른다. 에이전트 전용 뒷문은 없다.
4. **불변 산출물 + 계보.** Sample(content hash) → Annotation(출처) → DatasetVersion → Run → ModelVersion → Engine → Deployment 를 끝까지 역추적한다.
5. **무거운 것은 선택 사항.** FiftyOne·MLflow·CVAT·Triton 같은 외부 도구는 기본 설치에 넣지 않고 플러그인/내보내기로 붙인다 (3장).
6. **사람은 게이트에.** 오토라벨 중간 신뢰 샘플, 모델 승격, 프로덕션 배포는 사람이 승인한다.

## 3. 무엇을 직접 만들고 무엇을 붙이는가

"누구나 설치"가 기준이다. Docker·별도 DB 서버·별도 서비스 프로세스가 필요한 것은 기본 설치에서 뺐다.

| 영역 | 결정 | 뺀 것과 이유 |
|---|---|---|
| 분산 실행 | **Hub + Node Agent** (노드가 Hub로 outbound HTTP) | Ray — 멀티노드 클러스터가 Windows 미지원 |
| 워크플로 | **Job 큐** + 단계 목록형 파이프라인(Phase 5) | Prefect/범용 DAG 엔진 — 단계가 굵어서 불필요 |
| 제어 DB | **SQLite** (Postgres는 옵션) | 무설정 설치 |
| 카탈로그 데이터 | **SQLite** (0.4.0) — 항목·라벨 개정·릴리스. 박스는 개정마다 JSON 한 덩어리로 두고 필터용 값만 열로 둔다 | 별도 DB 서버. 박스를 한 줄씩 행으로 넣지 않는다 |
| 임베딩·벡터 검색 | **Lance** (Phase 2에서 도입) | Qdrant·MongoDB 서버. 임베딩이 필요해지기 전에는 의존성을 늘리지 않는다 |
| 실험 추적·레지스트리 | **자체** (지표 시계열 + 아티팩트 + alias) | MLflow 내장 — 무거운 의존성, 별도 서버·UI·인증, 레지스트리가 둘이 됨. 필요하면 **미러 플러그인**으로 MLflow/W&B에 복제 |
| 큐레이션 | 기본(near-dup·uniqueness·대표성)은 **임베딩으로 자체 계산**. FiftyOne Brain은 **플러그인 Job**(노드 venv에서 실행, 점수만 Hub에 기록) | FiftyOne을 Hub 의존성으로 두는 것 — MongoDB가 딸려오고 데이터 저장소가 둘이 됨 |
| 평가 | **자체** (pycocotools 계열 + 샘플 단위 TP/FP/FN) | 평가만을 위한 FiftyOne 의존 |
| 검수 | **내장 검수 에디터** — 승인/거절/클래스 변경/박스 조정/SAM 재프롬프트로 한정 | CVAT(Docker 전용). 처음부터 그리는 대량 수작업은 CVAT·Label Studio **내보내기/가져오기** |
| 저장소 | **저장 위치 4종** (4.3) | 자체 분산 스토리지, MinIO 기본 탑재 |
| 서빙 | **내장 추론 서비스** (ONNX Runtime / TensorRT), REST | Triton(사실상 Linux 전용)·gRPC — 필요해지면 exporter 플러그인 |
| Edge | **같은 Node Agent** (Jetson = arm64 노드) | 별도 Edge Agent 앱, 별도 Device 엔티티 |
| 인증·보안 | 내장 계정·역할, 노드/API **토큰**, TLS는 인증서 지정 또는 자체서명+지문 고정 | Keycloak, mTLS·사설 CA 운영 |
| 모니터링 | 노드가 보내는 지표 → **내장 화면**(GPU 자리표 · 타임라인, 확대 · 공유 십자선). 사내 Prometheus · Grafana에는 `/metrics`로 내보냄 | Prometheus/Grafana 스택을 함께 설치 |
| 프론트 | **React + Vite SPA** (Hub가 정적 서빙) | Next.js — 런타임에 Node.js 필요 |
| 설치 | **uv 기반 부트스트랩 스크립트** 하나로 Hub·노드 설치. `.deb`/EXE는 같은 스크립트 + 오프라인 wheelhouse를 감싼 래퍼 | Python 런타임 직접 번들, GPU 감지 로직 자작, Tauri 트레이 앱 |
| 에이전트 | **Hub = MCP 서버** + 내장 런타임(PydanticAI) | Claude 전용 SDK에 묶이는 구조, stdio 브리지 |

> 사용자 최초 구상에 있던 **FiftyOne Brain**은 버린 게 아니라 "코어 의존성"에서 "플러그인"으로 옮긴 것이다.
> MLflow·W&B·ClearML은 레퍼런스였으므로 기능만 가져오고 제품에는 넣지 않았다.

## 4. 구조

### 4.1 토폴로지

```
                 ┌──────────── BADAMOA Hub (1대, Ubuntu 또는 Windows) ───────────────┐
 Browser ──────▶ │ Web UI (SPA) · REST API · MCP Server (/mcp)                       │
 Claude Code,  ─▶│ services (비즈니스 로직) · Job 큐 · 노드 레지스트리 · 계정/역할     │
 기타 MCP 클라 ─▶│ SQLite(제어) · Lance(샘플·어노테이션·임베딩) · wheel 인덱스          │
                 └───────▲───────────────────▲───────────────────▲───────────────────┘
                         │ outbound HTTP(S): 등록 · heartbeat · Job long-poll · 로그
         ┌───────────────┴───┐   ┌───────────┴───────┐   ┌───────┴────────────────┐
         │ Node (compute)    │   │ Node (storage)    │   │ Node (inference)       │
         │ 학습·오토라벨 Job  │   │ 데이터가 있는 서버  │   │ 추론 서비스 · 스트리밍   │
         │ 플러그인 venv(uv)  │   │                   │   │ x86 GPU 또는 Jetson     │
         └───────────────────┘   └───────────────────┘   └────────────────────────┘
         한 머신이 여러 역할 가능 — 기본 설치는 All-in-one (Hub가 곧 노드)
```

노드는 항상 Hub로 **나가는** 연결만 쓴다. NAT·방화벽 뒤 서버나 현장의 Jetson도 포트를 열 필요가 없다.

### 4.2 Job과 Service

- **Job** (배치): 학습, 오토라벨, 임베딩, 평가, export. 큐 → 조건에 맞는 노드가 long-poll로 가져감 → 서브프로세스 실행 → 로그 스트리밍 → 결과. Phase 0에 구현됨.
- **배정 단위는 GPU다.** 서버를 여러 사람이 직접 쓰는 환경이므로 BADAMOA가 GPU를 독점한다고 가정하지 않는다. 작업은 "GPU n개 · VRAM 최소 x"를 요구하고, 다른 프로세스가 쓰고 있는 GPU는 바쁜 것으로 보아 배정하지 않는다. 노드는 배정받은 GPU만 보이게 해서 실행하고, 대기 중인 작업에는 이유를 표시한다 ([SCREEN_SPECS.md](SCREEN_SPECS.md) 3.3).
- **노드 자원 이력**: Node Agent가 CPU·메모리·GPU(사용률·메모리·온도·전력·점유자)·디스크·사용자별 사용량을 보내고, Hub가 1분 → 10분 → 1시간으로 줄여 30일 보관한다.
- **Service** (온라인): 텍스트 검색용 임베딩, 검수 화면의 SAM 클릭 보조, 샘플 테스트, RTSP 실시간 추론처럼 **즉시 응답**이 필요한 것. 노드가 모델을 띄워 두고, 노드↔Hub 사이 WebSocket 터널로 요청을 중계한다(브라우저가 노드에 직접 닿을 필요 없음). 첫 대화형 기능이 필요한 Phase 2에서 도입.
- **플러그인**: Trainer / Labeler / Generator / Exporter / Curator. 각자 `manifest`(지원 OS, 최소 VRAM, CUDA, 의존성 lock)와 독립 venv를 가진다. 스케줄러는 매니페스트에 맞는 노드에만 배정한다.
  - 구현(0.6.0): 학습 코드는 **모델 템플릿**이다 — 폴더 하나에 선언 파일 `badamoa.toml`(진입점 · 설정 항목 · 데이터 형식 · 기본 환경)과 코드. 내장 YOLO도 같은 형식(`badamoa/templates/yolo`)이고, 사내 코드는 `badamoa template push`로 Hub에 올리면 노드가 받아 해시로 캐시한다. **환경**은 자동 설치(`<노드 홈>/envs/<해시>`에 uv로, 같은 요구사항이면 공유) 또는 서버에 이미 있는 Python이다(`node/envs.py`). 코드는 BADAMOA를 import하지 않고, 노드 쪽 래퍼(`node/trainer.py`)와 표준 출력으로만 주고받는다(`@@badamoa {JSON}` 한 줄이 이벤트). 형식은 `core/templates.py`, 등록은 `hub/registry.py`.
  - 다음 단계(계획): Docker 이미지 환경, 평가·추론 진입점, 지표 외 기록(파일·이미지·시스템 지표) — [EXECUTION_PLAN.md](EXECUTION_PLAN.md).
  - 실행 방식 두 가지: **로컬 venv**(노드에서 직접) / **원격 엔드포인트**(이미 떠 있는 추론 서버·NIM을 호출). Cosmos Transfer나 대형 VLM은 12GB급 GPU에서 못 돌리므로 후자가 필요하다.

### 4.3 저장 위치

분산 파일시스템을 만들지 않는다. 데이터가 **어디에 있는지**만 등록하고 스케줄러가 그에 맞춘다.

| 유형 | 설명 | 노드가 읽는 방법 |
|---|---|---|
| Hub 로컬 | Hub 머신의 디렉터리 (기본값) | All-in-one은 직접, 원격 노드는 Hub HTTP로 받아 캐시 |
| 공유 마운트 | NAS(SMB/NFS). 노드별 마운트 경로 매핑 | 직접 — **멀티노드 권장 구성** |
| S3 호환 | S3, 사내 MinIO 등 | 직접 |
| 노드 로컬 폴더 | 특정 서버에 이미 있는 데이터를 복사 없이 등록 | 그 데이터를 쓰는 Job은 **그 노드에 배정** (필요하면 "노드로 동기화" Job) |

이미지는 content hash로 식별하므로 캐시와 중복 제거가 단순하다. 구현은 `badamoa.core.storage`(fsspec).

> 0.5.0까지 구현된 것은 **노드 로컬 폴더** 하나다(등록한 노드에서만 학습). 공유 마운트 · 서버 간 동기화 · S3를 등록과 학습 양쪽에 넣는 계획은 [EXECUTION_PLAN.md](EXECUTION_PLAN.md) 4장.

### 4.4 데이터 모델

> 사내 Dataset Catalog의 개념(계열 → 릴리스, 라벨 공간, 라벨 개정, 라벨 상태)을 참고해 같은 구조로 설계한다. 그 도구를 대체하거나 데이터를 옮겨 오지는 않는다.
> 상세와 근거는 [SCREEN_SPECS.md](SCREEN_SPECS.md) 2장. 아래에서 **DatasetVersion = 릴리스**, **Ontology = 라벨 공간**이다.

- **항목은 이미지 또는 시퀀스**(영상·스켈레톤)이고 sha256으로 식별한다. 사이트·카메라는 정해진 메타 필드다.
- **라벨은 덮어쓰지 않고 개정(Label revision)으로 쌓인다.** 개정은 라벨 공간, 출처, 날짜, **빠짐없이 라벨한 클래스 목록**, 라벨 상태(라벨 있음 / 배경으로 확인 / 미라벨)를 가진다. 릴리스는 항목마다 (split, 가리키는 개정)을 기록하므로, 한 이미지를 여러 릴리스가 서로 다른 라벨로 공유한다.
- **릴리스는 base 릴리스에서 파생**되고, 항목 → 포함된 릴리스 역참조를 지원한다.
- **제어 엔티티 (SQLite)**: User, ApiToken, Node, Job, JobLog, AuditLog, Project, Ontology, DatasetVersion, Run, ModelVersion, Engine, Deployment, Plugin, StorageLocation, LicenseRecord, NodeMetric(시계열 요약)
- **카탈로그 (SQLite, 0.4.0에 구현)**: `projects, label_spaces, storage_locations, batches, items(sha256, 크기, 사이트, 카메라, 위치), label_revisions(rev, 상태, boxes JSON, 빠짐없이 라벨한 클래스), revision_classes(클래스 필터용), datasets, releases(통계), release_items(split, 가리키는 개정)`. 코드: `hub/catalog_models.py`, `hub/catalog.py`
- **가져오기는 노드가 한다.** 데이터가 있는 노드가 `data.import` 작업으로 폴더를 읽어 해시·크기·라벨·썸네일만 Hub에 묶음 단위로 올린다(`node/importer.py`). 이미지 파일은 옮기지 않는다.
- **이미지 URL은 서명된다.** `<img>`는 인증 헤더를 못 보내므로 썸네일·원본 URL에 만료 시각과 서명을 붙인다. 원본은 Hub 머신에서 읽을 수 있는 경로일 때만 제공한다.
- **학습 (SQLite, 0.5.0에 구현)**: `runs(릴리스, 설정, 진행 epoch, 학습에 쓴 데이터 요약, 최종 지표)`, `run_metrics(epoch별 지표)`, `model_versions(이름, 버전, alias, 가중치 해시, 지표, 클래스)`. 실행의 상태는 그 실행을 돌리는 Job의 상태다. 코드: `hub/train_models.py`, `hub/training.py`
- **학습 데이터는 릴리스에서 만든다.** 노드가 Hub에서 릴리스의 목록(항목 · split · 가리키는 라벨 개정)을 받아 YOLO 형식 폴더를 만든다(`node/exporter.py`). 이미지는 링크하고 라벨은 릴리스가 기록한 개정으로 쓰므로, 이미지 옆의 라벨 파일이 나중에 바뀌어도 같은 릴리스는 같은 학습 데이터가 된다.
- **분류 라벨 (0.8.0)**: 같은 표를 쓴다. 박스는 `[클래스, cx, cy, w, h]`, 분류 라벨은 위치가 없는 `[클래스]` 하나다. 그래서 클래스 필터·통계·릴리스 비교·라벨 개정이 그대로 동작한다. 가져오기는 클래스별 폴더를 읽고(`imagefolder`), 학습용 변환은 `<split>/<클래스>/` 폴더를 만든다.
- **분할 라벨 (0.11.0)**: 박스 뒤에 윤곽 좌표를 이어 붙인다 — `[클래스, cx, cy, w, h, x1, y1, x2, y2, …]`. 박스만 읽는 코드는 앞의 다섯 값만 쓰면 되므로 탐지용 변환과 통계가 그대로 동작한다.
- **임베딩 (Lance, Phase 2)**: `embeddings(item_id, vector)` — 텍스트·이미지 검색과 큐레이션용
- **Folder → Project.** 폴더는 프로젝트를 묶는 선택적 1단 분류다. **Project** = 하나의 태스크 범위로, 온톨로지를 공유하는 데이터셋 여러 개·Run·모델·승인 요청을 묶는다.
- **Batch** = 가져오기 한 번(이름 · 태그 · 출처 · 시각). 모든 샘플은 자신이 들어온 배치를 기억한다 — 가져오기 취소, 배치 단위 오토라벨·검수 배정, 출처 필터에 쓴다.
- **split은 데이터셋 안에서 샘플의 속성**이고, 버전을 찍을 때 고정된다. **전처리·증강은 버전에 굽지 않고 학습 설정(Run)에 둔다** — 이미지 복제가 없고, 같은 버전으로 설정만 바꿔 실험할 수 있다.
- **버전끼리 비교**할 수 있다: 추가/제거된 샘플, 바뀐 라벨, 클래스 분포 변화.
- **샘플 라이브러리 → 데이터셋 → DatasetVersion** 3층. 이미지는 프로젝트의 샘플 라이브러리에 한 번만 들어가고(content hash), 데이터셋은 그 위의 작업용 묶음, 버전은 그 시점의 불변 스냅샷이다. 같은 이미지를 여러 데이터셋이 복사 없이 쓴다.
- **DatasetVersion** = 불변 manifest(샘플 hash 목록 + 라벨 스냅샷 + split + 온톨로지 버전 + 생성 Job). split은 카메라/장면/시간 **그룹 단위**로 나누고 train↔test near-dup 누수를 검사한다.
- **삭제는 휴지통으로.** 데이터셋·모델은 soft delete 후 일정 기간 복구할 수 있다. 실제 파일 정리는 어떤 버전도 참조하지 않는 blob만 대상으로 한다.
- **Run은 재현 가능해야 한다.** 플러그인 버전·설정·DatasetVersion을 기록하고, 화면에서 "같은 설정으로 다시 실행"과 CLI·MCP 호출 예시를 제공한다.
- **Annotation.source** = `human | autolabel:<model>@<ver> | synthetic:<generator>`
- **Model Contract** = 입력 크기·전처리·클래스·임계값·온톨로지 버전. SDK·추론 서비스·Edge가 같은 계약을 쓴다 (학습/배포 전처리 불일치 방지).

> 카탈로그를 SQLite에 둔 것은 150장 규모에서만 확인했다. 수십만 장에서 가져오기 속도와 필터 응답이 충분한지는 실제 데이터로 측정해야 하고, 부족하면 항목·개정을 열 기반 저장으로 옮긴다.

## 5. 기능

### Build — 데이터
- **Import**: COCO·YOLO·VOC, Roboflow Universe, HF Datasets, 영상 프레임 추출(장면 전환 기반), 폴더 등록
- **Search**: SigLIP2 임베딩 → Lance 벡터 검색. 텍스트/이미지 쿼리 + 메타 필터, 결과를 저장 뷰로
- **Scene tagging**: SigLIP2 zero-shot 주/야·날씨·실내외 → 평가 슬라이스·균형 샘플링
- **Curation**: near-dup 제거, uniqueness·대표성으로 라벨링 우선순위(자체) / mistakenness·hardness(FiftyOne Brain 플러그인)
- **Ontology**: 클래스 체계 + 클래스별 프롬프트·네거티브 프롬프트·임계값, 버전 관리, 클래스 매핑
- **Data QA**: 깨진 이미지, bbox 오류, 극소 박스, 클래스 불균형
- **Data generation**: Generator 플러그인 — Cosmos Transfer 계열 IAA(날씨·조명·시간대, 구조 보존 → 라벨 재사용), 희귀 클래스 합성. synthetic 태그, 테스트셋은 real-only, 효과는 A/B로 측정

### Annotate — 오토라벨
- Labeler 플러그인: OVD(Grounding DINO, OWLv2, Florence-2) / Seg(SAM 2·3) / Grounding VLM(Qwen-VL 계열) / Cls(SigLIP2 zero-shot). 후보는 착수 시점에 최신·라이선스·Windows 지원을 다시 확인한다.
- 흐름: 다중 프롬프트 OVD → (박스→SAM 마스크) → 앙상블 → VLM이 crop 재검증 → 신뢰도 라우팅(자동 승인 / 검수 / 폐기)
- 검수 결과로 클래스별 오토라벨 정밀도를 재고 임계값을 조정한다.

### Develop — 학습·평가·모델
- **학습 모드**: ① 일반 지도학습(fine-tune / scratch) ② 이어 학습 ③ 오토라벨 데이터로 학습 ④ 지식 증류 ⑤ semi-supervised (④⑤는 선택)
- **Trainer 플러그인**: Ultralytics YOLO(det/seg/cls), RT-DETRv2, D-FINE/DEIM, RF-DETR, timm, SigLIP2 linear probe + **Custom Trainer 템플릿**(사내 모델·외부 repo를 감싸 등록)
- DatasetVersion → YOLO/COCO/ImageFolder 자동 변환, 실시간 학습 곡선, 중단·재개
- **평가**: mAP/크기별/클래스별, mask mAP/mIoU, top-1/F1/ECE. 샘플 단위 TP/FP/FN → 실패 케이스 뷰 → 데이터 루프. Golden set + 슬라이스 평가, 승격 게이트, 클래스별 운영 임계값
- **레지스트리**: 버전 + alias(champion / challenger) + 계보 + Model Contract + 모델 카드
- **Export**: PyTorch → ONNX → TensorRT(FP16/INT8). 엔진은 대상 기기에서 빌드하고 그 기기에서 정확도·FPS를 잰다
- **Workspace**: 샘플 테스트·모델 비교, RTSP/웹캠 실시간 추론, YAML 파이프라인(detector → tracker → classifier)

### Automate — 자동화·제품 연동
- 파이프라인 템플릿: import → 오토라벨 → 검수 → 스냅샷 → 학습 → 평가 → 게이트 → (승인) → export → 배포
- 트리거: 검수 라벨 N개, 드리프트 경보, 일정, 사용자/에이전트 요청
- 배포: 추론 노드(서버·Jetson)에 모델 버전 지정 → 노드가 받아 교체, 롤백. Python SDK(모델 계약 기반 전/후처리)
- 피드백: 현장의 저신뢰·불일치 샘플 업로드 → import. 신뢰도·클래스 분포 드리프트 감시

### MOA — 에이전트 (MCP 중심)
**Hub가 곧 MCP 서버다.** 내장 에이전트도 외부 에이전트도 같은 `/mcp`의 클라이언트다.

- **Tools**: 세밀한 툴(`list_nodes, run_job, wait_for_job, get_job_logs …` — 기능이 생길 때마다 함께 추가) + 굵은 워크플로 툴(`run_pipeline`, 툴 호출이 약한 로컬 모델용). `readOnlyHint`/`destructiveHint`를 정확히 단다.
- **Resources**: `badamoa://nodes`, `badamoa://jobs/{id}`, `badamoa://jobs/{id}/logs` … (데이터셋·모델 카드·지표로 확장)
- **Prompts**: `diagnose-failed-job`, `node-health-report` … (Claude Code에서는 슬래시 커맨드)
- **Subagents**: `agents/*.md` 한 곳에서 정의(이름, 설명, 허용 툴, 선호 모델). ① 내장 런타임 ② Claude Code 플러그인의 `agents/` ③ **subagent-as-tool**(`delegate_to_curator(task)` — Hub에서 로컬 모델로 실행하고 결과만 반환)에서 같이 쓴다.
- **내장 런타임**: PydanticAI + 프로바이더 추상화 — 로컬(Ollama, llama.cpp, vLLM 등 OpenAI 호환) / Claude. 폐쇄망은 로컬만.
- **연결**: "에이전트 연결" 화면에서 토큰 발급 + 복붙 명령
  `claude mcp add --transport http badamoa http://<hub>:8470/mcp --header "Authorization: Bearer <token>"`
- **권한**: 에이전트는 토큰 주인의 역할을 그대로 따른다. 조회 = 자유 / GPU 소모 = 확인 / 승격·배포·삭제 = Hub 승인 요청(사람이 UI에서 승인할 때까지 `pending`). 모든 호출은 감사 로그에 남는다.

### 화면
사이드바는 제품 구조 그대로다: 홈 → **Build**(샘플 라이브러리, 데이터셋) → **Annotate**(오토라벨, 검수) → **Develop**(학습, 모델) → **Automate**(배포, 파이프라인) → **MOA**(노드, 작업, 에이전트).
아직 없는 화면은 흐리게 두고 추가될 Phase를 표시한다.

- 구현됨(Phase 0): 홈(상태 요약 · 시작하기 · 최근 활동 · 최근 작업), 노드, 작업, 에이전트, 밝은/어두운 테마
- 모든 엔티티 상세 화면은 같은 틀을 쓴다: 머리말(이름 · 배지) → 요약 줄 → 탭
  - 데이터셋: 샘플 / 클래스 / 분포 / 품질 / 모델 / 버전
  - 모델: 개요(지표 카드) / 학습(지표 · 콘솔 · 시스템) / 테스트 / 내보내기 / 배포 / 계보
- 버전 표시(오른쪽 위)를 누르면 릴리즈 노트가 열린다. 내용은 저장소의 `CHANGELOG.md` 하나에서 나온다.
- 전체 구성과 근거는 [REFERENCE_STUDY.md](REFERENCE_STUDY.md) 4장

## 6. 설치·운영

- **Hub 설치**: 부트스트랩 스크립트가 uv를 받고 → `badamoa[hub]` 설치 → 부팅 시 자동 실행 등록(Linux: systemd, Windows: 작업 스케줄러) → 브라우저에서 관리자 계정 생성.
- **노드 추가**: UI "노드 추가" → OS별 한 줄 명령 + 1회용 join token → 원격 서버에서 실행. Hub가 자기 wheel을 `/install/simple/`로 서빙하므로 **노드는 Hub에만 닿으면 된다** (서드파티 의존성은 PyPI, 폐쇄망은 오프라인 번들).
- **GPU 스택**: 플러그인 venv를 만들 때 uv가 노드의 드라이버에 맞는 PyTorch 빌드를 고른다(착수 시 검증). 커스텀 CUDA op가 필요한 모델은 매니페스트에 `platforms: [linux]`.
- **운영**(Phase 8): 자동 업데이트, DB 마이그레이션, 백업·복원, 진단 번들, 쿼터.

## 7. 라이선스

플랫폼은 사내용이라 번들 부담은 낮다. **제품에 탑재되는 모델·런타임**은 확인이 필요하다.
Ultralytics YOLO(AGPL-3.0)는 학습을 정식 지원하되 제품용 export·배포 시 경고를 띄운다.
모델 가중치와 데이터셋(Roboflow Universe는 데이터셋마다 다름)의 라이선스는 등록 시 기록한다.

## 8. 레포 구조

```
src/badamoa/
  core/     스키마 · Job 유형 · storage(fsspec) · 시스템 정보 · 경로
  hub/      FastAPI 앱 · services · REST(api) · 노드 프로토콜(agent_api) · MCP(mcp_server)
            설치 스크립트/wheel 인덱스(install) · static/ (빌드된 웹 UI)
  node/     Node Agent · 내장 Job · 데이터 가져오기 · 학습용 변환 · 환경 · 부팅 서비스 등록
  templates/  내장 모델 템플릿(yolo, torchvision, classify, dummy) · SDK(_sdk/badamoa_sdk.py) · 새 템플릿의 뼈대(_scaffold)
  cli.py    `badamoa hub …` / `badamoa node …`
web/        React + Vite (빌드 결과는 src/badamoa/hub/static 으로)
plugins/    (Phase 3~) 저장소 밖에서 추가하는 플러그인 — labelers/ generators/ exporters/ curators/
agents/     (Phase 6) 서브에이전트 정의
tests/      단위 · Hub API · 실제 Hub+노드+MCP E2E
docs/
```
