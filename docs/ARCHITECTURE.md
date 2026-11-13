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

최초 구상은 [PLANS.md](PLANS.md), 단계별 일정은 [ROADMAP.md](ROADMAP.md).

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
| 대량 데이터 | **Lance 테이블** (샘플 메타·어노테이션·임베딩·벡터 검색을 한 엔진에) | Qdrant·MongoDB 서버. SQLite에 수백만 어노테이션 행을 넣지 않는다 |
| 실험 추적·레지스트리 | **자체** (지표 시계열 + 아티팩트 + alias) | MLflow 내장 — 무거운 의존성, 별도 서버·UI·인증, 레지스트리가 둘이 됨. 필요하면 **미러 플러그인**으로 MLflow/W&B에 복제 |
| 큐레이션 | 기본(near-dup·uniqueness·대표성)은 **임베딩으로 자체 계산**. FiftyOne Brain은 **플러그인 Job**(노드 venv에서 실행, 점수만 Hub에 기록) | FiftyOne을 Hub 의존성으로 두는 것 — MongoDB가 딸려오고 데이터 저장소가 둘이 됨 |
| 평가 | **자체** (pycocotools 계열 + 샘플 단위 TP/FP/FN) | 평가만을 위한 FiftyOne 의존 |
| 검수 | **내장 검수 에디터** — 승인/거절/클래스 변경/박스 조정/SAM 재프롬프트로 한정 | CVAT(Docker 전용). 처음부터 그리는 대량 수작업은 CVAT·Label Studio **내보내기/가져오기** |
| 저장소 | **저장 위치 4종** (4.3) | 자체 분산 스토리지, MinIO 기본 탑재 |
| 서빙 | **내장 추론 서비스** (ONNX Runtime / TensorRT), REST | Triton(사실상 Linux 전용)·gRPC — 필요해지면 exporter 플러그인 |
| Edge | **같은 Node Agent** (Jetson = arm64 노드) | 별도 Edge Agent 앱, 별도 Device 엔티티 |
| 인증·보안 | 내장 계정·역할, 노드/API **토큰**, TLS는 인증서 지정 또는 자체서명+지문 고정 | Keycloak, mTLS·사설 CA 운영 |
| 모니터링 | 노드가 보내는 지표 → **내장 화면** | Prometheus/Grafana 스택 |
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
- **Service** (온라인): 텍스트 검색용 임베딩, 검수 화면의 SAM 클릭 보조, 샘플 테스트, RTSP 실시간 추론처럼 **즉시 응답**이 필요한 것. 노드가 모델을 띄워 두고, 노드↔Hub 사이 WebSocket 터널로 요청을 중계한다(브라우저가 노드에 직접 닿을 필요 없음). 첫 대화형 기능이 필요한 Phase 2에서 도입.
- **플러그인**: Trainer / Labeler / Generator / Exporter / Curator. 각자 `manifest`(지원 OS, 최소 VRAM, CUDA, 의존성 lock)와 독립 venv를 가진다. 스케줄러는 매니페스트에 맞는 노드에만 배정한다.
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

### 4.4 데이터 모델

- **제어 엔티티 (SQLite)**: User, ApiToken, Node, Job, JobLog, AuditLog, Project, Ontology, DatasetVersion, Run, ModelVersion, Engine, Deployment, Plugin, StorageLocation, LicenseRecord
- **대량 데이터 (Lance)**: `samples(hash, uri, 메타, 태그, embedding)`, `annotations(sample_hash, 형태, 클래스, source, confidence, review_status)`
- **DatasetVersion** = 불변 manifest(샘플 hash 목록 + 라벨 스냅샷 + split + 온톨로지 버전 + 생성 Job). split은 카메라/장면/시간 **그룹 단위**로 나누고 train↔test near-dup 누수를 검사한다.
- **Annotation.source** = `human | autolabel:<model>@<ver> | synthetic:<generator>`
- **Model Contract** = 입력 크기·전처리·클래스·임계값·온톨로지 버전. SDK·추론 서비스·Edge가 같은 계약을 쓴다 (학습/배포 전처리 불일치 방지).

> Lance가 100만 샘플 규모에서 동시 쓰기(오토라벨 Job)와 UI 조회를 감당하는지는 Phase 2 첫 주에 벤치마크로 확인한다.

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

### 관리 화면
노드(추가·역할·상태·GPU·디스크), 작업(큐·로그), 에이전트 연결 — Phase 0 구현.
이후 데이터셋, 검수, 실험 비교, 레지스트리, 배포 현황, 설정이 추가된다.

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
  node/     Node Agent · 내장 Job · 부팅 서비스 등록
  cli.py    `badamoa hub …` / `badamoa node …`
web/        React + Vite (빌드 결과는 src/badamoa/hub/static 으로)
plugins/    (Phase 1~) trainers/ labelers/ generators/ exporters/ curators/ — 각자 manifest + lock
agents/     (Phase 6) 서브에이전트 정의
tests/      단위 · Hub API · 실제 Hub+노드+MCP E2E
docs/
```
