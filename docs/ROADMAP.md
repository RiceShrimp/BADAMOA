# BADAMOA 로드맵

각 Phase는 **설치해서 쓸 수 있는 상태**로 끝낸다. 설계 배경은 [ARCHITECTURE.md](ARCHITECTURE.md).

순서의 기준은 두 가지다. **가장 위험한 기술을 먼저 검증**하고(Windows·Linux 노드에서 torch 플러그인 환경이 깔끔하게 서는가),
**쓸모 있는 결과를 빨리** 낸다(라벨된 데이터로 YOLO를 학습해 모델을 등록하는 것이 첫 목표).

| Phase | 내용 | 완료 기준 |
|---|---|---|
| **0. 골격** ✅ | 단일 패키지, Hub(계정·노드·Job·로그), Node Agent, MCP 서버, 웹 UI, 노드 설치 스크립트 | 아래 체크리스트 |
| **1. Develop: 첫 학습 E2E** | 플러그인 시스템(manifest·노드별 venv·GPU 스택), 저장 위치, 라벨된 데이터셋 import(COCO/YOLO) → DatasetVersion, Ultralytics YOLO det Trainer, 자체 실험 추적(지표·곡선·아티팩트), 평가(mAP), 모델 레지스트리(alias) | 데이터셋을 가져와 Windows·Ubuntu 노드 각각에서 YOLO 학습 → 학습 곡선 → 모델 등록 → 두 모델 비교, 계보 역추적. 같은 흐름을 MCP로도 수행 |
| **2. Build: 데이터 인텔리전스** | Lance 샘플 저장소(100만 장 벤치마크), 데이터 그리드, Service 터널, SigLIP2 임베딩·텍스트/이미지 검색, 씬 태깅, 큐레이션(near-dup·uniqueness), 온톨로지, Data QA, 그룹 split·누수 검사 | "night truck" 검색 → 뷰 저장 → 스냅샷 → 학습. near-dup 제거 전후 수치 확인 |
| **3. Annotate: 오토라벨** | Labeler 플러그인(OVD → SAM → VLM 검증 → SigLIP2 cls), 신뢰도 라우팅, 내장 검수 에디터, 출처 기록, 오토라벨 정밀도 리포트, CVAT/Label Studio 내보내기·가져오기 | 무라벨 1천 장 → 오토라벨 → 검수 → DatasetVersion → 학습. 정답 셋 대비 오토라벨 mAP 측정 |
| **4. Develop: 확장·Export·추론** | Trainer 추가(RT-DETR/D-FINE, Seg, Cls, Custom 템플릿), 증류·semi-supervised 모드, golden set·슬라이스 평가·실패 케이스 뷰, ONNX/TensorRT export, Model Contract, 추론 서비스(REST)·샘플 테스트·RTSP 스트리밍, Jetson 노드, Python SDK | x86 GPU에서 TensorRT 엔진의 FPS·정확도 리포트(Jetson 확보 시 Jetson도), 추론 API 호출, 실시간 스트림 표시 |
| **5. Automate: 플라이휠** | 파이프라인 템플릿·트리거, 승격 게이트 + 승인, 추론 노드 배포·롤백, 현장 샘플 피드백, 드리프트 감시 | 새 데이터를 넣기만 하면 재학습 → 게이트 → 승인 대기까지 자동으로 진행 |
| **6. MOA: 에이전트** | (MCP Tools·Resources·Prompts는 매 Phase 기능과 함께 추가) Claude Code 플러그인 생성, 승인 요청 게이트, 내장 런타임(로컬 LLM·Claude), 서브에이전트 + subagent-as-tool, 채팅 패널 | Claude Code에서, 그리고 로컬 모델만으로 "검색 → 오토라벨 → 학습 → 비교"를 수행하고 승격에서 승인 요청으로 멈춤 |
| **7. Build: 데이터 생성** | Generator 플러그인(Cosmos Transfer IAA — 로컬 또는 원격 엔드포인트, 희귀 클래스 합성), 생성 검증, A/B | 야간·우천 슬라이스의 성능 향상이 수치로 확인됨 |
| **8. 제품화** | Hub 설치 스크립트·네이티브 설치기(.deb/EXE), 오프라인 번들, 자동 업데이트, DB 마이그레이션, 백업·복원, TLS, 진단 번들 | 폐쇄망에서 설치 성공, 버전 업그레이드 후 데이터·모델 보존 |

데이터 생성이 뒤에 있는 이유: 효과를 재려면 학습·평가 루프가 먼저 있어야 하고 GPU 비용이 가장 크다.

## Phase 0 체크리스트

- [x] 단일 패키지 `badamoa` (`[hub]` extra) — 노드 설치에는 FastAPI·torch가 들어가지 않음
- [x] Hub: SQLite, 첫 관리자 설정, 로그인·API 토큰, 역할(admin/member), 감사 로그
- [x] 노드: join token(1회용·만료), 등록, heartbeat·지표, 역할 변경·비활성화·삭제
- [x] Job: 큐, long-poll 배정(대상 노드·역할 조건), 로그 스트리밍, 취소, 끊긴 노드의 Job 정리
- [x] Node Agent: `join` / `run`, Job 서브프로세스 실행, All-in-one(Hub가 곧 노드)
- [x] MCP `/mcp`: Bearer 토큰, Tools 9개 · Resources 3개 · Prompts 2개
- [x] 웹 UI: 초기 설정, 로그인, 노드(추가 명령·설정), 작업(실행·로그·취소), 에이전트 연결
- [x] Hub의 wheel 인덱스에서 노드 설치 (`uv tool install --index <hub>/install/simple/`)
- [x] 테스트 19개(Hub API · 실제 Hub+Node Agent · MCP JSON-RPC), CI(Ubuntu·Windows)
- [ ] 노드 설치 스크립트(`node.sh` / `node.ps1`)와 부팅 서비스 등록을 실제 원격 머신에서 검증 — 관리자 권한이 필요해 아직 실행해 보지 못함
- [ ] Hub 설치 스크립트 (지금은 `uv run badamoa hub run`)
- [ ] Claude Code에서 실제로 `claude mcp add` 후 사용 확인 (JSON-RPC 수준에서는 검증됨)

## "필요해지면" 목록

지금은 만들지 않기로 한 것들. 요구가 실제로 생기면 그때 넣는다.

- Postgres 모드, OIDC/LDAP 로그인
- Triton 저장소 export, gRPC 추론 API, C++ SDK
- 노드 간 분산 학습, Optuna 분산 HPO (단일 노드 멀티 GPU가 먼저)
- MLflow / W&B 미러 플러그인
- shadow·canary 배포 (우선은 기기를 골라 배포하고 지표를 비교, 롤백)
- 트레이 앱, stdio MCP 브리지

## 진행하면서 확인할 것

- 데이터 규모(이미지 수·영상 시간), GPU 서버 사양·OS, NAS/S3 유무 → 저장 위치 기본값
- Jetson 기종·JetPack 버전
- 내장 에이전트의 기본 로컬 모델 (Phase 6에서 툴 호출 성능으로 선정)
- 사내 SSO 유무, 기존 데이터·라벨 포맷
