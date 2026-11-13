# 최초 구상

프로젝트를 시작할 때 적은 기능·기술·레퍼런스 메모를 그대로 보존한 문서입니다 (오타만 고침).
오른쪽 표는 각 항목이 BADAMOA의 어디로 갔는지입니다. 자세한 설계는 [ARCHITECTURE.md](ARCHITECTURE.md), 일정은 [ROADMAP.md](ROADMAP.md).

## 원문

```
Function
- Data
	- Auto Labeling
		- OVD & Grounding VLM
	- Data Generation
		- Cosmos Data Factory - IAA(Image Attribute Augmentation)
	- Data Search
		- SigLIP2 - embedding vector
	- Data Curation
		- FiftyOne Brain
	- Dataset Versioning
- Model
	- Training
		- Teacher-Student
	- Evaluation
	- Model Versioning
- Workspace
	- Inference
		- Sample testing
		- Real-time streaming inference
		- Pipeline
- Agent
	- This Function & Program Management
	- Searching
- Product Connecting
- Management Dashboard

Ref
- Roboflow Universe
- W&B
- ClearML
- MLflow
- FiftyOne
- Hugging Face
- Unsloth
```

## 어디로 갔나

| 최초 구상 | BADAMOA | Phase |
|---|---|---|
| Auto Labeling (OVD & Grounding VLM) | **Annotate** — Labeler 플러그인 + 신뢰도 라우팅 + 내장 검수 | 3 |
| Data Generation (Cosmos IAA) | **Build** — Generator 플러그인 (로컬 또는 원격 엔드포인트) | 7 |
| Data Search (SigLIP2) | **Build** — Lance 벡터 검색, 텍스트·이미지 쿼리 | 2 |
| Data Curation (FiftyOne Brain) | **Build** — 기본 큐레이션은 자체 계산, FiftyOne Brain은 플러그인 | 2 |
| Dataset Versioning | **Build** — 불변 manifest(DatasetVersion) | 1 |
| Training (Teacher-Student) | **Develop** — 일반 지도학습이 기본, 증류는 선택 모드 | 1, 4 |
| Evaluation | **Develop** — 지표 + 슬라이스 + 실패 케이스 뷰 + 승격 게이트 | 1, 4 |
| Model Versioning | **Develop** — 레지스트리(alias, 계보, Model Contract) | 1 |
| Workspace / Inference | **Develop** — 샘플 테스트, 실시간 스트리밍, 파이프라인 | 4 |
| Agent | **MOA** — Hub가 MCP 서버, 서브에이전트, 내장 런타임 | 0~, 6 |
| Product Connecting | **Automate** — 추론 노드(서버·Jetson) 배포, SDK, 피드백 | 4, 5 |
| Management Dashboard | 웹 UI | 0~ |

## 구상에 없었지만 추가한 것

- **설치·노드 관리**: 한 줄 명령으로 원격 학습/데이터 서버 추가, All-in-one 설치
- **온톨로지**: 클래스 체계와 클래스별 프롬프트·임계값을 버전으로 관리
- **출처(provenance)와 계보**: 라벨이 사람/오토라벨/합성 중 어디서 왔는지, 배포된 모델이 어떤 데이터에서 나왔는지
- **Data QA, 그룹 단위 split, 누수 검사**
- **Model Contract**: 학습과 배포의 전처리·클래스·임계값 불일치 방지
- **Golden set·슬라이스 평가·승격 게이트**, 대상 기기에서의 실측 평가
- **피드백 루프와 드리프트 감시**, 권한·감사 로그, 라이선스 기록

## 레퍼런스에서 가져온 것

| 레퍼런스 | 참고한 점 |
|---|---|
| Roboflow Universe | 데이터셋 import, 오토라벨 → 학습 → 배포로 이어지는 흐름 |
| W&B · ClearML · MLflow | 실험 추적, 모델 레지스트리, 원격 에이전트가 작업을 가져가는 구조 (제품에 넣지는 않고 기능만) |
| FiftyOne | 임베딩 기반 탐색·큐레이션, 샘플 단위 평가 |
| Hugging Face | 모델·데이터셋 허브 연동, 모델 카드 |
| Unsloth | Teacher VLM 파인튜닝 (후순위) |
