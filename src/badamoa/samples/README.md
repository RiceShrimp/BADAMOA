# 처음 설치 때 넣는 샘플 데이터

Ultralytics의 COCO8 세 벌(COCO train2017의 처음 8장, train 4 · val 4): 검출(`coco8`) · 분할(`coco8-seg`) ·
키포인트(`coco8-pose`). 출처 https://github.com/ultralytics/assets/releases (v0.0.0), 라이선스는 각 폴더의 LICENSE.
`data.yaml`의 클래스 이름 · 키포인트 모양은 ultralytics `cfg/datasets/coco8*.yaml`에서 옮겼습니다.

Hub가 처음 관리자 계정을 만들 때 `hub/samples.py`가 이것을 Hub 폴더의 `samples/`로 복사해 "샘플" 프로젝트에
데이터셋으로 등록하고, 검수 예시로 일부러 틀린 라벨을 넣은 검출 버전을 하나 더 만듭니다.
