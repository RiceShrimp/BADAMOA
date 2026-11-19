# BADAMOA 모델 템플릿

이 폴더가 템플릿 하나입니다. `badamoa.toml`이 선언이고, 나머지는 그 선언이 가리키는 코드입니다.

1. `train.py`의 표시한 부분을 자신의 학습 코드로 바꿉니다.
2. 사용자가 고를 수 있는 값(하이퍼파라미터)을 `train.py`의 `bm.params(...)`에 선언합니다. 학습 대화상자의 입력 칸이 되고, `advanced=True`인 것은 "세부 설정"에 접혀 들어갑니다.
3. 진행 상황은 `bm.log(...)`, `bm.progress(...)`로, 결과는 `bm.log_model(...)`, `bm.finish(...)`로 알립니다. (`badamoa_sdk.py` 맨 위에 설명이 있습니다.)
4. 필요한 패키지를 `badamoa.toml`의 `[environment] requirements`에 적거나, Hub에 등록한 환경 이름을 `default`에 적습니다.
5. 이 PC에서 서버와 같은 시험 실행을 해 보고(`--run`), 올립니다(`--trial`이면 올린 뒤 서버의 시험 실행까지 시작하고 결과를 기다림).

```bash
badamoa template check . --run
```

```bash
badamoa template push . --trial --hub http://<hub>:8470 --token <API 토큰>
```

PyPI에 없는 사내 패키지는 wheel(.whl)로 만들어 Hub의 설정 → 환경 → "사내 패키지"에 올리고 requirements에 이름을 적습니다. 서버가 환경을 만들 때 Hub에서 받아 설치합니다.

올린 버전은 "시험 전"입니다. 관리자가 Hub의 설정 → 모델 종류에서 "시험 실행"하면 학습 서버 한 대가 작은 합성 데이터로 짧게 돌려 보고(학습 → 테스트 · 평가 · 내보내기), 통과한 것을 "허용"하면 모두의 학습 화면에 나옵니다. 실패하면 실패한 단계와 이유가 보입니다. 코드는 `run.check`(True)를 보고 몇 step만에 끝내도 됩니다. 시험 실행의 설정은 `badamoa.toml`의 `[check]`로 바꿉니다.

코드를 바꿔 다시 올릴 때는 `version`을 올립니다. 새 버전이 허용되기 전에는 다른 사람은 그 전 버전을 그대로 씁니다. 가중치나 데이터는 이 폴더에 넣지 않습니다.

`badamoa_sdk.py`는 BADAMOA가 실행할 때 자동으로 제공하므로 지워도 됩니다. 폴더에 두면 BADAMOA 밖에서도 `python train.py`로 돌려 볼 수 있습니다.
