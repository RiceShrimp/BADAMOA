# BADAMOA 모델 템플릿

이 폴더가 템플릿 하나입니다. `badamoa.toml`이 선언이고, 나머지는 그 선언이 가리키는 코드입니다.

1. `train.py`의 표시한 부분을 자신의 학습 코드로 바꿉니다.
2. 사용자가 고를 수 있는 값(하이퍼파라미터)을 `train.py`의 `bm.params(...)`에 선언합니다. 학습 대화상자의 입력 칸이 되고, `advanced=True`인 것은 "세부 설정"에 접혀 들어갑니다.
3. 진행 상황은 `bm.log(...)`, `bm.progress(...)`로, 결과는 `bm.log_model(...)`, `bm.finish(...)`로 알립니다. (`badamoa_sdk.py` 맨 위에 설명이 있습니다.)
4. 필요한 패키지를 `badamoa.toml`의 `[environment] requirements`에 적거나, Hub에 등록한 환경 이름을 `default`에 적습니다.
5. 확인하고 올립니다.

```bash
badamoa template check .
```

```bash
badamoa template push . --hub http://<hub>:8470 --token <API 토큰>
```

코드를 바꿔 다시 올릴 때는 `version`을 올립니다. 가중치나 데이터는 이 폴더에 넣지 않습니다.

`badamoa_sdk.py`는 BADAMOA가 실행할 때 자동으로 제공하므로 지워도 됩니다. 폴더에 두면 BADAMOA 밖에서도 `python train.py`로 돌려 볼 수 있습니다.
