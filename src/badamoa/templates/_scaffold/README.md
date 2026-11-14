# BADAMOA 모델 템플릿

이 폴더가 템플릿 하나입니다. `badamoa.toml`이 선언이고, 나머지는 그 선언이 가리키는 코드입니다.

1. `train.py`의 표시한 부분을 자신의 학습 코드로 바꿉니다. 설정은 `cfg`로 받고, 진행과 결과는 `badamoa_report`로 알립니다.
2. `badamoa.toml`의 `[params]`에 사용자가 고를 수 있는 값을 적습니다. 학습 대화상자의 입력 항목이 됩니다.
3. 필요한 패키지를 `[environment] requirements`에 적거나, Hub에 등록한 환경 이름을 `default`에 적습니다.
4. 확인하고 올립니다.

```bash
badamoa template check .
```

```bash
badamoa template push . --hub http://<hub>:8470 --token <API 토큰>
```

코드를 바꿔 다시 올릴 때는 `version`을 올립니다. 가중치나 데이터는 이 폴더에 넣지 않습니다.
