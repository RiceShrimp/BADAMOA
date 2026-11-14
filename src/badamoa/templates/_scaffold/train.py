"""학습 진입점. 이 뼈대는 그대로 실행되지만 실제로 학습하지는 않습니다 — 표시한 곳을 자신의 코드로 바꾸세요.

`badamoa_sdk` 사용법은 그 파일의 맨 위 설명에 있습니다.
"""

from __future__ import annotations

import badamoa_sdk as bm

# 설정 선언: 학습 대화상자의 입력 칸이 됩니다. advanced=True 는 "세부 설정"에, group 은 그 안의 묶음 제목입니다.
# 값은 직접 적어야 합니다 (BADAMOA는 이 파일을 실행하지 않고 읽어서 입력 칸을 만듭니다).
hp = bm.params(
    model=bm.choice(["small", "large"], label="모델"),
    epochs=bm.integer(10, min=1, label="epoch"),
    lr=bm.number(0.001, min=0, label="학습률", group="최적화", advanced=True),
    weight_decay=bm.number(0.0005, min=0, label="weight decay", group="최적화", advanced=True),
    batch=bm.integer(16, min=1, label="배치", group="데이터", advanced=True),
    augment=bm.boolean(True, label="증강 사용", group="데이터", advanced=True),
)


def main() -> None:
    run = bm.init()  # run.data, run.data_dir, run.out, run.cache, run.device, run.classes
    print(f"모델 {hp.model} · 학습률 {hp.lr} · 클래스 {len(run.classes)}개 · 데이터 {run.data}")

    # ---- 여기부터 자신의 학습 코드로 바꿉니다 -------------------------------------------
    images = len(list((run.data_dir / "images" / "train").glob("*")))
    score = 0.0
    for epoch in range(1, hp.epochs + 1):
        score = 0.5 * epoch / hp.epochs  # 한 epoch 학습하고 검증한 결과라고 가정
        bm.log({"train_loss": 1.0 / epoch, "map50_95": score}, step=epoch)
        bm.progress(epoch, hp.epochs)
        print(f"epoch {epoch}/{hp.epochs} · 학습 이미지 {images}장")
    weights = run.out / "best.pt"
    weights.write_bytes(b"replace this with real weights")
    # ---- 여기까지 -------------------------------------------------------------------------

    bm.log_model(weights)
    bm.finish({"map50_95": score})


if __name__ == "__main__":
    main()
