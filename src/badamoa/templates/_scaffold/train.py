"""학습 진입점. 이 뼈대는 그대로 실행되지만 실제로 학습하지는 않습니다 — 표시한 곳을 자신의 코드로 바꾸세요.

BADAMOA가 정해 주는 것은 `run` 에 들어 있습니다.
    run.config    badamoa.toml의 [params]에서 사용자가 고른 값
    run.data      학습 데이터 (YOLO 형식이면 data.yaml 경로)
    run.data_dir  그 데이터가 있는 폴더 (images/train, labels/train, …)
    run.out       결과를 쓸 폴더
    run.cache     실행 사이에 유지되는 폴더 (내려받은 사전학습 가중치 등)
    run.device    "cpu" 또는 "0", "0,1"
    run.classes   클래스 이름 목록 (학습 순서)
"""

from __future__ import annotations

import badamoa_report as report


def main() -> None:
    run = report.init()
    epochs = int(run.config.get("epochs", 10))
    print(f"모델 {run.config.get('model')} · 클래스 {len(run.classes)}개 · 데이터 {run.data}")

    # ---- 여기부터 자신의 학습 코드로 바꿉니다 -------------------------------------------
    images = len(list((run.data_dir / "images" / "train").glob("*")))
    score = 0.0
    for epoch in range(1, epochs + 1):
        score = 0.5 * epoch / epochs  # 한 epoch 학습하고 검증한 결과라고 가정
        report.log({"train_loss": 1.0 / epoch, "map50_95": score}, step=epoch)
        report.progress(epoch, epochs)
        print(f"epoch {epoch}/{epochs} · 학습 이미지 {images}장")
    weights = run.out / "best.pt"
    weights.write_bytes(b"replace this with real weights")
    # ---- 여기까지 -------------------------------------------------------------------------

    report.log_model(weights)
    report.finish({"map50_95": score})


if __name__ == "__main__":
    main()
