"""A trainer that trains nothing: fake curves and a small weights file.

It exercises everything around training (dataset export, metric reporting,
weights upload, model registration) without torch — used by the tests and to
check a new node. Standard library only; see ``badamoa.templates`` for the contract.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path


def emit(event: str, **data) -> None:
    print("@@badamoa " + json.dumps({"event": event, **data}), flush=True)


def main(argv: list[str]) -> int:
    cfg = json.loads(Path(argv[argv.index("--config") + 1]).read_text(encoding="utf-8"))
    params = cfg["params"]
    if params.get("fail"):
        print("fail 설정으로 요청된 의도적 실패", file=sys.stderr)
        return 1
    epochs = int(params.get("epochs", 5))
    pause = float(params.get("sleep", 0.02))
    ceiling = float(params.get("ceiling", 0.6))  # where mAP50-95 levels off
    emit("info", trainer="dummy", device=cfg["device"])
    print(f"가짜 학습 {epochs} epoch · 데이터 {cfg['data']}")
    if cfg.get("init_weights"):
        print(f"이어서 학습: {Path(cfg['init_weights']).read_text(encoding='utf-8')[:60]}")
    metrics: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        progress = 1 - math.exp(-3 * epoch / epochs)
        metrics = {
            "map50_95": round(ceiling * progress, 4),
            "map50": round(min(0.99, ceiling * 1.4) * progress, 4),
            "precision": round(0.9 * progress, 4),
            "recall": round(0.8 * progress, 4),
            "train_box_loss": round(1.6 * (1 - progress) + 0.4, 4),
            "val_box_loss": round(1.7 * (1 - progress) + 0.5, 4),
        }
        emit("epoch", epoch=epoch, epochs=epochs, metrics=metrics)
        print(f"epoch {epoch}/{epochs} mAP50-95 {metrics['map50_95']}")
        time.sleep(pause)

    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    weights = out / "best.pt"
    content = {"dummy": True, "model": params.get("model"), "classes": cfg["classes"]}
    weights.write_text(json.dumps(content), encoding="utf-8")
    classes = [
        {"id": i, "name": name, "precision": 0.9, "recall": 0.8, "map50": metrics["map50"], "map50_95": metrics["map50_95"]}
        for i, name in enumerate(cfg["classes"])
    ]
    emit("result", weights=str(weights), metrics=metrics, classes=classes)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
