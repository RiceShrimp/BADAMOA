"""Evaluate, export and test for the ``dummy`` template: the same paths a real template takes, without torch."""

from __future__ import annotations

import json
import shutil

import badamoa_sdk as bm

run = bm.init()
action = run.cfg["action"]
assert run.weights is not None and run.weights.is_file(), "the node hands over the model's weights"
if action == "eval":
    images = len([p for p in (run.data_dir / "images" / run.split).iterdir() if p.is_file()])
    print(f"평가: {run.split} split {images}장 · 학습 때 설정 {run.train_params}")
    per_class = [{"id": i, "name": name, "map50_95": 0.4, "map50": 0.6} for i, name in enumerate(run.classes)]
    # per-image results as the real templates record them: all images by name, the worst as pictures with boxes
    pictures = sorted(p for p in (run.data_dir / "images" / run.split).iterdir() if p.is_file())
    registered = json.loads((run.data_dir / "files.json").read_text(encoding="utf-8"))
    columns = ["image", "truth", "predicted", "found", "missed", "extra"]
    bm.log_table("per_image", columns, [[registered.get(p.stem, p.name).split("/")[-1], 1, 1, 1, 0, 0] for p in pictures])
    bm.log_table("failures", columns, [[bm.overlay(pictures[0], boxes=[[0, 0.9, 0.5, 0.5, 0.2, 0.2]]), 1, 1, 0, 1, 1]])
    bm.log_predictions(
        [
            {
                "image": p.stem,
                "labels": [[0, 0.5, 0.5, 0.2, 0.2]],
                "scores": [0.9],
                "found": 0 if n == 0 else 1,
                "missed": 2 if n == 0 else 0,
                "extra": 1 if n == 0 else 0,
            }
            for n, p in enumerate(pictures)
        ]
    )
    bm.finish({"map50_95": 0.4, "map50": 0.6, "images": images}, classes=per_class)
elif action == "export":
    exported = run.out / f"model.{run.config.get('format', 'onnx')}"
    exported.write_bytes(b"exported:" + run.weights.read_bytes())
    bm.log_artifact(exported)
    bm.finish(file=exported.name)
else:
    predictions = []
    for image in sorted(run.source.iterdir()):
        shutil.copy(image, run.out / image.name)  # a real template draws its boxes on this copy
        bm.log_artifact(run.out / image.name)
        predictions.append({"image": image.name, "boxes": [[0, float(run.config.get("conf", 0.25)), 0.5, 0.5, 0.2, 0.2]]})
    bm.finish(predictions=predictions)
