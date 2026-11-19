"""Evaluate, export and test a trained classifier: the ``eval`` / ``export`` / ``predict`` entry points."""

from __future__ import annotations

import json
from pathlib import Path

import badamoa_sdk as bm
from train import MEAN, STD, build, evaluate, save_confusion

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FAILURES = 48  # the most confident mistakes, kept as pictures
COLUMNS = ["image", "truth", "prediction", "confidence", "correct"]


def per_image(model, data, loader, device, run) -> None:
    """Every image of the evaluated split with its prediction, mistakes first; the most confident mistakes as pictures."""
    import torch
    from PIL import Image, ImageOps

    try:  # the exported images are named by their hash: show the names they were registered under
        registered = json.loads((run.data_dir / "files.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        registered = {}
    rows, wrong, index, predicted = [], [], 0, []
    with torch.no_grad():
        for images, targets in loader:
            scores, guesses = torch.softmax(model(images.to(device)), dim=1).max(1)
            for score, guess, truth in zip(scores.tolist(), guesses.tolist(), targets.tolist(), strict=True):
                path = Path(data.samples[index][0])
                index += 1
                name = Path(registered.get(path.stem, path.name)).name
                row = [name, run.classes[truth], run.classes[guess], round(score, 4), "O" if guess == truth else "X"]
                rows.append(row)
                right = int(guess == truth)
                predicted.append(
                    {
                        "image": path.stem,
                        "labels": [[guess]],
                        "scores": [round(score, 4)],
                        "found": right,
                        "missed": 1 - right,
                        "extra": 0,
                    }
                )
                if guess != truth:
                    wrong.append((score, path, row))
    rows.sort(key=lambda r: (r[4] == "O", -r[3]))
    bm.log_table("per_image", COLUMNS, rows)
    if hasattr(bm, "log_predictions"):  # every image's guess: the dataset screen sorts by likely wrong labels
        bm.log_predictions(predicted)
    folder = run.out / ".badamoa" / "failures"
    folder.mkdir(parents=True, exist_ok=True)
    cells = []
    for n, (_, path, row) in enumerate(sorted(wrong, key=lambda w: -w[0])[:FAILURES]):
        copy = folder / f"{n:03d}.jpg"
        with Image.open(path) as opened:
            picture = ImageOps.exif_transpose(opened).convert("RGB")
            picture.thumbnail((256, 256))
            picture.save(copy, quality=85)
        cells.append([bm.overlay(copy), *row[1:]])
    if cells:
        bm.log_table("failures", COLUMNS, cells)
    print(f"틀린 이미지 {len(wrong):,}장 / {len(rows):,}장 (그림으로 남긴 것 {len(cells)}장)")


def main() -> None:
    import torch
    from torchvision import transforms as T

    run = bm.init(torch=torch.__version__)
    torch.hub.set_dir(str(run.cache / "torch"))
    settings = run.train_params
    size = int(settings.get("imgsz", 224))
    device = torch.device("cpu" if run.device == "cpu" else "cuda:0")
    model = build(str(settings.get("model", "resnet18")), len(run.classes), pretrained=False, freeze=False)
    model.load_state_dict(torch.load(run.weights, map_location="cpu"))
    model.eval().to(device)
    plain = T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])
    action = run.cfg["action"]

    if action == "eval":
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder

        data = ImageFolder(str(run.data_dir / run.split), plain, allow_empty=True)
        loader = DataLoader(data, batch_size=int(settings.get("batch", 32)))
        loss = torch.nn.CrossEntropyLoss()
        scores, precision, recall, f1, confusion = evaluate(model, loader, device, len(run.classes), loss)
        per_class = [
            {"id": i, "name": name, "precision": precision[i], "recall": recall[i], "f1": f1[i]}
            for i, name in enumerate(run.classes)
        ]
        if len(run.classes) <= 60:
            save_confusion(confusion, run.classes, run.out / "confusion_matrix.png")
            bm.log_artifact(run.out / "confusion_matrix.png")
        try:
            per_image(model, data, loader, device, run)
        except Exception as e:  # the metrics stand without it
            print(f"이미지별 결과를 남기지 못했습니다: {e}", flush=True)
        bm.finish(scores, classes=per_class)
    elif action == "export":
        example = torch.zeros(1, 3, size, size)
        model.cpu()
        if run.config.get("format") == "torchscript":
            exported = run.out / "model.torchscript"
            torch.jit.trace(model, example).save(str(exported))
        else:
            exported = run.out / "model.onnx"
            names = {"input_names": ["images"], "output_names": ["logits"]}
            torch.onnx.export(model, example, str(exported), dynamic_axes={"images": {0: "batch"}}, dynamo=False, **names)
        print(f"내보낸 파일: {exported.name} ({exported.stat().st_size / 1024**2:.1f} MB)")
        bm.log_artifact(exported)
        bm.finish(file=exported.name, bytes=exported.stat().st_size)
    elif action == "autolabel":  # a labeling set's images, named by their hash: the top class of each
        from PIL import Image

        paths = sorted(p for p in run.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        print(f"오토라벨: {len(paths):,}장의 클래스를 고릅니다", flush=True)
        predicted = []
        for n, path in enumerate(paths, 1):
            with Image.open(path) as opened:
                batch = plain(opened.convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                probabilities = model(batch).softmax(1)[0].cpu()
            score, guess = probabilities.max(0)
            predicted.append({"image": path.stem, "labels": [[int(guess)]], "scores": [round(float(score), 4)]})
            if n % 200 == 0 or n == len(paths):
                print(f"{n:,}/{len(paths):,}장", flush=True)
        bm.log_predictions(predicted)
        bm.finish(images=len(predicted), labels=len(predicted))
    else:
        from PIL import Image

        predictions = []
        for path in sorted(p for p in run.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            with Image.open(path) as opened:
                batch = plain(opened.convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                probabilities = model(batch).softmax(1)[0].cpu()
            best = probabilities.topk(min(5, len(run.classes)))
            top = [[int(i), round(float(p), 4)] for i, p in zip(best.indices.tolist(), best.values.tolist(), strict=True)]
            predictions.append({"image": path.name, "top": top})
            print(f"{path.name}: {run.classes[top[0][0]]} ({top[0][1]:.2f})")
        bm.finish(predictions=predictions)


if __name__ == "__main__":
    main()
