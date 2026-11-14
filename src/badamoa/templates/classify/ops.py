"""Evaluate, export and test a trained classifier: the ``eval`` / ``export`` / ``predict`` entry points."""

from __future__ import annotations

import badamoa_sdk as bm
from train import MEAN, STD, build, evaluate, save_confusion

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
