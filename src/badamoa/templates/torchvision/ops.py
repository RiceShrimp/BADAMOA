"""Evaluate, export and test a trained torchvision detector: the ``eval`` / ``export`` / ``predict`` entry points."""

from __future__ import annotations

import badamoa_sdk as bm
from train import IMAGE_SUFFIXES, YoloFolder, build, collate, evaluate

COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#800000", "#000075", "#f032e6"]


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    run = bm.init(torch=torch.__version__)
    torch.hub.set_dir(str(run.cache / "torch"))
    settings = run.train_params
    size = int(settings.get("imgsz", 640))
    name = str(settings.get("model", "fasterrcnn_resnet50_fpn"))
    device = torch.device("cpu" if run.device == "cpu" else "cuda:0")
    model = build(name, len(run.classes), pretrained=False, imgsz=size)
    model.load_state_dict(torch.load(run.weights, map_location="cpu"))
    model.eval().to(device)
    action = run.cfg["action"]

    if action == "eval":
        data = YoloFolder(run.data_dir, run.split)
        loader = DataLoader(data, batch_size=int(settings.get("batch", 4)), collate_fn=collate)
        scores = evaluate(model, loader, device, per_class=True)
        per_class = []
        ids, values = scores["classes"].reshape(-1).tolist(), scores["map_per_class"].reshape(-1).tolist()
        for class_id, value in zip(ids, values, strict=False):
            index = int(class_id) - 1
            if 0 <= index < len(run.classes):
                per_class.append({"id": index, "name": run.classes[index], "map50_95": max(0.0, float(value))})
        metrics = {
            "map50_95": max(0.0, float(scores["map"])),
            "map50": max(0.0, float(scores["map_50"])),
            "recall": max(0.0, float(scores["mar_100"])),
        }
        bm.finish(metrics, classes=per_class)
    elif action == "export":
        model.cpu()
        if run.config.get("format") == "torchscript":
            exported = run.out / "model.torchscript"
            torch.jit.script(model).save(str(exported))
        else:
            exported = run.out / "model.onnx"
            shape = 320 if name.startswith("ssdlite") else size
            names = {"input_names": ["images"], "output_names": ["boxes", "labels", "scores"]}
            example = ([torch.rand(3, shape, shape)],)  # the model takes one argument: a list of images
            torch.onnx.export(model, example, str(exported), opset_version=17, dynamo=False, **names)
        print(f"내보낸 파일: {exported.name} ({exported.stat().st_size / 1024**2:.1f} MB)")
        bm.log_artifact(exported)
        bm.finish(file=exported.name, bytes=exported.stat().st_size)
    else:
        from PIL import Image, ImageDraw
        from torchvision.transforms.functional import pil_to_tensor

        threshold = float(run.config.get("conf", 0.25))
        predictions = []
        for path in sorted(p for p in run.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            with Image.open(path) as opened:
                picture = opened.convert("RGB")
            width, height = picture.size
            with torch.no_grad():
                found = model([(pil_to_tensor(picture).float() / 255).to(device)])[0]
            draw = ImageDraw.Draw(picture)
            boxes = []
            rows = zip(found["boxes"].tolist(), found["labels"].tolist(), found["scores"].tolist(), strict=True)
            for (x0, y0, x1, y1), label, score in rows:
                index = int(label) - 1
                if score < threshold or not 0 <= index < len(run.classes):
                    continue
                centre = [(x0 + x1) / 2 / width, (y0 + y1) / 2 / height]
                boxes.append([index, round(score, 4), *centre, (x1 - x0) / width, (y1 - y0) / height])
                color = COLORS[index % len(COLORS)]
                draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
                title = run.classes[index] if run.classes[index].isascii() else f"#{index}"
                draw.text((x0 + 4, max(0, y0 - 12)), f"{title} {score:.2f}", fill=color)
            picture.save(run.out / path.name)
            bm.log_artifact(run.out / path.name)
            predictions.append({"image": path.name, "boxes": [[b[0], b[1], *(round(v, 5) for v in b[2:])] for b in boxes]})
            print(f"{path.name}: {len(boxes)}개 검출")
        bm.finish(predictions=predictions)


if __name__ == "__main__":
    main()
