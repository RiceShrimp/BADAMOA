"""Image classification with torchvision models (ResNet, EfficientNet, ConvNeXt, ViT …): the ``classify`` task of the
``torchvision`` template, and the code of the earlier ``classify`` template that it replaced.

The data arrives as ``<split>/<class>/<image>`` folders. Any torchvision classification model works: the last
linear layer is replaced for our classes.
"""

from __future__ import annotations

from pathlib import Path

import badamoa_sdk as bm
from common import IMAGE_SUFFIXES, device_of, export, fit, optimizer_for, registered_names, start_from

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics the pretrained weights expect
SAMPLES = 48  # validation images shown with their predictions
FAILURES = 48  # the most confident mistakes, kept as pictures
COLUMNS = ["image", "truth", "prediction", "confidence", "correct"]


def build(name: str, classes: int, pretrained: bool, freeze: bool):
    import torch
    import torchvision

    model = torchvision.models.get_model(name, weights="DEFAULT" if pretrained else None)
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad = False
    # The classifier is the last linear layer, whatever the architecture calls it (fc, classifier[-1], heads.head …).
    path, last = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)][-1]
    parent = model
    *parents, leaf = path.split(".")
    for part in parents:
        parent = getattr(parent, part)
    setattr(parent, leaf, torch.nn.Linear(last.in_features, classes))
    return model


def evaluate(model, loader, device, classes: int, criterion):
    import torch

    confusion = torch.zeros(classes, classes, dtype=torch.long)  # rows: truth, columns: prediction
    loss_sum, top5, seen = 0.0, 0, 0
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device)).cpu()
            loss_sum += float(criterion(logits, targets)) * len(targets)
            for truth, guess in zip(targets.tolist(), logits.argmax(1).tolist(), strict=True):
                confusion[truth, guess] += 1
            top5 += int((logits.topk(min(5, classes), dim=1).indices == targets[:, None]).any(1).sum())
            seen += len(targets)
    hits = confusion.diag().float()
    precision = hits / confusion.sum(0).clamp(min=1)
    recall = hits / confusion.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
    present = confusion.sum(1) > 0
    scores = {
        "top1": float(hits.sum() / max(1, seen)),
        "top5": top5 / max(1, seen),
        "f1": float(f1[present].mean()) if present.any() else 0.0,
        "val_loss": loss_sum / max(1, seen),
    }
    return scores, precision.tolist(), recall.tolist(), f1.tolist(), confusion


def save_confusion(confusion, names: list[str], path) -> None:
    """The confusion matrix as a picture, drawn with Pillow (no plotting library needed)."""
    from PIL import Image, ImageDraw

    n = len(names)
    # The built-in font only has Latin letters: other names are shown by their number.
    names = [name if name.isascii() else f"#{i}" for i, name in enumerate(names)]
    cell = max(14, min(48, 640 // max(1, n)))
    margin = 12 + 7 * max(len(name) for name in names)
    picture = Image.new("RGB", (margin + n * cell + 12, margin + n * cell + 12), "white")
    draw = ImageDraw.Draw(picture)
    rows = confusion.float() / confusion.sum(1, keepdim=True).clamp(min=1)
    for r in range(n):
        draw.text((6, margin + r * cell + cell // 2 - 5), names[r][:24], fill="black")
        for c in range(n):
            share = float(rows[r, c])
            shade = int(255 - 205 * share)
            box = (margin + c * cell, margin + r * cell, margin + (c + 1) * cell - 1, margin + (r + 1) * cell - 1)
            draw.rectangle(box, fill=(shade, shade, 255) if r != c else (shade, 255, shade), outline=(225, 225, 225))
            if cell >= 22 and int(confusion[r, c]):
                draw.text((box[0] + 4, box[1] + cell // 2 - 5), str(int(confusion[r, c])), fill="black")
    draw.text((margin, 4), "columns: predicted   rows: truth", fill="black")
    picture.save(path)


def log_samples(model, data, transform, device, run) -> None:
    """A table of validation images with the truth and the prediction of the best weights, wrong ones first."""
    import torch
    from PIL import Image

    if len(data) == 0:
        return
    picked = [data.samples[i] for i in range(0, len(data), max(1, len(data) // SAMPLES))][:SAMPLES]
    folder = run.out / "samples"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    model.eval()
    with torch.no_grad():
        for path, truth in picked:
            image = Image.open(path).convert("RGB")
            probabilities = torch.softmax(model(transform(image).unsqueeze(0).to(device)), dim=1)[0]
            score, guess = (float(v) for v in probabilities.max(0))
            thumb = folder / f"{len(rows):03d}.jpg"
            image.thumbnail((128, 128))
            image.save(thumb, quality=85)
            truth_name, guess_name = run.classes[truth], run.classes[int(guess)]
            name = bm.log_image(thumb, key="val_samples", caption=f"{truth_name} → {guess_name}")
            rows.append([name, truth_name, guess_name, round(score, 4), "O" if int(guess) == truth else "X"])
    rows.sort(key=lambda row: (row[4] == "O", -row[3]))
    bm.log_table("val_predictions", COLUMNS, rows)


def plain_transform(size: int):
    from torchvision import transforms as T

    return T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])


def train(run, hp: bm.Params) -> None:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from torchvision.datasets import ImageFolder

    torch.hub.set_dir(str(run.cache / "torch"))  # pretrained weights are kept between runs
    device = device_of(run)
    classes = len(run.classes)

    augment = T.Compose(
        [
            T.RandomResizedCrop(hp.imgsz, scale=(hp.crop_scale, 1.0)),
            T.RandomHorizontalFlip(hp.fliplr),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )
    plain = plain_transform(hp.imgsz)
    val_split = "val" if (run.data_dir / "val").is_dir() else "train"
    # allow_empty: with a class subset or a small val split, some class folders have no images
    train_data = ImageFolder(str(run.data_dir / "train"), augment, allow_empty=True)
    val_data = ImageFolder(str(run.data_dir / val_split), plain, allow_empty=True)
    drop_last = len(train_data) > hp.batch  # a last batch of one image breaks batch normalisation
    # Workers kept between epochs: starting them again each epoch costs seconds (Windows starts a new Python).
    workers = {"num_workers": hp.workers, "persistent_workers": hp.workers > 0}
    loader = DataLoader(train_data, batch_size=hp.batch, shuffle=True, drop_last=drop_last, **workers)
    val = DataLoader(val_data, batch_size=hp.batch, **workers)
    print(f"{hp.model} · 클래스 {classes}개 · train {len(train_data)}장 · val {len(val_data)}장 · 장치 {device}")

    model = build(hp.model, classes, hp.pretrained, hp.freeze)
    if run.init_weights:
        start_from(model, run.init_weights)
    model = model.to(device)
    optimizer = optimizer_for(hp, [p for p in model.parameters() if p.requires_grad])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=hp.label_smoothing)
    plain_loss = torch.nn.CrossEntropyLoss()

    def one_epoch(epoch: int) -> float:
        model.train()
        total, seen = 0.0, 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            loss = criterion(model(images), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, seen = total + float(loss.detach()) * len(targets), seen + len(targets)
        return total / max(1, seen)

    def validate() -> dict[str, float]:
        return evaluate(model, val, device, classes, plain_loss)[0]

    def shown(metrics: dict[str, float]) -> str:
        return f"top-1 {metrics['top1']:.3f} · top-5 {metrics['top5']:.3f}"

    best = fit(run, hp, model, optimizer, one_epoch, validate, "top1", shown)

    # Per-class numbers and the confusion matrix of the best weights.
    _, precision, recall, f1, confusion = evaluate(model, val, device, classes, plain_loss)
    per_class = [
        {"id": i, "name": name, "precision": precision[i], "recall": recall[i], "f1": f1[i]}
        for i, name in enumerate(run.classes)
    ]
    if classes <= 60:
        picture = run.out / "confusion_matrix.png"
        save_confusion(confusion, run.classes, picture)
        bm.log_artifact(picture)
    log_samples(model, val_data, plain, device, run)
    bm.log_model(run.out / "best.pt", classes=per_class)
    bm.finish(best)


# ---- evaluate · export · test · 오토라벨 ---------------------------------------------------------


def per_image(model, data, loader, device, run) -> None:
    """Every image of the evaluated split with its prediction, mistakes first; the most confident mistakes as pictures."""
    import torch
    from PIL import Image, ImageOps

    registered = registered_names(run.data_dir)
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


def operate(run, settings: bm.Params) -> None:
    import torch

    torch.hub.set_dir(str(run.cache / "torch"))
    size = int(settings.imgsz)
    device = device_of(run)
    model = build(str(settings.model), len(run.classes), pretrained=False, freeze=False)
    model.load_state_dict(torch.load(run.weights, map_location="cpu"))
    model.eval().to(device)
    plain = plain_transform(size)
    action = run.cfg["action"]

    if action == "eval":
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder

        data = ImageFolder(str(run.data_dir / run.split), plain, allow_empty=True)
        loader = DataLoader(data, batch_size=int(settings.batch))
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
        names = {"input_names": ["images"], "output_names": ["logits"]}
        export(run, model, torch.zeros(1, 3, size, size), names, trace=True, dynamic_axes={"images": {0: "batch"}})
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
