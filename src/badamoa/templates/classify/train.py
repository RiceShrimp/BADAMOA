"""Image classification with torchvision models (ResNet, EfficientNet, ConvNeXt, ViT …), the built-in ``classify`` template.

The data arrives as ``<split>/<class>/<image>`` folders. Any torchvision
classification model works: the last linear layer is replaced for our classes.
"""

from __future__ import annotations

import time

import badamoa_sdk as bm

hp = bm.params(
    model=bm.choice(
        [
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "efficientnet_b0",
            "efficientnet_v2_s",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "convnext_tiny",
            "regnet_y_400mf",
            "vit_b_16",
            "swin_t",
        ],
        label="모델",
        custom=True,
        help="torchvision의 다른 분류 모델 이름도 직접 입력할 수 있습니다.",
    ),
    epochs=bm.integer(30, min=1, label="epoch"),
    imgsz=bm.integer(224, min=32, max=1024, label="이미지 크기", group="학습", advanced=True),
    batch=bm.integer(32, min=1, label="배치", group="학습", advanced=True),
    workers=bm.integer(2, min=0, max=32, label="데이터 로더 수", group="학습", advanced=True),
    pretrained=bm.boolean(True, label="사전학습 가중치에서 시작", group="학습", advanced=True),
    freeze=bm.boolean(
        False, label="마지막 층만 학습", group="학습", advanced=True, help="백본을 고정합니다. 데이터가 적을 때 빠릅니다."
    ),
    optimizer=bm.choice(["AdamW", "SGD"], label="옵티마이저", group="최적화", advanced=True),
    lr=bm.number(0.001, min=0, label="학습률", group="최적화", advanced=True),
    momentum=bm.number(
        0.9, min=0, max=1, label="모멘텀 (SGD)", group="최적화", advanced=True, unused_when={"optimizer": ["AdamW"]}
    ),
    weight_decay=bm.number(0.01, min=0, label="weight decay", group="최적화", advanced=True),
    label_smoothing=bm.number(0.0, min=0, max=0.5, label="label smoothing", group="최적화", advanced=True),
    cos_lr=bm.boolean(True, label="코사인 학습률 스케줄", group="최적화", advanced=True),
    fliplr=bm.number(
        0.5,
        min=0,
        max=1,
        label="좌우 뒤집기 확률",
        group="증강",
        advanced=True,
        help="숫자·글자처럼 방향이 뜻을 갖는 데이터는 0으로 둡니다.",
    ),
    crop_scale=bm.number(0.6, min=0.08, max=1, label="무작위 자르기 최소 비율", group="증강", advanced=True),
)

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics the pretrained weights expect


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


SAMPLES = 48  # validation images shown with their predictions


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
    bm.log_table("val_predictions", ["image", "truth", "prediction", "confidence", "correct"], rows)


def main() -> None:
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import transforms as T
    from torchvision.datasets import ImageFolder

    run = bm.init(torch=torch.__version__, torchvision=torchvision.__version__)
    torch.hub.set_dir(str(run.cache / "torch"))  # pretrained weights are kept between runs
    device = torch.device("cpu" if run.device == "cpu" else "cuda:0")
    classes = len(run.classes)

    augment = T.Compose(
        [
            T.RandomResizedCrop(hp.imgsz, scale=(hp.crop_scale, 1.0)),
            T.RandomHorizontalFlip(hp.fliplr),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )
    plain = T.Compose([T.Resize(int(hp.imgsz * 1.14)), T.CenterCrop(hp.imgsz), T.ToTensor(), T.Normalize(MEAN, STD)])
    val_split = "val" if (run.data_dir / "val").is_dir() else "train"
    # allow_empty: with a class subset or a small val split, some class folders have no images
    train_data = ImageFolder(str(run.data_dir / "train"), augment, allow_empty=True)
    val_data = ImageFolder(str(run.data_dir / val_split), plain, allow_empty=True)
    drop_last = len(train_data) > hp.batch  # a last batch of one image breaks batch normalisation
    train = DataLoader(train_data, batch_size=hp.batch, shuffle=True, num_workers=hp.workers, drop_last=drop_last)
    val = DataLoader(val_data, batch_size=hp.batch, num_workers=hp.workers)
    print(f"{hp.model} · 클래스 {classes}개 · train {len(train_data)}장 · val {len(val_data)}장 · 장치 {device}")

    model = build(hp.model, classes, hp.pretrained, hp.freeze)
    if run.init_weights:  # continue from a registered model: every layer whose shape still fits
        state = torch.load(run.init_weights, map_location="cpu")
        own = model.state_dict()
        fitting = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
        model.load_state_dict(fitting, strict=False)
        print(f"등록된 모델에서 시작: {len(fitting)}/{len(own)}개 층을 불러왔습니다.")
    model = model.to(device)
    weights = [p for p in model.parameters() if p.requires_grad]
    if hp.optimizer == "SGD":
        optimizer = torch.optim.SGD(weights, lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay)
    else:
        optimizer = torch.optim.AdamW(weights, lr=hp.lr, weight_decay=hp.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.epochs) if hp.cos_lr else None
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=hp.label_smoothing)
    plain_loss = torch.nn.CrossEntropyLoss()

    best, best_path, best_scores = -1.0, run.out / "best.pt", {}
    start = 1
    if run.resume_checkpoint and run.resume_checkpoint.is_file():  # continue an interrupted run
        state = torch.load(run.resume_checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        start, best, best_scores = state["epoch"] + 1, state["best"], state["best_metrics"]
        if state.get("best_model"):
            torch.save(state["best_model"], best_path)
        print(f"체크포인트에서 이어 갑니다: epoch {start}부터")
    for epoch in range(start, hp.epochs + 1):
        started = time.monotonic()
        model.train()
        total, seen = 0.0, 0
        for images, targets in train:
            images, targets = images.to(device), targets.to(device)
            loss = criterion(model(images), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total, seen = total + float(loss) * len(targets), seen + len(targets)
        rate = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        scores, *_ = evaluate(model, val, device, classes, plain_loss)
        metrics = {**scores, "train_loss": total / max(1, seen), "lr": rate}
        bm.epoch(epoch, hp.epochs, metrics)
        print(
            f"epoch {epoch}/{hp.epochs} · loss {metrics['train_loss']:.3f} · top-1 {scores['top1']:.3f} · "
            f"top-5 {scores['top5']:.3f} · {time.monotonic() - started:.0f}초"
        )
        if scores["top1"] > best:
            best, best_scores = scores["top1"], {k: v for k, v in metrics.items() if k != "lr"}
            torch.save(model.state_dict(), best_path)

        # Everything needed to continue from here, should the run be interrupted.
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best": best,
            "best_metrics": best_scores,
            "best_model": torch.load(best_path, map_location="cpu") if best_path.is_file() else None,
        }
        torch.save(checkpoint, run.out / "last.pt.part")
        (run.out / "last.pt.part").replace(run.out / "last.pt")  # never a half-written file under the real name
        bm.save_checkpoint(run.out / "last.pt", epoch)

    # Per-class numbers and the confusion matrix of the best weights.
    model.load_state_dict(torch.load(best_path, map_location=device))
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
    bm.log_model(best_path, classes=per_class)
    bm.finish(best_scores)


if __name__ == "__main__":  # required: dataloader workers re-import this file on Windows
    main()
