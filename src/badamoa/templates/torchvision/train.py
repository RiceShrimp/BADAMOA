"""torchvision detection models (Faster R-CNN, RetinaNet, FCOS, SSDlite), the built-in ``torchvision`` template.

Also the reference for writing a template with the SDK: settings are declared in
code with ``bm.params`` and progress is reported with ``bm.epoch`` / ``bm.finish``.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import badamoa_sdk as bm

hp = bm.params(
    model=bm.choice(
        [
            "fasterrcnn_resnet50_fpn",
            "fasterrcnn_mobilenet_v3_large_fpn",
            "retinanet_resnet50_fpn",
            "fcos_resnet50_fpn",
            "ssdlite320_mobilenet_v3_large",
        ],
        label="모델",
        help="resnet50은 정확하지만 무겁고, mobilenet과 ssdlite는 가볍고 빠릅니다.",
    ),
    epochs=bm.integer(30, min=1, label="epoch"),
    imgsz=bm.integer(640, min=64, max=2048, label="이미지 크기", group="학습", advanced=True),
    batch=bm.integer(4, min=1, label="배치", group="학습", advanced=True),
    workers=bm.integer(2, min=0, max=32, label="데이터 로더 수", group="학습", advanced=True),
    pretrained=bm.boolean(True, label="사전학습 가중치에서 시작", group="학습", advanced=True),
    optimizer=bm.choice(["SGD", "AdamW"], label="옵티마이저", group="최적화", advanced=True),
    lr=bm.number(0.005, min=0, label="학습률", group="최적화", advanced=True),
    clip=bm.number(10.0, min=0.1, label="기울기 자르기 (norm)", group="최적화", advanced=True),
    momentum=bm.number(0.9, min=0, max=1, label="모멘텀 (SGD)", group="최적화", advanced=True),
    weight_decay=bm.number(0.0001, min=0, label="weight decay", group="최적화", advanced=True),
    cos_lr=bm.boolean(True, label="코사인 학습률 스케줄", group="최적화", advanced=True),
    fliplr=bm.number(0.5, min=0, max=1, label="좌우 뒤집기 확률", group="증강", advanced=True),
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YoloFolder:
    """``images/<split>`` + ``labels/<split>`` in YOLO format, as torchvision detection targets."""

    def __init__(self, root: Path, split: str, flip: float = 0.0) -> None:
        self.images = sorted(p for p in (root / "images" / split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        self.labels = root / "labels" / split
        self.flip = flip

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image
        from torchvision.transforms.functional import hflip, pil_to_tensor

        path = self.images[index]
        with Image.open(path) as opened:
            picture = opened.convert("RGB")
        width, height = picture.size
        boxes, labels = [], []
        label_file = self.labels / f"{path.stem}.txt"
        if label_file.is_file():
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, w, h = (float(v) for v in parts[1:5])
                x0, y0, x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height
                if x1 - x0 >= 1 and y1 - y0 >= 1:
                    boxes.append([x0, y0, x1, y1])
                    labels.append(int(parts[0]) + 1)  # 0 is the background class in torchvision
        image = pil_to_tensor(picture).float() / 255
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.flip and random.random() < self.flip:
            image = hflip(image)
            box_tensor[:, [0, 2]] = width - box_tensor[:, [2, 0]]
        return image, {"boxes": box_tensor, "labels": torch.tensor(labels, dtype=torch.int64)}


def collate(batch):
    return tuple(zip(*batch, strict=True))


def build(name: str, classes: int, pretrained: bool, imgsz: int):
    import torchvision.models.detection as detection
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    factory = getattr(detection, name)
    size = {} if name.startswith("ssdlite") else {"min_size": imgsz, "max_size": imgsz}  # SSDlite is fixed at 320
    if pretrained and name.startswith("fasterrcnn"):
        # The whole detector comes pretrained on COCO; only the last layer is replaced for our classes.
        model = factory(weights="DEFAULT", **size)
        features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(features, classes + 1)
        return model
    backbone = "DEFAULT" if pretrained else None  # ImageNet backbone, new detection head
    return factory(weights=None, weights_backbone=backbone, num_classes=classes + 1, **size)


def evaluate(model, loader, device, per_class: bool = False) -> dict:
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(class_metrics=per_class)
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            metric.update([{k: v.cpu() for k, v in out.items()} for out in outputs], list(targets))
    return metric.compute()


def main() -> None:
    import torch
    import torchvision
    from torch.utils.data import DataLoader

    run = bm.init(torch=torch.__version__, torchvision=torchvision.__version__)
    torch.hub.set_dir(str(run.cache / "torch"))  # pretrained weights are kept between runs
    device = torch.device("cpu" if run.device == "cpu" else "cuda:0")
    val_split = "val" if (run.data_dir / "images" / "val").is_dir() else "train"
    train = DataLoader(
        YoloFolder(run.data_dir, "train", flip=hp.fliplr),
        batch_size=hp.batch,
        shuffle=True,
        num_workers=hp.workers,
        collate_fn=collate,
    )
    val_data = YoloFolder(run.data_dir, val_split)
    val = DataLoader(val_data, batch_size=hp.batch, num_workers=hp.workers, collate_fn=collate)
    sizes = f"train {len(train.dataset)}장 · val {len(val_data)}장"
    print(f"{hp.model} · 클래스 {len(run.classes)}개 · {sizes} · 장치 {device}")

    model = build(hp.model, len(run.classes), hp.pretrained, hp.imgsz)
    if run.init_weights:  # continue from a registered model: every layer whose shape still fits
        state = torch.load(run.init_weights, map_location="cpu")
        own = model.state_dict()
        fitting = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
        model.load_state_dict(fitting, strict=False)
        print(f"등록된 모델에서 시작: {len(fitting)}/{len(own)}개 층을 불러왔습니다.")
    model = model.to(device)
    weights = [p for p in model.parameters() if p.requires_grad]
    if hp.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(weights, lr=hp.lr, weight_decay=hp.weight_decay)
    else:
        optimizer = torch.optim.SGD(weights, lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.epochs) if hp.cos_lr else None

    best, best_path = -1.0, run.out / "best.pt"
    best_metrics: dict[str, float] = {}
    start = 1
    if run.resume_checkpoint and run.resume_checkpoint.is_file():  # continue an interrupted run
        state = torch.load(run.resume_checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        start, best, best_metrics = state["epoch"] + 1, state["best"], state["best_metrics"]
        if state.get("best_model"):
            torch.save(state["best_model"], best_path)
        print(f"체크포인트에서 이어 갑니다: epoch {start}부터")
    for epoch in range(start, hp.epochs + 1):
        started = time.monotonic()
        model.train()
        total, batches = 0.0, 0
        for images, targets in train:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            loss = sum(model(images, targets).values())
            if not torch.isfinite(loss):
                raise SystemExit(f"loss가 발산했습니다 (epoch {epoch}). 학습률을 낮춰 보세요.")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(weights, hp.clip)  # small batches make detectors blow up without it
            optimizer.step()
            total, batches = total + float(loss.detach()), batches + 1
        rate = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        scores = evaluate(model, val, device)
        metrics = {
            "map50_95": max(0.0, float(scores["map"])),
            "map50": max(0.0, float(scores["map_50"])),
            "recall": max(0.0, float(scores["mar_100"])),
            "train_loss": total / max(1, batches),
            "lr": rate,
        }
        bm.epoch(epoch, hp.epochs, metrics)
        print(
            f"epoch {epoch}/{hp.epochs} · loss {metrics['train_loss']:.3f} · mAP50-95 {metrics['map50_95']:.3f} · "
            f"mAP50 {metrics['map50']:.3f} · {time.monotonic() - started:.0f}초"
        )
        if metrics["map50_95"] > best:
            best, best_metrics = metrics["map50_95"], metrics
            torch.save(model.state_dict(), best_path)

        # Everything needed to continue from here, should the run be interrupted.
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best": best,
            "best_metrics": best_metrics,
            "best_model": torch.load(best_path, map_location="cpu") if best_path.is_file() else None,
        }
        torch.save(checkpoint, run.out / "last.pt.part")
        (run.out / "last.pt.part").replace(run.out / "last.pt")  # never a half-written file under the real name
        bm.save_checkpoint(run.out / "last.pt", epoch)

    # Per-class numbers for the best weights.
    model.load_state_dict(torch.load(best_path, map_location=device))
    scores = evaluate(model, val, device, per_class=True)
    per_class = []
    ids = scores["classes"].reshape(-1).tolist()
    values = scores["map_per_class"].reshape(-1).tolist()
    for class_id, value in zip(ids, values, strict=False):
        index = int(class_id) - 1
        if 0 <= index < len(run.classes):
            per_class.append({"id": index, "name": run.classes[index], "map50_95": max(0.0, float(value))})
    bm.log_model(best_path, classes=per_class)
    bm.finish({k: v for k, v in best_metrics.items() if k != "lr"})


if __name__ == "__main__":  # required: dataloader workers re-import this file on Windows
    main()
