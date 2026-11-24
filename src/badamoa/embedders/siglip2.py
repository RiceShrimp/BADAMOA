"""SigLIP2 image vectors (Hugging Face ``transformers``): the images listed, read by several workers while the GPU
works, a float16 vector each, L2-normalised, written a part at a time (see ``badamoa.embedders``)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

BATCH = 64


class Images(Dataset):
    """The listed images, each as the model's input (None for one that cannot be read)."""

    def __init__(self, items: list[dict[str, Any]], processor: Any) -> None:
        self.items = items
        self.processor = processor

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[int, Any]:
        try:
            with Image.open(self.items[index]["path"]) as image:
                pixels = self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
            return index, pixels
        except Exception:  # noqa: BLE001 - a broken or vanished image is left out, the rest go on
            return index, None


def collate(batch: list[tuple[int, Any]]) -> tuple[list[int], Any, int]:
    good = [(i, p) for i, p in batch if p is not None]
    pixels = torch.stack([p for _, p in good]) if good else None
    return [i for i, _ in good], pixels, len(batch) - len(good)


def pooled(output: Any) -> Any:
    """The image vectors: transformers 5 returns the vision tower's output (its pooled vectors), older ones a tensor."""
    if torch.is_tensor(output):
        return output
    found = getattr(output, "pooler_output", None)
    return found if found is not None else output[0]


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out, size, dim = Path(config["out"]), int(config["part"]), int(config["dim"])
    items = [json.loads(line) for line in Path(config["list"]).read_text(encoding="utf-8").splitlines() if line]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"{config['checkpoint']}를 불러옵니다 ({'GPU' if device == 'cuda' else 'CPU'})", flush=True)
    model = AutoModel.from_pretrained(config["checkpoint"], dtype=dtype).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(config["checkpoint"])
    workers = min(8, os.cpu_count() or 1)
    loader = DataLoader(
        Images(items, processor), batch_size=BATCH, num_workers=workers, collate_fn=collate, pin_memory=device == "cuda"
    )

    shas: list[str] = []
    rows: list[np.ndarray] = []
    parts = done = failed = 0
    started = shown = time.monotonic()

    def flush() -> None:
        nonlocal shas, rows, parts
        if not shas:
            return
        parts += 1
        vectors = np.ascontiguousarray(np.concatenate(rows), dtype="<f2")
        if vectors.shape != (len(shas), dim):
            raise RuntimeError(f"벡터 모양이 {vectors.shape}입니다 ({dim}차원이어야 함).")
        head = json.dumps({"model": config["model"], "dim": dim, "shas": shas}).encode() + b"\n"
        temp = out / f"part-{parts:04d}.bin.tmp"
        temp.write_bytes(head + vectors.tobytes())
        temp.replace(out / f"part-{parts:04d}.bin")
        print(f"벡터 {len(shas):,}개 완료 (모두 {done:,}/{len(items):,}장)", flush=True)
        shas, rows = [], []

    with torch.inference_mode():
        for indices, pixels, broken in loader:
            failed += broken
            if pixels is None:
                continue
            features = pooled(model.get_image_features(pixel_values=pixels.to(device, dtype=dtype)))
            features = torch.nn.functional.normalize(features.float(), dim=-1)
            rows.append(features.cpu().numpy())
            shas.extend(items[i]["sha256"] for i in indices)
            done += len(indices)
            if len(shas) >= size:
                flush()
            if time.monotonic() - shown >= 30:
                rate = done / max(time.monotonic() - started, 1e-6)
                print(f"임베딩 {done:,}/{len(items):,}장 ({rate:,.0f}장/초)", flush=True)
                shown = time.monotonic()
    flush()
    took = time.monotonic() - started
    note = f" (읽지 못한 이미지 {failed:,}장)" if failed else ""
    print(f"이미지 {done:,}장의 벡터를 {took:,.0f}초에 만들었습니다{note}", flush=True)


if __name__ == "__main__":
    main()
