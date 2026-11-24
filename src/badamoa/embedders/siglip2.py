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


TEXT_LENGTH = 64  # the tokens a query is padded or cut to, as SigLIP2 was trained
PHRASES = ("a photo of a cat", "a truck on a road at night", "자전거를 타는 사람")


def export_text(checkpoint: str, folder: Path) -> None:
    """The text encoder as ONNX for the Hub (8-bit weights when the vectors stay as good), its tokenizer, and the
    phrases' token ids and vectors as made here, which the Hub checks it reproduces."""
    import onnxruntime
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer

    folder.mkdir(parents=True, exist_ok=True)
    print("글 검색에 쓸 텍스트 인코더를 만듭니다 (처음 한 번).", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, dtype=torch.float32).eval()

    class Text(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, input_ids: Any) -> Any:
            return pooled(self.model.get_text_features(input_ids=input_ids))

    def ids_of(text: str) -> Any:
        found = tokenizer(text.lower(), padding="max_length", max_length=TEXT_LENGTH, truncation=True, return_tensors="pt")
        return found["input_ids"]

    full = folder / "text-full.onnx"
    args = {
        "input_names": ["input_ids"],
        "output_names": ["embeds"],
        "dynamic_axes": {"input_ids": {0: "batch"}, "embeds": {0: "batch"}},
        "opset_version": 17,
    }
    with torch.no_grad():
        try:
            torch.onnx.export(Text(), (ids_of(PHRASES[0]),), str(full), dynamo=False, **args)
        except TypeError:  # a torch without the choice
            torch.onnx.export(Text(), (ids_of(PHRASES[0]),), str(full), **args)
        wanted = [torch.nn.functional.normalize(Text()(ids_of(t)).float(), dim=-1)[0].numpy() for t in PHRASES]

    def vectors(path: Path) -> list[Any]:
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = session.get_inputs()[0].name
        made = [session.run(None, {name: ids_of(t).numpy().astype(np.int64)})[0][0] for t in PHRASES]
        return [v / (np.linalg.norm(v) or 1.0) for v in made]

    def worst(path: Path) -> float:
        return min(float(np.dot(a, b)) for a, b in zip(vectors(path), wanted, strict=True))

    if worst(full) < 0.999:
        raise RuntimeError(f"ONNX로 바꾼 텍스트 인코더가 원래와 다릅니다 (코사인 {worst(full):.4f}).")
    small = folder / "text-int8.onnx"
    chosen = full
    try:
        quantize_dynamic(str(full), str(small), weight_type=QuantType.QInt8)
        if worst(small) >= 0.99:
            chosen = small
    except Exception as e:  # noqa: BLE001 - the full one is good too, only bigger
        print(f"8비트로 줄이지 못해 원래 크기로 보냅니다: {type(e).__name__}: {e}", flush=True)
    chosen.replace(folder / "text.onnx")
    tokenizer.backend_tokenizer.save(str(folder / "tokenizer.json"))
    checks = {
        "max_length": TEXT_LENGTH,
        "pad_id": int(tokenizer.pad_token_id),
        "pad_token": str(tokenizer.pad_token),
        "padding_side": str(getattr(tokenizer, "padding_side", "right") or "right"),
        "lowercase": True,
        "phrases": [
            {"text": t, "ids": [int(i) for i in ids_of(t)[0].tolist()], "vector": [float(x) for x in v]}
            for t, v in zip(PHRASES, wanted, strict=True)
        ],
    }
    (folder / "checks.json").write_text(json.dumps(checks, ensure_ascii=False), encoding="utf-8")
    size = (folder / "text.onnx").stat().st_size / 1e6
    print(f"글 검색 모델을 만들었습니다 ({size:,.0f} MB, {'8비트' if chosen == small else '원래 크기'}).", flush=True)


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
    if config.get("export_text"):  # the first job: the text encoder the Hub reads 글 검색 queries with
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        try:
            export_text(config["checkpoint"], out / "text")
        except Exception as e:  # noqa: BLE001 - the vectors are made; 글 검색 waits for the next job
            print(f"글 검색 모델을 만들지 못했습니다(다음 임베딩 때 다시): {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
