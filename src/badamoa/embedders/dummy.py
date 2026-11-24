"""Test vectors from the node's own Python: a few numbers of the image (its colours, its shape), no model."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

from PIL import Image


def main() -> None:
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    dim, out, size = int(config["dim"]), Path(config["out"]), int(config["part"])
    shas: list[str] = []
    rows: list[list[float]] = []
    parts = done = failed = 0

    def flush() -> None:
        nonlocal shas, rows, parts
        if not shas:
            return
        parts += 1
        head = json.dumps({"model": config["model"], "dim": dim, "shas": shas}).encode() + b"\n"
        body = b"".join(struct.pack(f"<{dim}e", *row) for row in rows)
        temp = out / f"part-{parts:04d}.bin.tmp"
        temp.write_bytes(head + body)
        temp.replace(out / f"part-{parts:04d}.bin")
        print(f"벡터 {len(shas):,}개 완료", flush=True)
        shas, rows = [], []

    for line in Path(config["list"]).read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        try:
            with Image.open(item["path"]) as image:
                width, height = image.size
                pixels = list(image.convert("RGB").resize((2, 1)).getdata())
        except OSError:
            failed += 1
            continue
        values = [c / 255 for pixel in pixels for c in pixel] + [width / (width + height), height / (width + height)]
        values = (values + [0.0] * dim)[:dim]
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        rows.append([v / norm for v in values])
        shas.append(item["sha256"])
        done += 1
        if len(shas) >= size:
            flush()
    flush()
    print(f"이미지 {done:,}장의 벡터를 만들었습니다" + (f" (읽지 못한 이미지 {failed:,}장)" if failed else ""), flush=True)


if __name__ == "__main__":
    main()
