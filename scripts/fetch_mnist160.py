"""Fetch MNIST160: 160 handwritten digits in class folders (about 70 KB), for light classification checks.

    uv run python scripts/fetch_mnist160.py

It lands in ``.badamoa/data/mnist160`` as ``train/<digit>/*.png`` and ``test/<digit>/*.png``.
Register it in a classification project with "데이터셋 가져오기".
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/mnist160.zip"
TARGET = Path(__file__).parent.parent / ".badamoa" / "data"


def main() -> int:
    out = TARGET / "mnist160"
    if out.is_dir() and any(p for p in out.rglob("*") if p.suffix.lower() in (".png", ".jpg")):
        print(f"이미 있습니다: {out}")
        return 0
    print(f"내려받는 중: {URL}")
    with urllib.request.urlopen(URL, timeout=120) as response:  # noqa: S310 - a fixed https URL
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    TARGET.mkdir(parents=True, exist_ok=True)
    root = TARGET.resolve()
    for member in archive.infolist():
        if root not in (root / member.filename).resolve().parents and (root / member.filename).resolve() != root:
            print(f"허용되지 않는 경로가 들어 있습니다: {member.filename}", file=sys.stderr)
            return 1
    archive.extractall(TARGET)
    images = sum(p.suffix.lower() in (".png", ".jpg") for p in out.rglob("*"))
    print(f"완료: {out} · 이미지 {images}장")
    return 0


if __name__ == "__main__":
    sys.exit(main())
