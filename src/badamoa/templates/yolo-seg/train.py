"""Ultralytics YOLO segmentation: the same code as the ``yolo`` template, which handles both tasks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "yolo"))

from train import main  # noqa: E402

if __name__ == "__main__":  # required: dataloader workers re-import this file on Windows
    sys.exit(main(sys.argv))
