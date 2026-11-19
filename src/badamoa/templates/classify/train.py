"""torchvision classification: the code of the ``torchvision`` template, which trains classification among its tasks.

This kind stays for the models trained with it; its settings are in ``badamoa.toml`` as they were.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "torchvision"))

from train import main  # noqa: E402

if __name__ == "__main__":  # required: dataloader workers re-import this file on Windows
    main("classify", Path(__file__).resolve().with_name("badamoa.toml"))
