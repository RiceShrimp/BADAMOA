"""torchvision classification: the code of the ``torchvision`` template, which trains classification among its tasks.

This kind stays for the models trained with it: they are evaluated, tested, exported and 오토라벨 as before.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "torchvision"))

from ops import main  # noqa: E402

if __name__ == "__main__":
    main("classify", Path(__file__).resolve().with_name("badamoa.toml"))
