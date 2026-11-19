"""torchvision models for four tasks, the built-in ``torchvision`` template: the ``train`` entry point.

The run's task (its dataset's, ``task`` in the config) picks the code: ``detection.py`` trains Faster R-CNN ·
RetinaNet · FCOS · SSDlite (detect), Mask R-CNN (segment) and Keypoint R-CNN (pose), ``classification.py`` the
classification models (classify). The settings are declared in ``badamoa.toml``, some of them per task.
"""

from __future__ import annotations

from pathlib import Path

import badamoa_sdk as bm
from common import MANIFEST, settings, task_of


def main(task: str | None = None, manifest: Path = MANIFEST) -> None:
    """``task`` and ``manifest``: what a template that runs this code for one task of its own gives (``classify``)."""
    import torch
    import torchvision

    run = bm.init(torch=torch.__version__, torchvision=torchvision.__version__)
    task = task or task_of(run.cfg)
    hp = settings(run.config, task, manifest)
    if task == "classify":
        from classification import train

        train(run, hp)
    else:
        from detection import train

        train(run, hp, task)


if __name__ == "__main__":  # required: dataloader workers re-import this file on Windows
    main()
