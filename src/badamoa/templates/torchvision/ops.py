"""Evaluate, export, test and 오토라벨 with a trained torchvision model: the ``eval`` / ``export`` / ``predict``
entry points (오토라벨 is predict with ``action = "autolabel"``). The model's task picks the code, as in training."""

from __future__ import annotations

from pathlib import Path

import badamoa_sdk as bm
from common import MANIFEST, settings, task_of


def main(task: str | None = None, manifest: Path = MANIFEST) -> None:
    """``task`` and ``manifest``: what a template that runs this code for one task of its own gives (``classify``)."""
    import torch

    run = bm.init(torch=torch.__version__)
    task = task or task_of(run.cfg)
    trained = settings(run.train_params, task, manifest)  # how the model was trained, defaults for what it lacks
    if task == "classify":
        from classification import operate

        operate(run, trained)
    else:
        from detection import operate

        operate(run, trained, task)


if __name__ == "__main__":
    main()
