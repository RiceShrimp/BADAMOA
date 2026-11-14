"""Built-in model templates and the skeleton for new ones.

Each folder here is a template: ``badamoa.toml`` plus the code it declares
(``badamoa.core.templates`` describes the format). The code runs in the
template's environment with that environment's Python, so it must not import
``badamoa`` — only the standard library and its own requirements.

The contract with the node:

- input: ``--config <file>``, a JSON file with ``data`` (the dataset, e.g. the
  path of ``data.yaml``), ``data_dir``, ``out`` (write results here), ``cache``
  (kept between runs), ``device`` (``cpu``, ``0`` or ``0,1``), ``classes``
  (names, in training order) and ``params`` (what the user chose)
- output: ordinary stdout lines are the job log; a line starting with
  ``@@badamoa `` carries one JSON event — ``info``, ``epoch`` or ``result``
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent
SCAFFOLD = ROOT / "_scaffold"


def builtin() -> dict[str, Path]:
    """Folders of the templates shipped with BADAMOA, by template name."""
    return {p.name: p for p in sorted(ROOT.iterdir()) if p.is_dir() and not p.name.startswith(("_", "."))}
