"""Plugins: ML code that runs on a node in its own virtual environment.

The base install carries no ML libraries. A plugin declares what it needs, and
the node builds an isolated environment for it the first time a job asks
(``badamoa.node.envs``). Different plugins can therefore pin conflicting
versions of torch without touching each other or the Node Agent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PluginSpec(BaseModel):
    name: str
    title: str
    version: str  # bump to rebuild the environment on every node
    python: str = "3.12"
    requirements: list[str] = Field(default_factory=list)
    # Pick the PyTorch build that matches the node's GPU driver (CPU build when there is none).
    torch: bool = False
    platforms: list[str] = Field(default_factory=lambda: ["linux", "windows"])
    license: str = ""


PLUGINS: dict[str, PluginSpec] = {
    spec.name: spec
    for spec in [
        PluginSpec(
            name="yolo",
            title="Ultralytics YOLO",
            version="1",
            requirements=["ultralytics>=8.3,<9"],
            torch=True,
            license="AGPL-3.0",
        ),
    ]
}
