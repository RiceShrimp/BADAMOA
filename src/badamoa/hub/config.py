from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PORT = 8470


@dataclass
class HubSettings:
    home: Path
    host: str = "0.0.0.0"
    port: int = DEFAULT_PORT
    # URL remote nodes / agents should use. None → derived from the incoming request.
    public_url: str | None = None
    database_url: str | None = None
    wheels_dir: Path | None = None
    web_dir: Path | None = None
    # Run a node agent on this machine as well (All-in-one install).
    local_node: bool = False
    node_offline_after_s: float = 30.0
    node_lost_after_s: float = 90.0
    session_ttl_hours: int = 24 * 14
    # A GPU nobody reserved through BADAMOA still counts as busy above these levels
    # (someone is using the server directly).
    gpu_busy_memory_pct: float = 10.0
    gpu_busy_util_pct: float = 50.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.home = Path(self.home)
        self.home.mkdir(parents=True, exist_ok=True)
        if self.database_url is None:
            self.database_url = f"sqlite:///{(self.home / 'hub.db').as_posix()}"
        if self.wheels_dir is None:
            self.wheels_dir = self.home / "wheels"
        self.wheels_dir.mkdir(parents=True, exist_ok=True)

    @property
    def storage_dir(self) -> Path:
        path = self.home / "storage"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def local_node_home(self) -> Path:
        return self.home / "local-node"
