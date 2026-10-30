import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str
    artifact_root: Path
    data_root: Path
    worker_resource_class: str = "cpu"
    worker_concurrency: int = 1
    lease_seconds: int = 30
    model_cache: Path = Path("var/model-cache")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.getenv("DOCATTENTIONLENS_DATABASE_URL", "postgresql://docattentionlens:docattentionlens-local-only@localhost:5432/docattentionlens"),
            artifact_root=Path(os.getenv("DOCATTENTIONLENS_ARTIFACT_ROOT", "var/artifacts")),
            data_root=Path(os.getenv("DOCATTENTIONLENS_DATA_ROOT", "data")),
            worker_resource_class=os.getenv("DOCATTENTIONLENS_WORKER_RESOURCE_CLASS", "cpu"),
            worker_concurrency=int(os.getenv("DOCATTENTIONLENS_WORKER_CONCURRENCY", "1")),
            lease_seconds=int(os.getenv("DOCATTENTIONLENS_LEASE_SECONDS", "30")),
            model_cache=Path(os.getenv("DOCATTENTIONLENS_MODEL_CACHE", "var/model-cache")),
        )

    def __post_init__(self) -> None:
        if not self.database_url.startswith(("postgresql://", "postgres://")):
            raise ValueError("DOCATTENTIONLENS_DATABASE_URL must be a PostgreSQL connection URI")
        if self.worker_resource_class not in ("cpu", "gpu") or self.worker_concurrency != 1:
            raise ValueError("CPU/GPU workers support concurrency 1 only")
        if self.lease_seconds < 10:
            raise ValueError("Lease must allow at least 10 seconds for heartbeat renewal")
