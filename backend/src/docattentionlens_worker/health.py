from pathlib import Path
import os
import tempfile

from docattentionlens_api.config import Settings
from docattentionlens_api.repositories.jobs import JobRepository


def worker_id_file() -> Path:
    """Where a running worker records its ID for this health check."""
    return Path(os.getenv("DOCATTENTIONLENS_WORKER_ID_FILE", str(Path(tempfile.gettempdir()) / "docattentionlens-worker-id")))


def main() -> None:
    worker_id = worker_id_file().read_text(encoding="utf-8").strip()
    available = any(w["worker_id"] == worker_id and w["available"] for w in JobRepository(Settings.from_env()).workers())
    if not available:
        raise SystemExit("Worker heartbeat is stale")


if __name__ == "__main__":
    main()
