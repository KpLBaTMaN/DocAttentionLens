"""The container health check reads the worker ID the worker wrote, from the same configurable file."""
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS

import pytest

import docattentionlens_worker.health as health


def workers(*rows):
    return lambda settings: NS(workers=lambda: list(rows))


def test_health_reads_the_configured_worker_id_file(monkeypatch, tmp_path):
    path = tmp_path / "worker-id"
    path.write_text("worker_fixture\n", encoding="utf-8")
    monkeypatch.setenv("DOCATTENTIONLENS_WORKER_ID_FILE", str(path))
    monkeypatch.setattr(health, "JobRepository", workers({"worker_id": "worker_fixture", "available": True}))
    health.main()
    monkeypatch.setattr(health, "JobRepository", workers({"worker_id": "worker_fixture", "available": False}, {"worker_id": "worker_other", "available": True}))
    with pytest.raises(SystemExit, match="Worker heartbeat is stale"):
        health.main()


def test_worker_id_file_defaults_to_the_temporary_directory(monkeypatch):
    monkeypatch.delenv("DOCATTENTIONLENS_WORKER_ID_FILE", raising=False)
    read = []
    monkeypatch.setattr(Path, "read_text", lambda self, encoding=None: read.append(self) or "worker_fixture")
    monkeypatch.setattr(health, "JobRepository", workers({"worker_id": "worker_fixture", "available": True}))
    health.main()
    assert read == [Path(tempfile.gettempdir()) / "docattentionlens-worker-id"]
