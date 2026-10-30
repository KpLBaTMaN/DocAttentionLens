import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from docattentionlens_api.main import create_app
from docattentionlens_api.migrate import migrate
from docattentionlens_api.repositories.jobs import ConflictError, JobRepository
from docattentionlens_worker.main import run_once
from docattentionlens.artifacts.records import OperationalReceipt


def submit(repo, key="check"):
    return repo.submit(key, {"kind": "integrity_check", "resource_class": "cpu", "inputs": {"text": "DocAttentionLens CPU check"}, "operation_version": "utf8_sha256_v1"})


def receipt(job):
    return OperationalReceipt(record_id="receipt_" + "b" * 32, created_at=datetime.now(UTC), producer_version="test", dependencies=[job["job_id"]], provenance={}, output={"operational": True})


def test_separate_worker_process_and_fresh_api_reopen(database, tmp_path):
    with TestClient(create_app(database)) as api:
        assert api.get("/health/ready").status_code == 200
        request = {"kind": "integrity_check", "idempotency_key": "process-test"}
        response = api.post("/api/v1/jobs", json=request)
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        assert api.post("/api/v1/jobs", json=request).json()["job_id"] == job_id
        request["inputs"] = {"text": "different"}
        assert api.post("/api/v1/jobs", json=request).status_code == 409
    env = {**os.environ, "DOCATTENTIONLENS_DATABASE_URL": database.database_url, "DOCATTENTIONLENS_ARTIFACT_ROOT": str(database.artifact_root), "DOCATTENTIONLENS_WORKER_ID_FILE": str(tmp_path / "worker-id")}
    result = subprocess.run([sys.executable, "-m", "docattentionlens_worker.main", "--once"], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    # The health check reads the ID from this same configured file.
    assert (tmp_path / "worker-id").read_text(encoding="utf-8").startswith("worker_")
    with TestClient(create_app(database)) as reopened:
        detail = reopened.get(f"/api/v1/jobs/{job_id}").json()
        assert detail["state"] == "completed"
        assert [e["state"] for e in detail["events"]] == ["queued", "running", "completed"]
        assert len(detail["attempts"]) == 1
        saved = reopened.get(f"/api/v1/artifacts/operational_receipt/{detail['result_id']}")
        assert saved.status_code == 200
        assert saved.json()["output"]["byte_count"] == len(b"DocAttentionLens CPU check")
        assert saved.json()["record_kind"] == "operational_receipt"
        assert reopened.get("/api/v1/runtime").json()["workers"][0]["available"]
    path = database.artifact_root / "receipts" / f"{detail['result_id']}.json"
    path.write_text("corrupted", encoding="utf-8")
    with TestClient(create_app(database)) as reopened:
        assert reopened.get(f"/api/v1/artifacts/operational_receipt/{detail['result_id']}").status_code == 409


def test_concurrent_idempotency_and_claims(database):
    repo = JobRepository(database)
    with ThreadPoolExecutor(max_workers=4) as pool:
        duplicates = list(pool.map(lambda _: submit(repo), range(4)))
    assert len({j["job_id"] for j in duplicates}) == 1
    submit(repo, "second")
    for worker in ("worker_a", "worker_b"):
        repo.heartbeat(worker)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(repo.claim, ["worker_a", "worker_b"]))
    assert len({j["job_id"] for j in claims}) == 2
    assert len({j["attempt_id"] for j in claims}) == 2
    assert repo.claim("worker_a") is None
    with pytest.raises(ConflictError):
        repo.submit("check", {"kind": "integrity_check", "inputs": {"text": "changed"}})


def test_lease_expiry_recovery_fences_old_publication(database):
    repo = JobRepository(database)
    job = submit(repo)
    repo.heartbeat("worker_old")
    repo.heartbeat("worker_new")
    first = repo.claim("worker_old")
    assert repo.renew(first["job_id"], first["attempt_id"])
    with repo.connect() as conn:
        conn.execute("UPDATE jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE job_id=%s", (job["job_id"],))
    assert not repo.finish(first, receipt(first), "old-checksum")
    second = repo.claim("worker_new")
    assert second["attempt_id"] != first["attempt_id"]
    assert not repo.renew(first["job_id"], first["attempt_id"])
    assert not repo.finish(first, receipt(first), "old-checksum")
    assert repo.finish(second, receipt(second), "new-checksum")
    saved = repo.get(job["job_id"])
    assert saved["state"] == "completed"
    assert [a["state"] for a in saved["attempts"]] == ["expired", "completed"]


def test_cancellation_boundaries_and_crash_recovery(database):
    repo = JobRepository(database)
    queued = submit(repo, "queued-cancel")
    assert repo.cancel(queued["job_id"])["state"] == "cancelled"
    active = submit(repo, "running-cancel")
    repo.heartbeat("worker")
    attempt = repo.claim("worker")
    assert repo.cancel(active["job_id"])["state"] == "cancellation_requested"
    assert repo.finish(attempt, receipt(attempt), "unused")
    assert repo.get(active["job_id"])["state"] == "cancelled"
    assert repo.get(active["job_id"])["result_id"] is None
    crashed = submit(repo, "crashed-cancel")
    repo.claim("worker")
    repo.cancel(crashed["job_id"])
    with repo.connect() as conn:
        conn.execute("UPDATE jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE job_id=%s", (crashed["job_id"],))
    assert repo.claim("worker") is None
    assert repo.get(crashed["job_id"])["state"] == "cancelled"


def test_failed_dispatch_does_not_publish_receipt(database):
    repo = JobRepository(database)
    invalid = repo.submit("invalid-version", {"kind": "integrity_check", "inputs": {}, "operation_version": "unsupported"})
    assert run_once(database, "worker")
    result = repo.get(invalid["job_id"])
    assert result["state"] == "failed"
    assert result["result_id"] is None
    assert result["attempts"][0]["error_code"] == "execution_failed"


def test_kernel_tolerance_failure_reaches_the_job(database, monkeypatch):
    def fail(_specification):
        raise ValueError("Retriever eager/SDPA query embeddings exceed tolerance: mean per-position L2 distance 0.0402 > 0.04 "
                         "(max absolute coordinate difference 0.1567)")
    monkeypatch.setattr("docattentionlens_worker.main.execute", fail)
    repo = JobRepository(database)
    job = submit(repo, "kernel-tolerance")
    assert run_once(database, "worker")
    result = repo.get(job["job_id"])
    assert result["state"] == "failed"
    assert result["error_message"] == ("Attention extraction and scoring disagreed beyond the calibrated tolerance: Retriever eager/SDPA query "
                                       "embeddings exceed tolerance: mean per-position L2 distance 0.0402 > 0.04 (max absolute coordinate "
                                       "difference 0.1567). No capture was published.")


def test_migration_is_repeatable_and_detects_drift(database):
    migrate(database)
    repo = JobRepository(database)
    with repo.connect() as conn:
        assert conn.execute("SELECT count(*) AS count FROM schema_migrations").fetchone()["count"] == 12
        conn.execute("UPDATE schema_migrations SET checksum='changed'")
    with pytest.raises(RuntimeError, match="Applied migration changed"):
        migrate(database)


def test_cursor_history_and_stale_worker(database):
    repo = JobRepository(database)
    identifiers = [submit(repo, f"request-{i}")["job_id"] for i in range(5)]
    with TestClient(create_app(database)) as api:
        first = api.get("/api/v1/jobs?limit=2").json()
        second = api.get("/api/v1/jobs", params={"limit": 2, "before": first["next_cursor"]}).json()
        third = api.get("/api/v1/jobs", params={"limit": 2, "before": second["next_cursor"]}).json()
        assert [j["job_id"] for page in [first, second, third] for j in page["items"]] == identifiers[::-1]
        assert third["next_cursor"] is None
        assert api.get("/api/v1/jobs?before=missing").status_code == 404
    repo.heartbeat("worker")
    with repo.connect() as conn:
        conn.execute("UPDATE workers SET heartbeat_at=clock_timestamp()-interval '20 seconds'")
    assert not repo.workers()[0]["available"]
