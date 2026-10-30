from fastapi.testclient import TestClient

from docattentionlens_api.config import Settings
from docattentionlens_api.main import create_app
from docattentionlens_api.repositories.receipts import FileReceiptStore
from docattentionlens.artifacts.records import OperationalReceipt


def test_catalogue_and_planned_execution_do_not_require_database(tmp_path):
    app = create_app(Settings("postgresql://unused:unused@127.0.0.1:1/unused", tmp_path, tmp_path))
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        response = client.get("/api/v1/methods")
        assert response.status_code == 200
        for method in response.json()["items"]:
            result = client.post("/api/v1/jobs", json={"kind": "calculation", "idempotency_key": "test", "method_id": method["method_id"], "variant_id": method["variant_id"]})
            assert result.status_code == 422
            assert result.json()["error"]["code"] == ("capture_required" if method["status"]!="planned" else "method_planned")
        assert len(client.get("/api/v1/methods?evidence_kind=retrieval_scoring").json()["items"]) == 1
        assert client.get("/api/v1/methods/missing").status_code == 404
        assert client.post("/api/v1/jobs", json={"kind": "unknown", "idempotency_key": "test"}).status_code == 422
        assert response.headers["X-Request-ID"]


def test_receipt_is_exclusive_verified_and_confined(tmp_path):
    from datetime import UTC, datetime
    import pytest

    store = FileReceiptStore(tmp_path)
    record = OperationalReceipt(record_id="receipt_" + "a" * 32, created_at=datetime.now(UTC), producer_version="test", dependencies=[], provenance={}, output={"note": "operational"})
    checksum = store.put(record)
    assert store.read(record.record_id, checksum) == record
    with pytest.raises(FileExistsError):
        store.put(record)
    with pytest.raises(ValueError):
        store.read("../../external", checksum)
    (tmp_path / "receipts" / f"{record.record_id}.json").write_text("{}")
    with pytest.raises(ValueError):
        store.read(record.record_id, checksum)
