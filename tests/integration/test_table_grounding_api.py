"""Table grounding registration, table resources, table cases and the single-page generation plan."""
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from psycopg.types.json import Jsonb

from docattentionlens_api.main import create_app
from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens_worker.main import run_once

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "TableGroundQA-Pilot-v1"
pytestmark = pytest.mark.skipif(not (ROOT / "data" / SOURCE / "manifest.json").is_file(), reason="TableGroundQA-Pilot-v1 export is not present")


@pytest.fixture
def registered(database):
    """Registers the table export through the real job path."""
    settings = replace(database, data_root=ROOT / "data", lease_seconds=60)
    with TestClient(create_app(settings)) as api:
        accepted = api.post("/api/v1/datasets/register", json={"source": SOURCE, "idempotency_key": "register-tables"})
        assert accepted.status_code == 202, accepted.json()
    assert run_once(settings, "dataset-worker")
    with TestClient(create_app(settings)) as api:
        job = api.get(f"/api/v1/jobs/{accepted.json()['job_id']}").json()
        assert job["state"] == "completed", job
        return settings, job["output_resource_id"]


def test_registration_lists_sources_tables_and_page_geometry(registered):
    settings, dataset_id = registered
    with TestClient(create_app(settings)) as api:
        sources = {s["source"]: s for s in api.get("/api/v1/datasets/sources").json()["items"]}
        assert sources[SOURCE]["present"] and sources[SOURCE]["structure_kind"] == "table_grounding"
        assert sources["doclayout_mm_rag_bench"]["structure_kind"] == "document_layout"
        assert api.post("/api/v1/datasets/register", json={"source": "../elsewhere", "idempotency_key": "bad"}).status_code == 422
        summary = api.get(f"/api/v1/datasets/{dataset_id}/summary").json()
        assert summary["structure_kind"] == "table_grounding" and summary["counts"]["cells"] == 975
        assert [s["original_page_number"] for s in summary["source_manifest"]["page_sources"]] == [7, 4, 4, 7, 4]
        tables = api.get("/api/v1/tables", params={"dataset_id": dataset_id}).json()["items"]
        assert [t["question_count"] for t in tables] == [20] * 5
        assert {t["title"]: t["questions_needing_review"] for t in tables}["Cover-crop seed weight and temperatures"] == 4
        lactate = next(t for t in tables if t["title"] == "Lactate cross-tabulation")
        assert (lactate["row_count"], lactate["column_count"], lactate["cell_count"], lactate["source_page_number"], lactate["source_page_count"]) == (9, 5, 34, 7, 10)
        page = api.get(f"/api/v1/pages/{lactate['page_id']}").json()
        assert (page["width"], page["height"], page["page_number"]) == (1211, 1684, 7)
        located = api.get(f"/api/v1/pages/{lactate['page_id']}/tables").json()
        assert located["total"] == 1 and len(located["items"][0]["cells"]) == 34
        assert api.get(f"/api/v1/pages/{lactate['page_id']}/elements").json()["total"] == 0
        image = api.get(f"/api/v1/pages/{lactate['page_id']}/image")
        assert image.status_code == 200 and image.headers["content-type"] == "image/png" and image.headers["etag"] == f'"{page["image_sha256"]}"'
        document = api.get(f"/api/v1/documents/{lactate['document_id']}").json()
        assert document["filename"] == "bm-28-2-020701.pdf" and document["table_ids"] == [lactate["table_id"]]


def test_table_cases_freeze_grounding_and_take_the_full_page(registered):
    settings, dataset_id = registered
    with TestClient(create_app(settings)) as api:
        table = next(t for t in api.get("/api/v1/tables", params={"dataset_id": dataset_id}).json()["items"] if t["title"] == "Lactate cross-tabulation")
        questions = api.get("/api/v1/questions", params={"document_id": table["document_id"]}).json()["items"]
        question = next(q for q in questions if q["native_id"] == "pt1m-question-demo-0001")
        assert question["table_grounding"]["answer_cell_ids"] == ["cell-00015"]
        assert api.post("/api/v1/cases", json={"question_id": question["question_id"]}).json()["error"]["code"] == "invalid_scope"
        assert api.post("/api/v1/cases", json={"question_id": question["question_id"], "query_variant": "original", "task_scope": "document"}).status_code == 422
        created = api.post("/api/v1/cases", json={"question_id": question["question_id"], "query_variant": "original", "task_scope": "table"})
        assert created.status_code == 201, created.json()
        case = created.json()
        assert case["page_ids"] == [table["page_id"]] and case["support_page_ids"] == [table["page_id"]] and case["support_element_ids"] == []
        assert case["reference_answer"] == "1" and case["label_exposure"] == "evaluation_only"
        grounding = case["table_grounding"]
        assert grounding["row_header_path"] == ["cell-00008", "cell-00013"] and grounding["column_header_path"] == ["cell-00003", "cell-00006"]
        assert case["provenance"]["supplied_input"] == "full_page" and case["provenance"]["authored_input"] == "table_crop"
        assert table["table_id"] in case["dependencies"]
        assert api.get(f"/api/v1/cases/{case['case_id']}").json()["table_grounding"] == grounding


def test_single_page_generation_plan_is_hashed_into_the_job(registered):
    settings, dataset_id = registered
    jobs = JobRepository(settings)
    with jobs.connect() as conn:
        conn.execute("INSERT INTO workers(worker_id,resource_class,heartbeat_at,producer_version,capabilities) VALUES ('generator-fixture','gpu',clock_timestamp(),'test',%s)",
                     (Jsonb({"adapters": ["qwen25vl_prediction_rows_v1"], "free_mib": 15000}),))
    with TestClient(create_app(settings)) as api:
        assert api.get("/api/v1/generation/plan").json()["single_page_image_positions"] == 1024
        table = api.get("/api/v1/tables", params={"dataset_id": dataset_id}).json()["items"][0]
        question = api.get("/api/v1/questions", params={"document_id": table["document_id"]}).json()["items"][0]
        case = api.post("/api/v1/cases", json={"question_id": question["question_id"], "query_variant": "original", "task_scope": "table"}).json()
        request = {"case_id": case["case_id"], "page_ids": case["page_ids"], "idempotency_key": "table-detail"}
        assert api.post("/api/v1/generation/runs", json={**request, "image_token_limit": 1024, "page_ids": case["page_ids"] * 2}).status_code == 422
        assert api.post("/api/v1/generation/runs", json={**request, "image_token_limit": 2048}).status_code == 422
        detail = api.post("/api/v1/generation/runs", json={**request, "image_token_limit": 1024})
        standard = api.post("/api/v1/generation/runs", json={**request, "idempotency_key": "table-standard"})
        assert detail.status_code == standard.status_code == 202
    with jobs.connect() as conn:
        specs = {row["job_id"]: row["specification"] for row in conn.execute("SELECT job_id,specification FROM jobs WHERE kind='generation_capture'").fetchall()}
    assert specs[detail.json()["job_id"]]["image_token_limit"] == 1024
    # The standard plan keeps the earlier specification, so replay hashes do not change.
    assert "image_token_limit" not in specs[standard.json()["job_id"]]
