"""The committed examples register through the real job path and open as cases, with no benchmark data."""
from dataclasses import replace
from pathlib import Path
import shutil

from fastapi.testclient import TestClient
import pytest

from docattentionlens_api.main import create_app
from docattentionlens_worker.main import run_once

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS, TABLES = "doclayout_mm_rag_examples", "TableGroundQA-Examples-v1"


@pytest.fixture
def examples(database, tmp_path):
    """Both examples registered from a data root that holds nothing else."""
    root = tmp_path / "data"
    for name in (DOCUMENTS, TABLES):
        shutil.copytree(ROOT / "data" / name, root / name)
    settings = replace(database, data_root=root, lease_seconds=60)
    datasets = {}
    for name in (DOCUMENTS, TABLES):
        with TestClient(create_app(settings)) as api:
            accepted = api.post("/api/v1/datasets/register", json={"source": name, "idempotency_key": f"register-{name}"})
            assert accepted.status_code == 202, accepted.json()
        assert run_once(settings, "dataset-worker")
        with TestClient(create_app(settings)) as api:
            job = api.get(f"/api/v1/jobs/{accepted.json()['job_id']}").json()
            assert job["state"] == "completed", job
            datasets[name] = job["output_resource_id"]
    return settings, datasets


def test_examples_are_offered_and_register_alone(examples):
    settings, datasets = examples
    with TestClient(create_app(settings)) as api:
        sources = {s["source"]: s for s in api.get("/api/v1/datasets/sources").json()["items"]}
        assert sources[DOCUMENTS]["present"] and sources[DOCUMENTS]["structure_kind"] == "document_layout"
        assert sources[TABLES]["present"] and sources[TABLES]["structure_kind"] == "table_grounding"
        assert not sources["doclayout_mm_rag_bench"]["present"] and not sources["TableGroundQA-Pilot-v1"]["present"]
        registered = {d["name"]: d for d in api.get("/api/v1/datasets").json()["items"]}
        assert registered[DOCUMENTS]["counts"] == {"documents": 1, "pages": 8, "elements": 100, "questions": 29}
        assert registered[TABLES]["counts"]["tables"] == 3 and registered[TABLES]["counts"]["cells"] == 270
        document = api.get("/api/v1/documents", params={"dataset_id": datasets[DOCUMENTS]}).json()["items"][0]
        page = api.get(f"/api/v1/pages/{document['page_ids'][3]}").json()
        assert api.get(f"/api/v1/pages/{page['page_id']}/image").status_code == 200
        assert any(e["element_type"] == "table" for e in api.get(f"/api/v1/pages/{page['page_id']}/elements").json()["items"])


def test_a_table_question_takes_only_its_own_page(examples):
    settings, datasets = examples
    with TestClient(create_app(settings)) as api:
        tables = api.get("/api/v1/tables", params={"dataset_id": datasets[TABLES]}).json()["items"]
        assert sorted(t["cell_count"] for t in tables) == [35, 85, 150]
        small = next(t for t in tables if t["cell_count"] == 35)
        assert (small["source_page_number"], small["source_page_count"], small["question_count"]) == (6, 8, 6)
        assert api.get(f"/api/v1/pages/{small['page_id']}/tables").json()["total"] == 2
        question = next(q for q in api.get("/api/v1/questions", params={"document_id": small["document_id"]}).json()["items"]
                        if q["table_grounding"]["table_id"] == small["table_id"])
        created = api.post("/api/v1/cases", json={"question_id": question["question_id"], "query_variant": "original", "task_scope": "table"})
        assert created.status_code == 201, created.json()
        case = created.json()
        assert case["page_ids"] == [small["page_id"]] and case["provenance"]["authored_input"] == "full_page"
        assert case["provenance"]["input_note"] == "Question authored against the full page; supplied as that page."
        assert case["reference_answer"] == question["answer"]
