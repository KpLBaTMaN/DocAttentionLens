"""Table-aware analyses through the API on fixture captures: explicitly not model or research evidence."""
from datetime import datetime, UTC
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient

from docattentionlens_api.main import create_app
from docattentionlens_api.repositories.generations import GenerationStore
from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens.generation.contracts import (ContextManifest, ContextPage, GenerationCapture, GenerationPosition,
                                                                 PredictionStep, QWEN25_VL_3B)
from test_table_grounding_api import registered  # noqa: F401  (fixture)

GRID = 8


def case_for(api, question_id):
    created = api.post("/api/v1/cases", json={"question_id": question_id, "query_variant": "original", "task_scope": "table"})
    assert created.status_code == 201, created.text
    return created.json()


def capture(settings, case, page, seed):
    """A fixture capture: an 8 × 8 grid of image positions over the full page, then a question token and the source."""
    identifier = "generation_"+uuid4().hex
    n = GRID*GRID+2
    w, h = page["width"]/GRID, page["height"]/GRID
    positions = [GenerationPosition(index=i, token_id=i, piece="fixture", role="image", page_id=page["page_id"],
                                    bbox=[(i % GRID)*w, (i//GRID)*h, (i % GRID+1)*w, (i//GRID+1)*h], ownership={"background": 1.0}) for i in range(GRID*GRID)]
    positions += [GenerationPosition(index=GRID*GRID, token_id=1, piece="q", role="query", ownership={"non_image:query": 1.0}),
                  GenerationPosition(index=GRID*GRID+1, token_id=2, piece="s", role="special", ownership={"non_image:special": 1.0})]
    record = GenerationCapture(capture_id=identifier, case_id=case["case_id"], created_at=datetime.now(UTC).isoformat(), producer_version="fixture",
        dependencies=[], context=ContextManifest(retrieved_page_ids=[], selected_page_ids=[page["page_id"]], omitted_page_ids=[], query=case["query"],
            prompt="fixture not model evidence", input_token_ids=list(range(n)), input_attention_mask=[1]*n,
            pages=[ContextPage(page_id=page["page_id"], page_number=page["page_number"], image_sha256=page["image_sha256"], source_width=page["width"],
                               source_height=page["height"], resized_width=page["width"], resized_height=page["height"], grid_thw=[1, GRID*2, GRID*2],
                               position_indices=list(range(GRID*GRID)), pixel_array=None, preview_url=page["image_url"])]),
        positions=positions, output_token_ids=[n], answer="1", stop_reason="eos",
        steps=[PredictionStep(step=0, pass_kind="prefill", source_index=n-1, consumed_token_id=2, consumed_piece="s", predicted_index=n,
                              predicted_token_id=n, predicted_piece="1", target_count=n, array="step_0")],
        layer_count=2, query_head_count=2, kv_head_count=1,
        provenance={"purpose": "unit_fixture_not_model_evidence", "model_id": QWEN25_VL_3B.model_id, "model_revision": QWEN25_VL_3B.revision,
                    "model_registration_id": "builtin"},
        coverage={"quantity": "post_softmax", "all_causal_targets": True, "ownership": "fixture"}, validation={})
    values = np.random.default_rng(seed).random((2, 2, n)).astype(np.float64)
    values = (values/values.sum(axis=-1, keepdims=True)).astype(np.float32)
    store = GenerationStore(settings.artifact_root)
    digest = store.put(record, {"step_0": values})
    jobs = JobRepository(settings)
    job = jobs.submit("fixture-"+uuid4().hex, {"kind": "generation_capture"})
    with jobs.connect() as conn:
        conn.execute("INSERT INTO generation_captures(capture_id,case_id,job_id,content_hash) VALUES (%s,%s,%s,%s)", (identifier, case["case_id"], job["job_id"], digest))
        conn.execute("UPDATE jobs SET state='completed' WHERE job_id=%s", (job["job_id"],))
    return identifier


def test_table_methods_run_with_baseline_and_matched_contrast(registered):  # noqa: F811
    settings, dataset_id = registered
    with TestClient(create_app(settings)) as api:
        table = next(t for t in api.get("/api/v1/tables", params={"dataset_id": dataset_id}).json()["items"] if t["title"] == "Lactate cross-tabulation")
        page = api.get(f"/api/v1/pages/{table['page_id']}").json()
        questions = api.get("/api/v1/questions", params={"document_id": table["document_id"], "limit": 100}).json()["items"]
        first = next(q for q in questions if q["native_id"] == "pt1m-question-demo-0001")
        g = first["table_grounding"]
        same_row = next(q for q in questions if q["table_grounding"]["answer_row_ids"] == g["answer_row_ids"]
                        and q["table_grounding"]["answer_column_ids"] != g["answer_column_ids"])
        other = next(q for q in questions if q["table_grounding"]["answer_row_ids"] != g["answer_row_ids"]
                     and q["table_grounding"]["answer_column_ids"] != g["answer_column_ids"] and q["table_grounding"]["binding_status"] != "needs_source_qa_review")
        primary = capture(settings, case_for(api, first["question_id"]), page, 1)
        partner = capture(settings, case_for(api, same_row["question_id"]), page, 2)
        third = capture(settings, case_for(api, other["question_id"]), page, 3)

        related = api.get(f"/api/v1/generation/captures/{primary}/table-related").json()
        assert set(related["baseline_capture_ids"]) == {partner, third}
        assert related["matched_capture_ids"] == [partner]
        assert next(i for i in related["items"] if i["capture_id"] == partner)["relation"] == "same_row"

        prepared = api.get("/api/v1/methods/availability", params={"generation_capture_id": primary, "case_id": json_case(api, primary)}).json()
        assert prepared["table_structure"]["cell_count"] == 34
        ready = {p["method"]["method_id"]: p["available"] for p in prepared["items"] if p["method"]["perspective"] == "table_aware"}
        assert ready["table.cell_allocation"] and ready["table.lookup_path"] and ready["table.matched_contrast"]

        methods = {m["method_id"]: m for m in api.get("/api/v1/methods").json()["items"]}
        body = lambda identifier, **extra: {"generation_capture_id": primary, "method_id": identifier,
                                             "variant_id": methods[identifier]["variant_id"], "parameters": {}, **extra}
        for identifier in ("table.page_regions", "table.mapping_audit", "table.cell_allocation", "table.answer_distractors"):
            response = api.post("/api/v1/methods/analyses", json=body(identifier))
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["schema_version"] == "0.10.0" and len(result["table_allocation"]["cells"]) == 34
            assert result["input_manifest"]["inputs"][primary]["table_annotation_sha256"]
        lookup = api.post("/api/v1/methods/analyses", json=body("table.lookup_path", sample_capture_ids=[primary, partner, third]))
        assert lookup.status_code == 200, lookup.text
        swap = lookup.json()["tables"][0]
        assert swap["title"] == "Question-swap baseline"
        assert set(lookup.json()["input_manifest"]["sample_capture_ids"]) == {primary, partner, third}
        contrast = api.post("/api/v1/methods/analyses", json=body("table.matched_contrast", comparison_capture_id=partner))
        assert contrast.status_code == 200, contrast.text
        assert any("row" in n for n in contrast.json()["notices"])
        unmatched = api.post("/api/v1/methods/analyses", json=body("table.matched_contrast", comparison_capture_id=third))
        assert unmatched.status_code == 422 and "not matched" in unmatched.text


def json_case(api, capture_id):
    return api.get(f"/api/v1/generation/captures/{capture_id}").json()["case_id"]
