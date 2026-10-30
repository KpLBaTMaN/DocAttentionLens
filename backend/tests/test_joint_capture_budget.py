"""Resource admission for all three readings; no database or GPU is needed."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from docattentionlens_api.routers.retrieval import router
from docattentionlens_api.services.imports import ModelStore
from docattentionlens.retrieval.contracts import RETRIEVER_PROFILES


@pytest.mark.parametrize("registration", RETRIEVER_PROFILES)
def test_joint_plan_and_storage_admission_bound_all_three_readings(registration, monkeypatch, tmp_path):
    profile = RETRIEVER_PROFILES[registration]
    jobs = NS(workers=lambda: [{"available": True, "resource_class": "gpu", "capabilities": {
        "free_mib": profile["required_free_mib"], "adapters": [profile["adapter_version"]],
        "retrieval_attention_adapters": [profile["attention_adapter_version"]]}}],
        existing_submission=lambda *_: None, submit=Mock(return_value={"job_id": "joint-job", "state": "queued"}))
    app = FastAPI()
    app.include_router(router)
    app.state.repository = jobs
    app.state.settings = NS(artifact_root=tmp_path)
    app.state.documents = NS(case=lambda _: NS(case_id="case", page_ids=["page"], content_sha256="hash"))
    monkeypatch.setattr(ModelStore, "get", lambda *_: {"role": "retrieval", "registration_id": registration})
    # Independent retained-array accounting: one 256x256 query and TWO 1100x1100 page matrices,
    # for every query head at every language layer, all FP32. Vectors/manifests need headroom.
    matrix_count = profile["layers"] * profile["query_heads"]
    query_bytes = matrix_count * 256 * 256 * 4
    page_bytes = matrix_count * 1100 * 1100 * 4
    with TestClient(app, raise_server_exceptions=False) as client:
        ordinary = client.get('/api/v1/retrieval/plan', params={"model_registration_id": registration}).json()
        joint = client.get('/api/v1/retrieval/plan', params={"model_registration_id": registration, "capture_joint_attention": True}).json()
        assert ordinary["maximum_retained_attention_bytes"] == query_bytes + page_bytes
        assert joint["maximum_retained_attention_bytes"] == query_bytes + 2 * page_bytes
        assert joint["required_free_gpu_mib"] == ordinary["required_free_gpu_mib"], "passes run sequentially"
        body = {"case_id": "case", "page_ids": ["page"], "model_registration_id": registration,
                "capture_attention": True, "capture_joint_attention": True, "attention_page_id": "page",
                "idempotency_key": "three-reading-test"}
        # Enough for query+page, insufficient for the third reading: do not queue the job.
        monkeypatch.setattr('shutil.disk_usage', lambda _: NS(free=ordinary["maximum_retained_attention_bytes"] + 128 * 1024**2))
        response = client.post('/api/v1/retrieval/runs', json=body)
        assert response.status_code == 422, response.text
        jobs.submit.assert_not_called()
        monkeypatch.setattr('shutil.disk_usage', lambda _: NS(free=joint["maximum_retained_attention_bytes"] + 128 * 1024**2))
        response = client.post('/api/v1/retrieval/runs', json=body)
        assert response.status_code == 202, response.text
        specification = jobs.submit.call_args.args[1]
        assert specification["capture_joint_attention"] and specification["joint_capture_policy"] == "retain_document_v1"


def _app(registration, monkeypatch, tmp_path, reference_answer):
    profile = RETRIEVER_PROFILES[registration]
    jobs = NS(workers=lambda: [{"available": True, "resource_class": "gpu", "capabilities": {
        "free_mib": profile["required_free_mib"], "adapters": [profile["adapter_version"]],
        "retrieval_attention_adapters": [profile["attention_adapter_version"]]}}],
        existing_submission=lambda *_: None, submit=Mock(return_value={"job_id": "answer-job", "state": "queued"}))
    app = FastAPI()
    app.include_router(router)
    app.state.repository = jobs
    app.state.settings = NS(artifact_root=tmp_path)
    app.state.documents = NS(case=lambda _: NS(case_id="case", page_ids=["page"], content_sha256="hash", reference_answer=reference_answer))
    monkeypatch.setattr(ModelStore, "get", lambda *_: {"role": "retrieval", "registration_id": registration})
    return app, jobs, profile


@pytest.mark.parametrize("registration", RETRIEVER_PROFILES)
def test_joint_answer_pass_is_admitted_as_a_third_page_reading(registration, monkeypatch, tmp_path):
    app, jobs, profile = _app(registration, monkeypatch, tmp_path, "$4,130,000")
    matrix_count = profile["layers"] * profile["query_heads"]
    query_bytes, page_bytes = matrix_count * 256 * 256 * 4, matrix_count * 1100 * 1100 * 4
    body = {"case_id": "case", "page_ids": ["page"], "model_registration_id": registration, "capture_attention": True,
            "capture_joint_attention": True, "joint_answer_source": "reference", "attention_page_id": "page", "idempotency_key": "joint-answer-test"}
    with TestClient(app, raise_server_exceptions=False) as client:
        answered = client.get('/api/v1/retrieval/plan', params={"model_registration_id": registration,
                                                                 "capture_joint_attention": True, "joint_answer": True}).json()
        assert answered["maximum_retained_attention_bytes"] == query_bytes + 3 * page_bytes
        # Room for two page readings only: the answer pass is not queued.
        monkeypatch.setattr('shutil.disk_usage', lambda _: NS(free=query_bytes + 2 * page_bytes + 128 * 1024**2))
        assert client.post('/api/v1/retrieval/runs', json=body).status_code == 422
        jobs.submit.assert_not_called()
        monkeypatch.setattr('shutil.disk_usage', lambda _: NS(free=answered["maximum_retained_attention_bytes"] + 128 * 1024**2))
        response = client.post('/api/v1/retrieval/runs', json=body)
        assert response.status_code == 202, response.text
        assert jobs.submit.call_args.args[1]["joint_answer_source"] == "reference"


def test_joint_answer_needs_the_joint_pass_and_a_reference_answer(monkeypatch, tmp_path):
    registration = next(iter(RETRIEVER_PROFILES))
    base = {"case_id": "case", "page_ids": ["page"], "model_registration_id": registration, "capture_attention": True,
            "attention_page_id": "page", "joint_answer_source": "reference", "idempotency_key": "joint-answer-refusal"}
    app, jobs, _ = _app(registration, monkeypatch, tmp_path, "$4,130,000")
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post('/api/v1/retrieval/runs', json=base)
        assert response.status_code == 422 and "joint capture" in response.text
    app, jobs, _ = _app(registration, monkeypatch, tmp_path, "  ")
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post('/api/v1/retrieval/runs', json={**base, "capture_joint_attention": True})
        assert response.status_code == 422 and "no reference answer" in response.text
    jobs.submit.assert_not_called()
