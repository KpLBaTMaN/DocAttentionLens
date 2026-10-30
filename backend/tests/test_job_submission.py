"""Idempotent submission on every model run endpoint; no database or GPU is needed.

A new key is admitted only when a capable worker is live; a replay of the same inputs returns the
existing job without admission checks; a key naming different inputs is a 409 with a route-specific
message, also when the conflict appears between the replay check and the insert."""
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from docattentionlens_api.config import Settings
from docattentionlens_api.main import create_app
from docattentionlens_api.repositories.jobs import ConflictError, input_hash
from docattentionlens_api.services.imports import ModelStore
from docattentionlens.experiments import attention_interventions as interventions, document_studies as studies
from docattentionlens.experiments.contracts import VERSION as LAYOUT_VERSION
from docattentionlens.generation.contracts import ADAPTER_VERSIONS, QWEN25_VL_3B
from docattentionlens.retrieval.contracts import RETRIEVER_PROFILES

GPU = {"available": True, "resource_class": "gpu", "capabilities": {
    "adapters": [*(p["adapter_version"] for p in RETRIEVER_PROFILES.values()), *ADAPTER_VERSIONS],
    "retrieval_attention_adapters": [p["attention_adapter_version"] for p in RETRIEVER_PROFILES.values()], "free_mib": 100_000,
    "document_experiment_operations": [studies.VERSION], "attention_intervention_operations": [interventions.VERSION]}}
COLPALI = RETRIEVER_PROFILES["builtin-retrieval"]
CASE = NS(case_id="case_x", page_ids=["p1"], content_sha256="c"*64, reference_answer="")
KEY = "submission-key"


class Jobs:
    """Fake job repository: stored keys map to (input hash, job row); records each call in order."""
    def __init__(self, workers=(), stored=None, race=False):
        self.live, self.stored, self.race, self.calls, self.submitted = list(workers), stored or {}, race, [], None

    def workers(self):
        self.calls.append("workers")
        return self.live

    def existing_submission(self, key, specification):
        self.calls.append("existing")
        if key not in self.stored:
            return None
        digest, job = self.stored[key]
        if digest != input_hash(specification):
            raise ConflictError("Idempotency key already names a different input specification.")
        return job

    def submit(self, key, specification):
        self.calls.append("submit")
        if self.race:
            raise ConflictError("Idempotency key already names a different input specification.")
        self.submitted = (key, specification)
        return {"job_id": "job_new", "state": "queued"}


def retrieval(monkeypatch):
    monkeypatch.setattr(ModelStore, "get", lambda *_: {"role": "retrieval", "registration_id": "builtin-retrieval"})
    return ("/api/v1/retrieval/runs", {"case_id": "case_x", "page_ids": ["p1"], "idempotency_key": KEY},
            {"kind": "retrieval_capture", "case_id": "case_x", "page_ids": ["p1"], "adapter_version": COLPALI["adapter_version"],
             "model_revision": COLPALI["revision"], "case_sha256": "c"*64},
            "ColPali v1.3 GPU worker is unavailable or lacks required memory", "Key already names different retrieval inputs", ["existing"])


def generation(monkeypatch):
    monkeypatch.setattr(ModelStore, "get", lambda *_: {"role": "generation", "registration_id": "builtin", "revision": QWEN25_VL_3B.revision,
                                                        "capabilities": {"can_run": True}})
    return ("/api/v1/generation/runs", {"case_id": "case_x", "page_ids": ["p1"], "idempotency_key": KEY},
            {"kind": "generation_capture", "case_id": "case_x", "case_sha256": "c"*64, "retrieval_capture_id": None, "retrieval_sha256": None,
             "page_ids": ["p1"], "adapter_version": QWEN25_VL_3B.adapter_version, "model_revision": QWEN25_VL_3B.revision},
            "Generator worker unavailable or lacks required memory", "Key already names different generation inputs", ["existing"])


def layout_experiment(monkeypatch):
    import docattentionlens_api.routers.experiments as router
    from docattentionlens.generation.contracts import ADAPTER_VERSION, MODEL_REVISION
    from docattentionlens.retrieval.contracts import ADAPTER_VERSION as RETRIEVAL_ADAPTER, MODEL_REVISION as RETRIEVAL_REVISION
    protocol = NS(record_kind="experiment_protocol", review=NS(), content_sha256="p"*64)
    monkeypatch.setattr(router, "ExperimentRepository", lambda jobs: NS(get=lambda _: protocol))
    identifier = "experiment_"+"a"*32
    return (f"/api/v1/experiments/{identifier}/runs", {"idempotency_key": KEY},
            {"kind": "layout_experiment", "experiment_id": identifier, "protocol_sha256": "p"*64, "operation_version": LAYOUT_VERSION,
             "retrieval_adapter": RETRIEVAL_ADAPTER, "retrieval_revision": RETRIEVAL_REVISION, "generator_adapter": ADAPTER_VERSION,
             "generator_revision": MODEL_REVISION},
            "A GPU worker with 12 GiB available is required", "Key names different experiment inputs", ["existing"])


def document_study(monkeypatch):
    import docattentionlens_api.routers.document_studies as router
    study = NS(kind="prepared", model_registration_id="builtin", content_sha256="s"*64)
    monkeypatch.setattr(router, "DocumentStudyRepository", lambda jobs: NS(get=lambda _: study))
    monkeypatch.setattr(router, "validate_execution_code", lambda _: None)
    identifier = "document_study_"+"b"*32
    return (f"/api/v1/document-studies/{identifier}/runs", {"idempotency_key": KEY},
            {"kind": "document_experiment", "study_id": identifier, "study_sha256": "s"*64, "operation_version": studies.VERSION,
             "generator_adapter": QWEN25_VL_3B.adapter_version, "generator_revision": QWEN25_VL_3B.revision},
            "A GPU worker with the document experiment adapter is required", "Key names different document experiment inputs", ["existing"])


def attention_intervention(role, registration):
    def setup(monkeypatch):
        import docattentionlens_api.routers.attention_interventions as router
        experiment = NS(kind="prepared", role=role, model_registration_id=registration, content_sha256="e"*64)
        monkeypatch.setattr(router, "AttentionExperimentRepository", lambda jobs: NS(get=lambda _: experiment))
        monkeypatch.setattr(router, "validate_code", lambda _: None)
        identifier = "attention_experiment_"+"c"*32
        # Availability is read before the replay check, so a replay also asks for the live workers.
        return (f"/api/v1/attention-interventions/{identifier}/runs", {"idempotency_key": KEY},
                {"kind": "attention_intervention", "experiment_id": identifier, "experiment_sha256": "e"*64, "operation_version": interventions.VERSION},
                "A compatible GPU worker with the attention-intervention adapter is required", "Key names different attention experiment inputs",
                ["workers", "existing"])
    return setup


ENDPOINTS = [retrieval, generation, layout_experiment, document_study,
             attention_intervention("generation", "builtin"), attention_intervention("retrieval", "builtin-retrieval")]


def client(tmp_path, jobs):
    app = create_app(Settings("postgresql://unused@localhost/unused", tmp_path, tmp_path))
    app.state.repository = jobs
    app.state.documents = NS(case=lambda _: CASE)
    return TestClient(app, raise_server_exceptions=False)


def error(code, message):
    return {"error": {"code": code, "message": message}}


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_new_key_is_queued_only_with_a_capable_worker(endpoint, monkeypatch, tmp_path):
    url, body, specification, unavailable, _, _ = endpoint(monkeypatch)
    jobs = Jobs()
    response = client(tmp_path, jobs).post(url, json=body)
    assert (response.status_code, response.json()) == (503, error("model_unavailable", unavailable))
    assert "submit" not in jobs.calls
    jobs = Jobs([GPU])
    response = client(tmp_path, jobs).post(url, json=body)
    assert (response.status_code, response.json()) == (202, {"job_id": "job_new", "state": "queued", "status_url": "/api/v1/jobs/job_new"})
    assert jobs.submitted == (KEY, specification) and jobs.calls[-1] == "submit"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_replay_returns_the_existing_job_without_admission(endpoint, monkeypatch, tmp_path):
    url, body, specification, _, _, replay_calls = endpoint(monkeypatch)
    jobs = Jobs(stored={KEY: (input_hash(specification), {"job_id": "job_old", "state": "completed"})})
    response = client(tmp_path, jobs).post(url, json=body)
    assert (response.status_code, response.json()) == (202, {"job_id": "job_old", "state": "completed", "status_url": "/api/v1/jobs/job_old"})
    assert jobs.calls == replay_calls


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_key_naming_different_inputs_is_a_conflict(endpoint, monkeypatch, tmp_path):
    url, body, _, _, conflict, replay_calls = endpoint(monkeypatch)
    jobs = Jobs([GPU], stored={KEY: ("different", {"job_id": "job_old", "state": "completed"})})
    response = client(tmp_path, jobs).post(url, json=body)
    assert (response.status_code, response.json()) == (409, error("idempotency_conflict", conflict))
    assert jobs.calls == replay_calls
    # A concurrent submission of the same key between the replay check and the insert.
    jobs = Jobs([GPU], race=True)
    response = client(tmp_path, jobs).post(url, json=body)
    assert (response.status_code, response.json()) == (409, error("idempotency_conflict", conflict))
    assert jobs.calls[-1] == "submit"


def test_document_study_also_needs_the_model_memory_budget(monkeypatch, tmp_path):
    url, body, *_ = document_study(monkeypatch)
    jobs = Jobs([{**GPU, "capabilities": {**GPU["capabilities"], "free_mib": 1024}}])
    response = client(tmp_path, jobs).post(url, json=body)
    assert (response.status_code, response.json()) == (503, error("model_unavailable",
        "A compatible GPU worker and the selected model's memory budget are required"))
    assert "submit" not in jobs.calls
