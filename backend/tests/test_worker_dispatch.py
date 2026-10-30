"""What the worker publishes for each job kind, with the model adapters replaced; no database or GPU is needed."""
from datetime import UTC, datetime
from time import monotonic, sleep
from types import SimpleNamespace as NS

import numpy as np
import pytest

import docattentionlens_worker.main as worker
from docattentionlens_api.repositories.captures import CaptureStore
from docattentionlens_api.repositories.receipts import FileReceiptStore
from docattentionlens.retrieval.contracts import RETRIEVER_PROFILES, RetrievalCapture

COLPALI = RETRIEVER_PROFILES["builtin-retrieval"]
CASE = NS(case_id="case_x", content_sha256="c"*64, dataset_id="d", page_ids=["p1", "p2"])


class Jobs:
    """Fake job repository serving one claimed job; records publication."""
    def __init__(self, kind, specification, renewed=True):
        self.job = {"job_id": "job_1", "attempt_id": "attempt_1", "input_hash": "h", "kind": kind, "specification": specification}
        self.renewed, self.finished = renewed, []

    def __call__(self, settings):
        return self

    def heartbeat(self, worker_id):
        pass

    def claim(self, worker_id):
        return self.job

    def renew(self, job_id, attempt_id):
        return self.renewed

    def finish(self, job, *args, **kwargs):
        self.finished.append((args, kwargs))
        return True


def run(monkeypatch, tmp_path, jobs, lease_seconds=30):
    monkeypatch.setattr(worker, "JobRepository", jobs)
    assert worker.run_once(NS(lease_seconds=lease_seconds, model_cache=tmp_path / "models", artifact_root=tmp_path, data_root=tmp_path), "worker_x")
    return jobs.finished


def receipt(tmp_path, finished):
    (published, checksum), _ = finished
    return FileReceiptStore(tmp_path).read(published.record_id, checksum)


def retrieval_spec(**options):
    return {"kind": "retrieval_capture", "case_id": "case_x", "page_ids": ["p2"], "adapter_version": COLPALI["adapter_version"],
            "model_revision": COLPALI["revision"], "case_sha256": "c"*64, **options}


def fake_retriever(monkeypatch):
    import docattentionlens_api.repositories.documents as documents
    from docattentionlens.retrieval import colpali
    monkeypatch.setattr(documents, "DocumentRepository", lambda settings: NS(case=lambda _: CASE,
        snapshot=lambda _: NS(pages=[NS(page_id="p1"), NS(page_id="p2")], elements=[NS(page_id="p2", element_id="e")]),
        image_bytes=lambda snapshot, page: b"image:"+page.page_id.encode()))
    calls = []
    def capture(case, inputs, cache, interrupted, **options):
        calls.append((case, inputs, options))
        record = RetrievalCapture(capture_id="capture_"+"a"*32, case_id=case.case_id, created_at=datetime.now(UTC).isoformat(), producer_version="test",
            dependencies=[case.case_id], dataset_version="test", annotation_version="test", query="q", query_positions=[], query_array="query",
            pages=[], provenance={"purpose": "unit_fixture_not_model_evidence"}, scorer={}, geometry={}, validation={})
        return record, {"query": np.array([[1, -2]], dtype=np.float32)}
    monkeypatch.setattr(colpali, "capture", capture)
    return calls


@pytest.mark.parametrize("options,expected", [
    ({}, {}),
    ({"capture_attention": True, "attention_page_id": "p2", "attention_adapter_version": COLPALI["attention_adapter_version"],
      "capture_joint_attention": True, "joint_capture_policy": "retain_document_v1", "joint_answer_source": "reference"},
     {"capture_attention": True, "attention_page_id": "p2", "capture_joint_attention": True, "joint_answer_source": "reference"})])
def test_retrieval_capture_is_stored_with_job_lineage_and_published(monkeypatch, tmp_path, options, expected):
    calls = fake_retriever(monkeypatch)
    finished = run(monkeypatch, tmp_path, Jobs("retrieval_capture", retrieval_spec(**options)))
    case, inputs, passed = calls[0]
    assert case is CASE and passed == expected
    assert [(page.page_id, raw, [e.element_id for e in elements]) for page, raw, elements in inputs] == [("p2", b"image:p2", ["e"])]
    (_, checksum), kwargs = finished[0]
    capture = kwargs["capture"]
    assert kwargs == {"dataset": None, "capture": capture, "experiment": None}
    stored = CaptureStore(tmp_path).read(capture["capture_id"], capture["content_hash"])
    assert capture == {"capture_id": "capture_"+"a"*32, "case_id": "case_x", "content_hash": capture["content_hash"]}
    assert stored.dependencies == ["case_x", "job_1"]
    assert stored.provenance == {"purpose": "unit_fixture_not_model_evidence", "job_id": "job_1", "attempt_id": "attempt_1", "input_hash": "h"}
    published = receipt(tmp_path, finished[0])
    assert published.output == {**capture, "purpose": "retrieval_execution", "candidate_pages": ["p2"], "quantity": "late_interaction_similarity"}
    assert published.provenance == {"job_id": "job_1", "attempt_id": "attempt_1", "input_hash": "h", "purpose": "retrieval_execution"}
    assert published.dependencies == ["job_1"]


@pytest.mark.parametrize("change", [{"model_revision": "other"}, {"case_sha256": "d"*64}, {"page_ids": ["p3"]}, {"page_ids": ["p2", "p2"]},
                                    {"capture_attention": True, "attention_page_id": "p2", "attention_adapter_version": "other"}])
def test_retrieval_capture_refuses_a_specification_that_differs_from_frozen_inputs(monkeypatch, tmp_path, change):
    calls = fake_retriever(monkeypatch)
    finished = run(monkeypatch, tmp_path, Jobs("retrieval_capture", {**retrieval_spec(), **change}))
    assert calls == [] and finished == [((None, None, "execution_failed"), {"error_message": None})]
    assert not (tmp_path / "retrieval").exists() and not (tmp_path / "receipts").exists()


@pytest.mark.parametrize("kind,module,function,purpose", [
    ("layout_experiment", "docattentionlens_worker.experiments", "execute_experiment", "controlled_layout_execution"),
    ("document_experiment", "docattentionlens_worker.document_studies", "execute_document_study", "controlled_document_execution"),
    ("attention_intervention", "docattentionlens_worker.attention_interventions", "execute_attention_experiment", "direct_attention_intervention")])
def test_experiment_jobs_publish_their_executor_metadata(monkeypatch, tmp_path, kind, module, function, purpose):
    jobs = Jobs(kind, {"kind": kind})
    metadata = {"experiment_id": "experiment_x", "source_case_id": "case_x", "content_hash": "e"*64, "kind": kind, "capture_refs": []}
    calls = []
    monkeypatch.setattr(f"{module}.{function}", lambda settings, repository, job, interrupted: calls.append((repository, job)) or metadata)
    finished = run(monkeypatch, tmp_path, jobs)
    assert calls == [(jobs, jobs.job)]
    assert finished[0][1] == {"dataset": None, "capture": None, "experiment": metadata}
    published = receipt(tmp_path, finished[0])
    assert published.output == {**metadata, "purpose": purpose} and published.provenance["purpose"] == purpose


def test_a_lost_lease_publishes_nothing(monkeypatch, tmp_path):
    def execute(settings, repository, job, interrupted):
        deadline = monotonic() + 5
        while not interrupted() and monotonic() < deadline:
            sleep(0.005)
        assert interrupted()
        return {"experiment_id": "experiment_x", "kind": "layout_experiment"}
    monkeypatch.setattr("docattentionlens_worker.experiments.execute_experiment", execute)
    finished = run(monkeypatch, tmp_path, Jobs("layout_experiment", {}, renewed=False), lease_seconds=0.03)
    assert finished == [] and not (tmp_path / "receipts").exists()
