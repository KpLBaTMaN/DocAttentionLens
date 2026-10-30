"""Full-attention generator captures: lossless uint16 storage, request plumbing and worker hand-off."""
from datetime import UTC, datetime
from types import SimpleNamespace as NS
import json
import re
from hashlib import sha256
import numpy as np
import pytest

from docattentionlens_api.repositories.generations import GenerationStore
from docattentionlens.generation import full_attention as fa
from docattentionlens.generation.contracts import (MAX_NEW_TOKENS, QWEN25_VL_3B, SYSTEM_PROMPT, ContextManifest,
    GenerationCapture, GenerationPosition, GenerationRequest, PredictionStep)
from docattentionlens.generation.qwen import full_attention_record

N, HEADS, LAYERS = 5, 2, 2


def square(seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(HEADS, N, N)).astype(np.float32)
    logits[:, ~np.tri(N, dtype=bool)] = -np.inf
    weights = np.exp(logits - logits.max(-1, keepdims=True)); weights /= weights.sum(-1, keepdims=True)
    return (weights.astype(np.float32).view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def capture(identifier, arrays, full=True, count=1, schema_version="0.8.0", outputs=None):
    steps = [PredictionStep(step=k, pass_kind="prefill" if k == 0 else "decode", source_index=N+k-1, consumed_token_id=0, consumed_piece="x",
        predicted_index=N+k, predicted_token_id=0, predicted_piece="x", target_count=N+k, array=f"step_{k}") for k in range(count)]
    return GenerationCapture(schema_version=schema_version, capture_id="generation_"+identifier*32, case_id="storage_fixture",
        created_at=datetime.now(UTC).isoformat(),
        producer_version="test", dependencies=[], context=ContextManifest(retrieved_page_ids=[], selected_page_ids=[], omitted_page_ids=[],
            query="q", prompt="p", pages=[], input_token_ids=list(range(N)), input_attention_mask=[1]*N),
        positions=[GenerationPosition(index=i, token_id=i, piece="x", role="prompt" if i < N else "generated",
            ownership={"non_image:prompt" if i < N else "non_image:generated": 1.0}) for i in range(N + len(outputs or range(count)))],
        output_token_ids=list(range(count)) if outputs is None else outputs, answer="a", stop_reason="eos", steps=steps,
        layer_count=LAYERS, query_head_count=HEADS,
        kv_head_count=1, full_attention=full_attention_record(arrays, N, LAYERS, HEADS, len(steps)) if full else None,
        provenance={"purpose": "unit_fixture_not_model_evidence"}, coverage={}, validation={})


def payload():
    squares = [square(seed) for seed in range(LAYERS)]
    return squares, {"step_0": np.stack([s[:, -1, :] for s in squares]), **{fa.array_name(l): fa.pack(s) for l, s in enumerate(squares)}}


def decode_rows(count, seed=7):
    """Normalized (layers, heads, N+k) rows for prediction steps 1..count-1."""
    rng = np.random.default_rng(seed)
    rows = {}
    for k in range(1, count):
        weights = rng.random((LAYERS, HEADS, N + k)) + .1
        rows[f"step_{k}"] = (weights / weights.sum(-1, keepdims=True)).astype(np.float32)
    return rows


def test_all_step_full_capture_stores_every_generated_steps_row(tmp_path):
    store = GenerationStore(tmp_path)
    _, arrays = payload()
    arrays = {**arrays, **decode_rows(12)}
    record = capture("b", arrays, count=12)
    reopened = store.read(record.capture_id, store.put(record, arrays))
    assert [s.step for s in reopened.steps] == list(range(12)) and reopened.full_attention.square_limit == N + 11
    refs = {r.name: r for r in reopened.arrays}
    for k in range(12):
        assert refs[f"step_{k}"].shape == [LAYERS, HEADS, N + k] and refs[f"step_{k}"].axes == ["layer", "query_head", "causal_target_position"]
        assert np.array_equal(store.array(reopened, f"step_{k}"), arrays[f"step_{k}"])
    # Names follow the retained steps: step_0..step_63, never beyond MAX_NEW_TOKENS - 1.
    assert all(re.fullmatch(GenerationStore.array_pattern, f"step_{k}") for k in range(MAX_NEW_TOKENS))
    assert not any(re.fullmatch(GenerationStore.array_pattern, name) for name in ("step_64", "step_08", "step_100"))
    # A full capture must store every declared step's row, on its declared axes.
    with pytest.raises(ValueError, match="prediction-step rows"):
        store.put(capture("c", arrays, count=12), {k: v for k, v in arrays.items() if k != "step_11"})
    with pytest.raises(ValueError, match="prediction-step rows"):
        store.put(capture("c", arrays, count=12), {**arrays, "step_5": arrays["step_6"]})
    assert not store.path("generation_" + "c" * 32).exists()


def test_earlier_eight_step_full_attention_manifests_are_still_readable(tmp_path):
    store = GenerationStore(tmp_path)
    _, arrays = payload()
    arrays = {**arrays, **decode_rows(8)}
    record = capture("a", arrays, count=8, schema_version="0.7.0", outputs=list(range(36)))
    reopened = store.read(record.capture_id, store.put(record, arrays))
    assert reopened.schema_version == "0.7.0" and len(reopened.steps) == 8 < len(reopened.output_token_ids)
    assert reopened.full_attention.square_limit == N + 7


def test_uint16_full_attention_arrays_store_losslessly_with_triangle_axes(tmp_path):
    store = GenerationStore(tmp_path)
    squares, arrays = payload()
    record = capture("d", arrays)
    reopened = store.read(record.capture_id, store.put(record, arrays))
    refs = {r.name: r for r in reopened.arrays}
    assert refs["full_l0"].axes == ["query_head", "causal_triangle"] and refs["full_l1"].shape == [HEADS, fa.triangle_length(N)]
    assert refs["step_0"].axes == ["layer", "query_head", "causal_target_position"]
    # The manifest records the stored dtype: bit patterns are uint16, everything else float32.
    assert refs["full_l0"].dtype == "uint16" and refs["step_0"].dtype == "float32"
    assert reopened.full_attention.bytes == refs["full_l0"].bytes + refs["full_l1"].bytes
    for layer, expected in enumerate(squares):
        values = store.array(reopened, fa.array_name(layer))
        assert values.dtype == np.uint16 and np.array_equal(values, arrays[fa.array_name(layer)])
        assert np.array_equal(fa.unpack(values, N), expected)
    assert store.array(reopened, "step_0").dtype == np.float32
    # Hash and length checks still guard the bit patterns.
    (store.path(record.capture_id)/"full_l1.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        store.array(reopened, "full_l1")


def test_uint16_is_refused_for_every_other_array_and_floats_for_full_arrays(tmp_path):
    store = GenerationStore(tmp_path)
    _, arrays = payload()
    with pytest.raises(ValueError, match="uint16"):
        store.put(capture("e", arrays, full=False), {"step_0": arrays["full_l0"]})
    with pytest.raises(ValueError, match="uint16"):
        store.put(capture("e", arrays, full=False), {"pixels_0": np.zeros((4, 4), np.uint16)})
    as_float = {**arrays, "full_l0": arrays["full_l0"].astype(np.float32)}
    with pytest.raises(ValueError):
        store.put(capture("e", arrays), as_float)
    for name in ("full_l01", "full_lx", "full_l1000"):
        with pytest.raises(ValueError, match="Invalid array name"):
            store.put(capture("e", arrays, full=False), {name: arrays["full_l0"]})
    assert not store.path("generation_"+"e"*32).exists()


def test_full_arrays_must_match_their_declaration(tmp_path):
    store = GenerationStore(tmp_path)
    _, arrays = payload()
    with pytest.raises(ValueError, match="declaration"):
        store.put(capture("f", arrays, full=False), arrays)
    missing = {k: v for k, v in arrays.items() if k != "full_l1"}
    with pytest.raises(ValueError, match="declaration"):
        store.put(capture("f", arrays), missing)
    with pytest.raises(ValueError, match="declaration"):
        store.put(capture("f", arrays), {**arrays, "full_l1": arrays["full_l1"][:, :-1]})
    wrong_bytes = capture("f", arrays)
    wrong_bytes = wrong_bytes.model_copy(update={"full_attention": wrong_bytes.full_attention.model_copy(update={"bytes": 1})})
    with pytest.raises(ValueError, match="stored bytes"):
        store.put(wrong_bytes, arrays)
    assert not (store.path(wrong_bytes.capture_id)/"manifest.json").exists()


def test_older_float_manifests_are_still_readable(tmp_path):
    store = GenerationStore(tmp_path)
    _, arrays = payload()
    record = capture("a", arrays, full=False).model_copy(update={"schema_version": "0.6.0"})
    reopened = store.read(record.capture_id, store.put(record, {"step_0": arrays["step_0"]}))
    assert reopened.schema_version == "0.6.0" and reopened.full_attention is None


def input_hash(spec):
    return sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def test_full_attention_request_changes_the_job_specification_and_hash_only_when_requested():
    from docattentionlens_api.routers.generation import job_specification
    case = NS(case_id="case_x", content_sha256="c"*64)
    checkpoint = {"revision": QWEN25_VL_3B.revision}
    base = {"case_id": "case_x", "page_ids": ["p1"], "idempotency_key": "k"}
    default = job_specification(GenerationRequest.model_validate(base), case, None, QWEN25_VL_3B, checkpoint)
    full = job_specification(GenerationRequest.model_validate({**base, "capture_full_attention": True}), case, None, QWEN25_VL_3B, checkpoint)
    # Default requests keep their earlier specification, so replay hashes and idempotency are unchanged.
    assert "capture_full_attention" not in default and default == {"kind": "generation_capture", "case_id": "case_x", "case_sha256": "c"*64,
        "retrieval_capture_id": None, "retrieval_sha256": None, "page_ids": ["p1"], "adapter_version": QWEN25_VL_3B.adapter_version,
        "model_revision": QWEN25_VL_3B.revision}
    # Full-attention jobs keep every generated step; the marker separates them from the earlier eight-step jobs.
    earlier = {**default, "capture_full_attention": True}
    assert full == {**earlier, "full_attention_steps": "all"}
    assert len({input_hash(default), input_hash(earlier), input_hash(full)}) == 3


def test_a_new_all_step_request_with_a_fresh_key_queues_a_new_job_and_an_old_key_conflicts(monkeypatch, tmp_path):
    """Jobs are reused only by idempotency key, and a key is bound to its input hash."""
    from unittest.mock import Mock
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import docattentionlens_api.routers.generation as generation
    from docattentionlens_api.repositories.jobs import ConflictError
    from docattentionlens_api.services.imports import ModelStore
    earlier_key = "full-attention-run"
    case = NS(case_id="case_x", content_sha256="c"*64, page_ids=["p1"])
    checkpoint = {"role": "generation", "registration_id": "builtin", "revision": QWEN25_VL_3B.revision, "capabilities": {"can_run": True}}
    earlier = job_specification_for(case, checkpoint, {"case_id": "case_x", "page_ids": ["p1"], "idempotency_key": earlier_key})
    earlier.pop("full_attention_steps")
    submitted = {earlier_key: input_hash(earlier)}
    def existing(key, spec):
        if key in submitted and submitted[key] != input_hash(spec):
            raise ConflictError("different input")
        return {"job_id": "job_old", "state": "succeeded"} if key in submitted else None
    jobs = NS(existing_submission=existing, submit=Mock(return_value={"job_id": "job_new", "state": "queued"}))
    monkeypatch.setattr(ModelStore, "get", lambda *_: checkpoint)
    monkeypatch.setattr(generation, "generation_plan", lambda *_: NS(available=True))
    app = FastAPI(); app.include_router(generation.router)
    app.state.repository = jobs; app.state.settings = NS(artifact_root=tmp_path); app.state.documents = NS(case=lambda _: case)
    body = {"case_id": "case_x", "page_ids": ["p1"], "capture_full_attention": True}
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/generation/runs", json={**body, "idempotency_key": "full-attention-run-all-steps"})
        assert response.status_code == 202 and response.json()["job_id"] == "job_new"
        assert jobs.submit.call_args.args[1]["full_attention_steps"] == "all"
        # The earlier eight-step job's key now names a different input specification.
        assert client.post("/api/v1/generation/runs", json={**body, "idempotency_key": earlier_key}).status_code == 409
    assert jobs.submit.call_count == 1


def job_specification_for(case, checkpoint, body):
    from docattentionlens_api.routers.generation import job_specification
    return job_specification(GenerationRequest.model_validate({**body, "capture_full_attention": True}), case, None, QWEN25_VL_3B, checkpoint)


class Jobs:
    def __init__(self, specification):
        self.job = {"job_id": "job_1", "attempt_id": "attempt_1", "input_hash": "h", "kind": "generation_capture", "specification": specification}
        self.finished = []

    def __call__(self, settings):
        return self

    def heartbeat(self, worker_id):
        pass

    def claim(self, worker_id):
        return self.job

    def renew(self, job_id, attempt_id):
        return True

    def finish(self, job, *args, **kwargs):
        self.finished.append((args, kwargs))


ALL_STEPS = {"capture_full_attention": True, "full_attention_steps": "all"}


@pytest.mark.parametrize("option,expected", [({}, False), (ALL_STEPS, True), ({"capture_full_attention": "yes"}, None),
                                             ({"full_attention_steps": "all"}, None)])
def test_worker_passes_the_option_to_the_generator_capture(monkeypatch, tmp_path, option, expected):
    import docattentionlens_worker.main as worker
    import docattentionlens_api.repositories.documents as documents
    import docattentionlens_api.services.imports as imports
    from docattentionlens.generation import qwen
    case = NS(case_id="case_x", content_sha256="c"*64, dataset_id="d", page_ids=["p1"])
    jobs = Jobs({"kind": "generation_capture", "case_id": "case_x", "case_sha256": "c"*64, "retrieval_capture_id": None,
        "retrieval_sha256": None, "page_ids": ["p1"], "adapter_version": QWEN25_VL_3B.adapter_version, "model_revision": QWEN25_VL_3B.revision, **option})
    monkeypatch.setattr(worker, "JobRepository", jobs)
    monkeypatch.setattr(documents, "DocumentRepository", lambda settings: NS(case=lambda _: case,
        snapshot=lambda _: NS(pages=[NS(page_id="p1")], elements=[]), image_bytes=lambda snapshot, page: b""))
    monkeypatch.setattr(imports, "ModelStore", lambda root: NS(get=lambda _: {"registration_id": "builtin", "revision": QWEN25_VL_3B.revision}))
    calls = []
    def generate(*args, **kwargs):
        calls.append(kwargs)
        raise ValueError("Full attention triangle differs from the retained prefill rows")
    monkeypatch.setattr(qwen, "capture", generate)
    settings = NS(lease_seconds=30, model_cache=tmp_path, artifact_root=tmp_path)
    assert worker.run_once(settings, "worker_x")
    if expected is None:
        assert calls == []
    else:
        assert calls[0]["full_attention"] is expected and calls[0]["system_prompt"] == SYSTEM_PROMPT
        assert jobs.finished[0][1]["error_message"].startswith("Full attention capture failed")
    assert jobs.finished[0][0][2] == "execution_failed"


def worker_job(monkeypatch, tmp_path, option, generate):
    import docattentionlens_worker.main as worker
    import docattentionlens_api.repositories.documents as documents
    import docattentionlens_api.services.imports as imports
    from docattentionlens.generation import qwen
    case = NS(case_id="case_x", content_sha256="c"*64, dataset_id="d", page_ids=["p1"])
    jobs = Jobs({"kind": "generation_capture", "case_id": "case_x", "case_sha256": "c"*64, "retrieval_capture_id": None,
        "retrieval_sha256": None, "page_ids": ["p1"], "adapter_version": QWEN25_VL_3B.adapter_version, "model_revision": QWEN25_VL_3B.revision, **option})
    monkeypatch.setattr(worker, "JobRepository", jobs)
    monkeypatch.setattr(documents, "DocumentRepository", lambda settings: NS(case=lambda _: case,
        snapshot=lambda _: NS(pages=[NS(page_id="p1")], elements=[]), image_bytes=lambda snapshot, page: b""))
    monkeypatch.setattr(imports, "ModelStore", lambda root: NS(get=lambda _: {"registration_id": "builtin", "revision": QWEN25_VL_3B.revision}))
    monkeypatch.setattr(qwen, "capture", generate)
    assert worker.run_once(NS(lease_seconds=30, model_cache=tmp_path, artifact_root=tmp_path), "worker_x")
    return jobs


def test_worker_refuses_an_earlier_eight_step_full_attention_specification(monkeypatch, tmp_path):
    calls = []
    jobs = worker_job(monkeypatch, tmp_path, {"capture_full_attention": True}, lambda *a, **k: calls.append(k))
    assert calls == [] and jobs.finished[0][0][2] == "execution_failed"
    assert "before every answer step was kept" in jobs.finished[0][1]["error_message"]


def test_worker_fails_a_full_attention_capture_without_every_step(monkeypatch, tmp_path):
    _, arrays = payload()
    arrays = {**arrays, **decode_rows(8)}
    # A capture that kept eight of twelve steps (valid only as an earlier 0.7.0 manifest) is not published.
    partial = capture("f", arrays, count=8, schema_version="0.7.0", outputs=list(range(12)))
    jobs = worker_job(monkeypatch, tmp_path, ALL_STEPS, lambda *a, **k: (partial, arrays))
    assert jobs.finished[0][0][2] == "execution_failed"
    assert jobs.finished[0][1]["error_message"] == ("Full attention capture failed: not every generated answer token's attention row "
                                                    "was kept; no attention was published.")
    assert not GenerationStore(tmp_path).path(partial.capture_id).exists()


def test_worker_publishes_an_all_step_full_attention_capture(monkeypatch, tmp_path):
    _, arrays = payload()
    arrays = {**arrays, **decode_rows(12)}
    record = capture("e", arrays, count=12)
    jobs = worker_job(monkeypatch, tmp_path, ALL_STEPS, lambda *a, **k: (record, arrays))
    args, kwargs = jobs.finished[0]
    assert kwargs["capture"]["capture_id"] == record.capture_id and len(args) == 2
    store = GenerationStore(tmp_path)
    stored = store.read(record.capture_id, kwargs["capture"]["content_hash"])
    assert [s.step for s in stored.steps] == list(range(12)) and stored.provenance["job_id"] == "job_1"


def test_plan_and_model_limits_name_the_full_attention_step_count():
    from docattentionlens_api.routers.generation import GenerationPlan
    from docattentionlens_api.services.model_catalogue import builtin_models
    plan = GenerationPlan(available=True, reason="fixture")
    assert (plan.captured_steps, plan.full_attention_captured_steps, plan.max_new_tokens) == (8, MAX_NEW_TOKENS, MAX_NEW_TOKENS)
    generators = [m for m in builtin_models() if m["role"] == "generation"]
    assert generators and all(m["capabilities"]["limits"]["captured_steps"] == 8
                              and m["capabilities"]["limits"]["full_attention_captured_steps"] == MAX_NEW_TOKENS for m in generators)
