"""Full-attention generator analyses through the API service, with a real generation store on disk."""
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import pytest
from fastapi import HTTPException

import docattentionlens_api.services.analyses as analyses
from docattentionlens.datasets.adapter import canonical
from docattentionlens.methods import _full_matrix as fm
from docattentionlens.methods.contracts import AnalysisRequest
from docattentionlens_api.repositories.generations import GenerationStore

_spec = spec_from_file_location("full_matrix_fixture", Path(__file__).parents[2] / "library/tests/methods/test_full_matrix.py")
fixtures = module_from_spec(_spec); _spec.loader.exec_module(fixtures)


class Saved:
    def __init__(self):
        self.found = []

    def find(self, identifier):
        self.found.append(identifier)

    def put(self, result):
        return result


@pytest.fixture
def service(tmp_path, monkeypatch):
    capture, arrays, dense = fixtures.fixture()
    store = GenerationStore(tmp_path)
    digest = store.put(capture.model_copy(update={"arrays": []}), arrays)
    record = store.read(capture.capture_id, digest)
    saved = Saved()
    state = SimpleNamespace(record=record, saved=saved, dense=dense)
    monkeypatch.setattr(analyses, "GenerationRepository", lambda jobs: SimpleNamespace(get=lambda _: state.record, store=store))
    monkeypatch.setattr(analyses, "AnalysisRepository", lambda jobs: saved)
    monkeypatch.setattr(analyses, "calculation_sha256", lambda: "fixture")
    return state


def request(method, parameters=None, **extra):
    variant = next(m.variant_id for m in fixtures.list_methods() if m.method_id == method)
    return AnalysisRequest(generation_capture_id="generation_" + "c" * 32, method_id=method, variant_id=variant,
                           parameters=parameters or {}, **extra)


def test_prediction_step_graph_keeps_generator_identity(service):
    result = analyses.calculate(None, request("attention.graph", {"step": 2, "layers": [0, 1], "threshold": .05}))
    refs = {a.name: a.sha256 for a in service.record.arrays}
    assert (result.source_index, result.predicted_index, result.target_count) == (10, 11, 11)
    assert result.array_sha256 == refs["step_2"] and result.capture_sha256 == service.record.content_sha256
    assert result.parameters["step"] == 2 and result.valid_target_indices == list(range(11))
    assert result.coverage["full_attention"] == {"layout": fm.fa.LAYOUT, "position_count": 9, "square_limit": 11, "source_index": 10}
    assert any("truncated at source position 10" in n for n in result.notices)
    assert result.graph.edges and fixtures.edges(result) == fixtures.edges(fixtures.retriever("attention.graph", service.dense, 10,
                                                                                               {"layers": [0, 1], "threshold": .05}))


def test_image_patch_source_runs_vision_methods_with_scope_notice(service):
    result = analyses.calculate(None, request("attention.vision_to_vision", {"layer": 1, "head": 0}, source_index=4))
    assert result.source_index == 4 and result.target_count == 5
    assert result.array_sha256 == sha256(canonical(fm.identity(service.record, 4))).hexdigest()
    assert fm.IMAGE_SCOPE in result.notices
    rollout = analyses.calculate(None, request("attention.rollout", {"layer": 1}, source_index=6))
    # A retained question row keeps its own array identity.
    assert rollout.array_sha256 == next(a.sha256 for a in service.record.arrays if a.name == "question_6")


def test_rows_beyond_the_square_limit_and_multi_capture_requests_are_rejected(service):
    with pytest.raises(HTTPException) as error:
        analyses.calculate(None, request("attention.graph", source_index=11))
    assert error.value.status_code == 422 and "positions 0–10" in error.value.detail["message"]
    with pytest.raises(HTTPException) as error:
        analyses.calculate(None, request("attention.rollout", comparison_capture_id="generation_" + "d" * 32))
    assert error.value.status_code == 422 and "one capture" in error.value.detail["message"]


def test_full_array_hashes_enter_the_analysis_identity(service):
    original = service.record
    def identifier(record):
        service.record = record
        service.saved.found.clear()
        # Return a cached sentinel so only the identity is computed.
        service.saved.find = lambda i: (service.saved.found.append(i), "cached")[1]
        assert analyses.calculate(None, request("attention.rollout", {"step": 1, "layer": 1})) == "cached"
        return service.saved.found[-1]
    first = identifier(original)
    changed = [a.model_copy(update={"sha256": "0" * 64}) if a.name == "full_l1" else a for a in original.arrays]
    assert identifier(original.model_copy(update={"arrays": changed})) != first
    assert identifier(original) == first


def test_capture_without_full_attention_keeps_matrix_methods_unavailable(service):
    service.record = service.record.model_copy(update={"full_attention": None})
    with pytest.raises(HTTPException) as error:
        analyses.calculate(None, request("attention.rollout", {"layer": 1}))
    assert error.value.detail["code"] == "unsupported_capture" and "full_square_attention_matrices" in error.value.detail["message"]


def test_any_prompt_row_runs_row_methods_on_a_full_attention_capture(service):
    # Position 1 is a template/prompt token without a retained question row: its row comes from the triangle.
    assert not any(r.source_index == 1 for r in service.record.source_rows)
    result = analyses.calculate(None, request("attention.row", {"layer": 1, "head": 0}, source_index=1))
    assert (result.source_index, result.predicted_index, result.target_count) == (1, 2, 2)
    assert list(result.row.values[:2]) == list(service.dense[1, 0, 1, :2])
    assert result.array_sha256 == service_store_sha(service, "prefill_row_1")
    # A retained question row keeps reading its own array.
    retained = analyses.calculate(None, request("attention.row", {"layer": 1, "head": 0}, source_index=6))
    assert retained.array_sha256 == next(a.sha256 for a in service.record.arrays if a.name == "question_6")
    # Decode positions are prediction steps, not prompt rows.
    with pytest.raises(HTTPException) as error:
        analyses.calculate(None, request("attention.row", {"layer": 0, "head": 0}, source_index=9))
    assert error.value.status_code == 422


def service_store_sha(service, name):
    return GenerationStore.source_sha256(service.record, name)


@pytest.fixture
def long_service(tmp_path, monkeypatch):
    """A 12-token answer with every step retained (full-attention captures from generation schema 0.8.0)."""
    capture, arrays, dense = fixtures.fixture(12)
    store = GenerationStore(tmp_path)
    record = store.read(capture.capture_id, store.put(capture.model_copy(update={"arrays": []}), arrays))
    state = SimpleNamespace(record=record, saved=Saved(), dense=dense, store=store)
    monkeypatch.setattr(analyses, "GenerationRepository", lambda jobs: SimpleNamespace(get=lambda _: state.record, store=store))
    monkeypatch.setattr(analyses, "AnalysisRepository", lambda jobs: state.saved)
    monkeypatch.setattr(analyses, "calculation_sha256", lambda: "fixture")
    return state


def test_late_prediction_steps_run_matrix_and_row_methods(long_service):
    source = 9 + 10  # step 11 consumes position n+10
    rollout = analyses.calculate(None, request("attention.rollout", {"step": 11, "layer": 1}))
    assert (rollout.source_index, rollout.predicted_index, rollout.target_count) == (source, source + 1, source + 1)
    assert rollout.row.values == fixtures.retriever("attention.rollout", long_service.dense, source, {"layer": 1}).row.values
    assert rollout.array_sha256 == next(a.sha256 for a in long_service.record.arrays if a.name == "step_11")
    assert rollout.coverage["full_attention"]["square_limit"] == 20 and rollout.parameters["step"] == 11
    assert set(fm.identity(long_service.record, source)["decode_arrays"]) == {f"step_{k}" for k in range(1, 12)}
    # The ordinary row path reads step 11's own saved row.
    row = analyses.calculate(None, request("attention.row", {"step": 11, "layer": 0, "head": 1}))
    assert (row.source_index, row.target_count) == (source, source + 1)
    assert list(row.row.values) == list(long_service.dense[0, 1, source, :source + 1])
    with pytest.raises(HTTPException) as error:
        analyses.calculate(None, request("attention.row", {"step": 12, "layer": 0, "head": 0}))
    assert error.value.status_code == 422


def test_row_and_received_endpoints_accept_late_steps(long_service, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import docattentionlens_api.routers.generation as generation
    monkeypatch.setattr(generation, "GenerationRepository", lambda jobs: SimpleNamespace(get=lambda _: long_service.record, store=long_service.store))
    app = FastAPI(); app.include_router(generation.router); app.state.repository = None
    identifier = long_service.record.capture_id
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(f"/api/v1/generation/captures/{identifier}/rows", params={"step": 11, "layer": 1, "head": 0})
        assert response.status_code == 200, response.text
        body = response.json()
        assert (body["step"], body["source_index"], body["target_count"]) == (11, 19, 20)
        assert body["values"] == [float(v) for v in long_service.dense[1, 0, 19, :20]]
        assert client.get(f"/api/v1/generation/captures/{identifier}/rows", params={"step": 12}).json()["detail"]["code"] == "uncaptured_selection"
        assert client.get(f"/api/v1/generation/captures/{identifier}/rows", params={"step": 64}).status_code == 422
        # What attends to generated position 15: the rows of steps 7..11 (sources 15..19), and none before.
        received = client.get(f"/api/v1/generation/captures/{identifier}/received", params={"target_index": 15, "layer": 1, "head": 1})
        assert received.status_code == 200, received.text
        sources = received.json()["sources"]
        assert [s["source_index"] for s in sources] == [15, 16, 17, 18, 19]
        assert [s["value"] for s in sources] == [float(long_service.dense[1, 1, i, 15]) for i in range(15, 20)]
