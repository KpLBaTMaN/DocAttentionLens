"""Full-attention generator captures analysed as causal encodings truncated at one source.

Synthetic CPU fixture: 2 layers, 2 heads, 9 prefill positions with 4 image patches, 3 prediction
steps. The dense causal matrices are known, so results are checked against the retriever-encoding
path run on the same squares and against an independent rollout product.
"""
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
import numpy as np
import pytest

from docattentionlens.generation import full_attention as fa
from docattentionlens.generation.contracts import (ContextManifest, FullAttention, GenerationCapture,
    GenerationPosition, PredictionStep, SourceRow)
from docattentionlens.retrieval.contracts import ArrayRef, EncodingAttention, Position, RetrievalAttentionRequest
from docattentionlens.retrieval.attention import calculate as retriever_calculate
from docattentionlens.methods import _full_matrix as fm
from docattentionlens.methods.registry import get_definition, list_methods, prediction_capabilities, prepare_methods

LAYERS, HEADS, N, STEPS = 2, 2, 9, 3
ROLES = ["special", "prompt", "image", "image", "image", "image", "query", "query", "prompt"]
BOXES = {2: [0, 0, 10, 10], 3: [10, 0, 20, 10], 4: [0, 10, 10, 20], 5: [10, 10, 20, 20]}
MATRIX_METHODS = {"attention.rollout", "attention.partial_rollout", "attention.rollout_evolution", "attention.flow",
    "attention.graph", "attention.row_structure", "attention.column_structure", "attention.spatial_direction",
    "attention.local_global", "attention.vision_to_text", "attention.vision_to_vision"}


def bf16_causal(n, seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=2, size=(HEADS, n, n))
    logits[:, ~np.tri(n, dtype=bool)] = -np.inf
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return (weights.astype(np.float32).view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def ref(name, values):
    stream = BytesIO(); np.save(stream, values, allow_pickle=False); raw = stream.getvalue()
    return ArrayRef(name=name, sha256=sha256(raw).hexdigest(), shape=list(values.shape), axes=["fixture"], bytes=len(raw))


def fixture(count=STEPS):
    """Capture, its arrays and the dense (layers, heads, m, m) causal squares over m = N+count-1 positions.

    Every generated step is retained, as on full-attention captures from generation schema 0.8.0.
    """
    m = N + count - 1
    dense = np.stack([bf16_causal(m, seed) for seed in range(LAYERS)])
    arrays = {fa.array_name(layer): fa.pack(dense[layer, :, :N, :N]) for layer in range(LAYERS)}
    steps, positions = [], []
    for i, role in enumerate(ROLES):
        positions.append(GenerationPosition(index=i, token_id=100 + i, piece=f"t{i}", role=role,
            page_id="page_a" if role == "image" else None, bbox=BOXES.get(i),
            ownership={"background": 1.0} if role == "image" else {f"non_image:{role}": 1.0}))
    for k in range(count):
        positions.append(GenerationPosition(index=N + k, token_id=200 + k, piece=f"g{k}", role="generated",
            ownership={"non_image:generated": 1.0}))
        source = N + k - 1
        arrays[f"step_{k}"] = dense[:, :, source, :source + 1].copy()
        steps.append(PredictionStep(step=k, pass_kind="prefill" if k == 0 else "decode", source_index=source,
            consumed_token_id=positions[source].token_id, consumed_piece=positions[source].piece,
            predicted_index=source + 1, predicted_token_id=200 + k, predicted_piece=f"g{k}", target_count=source + 1,
            array=f"step_{k}"))
    rows = []
    for i in (6, 7):
        arrays[f"question_{i}"] = dense[:, :, i, :i + 1].copy()
        rows.append(SourceRow(source_index=i, token_id=100 + i, piece=f"t{i}", predicted_index=i + 1,
            predicted_token_id=101 + i, predicted_piece=f"t{i + 1}", target_count=i + 1, array=f"question_{i}"))
    full = FullAttention(layout=fa.LAYOUT, position_count=N, square_limit=N + count - 1,
        layer_arrays=[fa.array_name(layer) for layer in range(LAYERS)], query_head_count=HEADS,
        bytes=sum(ref(name, arrays[name]).bytes for name in map(fa.array_name, range(LAYERS))))
    capture = GenerationCapture(capture_id="generation_" + "c" * 32, case_id="fixture", created_at="2026-10-16T00:00:00Z",
        producer_version="test", dependencies=[],
        context=ContextManifest(retrieved_page_ids=[], selected_page_ids=["page_a"], omitted_page_ids=[], query="q",
            prompt="q", pages=[], input_token_ids=list(range(N)), input_attention_mask=[1] * N),
        positions=positions, output_token_ids=[200 + k for k in range(count)], answer="g", stop_reason="token_limit",
        steps=steps, source_rows=rows, layer_count=LAYERS, query_head_count=HEADS, kv_head_count=1,
        arrays=[ref(name, values) for name, values in arrays.items()], full_attention=full,
        provenance={"purpose": "unit_fixture_not_model_evidence"},
        coverage={"quantity": "post_softmax; dropout disabled", "all_causal_targets": True}, validation={})
    return capture, arrays, dense


def definition(method_id):
    method = next(m for m in list_methods() if m.method_id == method_id)
    return get_definition(method.method_id, method.variant_id)


def run(method_id, parameters=None, source_index=None, fixture_=None):
    """The backend adapter path without repositories."""
    capture, arrays, _ = fixture_ or fixture()
    item = definition(method_id)
    source, _, _, _, parsed, _ = fm.prepare(capture, item, parameters or {}, source_index)
    context, _ = fm.context(capture, source, arrays.__getitem__, fm.used_layers(parsed))
    return item.calculate(context, parsed)


def retriever(method_id, dense, source, parameters=None):
    """The same method on an equivalent causal retriever encoding built from the same squares."""
    m = source + 1
    roles = {"image": "visual_patch", "query": "query_text", "special": "prefix_or_special"}
    labels = [*ROLES, *["generated"] * max(0, m - N)]
    positions = [Position(index=i, token_id=i, piece=f"t{i}", role=roles.get(labels[i], "visual_prompt"), eligible=True,
        bbox=BOXES.get(i)) for i in range(m)]
    encoding = EncodingAttention(encoding_id="enc", pass_kind="document", page_id="page_a", layer_count=LAYERS,
        query_head_count=HEADS, position_count=m, attention_mask=[1] * m, layer_arrays=["a0", "a1"], causal=True,
        positions=positions, ownership=[{"background": 1.0} if labels[i] == "image" else {"non_page": 1.0} for i in range(m)])
    capture = SimpleNamespace(capture_id="capture_fixture", attention_encodings=[encoding], pages=[], query_positions=[])
    squares = {f"a{layer}": dense[layer, :, :m, :m] for layer in range(LAYERS)}
    return retriever_calculate(capture, RetrievalAttentionRequest(encoding_id="enc", source_index=source,
        method_id=method_id, parameters=parameters or {}), squares.__getitem__)


def edges(result):
    return [(e.source, e.target, e.layer, e.head, e.value) for e in result.graph.edges]


@pytest.mark.parametrize("step", range(STEPS))
def test_rollout_equals_retriever_path_and_an_independent_product(step):
    data = fixture(); dense = data[2]; source = N + step - 1
    result = run("attention.rollout", {"step": step, "layer": 1}, fixture_=data)
    expected = retriever("attention.rollout", dense, source, {"layer": 1})
    assert result.row.values == expected.row.values
    vector = np.eye(source + 1)[source]
    for layer in (1, 0):
        mixed = .5 * np.eye(source + 1) + .5 * dense[layer, :, :source + 1, :source + 1].astype(np.float64).mean(0)
        vector = vector @ (mixed / mixed.sum(1, keepdims=True))
    assert np.allclose(result.row.values, vector, atol=1e-12)
    assert result.row.source_index == source and result.row.target_count == source + 1


def test_graph_flow_and_partial_rollouts_match_retriever_path():
    data = fixture(); dense = data[2]
    graph = run("attention.graph", {"step": 2, "layers": [0, 1], "threshold": .05}, fixture_=data)
    assert edges(graph) == edges(retriever("attention.graph", dense, N + 1, {"layers": [0, 1], "threshold": .05}))
    assert graph.source_indices == list(range(N + 2))
    partial = {"start_layer": 1, "layer": 1, "heads": [1], "head_reduction": "max"}
    assert (run("attention.partial_rollout", {"step": 1, **partial}, fixture_=data).row.values
            == retriever("attention.partial_rollout", dense, N, partial).row.values)
    evolution = {"start_layer": 0, "layers": [0, 1], "residual_coefficient": .25}
    ours = run("attention.rollout_evolution", {"step": 2, **evolution}, fixture_=data)
    theirs = retriever("attention.rollout_evolution", dense, N + 1, evolution)
    assert [c.row.values for c in ours.comparisons] == [c.row.values for c in theirs.comparisons]
    assert run("attention.flow", {"layer": 1}, fixture_=data).table.values == retriever("attention.flow", dense, N - 1, {"layer": 1}).table.values


def test_spatial_structure_on_image_patch_sources_matches_retriever_path():
    data = fixture(); dense = data[2]
    for method in ("attention.row_structure", "attention.local_global"):
        ours = run(method, {"layer": 1, "head": 1}, source_index=5, fixture_=data)
        assert ours.table.values == retriever(method, dense, 5, {"layer": 1, "head": 1}).table.values
        assert ours.source_indices == [2, 3, 4, 5]  # every mapped patch up to the source; later patches are outside the view


def test_vision_methods_read_image_patch_rows_in_the_prefill_square():
    data = fixture(); dense = data[2]
    text = run("attention.vision_to_text", {"layer": 1, "head": 1}, source_index=4, fixture_=data)
    vision = run("attention.vision_to_vision", {"layer": 1, "head": 1}, source_index=4, fixture_=data)
    row = dense[1, 1, 4, :5]
    assert text.row.values == pytest.approx([row[0], row[1], 0, 0, 0], abs=0)
    assert vision.row.values == pytest.approx([0, 0, *row[2:5]], abs=0)
    assert text.statistics[0].value + vision.statistics[0].value == pytest.approx(float(row.sum()))
    with pytest.raises(ValueError, match="image-patch source"):
        run("attention.vision_to_text", source_index=6, fixture_=data)
    scope = definition("attention.vision_to_text").descriptor.limitations
    assert any("never to the question or the answer" in note for note in scope)


def test_sources_without_saved_rows_are_rejected():
    data = fixture()
    with pytest.raises(ValueError, match="positions 0–10"):
        run("attention.graph", source_index=N + STEPS - 1, fixture_=data)
    with pytest.raises(ValueError, match="prediction steps"):
        run("attention.rollout", {"step": STEPS}, fixture_=data)
    capture, arrays, _ = data
    shifted = capture.model_copy(update={"steps": [*capture.steps[:2], capture.steps[2].model_copy(update={"source_index": N})]})
    with pytest.raises(ValueError, match="not aligned"):
        fm.select(shifted, 0)
    with pytest.raises(ValueError, match="Save full attention matrices"):
        fm.select(capture.model_copy(update={"full_attention": None}), 0)


def test_source_rows_and_squares_are_the_saved_rows_exactly():
    capture, arrays, dense = fixture()
    for source in range(N + STEPS - 1):
        reader = fm.Reader(capture, source, arrays.__getitem__)
        for layer in range(LAYERS):
            square = reader.layer_matrix(layer)
            assert np.array_equal(square, dense[layer, :, :source + 1, :source + 1])
            assert np.array_equal(reader.source_row(layer), square[:, source])
    # Retained question and prediction rows equal the rebuilt rows (checked at capture as well).
    assert np.array_equal(fm.Reader(capture, 6, arrays.__getitem__).source_row(0), arrays["question_6"][0])
    assert fm.Reader(capture, N + 1, arrays.__getitem__).steps == capture.steps[1:3]
    assert fm.identity(capture, N + 1)["decode_arrays"].keys() == {"step_1", "step_2"}
    assert fm.identity(capture, 4)["decode_arrays"] == {}


LONG = 12  # an answer longer than the eight steps ordinary captures keep


def test_late_steps_of_an_all_step_capture_match_the_dense_reference():
    data = fixture(LONG); capture, arrays, dense = data
    assert len(capture.steps) == LONG == len(capture.output_token_ids) and capture.full_attention.square_limit == N + LONG - 1
    source = N + LONG - 2  # step 11 consumes the 11th generated token
    result = run("attention.rollout", {"step": LONG - 1, "layer": 1}, fixture_=data)
    assert result.row.values == retriever("attention.rollout", dense, source, {"layer": 1}).row.values
    vector = np.eye(source + 1)[source]
    for layer in (1, 0):
        mixed = .5 * np.eye(source + 1) + .5 * dense[layer, :, :source + 1, :source + 1].astype(np.float64).mean(0)
        vector = vector @ (mixed / mixed.sum(1, keepdims=True))
    assert np.allclose(result.row.values, vector, atol=1e-12)
    assert (result.row.source_index, result.row.target_count) == (source, source + 1)
    graph = run("attention.graph", {"step": LONG - 1, "layers": [0, 1], "threshold": .05}, fixture_=data)
    assert edges(graph) == edges(retriever("attention.graph", dense, source, {"layers": [0, 1], "threshold": .05}))
    # Squares at every retained source are the dense matrices; decode rows come from steps 1..k.
    for source in (N + 6, N + 7, N + LONG - 2):
        reader = fm.Reader(capture, source, arrays.__getitem__)
        assert reader.steps == capture.steps[1:source - N + 2]
        assert np.array_equal(reader.layer_matrix(0), dense[0, :, :source + 1, :source + 1])
    with pytest.raises(ValueError, match=f"positions 0–{N + LONG - 2}"):
        run("attention.graph", source_index=N + LONG - 1, fixture_=data)
    bounds = next(p for p in prepare_methods(prediction_capabilities(capture), capture) if p.method.method_id == "attention.rollout").control_bounds
    assert bounds["steps"] == list(range(LONG)) and bounds["full_attention"]["square_limit"] == N + LONG - 1


def test_saved_parameters_keep_the_selected_step():
    capture, _, _ = fixture()
    source, predicted, array, record, parsed, saved = fm.prepare(capture, definition("attention.partial_rollout"),
        {"step": 2, "layer": 1, "start_layer": 1})
    assert (source, predicted, array, parsed.step, saved["step"], saved["start_layer"]) == (N + 1, N + 2, "step_2", 0, 2, 1)
    assert len(record.positions) == N + 2 and record.attention_mask == [1] * (N + 2)
    _, _, array, *_ = fm.prepare(capture, definition("attention.vision_to_text"), {"step": 2}, source_index=3)
    assert array is None


def test_capabilities_with_and_without_full_attention():
    capture, _, _ = fixture()
    with_full = prediction_capabilities(capture)
    assert {"full_square_attention_matrices", "captured_image_sources", "aligned_layer_axes", "residual_policy",
            "masks", "eligible_mask", "complete_declared_support", "multiple_source_rows"} <= with_full
    available = {p.method.method_id for p in prepare_methods(with_full, capture) if p.available}
    assert MATRIX_METHODS <= available
    bounds = next(p for p in prepare_methods(with_full, capture) if p.method.method_id == "attention.graph").control_bounds
    assert bounds["full_attention"] == {"position_count": N, "square_limit": N + STEPS - 1}
    plain = capture.model_copy(update={"full_attention": None})
    without = prediction_capabilities(plain)
    assert not {"full_square_attention_matrices", "captured_image_sources", "aligned_layer_axes"} & without
    prepared = {p.method.method_id: p for p in prepare_methods(without, plain)}
    assert not any(prepared[m].available for m in MATRIX_METHODS)
    assert any("Save full attention matrices" in r for r in prepared["attention.graph"].reasons)
    assert fm.applies(prepared["attention.graph"].method, capture) and not fm.applies(prepared["attention.graph"].method, plain)
    assert not fm.applies(prepared["attention.row"].method, capture)
