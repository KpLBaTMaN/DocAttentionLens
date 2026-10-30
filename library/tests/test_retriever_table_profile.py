"""Element profiles and table-cell allocation on retriever encodings; known-answer fixtures only.

A causal joint encoding of one 4 × 2 page of unit image positions, then a prompt token and the source.
The table fixture is the one used for generator table methods: 2 rows × 3 unit cells over x 0–3.
"""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from docattentionlens.retrieval.contracts import Position, EncodingAttention, RetrievalAttentionRequest
from docattentionlens.retrieval.attention import calculate, methods
from docattentionlens.methods.contracts import TableInputs
from methods.test_table_methods import table_record, grounding, cell_id

N, LAYERS, HEADS, SOURCE = 10, 3, 2, 9


def matrices(seed=0):
    """Causal row-stochastic (heads, n, n) layers with every key at or before the source."""
    rng = np.random.default_rng(seed)
    out = {}
    for layer in range(LAYERS):
        m = rng.random((HEADS, N, N)) * np.tril(np.ones((N, N)))
        out[f"l{layer}"] = (m / m.sum(-1, keepdims=True)).astype(np.float32)
    return out


def encoding_capture(ownership=None):
    positions = [Position(index=i, token_id=i, piece="<|image_pad|>", role="visual_patch", eligible=True,
                          bbox=[i % 4, i//4, i % 4+1, i//4+1]) for i in range(8)]
    positions += [Position(index=8, token_id=8, piece="Total", role="query_text", eligible=True),
                  Position(index=9, token_id=9, piece="?", role="query_text", eligible=True)]
    # element_a owns the first row of image positions; half of position 4 is element_a, half background.
    ownership = ownership or ([{"element_a": 1.0}]*4 + [{"element_a": .5, "background": .5}] + [{"background": 1.0}]*3
                              + [{"non_page": 1.0}]*2)
    encoding = EncodingAttention(encoding_id="joint", pass_kind="joint", page_id="page_a", layer_count=LAYERS,
                                 query_head_count=HEADS, position_count=N, attention_mask=[1]*N, causal=True,
                                 layer_arrays=[f"l{i}" for i in range(LAYERS)], positions=positions, ownership=ownership)
    return NS(capture_id="capture_fixture", attention_encodings=[encoding], pages=[], query_positions=[])


def request(method, **parameters):
    return RetrievalAttentionRequest(encoding_id="joint", source_index=SOURCE, method_id=method, parameters=parameters)


def table_inputs():
    return TableInputs(NS(question_id="q1"), table_record(), NS(page_id="page_a", width=4, height=2), grounding(), "fixture_annotation")


def overlap_matrix():
    """w(c, j): unit position j lies wholly inside cell (j//4, j%4) for x < 3."""
    w = np.zeros((N, 6))
    for j in range(8):
        r, c = j//4, j % 4
        if c < 3:
            w[j, r*3+c] = 1.0
    return w


def test_new_methods_are_offered_on_retriever_encodings_without_steps():
    offered = {m.method_id: m for m in methods()}
    for identifier in ("attention.element_profile", "table.cell_allocation", "table.lookup_path", "table.page_regions", "table.mapping_audit"):
        assert identifier in offered and "step" not in offered[identifier].parameters_schema["properties"]
    assert "table.matched_contrast" not in offered


def test_element_profile_reads_every_layer_and_head_of_the_source_row():
    arrays = matrices()
    document = NS(elements=[NS(element_id="element_a", page_id="page_a", element_type="table", native_id="51", reading_order=0)])
    result = calculate(encoding_capture(), request("attention.element_profile"), arrays.__getitem__, document)
    profile = result.element_profile
    assert (profile.layer_count, profile.head_count, profile.image_positions) == (LAYERS, HEADS, 8)
    weights = np.array([1, 1, 1, 1, .5, 0, 0, 0, 0, 0])
    expected = np.stack([arrays[f"l{l}"][:, SOURCE, :] @ weights for l in range(LAYERS)])
    assert np.array(profile.elements[0].mass) == pytest.approx(expected, rel=1e-6)
    assert np.array(profile.image_mass) == pytest.approx(np.stack([arrays[f"l{l}"][:, SOURCE, :8].sum(-1) for l in range(LAYERS)]), rel=1e-6)
    assert profile.elements[0].area_share == pytest.approx(4.5/8)


def test_cell_allocation_at_one_layer_matches_the_raw_row():
    arrays = matrices(1)
    result = calculate(encoding_capture(), request("table.cell_allocation", layer=2, head=1), arrays.__getitem__, None, table_inputs())
    row = arrays["l2"][1, SOURCE].astype(np.float64)
    masses = [c.mass for c in result.table_allocation.cells]
    assert masses == pytest.approx(list(row @ overlap_matrix()), rel=1e-6)
    assert result.table_allocation.total_mass == pytest.approx(1.0, abs=1e-6)


def test_cell_allocation_over_a_rollout_window_matches_an_independent_rollout():
    arrays = matrices(2)
    result = calculate(encoding_capture(), request("table.cell_allocation", layer=2, layer_reduction="rollout", start_layer=1,
                                                   residual_coefficient=.5), arrays.__getitem__, None, table_inputs())
    r = np.zeros(N); r[SOURCE] = 1.0
    for layer in (2, 1):
        mixed = .5*arrays[f"l{layer}"].astype(np.float64).mean(0) + .5*np.eye(N)
        r = r @ (mixed / mixed.sum(-1, keepdims=True))
    masses = [c.mass for c in result.table_allocation.cells]
    assert masses == pytest.approx(list(r @ overlap_matrix()), rel=1e-6)
    assert result.row.quantity == "partial_rollout_connectivity"
    assert "rollout layers 1–2" in result.table_allocation.selection
    # The answer cell (1, 2) is identified by grounding, never sent to the model.
    path = calculate(encoding_capture(), request("table.lookup_path", layer=2, layer_reduction="rollout", start_layer=1,
                                                 include_head_matrix=False), arrays.__getitem__, None, table_inputs())
    answer = next(c for c in path.table_allocation.cells if c.cell_id == cell_id(1, 2))
    assert answer.mass == pytest.approx(float(r[6]), rel=1e-6)
    with pytest.raises(ValueError, match="start layer"):
        calculate(encoding_capture(), request("table.cell_allocation", layer=1, layer_reduction="rollout", start_layer=2),
                  arrays.__getitem__, None, table_inputs())


def test_table_methods_refuse_a_case_without_table_annotations():
    with pytest.raises(ValueError, match="table grounding case"):
        calculate(encoding_capture(), request("table.cell_allocation", layer=0), matrices().__getitem__)
