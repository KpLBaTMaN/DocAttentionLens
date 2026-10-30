"""Mathematical fixtures for attention columns and question-token source rows."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from docattentionlens.generation.contracts import GenerationCapture, GenerationPosition, PredictionStep, SourceRow
from docattentionlens.generation.inspection import source_view, received_column as generator_column
from docattentionlens.methods.attention_received import received
from docattentionlens.methods.contracts import Context
from docattentionlens.methods.registry import execute
from docattentionlens.retrieval.contracts import RetrievalReceivedRequest
from docattentionlens.retrieval.attention import received_column as encoding_column
from docattentionlens.methods.query_patch_similarity import similarity_column, similarity_row
from test_retriever_attention import fixture as encoding_fixture


def test_column_reads_each_row_without_renormalizing_and_omits_rows_that_cannot_see_the_target():
    rows = [(1, "question", "a", [.5, .5]), (2, "question", "b", [.2, .3, .5]), (0, "question", "c", [1.])]
    result = received("generation_" + "a" * 32, 1, 0, 0, rows)
    assert [(s.source_index, s.value) for s in result.sources] == [(1, .5), (2, pytest.approx(.3))]
    assert result.maximum == .5 and sum(s.value for s in result.sources) != pytest.approx(1)
    with pytest.raises(ValueError, match="normalized"):
        received("x", 0, 0, 0, [(0, "question", "a", [.5, .1])])
    with pytest.raises(ValueError, match="includes this target"):
        received("x", 3, 0, 0, rows)


def test_encoding_column_uses_active_source_rows_including_future_keys():
    capture, arrays = encoding_fixture()
    column = encoding_column(capture, RetrievalReceivedRequest(encoding_id="query", target_index=1), arrays.__getitem__)
    # Head 0 rows: [0,1,0] and [.25,.75,0]; the masked third row is not a source.
    assert [(s.source_index, s.value) for s in column.sources] == [(0, 1), (1, .75)]
    assert all(s.kind == "encoding" for s in column.sources)
    head = encoding_column(capture, RetrievalReceivedRequest(encoding_id="query", target_index=0, head=1), arrays.__getitem__)
    assert [s.value for s in head.sources] == [.5, 1]
    with pytest.raises(ValueError, match="source selection"):
        encoding_column(capture, RetrievalReceivedRequest(encoding_id="query", target_index=2), arrays.__getitem__)
    with pytest.raises(ValueError, match="outside"):
        encoding_column(capture, RetrievalReceivedRequest(encoding_id="query", target_index=0, layer=2), arrays.__getitem__)


def test_similarity_column_is_the_transpose_of_rows_and_reports_recorded_winners():
    q = np.array([[1, 0], [0, 1], [5, 5]], dtype=np.float32)
    d = np.array([[1, 0], [.5, .5], [-1, 0]], dtype=np.float32)
    record = NS(capture_id="numeric_fixture", query_positions=[NS(index=i, eligible=i < 2) for i in range(3)])
    page = NS(page_id="p", positions=[NS(index=i, eligible=i < 2) for i in range(3)], winner_indices=[0, 1])
    column = similarity_column(record, page, q, d, 1)
    assert column.similarities == [.5, .5, None] and column.winning_query_indices == [1]
    assert column.similarities[0] == similarity_row(record, page, q, d, 0).similarities[1]
    assert similarity_column(record, page, q, d, 0).winning_query_indices == [0]
    with pytest.raises(ValueError, match="ineligible"):
        similarity_column(record, page, q, d, 2)


def generator():
    roles = ["image", "image", "query", "query", "special"]
    positions = [GenerationPosition(index=i, token_id=i, piece=f"t{i}", role=r, page_id="page_a" if r == "image" else None,
        bbox=[0, 0, 1, 1] if r == "image" else None, ownership={"element_a": 1.0} if r == "image" else {"non_image:" + r: 1.0})
        for i, r in enumerate(roles)]
    record = GenerationCapture.model_construct(capture_id="generation_" + "b" * 32, positions=positions, layer_count=1, query_head_count=1,
        steps=[PredictionStep(step=0, pass_kind="prefill", source_index=4, consumed_token_id=4, consumed_piece="t4",
            predicted_index=5, predicted_token_id=9, predicted_piece="ans", target_count=5, array="step_0")],
        source_rows=[SourceRow(source_index=i, token_id=i, piece=f"t{i}", predicted_index=i + 1, predicted_token_id=i + 1,
            predicted_piece=f"t{i + 1}", target_count=i + 1, array=f"question_{i}") for i in (2, 3)],
        context=NS(input_attention_mask=[1] * 5, input_token_ids=list(range(5))),
        coverage={"all_causal_targets": True, "quantity": "post_softmax", "ownership": "reviewed"})
    arrays = {"step_0": np.array([[[.1, .1, .3, .3, .2]]], dtype=np.float32),
              "question_2": np.array([[[.6, .2, .2]]], dtype=np.float32),
              "question_3": np.array([[[.1, .5, .2, .2]]], dtype=np.float32)}
    return record, arrays


def test_question_token_rows_run_existing_methods_on_their_own_causal_support():
    record, arrays = generator()
    view, step = source_view(record, 3)
    assert step.source_index == 3 and step.target_count == 4 and step.predicted_index == 4
    row = execute("attention.row", "captured_weights_v1", {"step": 0, "layer": 0, "head": 0}, Context(view, arrays["question_3"])).row
    assert row.source_index == 3 and row.values == pytest.approx([.1, .5, .2, .2])
    assert row.image_mass == pytest.approx(.6)
    with pytest.raises(ValueError, match="not captured"):
        source_view(record, 4)


def test_generator_column_reads_prediction_and_question_rows_and_respects_causality():
    record, arrays = generator()
    patch = generator_column(record, 1, 0, 0, arrays.__getitem__)
    assert [(s.kind, s.source_index, s.value) for s in patch.sources] == [
        ("prediction", 4, pytest.approx(.1)), ("question", 2, pytest.approx(.2)), ("question", 3, pytest.approx(.5))]
    # Position 3 is after question token 2, so only later rows can attend to it.
    later = generator_column(record, 3, 0, 0, arrays.__getitem__)
    assert [s.source_index for s in later.sources] == [4, 3]
    with pytest.raises(ValueError, match="outside"):
        generator_column(record, 1, 1, 0, arrays.__getitem__)
