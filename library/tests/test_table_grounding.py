"""TableGroundQA adapter, additive table contracts and the single-page image position plan."""
from collections import Counter
from itertools import combinations
from pathlib import Path
import shutil

import pytest
from pydantic import ValidationError

from docattentionlens.datasets.contracts import (
    CaseRecord, DatasetSummary, DocumentRecord, QuestionRecord, SourceRef, Snapshot,
)
from docattentionlens.datasets.table_grounding import read_table_dataset
from docattentionlens.generation.contracts import (
    QWEN25_VL_3B, QWEN3_VL_8B, ContextManifest, GenerationRequest,
)
from docattentionlens.generation.qwen import image_budget

EXPORT = Path(__file__).resolve().parents[2] / "data/TableGroundQA-Pilot-v1"
needs_export = pytest.mark.skipif(not (EXPORT / "manifest.json").is_file(), reason="TableGroundQA-Pilot-v1 export is not present")


@pytest.fixture(scope="module")
def snapshot():
    return read_table_dataset(EXPORT)


def area(b):
    return max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)


def overlap(a, b):
    return max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))


@needs_export
def test_export_registers_tables_on_full_pages_without_layout_elements(snapshot):
    summary = snapshot.summary
    assert snapshot.schema_version == "0.3.0" and summary.structure_kind == "table_grounding"
    assert summary.adapter_version == "table_ground_qa_v1" and summary.name == "TableGroundQA-Pilot-v1"
    assert summary.counts == {"documents": 5, "pages": 5, "questions": 100, "elements": 0, "tables": 5, "cells": 975}
    assert snapshot.elements == [] and summary.element_types == []
    assert summary.quality_counts["questions_needing_source_qa_review"] == 4
    assert summary.quality_counts["logical_cells_without_text"] == 773
    assert summary.quality_counts["cells_with_pdf_text_only"] == 673
    assert all(p.width in (1211, 1275) and p.height in (1650, 1684) for p in snapshot.pages)
    # Page numbers are the source PDF pages; each document registers its one exported page.
    assert sorted(p.page_number for p in snapshot.pages) == [4, 4, 4, 7, 7]
    assert all(d.total_pages == 1 and len(d.table_ids) == 1 and d.element_count == 0 for d in snapshot.documents)


@needs_export
def test_cells_tile_each_table_on_its_page_and_keep_their_identities(snapshot):
    pages = {p.page_id: p for p in snapshot.pages}
    for table in snapshot.tables:
        page = pages[table.page_id]
        assert 0 <= table.bbox.x1 < table.bbox.x2 <= page.width and 0 <= table.bbox.y1 < table.bbox.y2 <= page.height
        # The caption-and-notes wrap contains the table.
        assert table.wrap_bbox.x1 <= table.bbox.x1 and table.wrap_bbox.y1 <= table.bbox.y1 and table.wrap_bbox.x2 >= table.bbox.x2
        assert sum(area(c.bbox) for c in table.cells) == pytest.approx(area(table.bbox), rel=1e-6)
        assert all(overlap(a.bbox, b.bbox) < 1e-6 for a, b in combinations(table.cells, 2))
        assert all(overlap(a.bbox, b.bbox) < 1e-6 for a, b in combinations(table.rows, 2))
        assert [r.index for r in table.rows] == list(range(table.row_count))
        assert len({c.native_id for c in table.cells}) == len(table.cells)
        assert all(c.pdf_bboxes for c in table.cells)
        assert 0.8 <= table.registration_correlation <= 1
    roles = Counter((r.role, r.status) for t in snapshot.tables for c in t.cells for r in c.roles)
    assert roles[("column_header", "source_supplied")] == 33 and roles[("row_header", "proposed")] > 0


@needs_export
def test_questions_carry_evaluation_only_grounding_bound_to_their_table(snapshot):
    tables = {t.table_id: t for t in snapshot.tables}
    flagged = []
    for q in snapshot.questions:
        g = q.table_grounding
        assert g is not None and g.support_role == "evaluation_annotation" and g.authored_input == "table_crop"
        assert q.question == q.document_scoped_question == q.collection_scoped_question
        assert q.support_page_ids == [tables[g.table_id].page_id] and q.support_element_ids == []
        cells = {c.native_id for c in tables[g.table_id].cells}
        assert set(g.answer_cell_ids + g.row_header_path + g.column_header_path + g.support_cell_ids) <= cells
        assert q.answer in g.accepted_answers
        if g.binding_status == "needs_source_qa_review":
            flagged.append(q.native_id)
            assert g.pdf_answer_text and g.pdf_answer_text != q.answer
    assert sorted(flagged) == [f"pt1m-question-pmc4993190-0-{n:04d}" for n in (5, 6, 14, 15)]


@needs_export
def test_snapshot_round_trips_and_content_version_follows_the_files(snapshot, tmp_path):
    assert Snapshot.model_validate_json(snapshot.model_dump_json()) == snapshot
    copy = tmp_path / EXPORT.name
    shutil.copytree(EXPORT, copy)
    assert read_table_dataset(copy).summary.dataset_version == snapshot.summary.dataset_version
    question = copy / "qa/questions.jsonl"
    question.write_bytes(question.read_bytes().replace(b"lookup", b"lookuq", 1))
    with pytest.raises(ValueError, match="SHA256SUMS"):
        read_table_dataset(copy)
    renamed = tmp_path / "renamed-export"
    shutil.copytree(EXPORT, renamed)
    with pytest.raises(ValueError, match="export_id"):
        read_table_dataset(renamed)


def test_additive_fields_leave_layout_records_unchanged():
    ref = SourceRef(dataset_id="d", dataset_version="v", domain="annual_reports", document_id=1)
    assert "table_id" not in ref.model_dump() and ref.model_copy(update={"table_id": "t"}).model_dump()["table_id"] == "t"
    question = QuestionRecord(key="q", question_id="q", document_id="d", ref=ref, native_id=1, question="Q", document_scoped_question="Q",
        collection_scoped_question="Q", answer="A", question_type="short", native_element_ids=[], support_element_ids=[], support_page_ids=[],
        support_element_refs=[], support_page_refs=[])
    assert "table_grounding" not in question.model_dump(mode="json")
    document = DocumentRecord(key="d", document_id="d", ref=ref, native_id=1, filename="f.pdf", total_pages=1, page_ids=[], question_count=0,
        element_count=0, annotation_version="a", source_pdf_sha256="s", ocr_model="m", ocr_status="complete")
    assert not {"title", "table_ids"} & set(document.model_dump())
    summary = DatasetSummary(dataset_id="d", name="n", dataset_version="v", counts={}, domain_counts={}, domain_aliases={}, element_types=[],
        element_categories=[], question_types=[], quality_counts={}, quality_issues=[], source_manifest={}, source_file_count=0, json_snapshot_sha256="j")
    assert "structure_kind" not in summary.model_dump() and summary.structure_kind == "document_layout"
    case = CaseRecord(key="c", case_id="c", dataset_id="d", dataset_version="v", document_id="d", question_id="q", query_variant="original",
        task_scope="document", query="Q", page_ids=[], page_image_sha256={}, annotation_version="a", support_page_ids=[], support_element_ids=[],
        reference_answer="A", producer_version="p")
    assert "table_grounding" not in case.model_dump(mode="json")


def test_single_page_plan_is_limited_to_one_page_and_recorded():
    base = {"case_id": "case", "idempotency_key": "k"}
    assert GenerationRequest(**base, page_ids=["p"]).image_token_limit == 512
    assert GenerationRequest(**base, page_ids=["p"], image_token_limit=1024).image_token_limit == 1024
    with pytest.raises(ValidationError, match="one supplied page"):
        GenerationRequest(**base, page_ids=["p", "q"], image_token_limit=1024)
    with pytest.raises(ValidationError):
        GenerationRequest(**base, page_ids=["p"], image_token_limit=768)
    # Earlier captures used the standard plan.
    assert ContextManifest(retrieved_page_ids=[], selected_page_ids=[], omitted_page_ids=[], query="q", prompt="p", pages=[],
                           input_token_ids=[], input_attention_mask=[]).image_token_limit == 512
    assert QWEN25_VL_3B.pixels_for(1024) == 1024 * 28 * 28 and QWEN3_VL_8B.pixels_for(1024) == 1024 * 32 * 32
    assert QWEN25_VL_3B.max_pixels == QWEN25_VL_3B.pixels_for(512)
    with pytest.raises(ValueError):
        QWEN25_VL_3B.pixels_for(2048)


def test_image_budget_restores_the_resting_processor_plan():
    class Processor:
        size = {"shortest_edge": QWEN25_VL_3B.min_pixels, "longest_edge": QWEN25_VL_3B.max_pixels}
        min_pixels, max_pixels = QWEN25_VL_3B.min_pixels, QWEN25_VL_3B.max_pixels
    ip = Processor()
    with image_budget(ip, QWEN25_VL_3B, 1024) as size:
        assert size["longest_edge"] == ip.max_pixels == 1024 * 28 * 28
    assert ip.max_pixels == ip.size["longest_edge"] == 512 * 28 * 28
    with pytest.raises(RuntimeError):
        with image_budget(ip, QWEN25_VL_3B, 1024):
            raise RuntimeError("processing failed")
    assert ip.max_pixels == 512 * 28 * 28
