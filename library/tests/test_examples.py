"""The committed examples: one annual report in the doclayout layout and three of its tables at cell level."""
from importlib.util import module_from_spec, spec_from_file_location
from itertools import combinations
import json
from pathlib import Path
import shutil

import pytest

from docattentionlens.datasets.adapter import read_dataset
from docattentionlens.datasets.table_grounding import read_table_dataset

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = ROOT / "data/doclayout_mm_rag_examples"
TABLES = ROOT / "data/TableGroundQA-Examples-v1"
BENCHMARK = ROOT / "data/doclayout_mm_rag_bench"
HEADERS = {"row_header", "column_header"}


@pytest.fixture(scope="module")
def tables():
    return read_table_dataset(TABLES)


def squash(text):
    return "".join((text or "").split())


def area(b):
    return (b.x2 - b.x1) * (b.y2 - b.y1)


def overlap(a, b):
    return max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))


def test_document_example_is_one_annual_report_named_by_its_source():
    snapshot = read_dataset(DOCUMENTS, name=DOCUMENTS.name)
    assert snapshot.summary.name == "doclayout_mm_rag_examples"
    assert snapshot.summary.counts == {"documents": 1, "pages": 8, "elements": 100, "questions": 29}
    assert [d.filename for d in snapshot.documents] == ["NASDAQ_AROW_2019"] and snapshot.summary.domain_counts.keys() == {"annual_reports"}
    assert "table" in snapshot.summary.element_types and snapshot.summary.quality_counts.get("bbox_outside_or_degenerate", 0) == 0
    # The source name enters the identity, so the example never collides with the benchmark.
    assert read_dataset(DOCUMENTS).summary.dataset_id != snapshot.summary.dataset_id


@pytest.mark.skipif(not (BENCHMARK / "manifest.json").is_file(), reason="doclayout_mm_rag_bench export is not present")
def test_document_example_files_are_the_benchmark_files():
    record = json.loads((DOCUMENTS / "annual_reports/documents.json").read_text(encoding="utf-8"))["documents"][0]
    relatives = [record[k] for k in ("source_pdf_file", "layout_file", "ocr_file", "annotations_file")] + [p["image_file"] for p in record["pages"]]
    for relative in relatives:
        assert (DOCUMENTS / "annual_reports" / relative).read_bytes() == (BENCHMARK / "annual_reports" / relative).read_bytes()


def test_table_example_holds_three_sizes_on_two_pages_of_one_document(tables):
    summary = tables.summary
    assert summary.name == "TableGroundQA-Examples-v1" and summary.structure_kind == "table_grounding"
    assert summary.counts == {"documents": 1, "pages": 2, "questions": 26, "elements": 0, "tables": 3, "cells": 270}
    assert summary.domain_counts.keys() == {"annual_reports"} and "300 DPI" in summary.coordinate_convention
    assert sorted(len(t.cells) for t in tables.tables) == [35, 85, 150]
    assert sorted((t.row_count, t.column_count) for t in tables.tables) == [(5, 7), (5, 17), (29, 6)]
    document = tables.documents[0]
    assert document.total_pages == 2 and len(document.table_ids) == 3 and document.question_count == 26
    assert sorted(p.page_number for p in tables.pages) == [4, 6] and len({p.page_id for p in tables.pages}) == 2
    assert sorted(sum(t.page_id == p.page_id for t in tables.tables) for p in tables.pages) == [1, 2]


def test_cells_match_the_pdf_text_and_tile_their_table(tables):
    pages = {p.page_id: p for p in tables.pages}
    for table in tables.tables:
        page = pages[table.page_id]
        assert all(squash(c.text) == squash(c.pdf_text) for c in table.cells)
        assert all(0 <= c.bbox.x1 < c.bbox.x2 <= page.width and 0 <= c.bbox.y1 < c.bbox.y2 <= page.height for c in table.cells)
        assert abs(sum(area(c.bbox) for c in table.cells) - area(table.bbox)) < 1e-6 * area(table.bbox)
        assert all(overlap(a.bbox, b.bbox) < 1e-6 for a, b in combinations(table.cells, 2))
        # Native boxes are the words' extents and lie inside their logical cells, except advisory overflows.
        flagged = {i for issue in table.quality_issues if issue.code == "text_crosses_logical_cell" for i in issue.cell_ids}
        for cell in table.cells:
            for box in cell.pdf_bboxes:
                inside = cell.bbox.x1 - 1 <= box.x1 and box.x2 <= cell.bbox.x2 + 1 and cell.bbox.y1 - 1 <= box.y1 and box.y2 <= cell.bbox.y2 + 1
                assert inside or cell.native_id in flagged


def test_questions_bind_one_answer_cell_through_header_paths(tables):
    by_table = {t.table_id: t for t in tables.tables}
    for question in tables.questions:
        ground = question.table_grounding
        table = by_table[ground.table_id]
        cells = {c.native_id: c for c in table.cells}
        answer = cells[ground.answer_cell_ids[0]]
        assert len(ground.answer_cell_ids) == 1 and question.answer == answer.text and squash(ground.pdf_answer_text) == squash(answer.text)
        assert ground.binding_status == "text_matched_not_human_approved" and ground.authored_input == "full_page"
        assert answer.row_ids == ground.answer_row_ids and answer.column_ids == ground.answer_column_ids
        assert question.answer in ground.accepted_answers and question.support_page_ids == [table.page_id]
        for path in (ground.row_header_path, ground.column_header_path):
            assert path and all(HEADERS & {r.role for r in cells[i].roles} for i in path)
        assert cells[ground.row_header_path[-1]].row_ids == answer.row_ids
        assert set(answer.column_ids) <= set(cells[ground.column_header_path[-1]].column_ids)


def test_a_changed_export_file_is_refused(tmp_path):
    copy = tmp_path / TABLES.name
    shutil.copytree(TABLES, copy)
    questions = copy / "qa/questions.jsonl"
    questions.write_bytes(questions.read_bytes().replace(b'"184.59"', b'"184.60"', 1))
    with pytest.raises(ValueError, match="SHA256SUMS"):
        read_table_dataset(copy)


def test_the_builder_refuses_text_the_pdf_does_not_contain():
    loaded = spec_from_file_location("build_examples", ROOT / "scripts/build_examples.py")
    builder = module_from_spec(loaded)
    loaded.loader.exec_module(builder)
    import pymupdf
    authoring = json.loads((TABLES / "authoring/tables.json").read_text(encoding="utf-8"))
    spec = next(t for t in authoring["tables"] if t["size"] == "small")
    layout = {e["id"]: e for e in json.loads((DOCUMENTS / "annual_reports/layouts/pdf_30.json").read_text(encoding="utf-8-sig"))["structured_layout_elements"]}
    with pymupdf.open(DOCUMENTS / "annual_reports/pdfs/pdf_30.pdf") as pdf:
        page = pdf[spec["page_number"] - 1]
        s = 1800 / page.rect.width
        b = layout[spec["element_id"]]["bbox"]
        region = [b["x1"] / s - 4, b["y1"] / s - 4, b["x2"] / s + 4, b["y2"] / s + 4]
        words = page.get_text("words")
    rows = builder.layout_table(spec, words, region)[0]
    assert sum(len(r["cells"]) for r in rows) == 35
    spec["rows"][1]["cells"][-1] = "184.95"
    with pytest.raises(ValueError, match="184.95"):
        builder.layout_table(spec, words, region)
