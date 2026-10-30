"""Known-answer table fixtures for table-aware methods; never scientific model results.

A 4 × 2 page of unit image positions. The table covers x 0–3 as 2 rows × 3 columns of unit cells, so
every cell has a position of its own; the caption-and-notes wrap adds x 3–3.5, and x 3.5–4 is the
rest of the page. Row 0 holds the column headers, cell (1, 0) is the row header "HP".
"""
from dataclasses import replace
from types import SimpleNamespace as NS
import numpy as np
import pytest
from docattentionlens.generation.contracts import GenerationPosition
from docattentionlens.methods.contracts import Context, TableInputs
from docattentionlens.methods.registry import execute, get_method, table_structure
from docattentionlens.methods._table import partition

BOX = lambda x1, y1, x2, y2: NS(x1=x1, y1=y1, x2=x2, y2=y2)
TEXT = {(0, 0): "Group", (0, 1): "A", (0, 2): "B", (1, 0): "HP", (1, 1): "1", (1, 2): "1"}
ROLES = {(0, 0): "column_header", (0, 1): "column_header", (0, 2): "column_header", (1, 0): "row_header"}


def cell_id(r, c):
    return f"cell-{r}{c}"


def table_record(offset=0.0):
    cells = [NS(native_id=cell_id(r, c), row_indices=[r], column_indices=[c], bbox=BOX(c+offset, r, c+1+offset, r+1),
                text=TEXT[r, c] if (r, c) != (1, 1) else None, pdf_text="1" if (r, c) == (1, 1) else None,
                roles=[NS(role=ROLES.get((r, c), "data"), status="proposed", method="fixture")], native_flags=[])
             for r in range(2) for c in range(3)]
    return NS(table_id="table_fixture", native_id="pt-fixture", table_sha256="f"*64, page_id="page_a", title="Fixture",
              bbox=BOX(offset, 0, 3+offset, 2), wrap_bbox=BOX(offset, 0, 3.5+offset, 2), cells=cells, row_count=2, column_count=3,
              rows=[NS(native_id=f"row-{r}", index=r, bbox=BOX(offset, r, 3+offset, r+1)) for r in range(2)],
              columns=[NS(native_id=f"column-{c}", index=c, bbox=BOX(c+offset, 0, c+1+offset, 2)) for c in range(3)])


def grounding(answer=(1, 2), status="text_matched_not_human_approved"):
    r, c = answer
    return NS(table_id="table_fixture", answer_cell_ids=[cell_id(r, c)], answer_row_ids=[f"row-{r}"], answer_column_ids=[f"column-{c}"],
              row_header_path=[cell_id(r, 0)], column_header_path=[cell_id(0, c)], support_cell_ids=[], accepted_answers=["1"],
              binding_status=status)


# Image positions 0–7 (row-major over the 4 × 2 page), then a question token and the source.
ROW = [.05, .05, .1, .1, .1, .1, .3, 0, .1, .1]


def capture(identifier="a", values=ROW, offset=0.0, answer=(1, 2), question="q1", status="text_matched_not_human_approved"):
    positions = [GenerationPosition(index=i, token_id=i, piece=str(i), role="image", page_id="page_a",
                                    bbox=[i % 4, i//4, i % 4+1, i//4+1], ownership={"background": 1.0}) for i in range(8)]
    positions += [GenerationPosition(index=8, token_id=8, piece="q", role="query", ownership={"non_image:query": 1.0}),
                  GenerationPosition(index=9, token_id=9, piece="s", role="special", ownership={"non_image:special": 1.0})]
    record = NS(capture_id="generation_"+identifier*32, layer_count=2, query_head_count=2, positions=positions,
                steps=[NS(step=0, target_count=10, source_index=9, predicted_index=10, array="step_0")],
                context=NS(input_attention_mask=[1]*10, input_token_ids=list(range(10))),
                coverage={"all_causal_targets": True, "quantity": "post_softmax", "ownership": "reviewed"},
                provenance={"model_id": "fixture", "model_revision": "v1"})
    uniform = [.1]*10
    array = np.array([[values, uniform], [uniform, values]], dtype=np.float32)
    table = TableInputs(NS(question_id=question), table_record(offset), NS(page_id="page_a", width=4, height=2),
                        grounding(answer, status), "fixture_annotation")
    return Context(record, array, table=table)


def run(identifier, context=None, **parameters):
    method = get_method(identifier)
    return execute(identifier, method.variant_id, parameters, context or capture())


def stats(result):
    return {s.name: s.value for s in result.statistics}


def test_partitions_conserve_every_position_and_report_isolable_cells():
    context = capture()
    part = partition(context.record, context.table, 10)
    total = part.cells.sum(1)+sum(v for k, v in part.regions.items() if k != "table")+sum(part.non_image.values())
    assert total == pytest.approx(np.ones(10))
    assert part.regions["table_wrap"][3] == pytest.approx(.5) and part.regions["rest_of_page"][3] == pytest.approx(.5)
    assert part.resolution.isolable_cells == 6 and part.resolution.positions_on_table == 6
    shifted = partition(capture(offset=.5).record, capture(offset=.5).table, 10)
    # Shifted by half a position, every cell straddles two positions: half inside each, so still isolable at 0.5,
    # but each position now feeds two cells.
    assert (shifted.cells > 0).sum(axis=1)[1] == 2
    assert shifted.cells.sum(1)+sum(v for k, v in shifted.regions.items() if k != "table")+sum(shifted.non_image.values()) == pytest.approx(np.ones(10))


def test_page_regions_and_cell_allocation_reconstruct_known_masses():
    regions = stats(run("table.page_regions"))
    assert regions["Table mass"] == pytest.approx(.7)
    assert regions["Caption and notes mass"] == pytest.approx(.05)
    assert regions["Rest-of-page mass"] == pytest.approx(.05)
    assert regions["Non-image mass"] == pytest.approx(.2)
    # Table area is 6/8 of the page and receives .7/.8 of image attention.
    assert regions["Table enrichment over its page area"] == pytest.approx((.7/.8)/(6/8))
    cells = run("table.cell_allocation")
    allocation = {c.cell_id: c for c in cells.table_allocation.cells}
    assert allocation["cell-12"].mass == pytest.approx(.3) and allocation["cell-12"].enrichment == pytest.approx(3.0)
    assert [b.mass for b in cells.table_allocation.rows] == pytest.approx([.2, .5])
    assert [b.mass for b in cells.table_allocation.columns] == pytest.approx([.15, .15, .4])
    assert cells.table.cell_ids[0] == "cell-12"
    mean = run("table.cell_allocation", head_reduction="head_mean")
    assert {c.cell_id: c.mass for c in mean.table_allocation.cells}["cell-12"] == pytest.approx(.2)


def test_mapping_audit_needs_no_labels():
    result = run("table.mapping_audit")
    assert stats(result)["Isolable share"] == 1
    assert [row[2] for row in result.table.values] == [1.0]*6


def test_lookup_path_parts_are_disjoint_and_sum_to_the_row():
    result = run("table.lookup_path")
    parts = {label.split(" (")[0]: row[0] for label, row in zip(result.table.labels, result.table.values)}
    assert parts["Answer cell"] == pytest.approx(.3)
    assert parts["Row-header path"] == pytest.approx(.1)
    assert parts["Column-header path"] == pytest.approx(.1)
    assert parts["Rest of the answer row"] == pytest.approx(.1)
    assert parts["Other header cells"] == pytest.approx(.1)
    assert sum(v or 0 for v in parts.values()) == pytest.approx(1)
    s = stats(result)
    assert s["Answer cell rank among cells"] == 1 and s["Answer row rank (mass)"] == 1 and s["Answer column rank (mass)"] == 1
    matrix = result.tables[-1].values
    assert matrix[0][0] == pytest.approx(.5) and matrix[0][1] == pytest.approx(.3)


def test_flagged_binding_is_refused_unless_included():
    flagged = capture(status="needs_source_qa_review")
    with pytest.raises(ValueError, match="Answer/cell mismatch"):
        run("table.lookup_path", flagged)
    assert "flagged" in " ".join(run("table.lookup_path", flagged, include_flagged_binding=True).notices)


def test_question_swap_baseline_excludes_questions_sharing_a_part():
    first = capture()
    other_row = [.05, .05, .05, .05, .1, .3, .1, .1, .1, .1]
    second = capture("b", other_row, answer=(1, 1), question="q2")
    result = run("table.lookup_path", replace(first, samples=(first, second)))
    swap = result.tables[0]
    rows = dict(zip(swap.labels, swap.values))
    # The second question's answer cell (1, 1) gets .3 there, while this answer cell (1, 2) gets .1.
    assert rows["Answer cell"][1] == pytest.approx(.1) and rows["Answer cell"][4] == pytest.approx(.2)
    # Both questions use the row header "HP", so the second cannot be its baseline.
    assert rows["Row-header path"][6] == 0 and rows["Row-header path"][1] is None
    assert rows["Column-header path"][6] == 1
    with pytest.raises(ValueError, match="different question"):
        run("table.lookup_path", replace(first, samples=(first, capture("c"))))


def test_answer_distractors_use_body_cells_and_same_value_text():
    result = run("table.answer_distractors")
    sets = dict(zip(result.table.labels, result.table.values))
    assert sets["Same row"][0] == 1 and sets["Same column"][0] == 0
    assert sets["Same value"][0] == 1 and sets["Same value"][4] == pytest.approx(.2)
    assert any("PDF text" in n for n in result.notices)
    assert stats(result)["Answer rank among body cells (mass)"] == 1


def test_matched_contrast_needs_a_shared_row_or_column():
    first = capture()
    second = capture("b", [.05, .05, .05, .05, .1, .3, .1, .1, .1, .1], answer=(1, 1), question="q2")
    result = run("table.matched_contrast", replace(first, comparison=second))
    s = stats(result)
    # (.3 − .1) + (.3 − .1): each capture favours its own answer cell.
    assert s["Answer-cell contrast"] == pytest.approx(.4)
    assert "row 2" in result.notices[0]
    with pytest.raises(ValueError, match="same question"):
        run("table.matched_contrast", replace(first, comparison=capture("c")))
    with pytest.raises(ValueError, match="same resolution"):
        run("table.matched_contrast", replace(first, comparison=replace(second, record=NS(**{**vars(second.record),
            "positions": [p.model_copy(update={"bbox": [v*1.01 for v in p.bbox]}) if p.bbox else p for p in second.record.positions]}))))


def test_table_methods_require_table_structure():
    with pytest.raises(ValueError, match="table annotations"):
        run("table.cell_allocation", replace(capture(), table=None))
    case = NS(table_grounding=NS(table_id="t", table_sha256="x", binding_status="text_matched_not_human_approved"))
    snapshot = NS(tables=[NS(table_id="t", table_sha256="x", title="T", revision="v1", cells=[1, 2], row_count=1, column_count=2,
                             quality_status="needs_review")], summary=NS(name="Tables", adapter_version="table_ground_qa_v1"))
    capabilities, structure = table_structure(case, snapshot)
    assert capabilities == {"table_structure_annotations", "table_grounding_annotations"} and structure.cell_count == 2
    assert table_structure(NS(table_grounding=None), snapshot) == (set(), None)


def test_real_table_cells_partition_their_tables_at_both_resolutions():
    """The pilot export's projected cells overhang their table box by sub-pixel rounding; clipped, they still conserve."""
    from pathlib import Path
    from docattentionlens.datasets.table_grounding import read_table_dataset
    root = Path(__file__).resolve().parents[3]/"data"/"TableGroundQA-Pilot-v1"
    if not (root/"manifest.json").is_file():
        pytest.skip("TableGroundQA-Pilot-v1 export is not present")
    snapshot = read_table_dataset(root)
    pages = {p.page_id: p for p in snapshot.pages}
    for table in snapshot.tables:
        page = pages[table.page_id]
        for columns, rows in ((27, 37), (19, 26)):
            w, h = page.width/columns, page.height/rows
            positions = [GenerationPosition(index=i, token_id=0, piece="x", role="image", page_id=page.page_id,
                                            bbox=[(i % columns)*w, (i//columns)*h, (i % columns+1)*w, (i//columns+1)*h], ownership={"b": 1.0})
                         for i in range(columns*rows)]
            part = partition(NS(positions=positions), TableInputs(None, table, page, None, "x"), len(positions))
            assert part.cells.sum() == pytest.approx(part.regions["table"].sum())
            assert 0 <= part.resolution.isolable_cells <= len(table.cells)
