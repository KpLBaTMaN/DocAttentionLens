"""Lookup-path allocation: page mass split into the question's answer cell, header paths and the rest,
with an optional question-swap baseline from saved captures of other questions on the same table."""
import numpy as np
from pydantic import Field
from .contracts import Calculation
from ._shared import statistic, selected_rows, selected_step
from ._table import (GroundingParameters, PATH_PARTS, measure, allocation, selection_label, grounding, path_groups, lookup_cells,
                     group_mass, rank, partition, masses, vector, matched_record, result_table, plot, resolution_notice,
                     define, EVALUATION_NOTICE, GROUNDING_REQUIRES)


class Parameters(GroundingParameters):
    include_head_matrix: bool = Field(default=True, title="Layer × head matrix of lookup-path mass")


TAIL = (("table_gap", "Inside the table, in no cell"), ("table_wrap", "Caption and notes"), ("rest_of_page", "Rest of the page"),
        ("other_page", "Other pages"))


def baseline(context, parameters, table, ground, part, observed):
    """Mass each other question's capture gives to this question's lookup cells.

    For each part, a capture counts only if its own question's lookup cells avoid that part, so a
    question that also needs a header is not used as that header's baseline."""
    others = [c for c in context.samples if c is not context and c.record.capture_id != context.record.capture_id]
    if not others:
        return [], []
    groups = path_groups(table.table, ground)
    own = {key: groups[key] for key in ("answer", "row_path", "column_path")}
    own["lookup"] = sorted(lookup_cells(table.table, ground))
    values = {k: [] for k in own}
    skipped, questions = [], set()
    for sample in others:
        sample_table = sample.table
        matched_record(context.record, sample.record, table, sample_table)
        other = sample_table.grounding
        if other is None or other.binding_status == "needs_source_qa_review":
            skipped.append(sample.record.capture_id); continue
        if sample_table.case.question_id == table.case.question_id:
            raise ValueError("Baseline captures must ask a different question about the table")
        if sample_table.case.question_id in questions:
            raise ValueError("Use one baseline capture per question")
        questions.add(sample_table.case.question_id)
        weights, mask = vector(sample, parameters)
        sample_part = partition(sample.record, table, selected_step(sample.record, parameters.step).target_count)
        sm = masses(sample_part, weights, mask)
        theirs = lookup_cells(table.table, other)
        for key, cells in own.items():
            if cells and not theirs & set(cells):
                mass = group_mass(sm.cells, cells)
                if mass is not None:
                    values[key].append(mass)
    labels = {"answer": "Answer cell", "row_path": "Row-header path", "column_path": "Column-header path", "lookup": "Lookup path (all three)"}
    rows = []
    for key, label in labels.items():
        v = np.array(values[key]); o = observed[key]
        rows.append([o, float(v.mean()) if v.size else None, float(v.min()) if v.size else None, float(v.max()) if v.size else None,
                     o-float(v.mean()) if v.size and o is not None else None,
                     float((v >= o).mean()) if v.size and o is not None else None, float(v.size)])
    notices = [f"Question-swap baseline: {len(questions)} captures of other questions on this table. For each part, a capture "
               "counts only when its own lookup cells avoid that part; n per part is shown."]
    if skipped:
        notices.append(f"{len(skipped)} baseline capture(s) with a binding flagged for review were left out.")
    return [result_table("Question-swap baseline", ["Observed", "Baseline mean", "Baseline min", "Baseline max",
                                                    "Observed − baseline mean", "Share of baselines ≥ observed", "Baseline captures"],
                         list(labels.values()), rows, "mass; mass; mass; mass; mass; fraction; captures")], notices


def compute(context, parameters):
    table, part, m, row, mask = measure(context, parameters)
    ground = grounding(context, parameters)
    groups = path_groups(table.table, ground)
    parts = [(label, group_mass(m.cells, groups[key]), len(groups[key])) for key, label in PATH_PARTS]
    parts += [(label, m.regions[key], None) for key, label in TAIL if key != "other_page" or m.regions[key] > 0]
    parts.append(("Non-image positions", float(sum(m.non_image.values())), None))
    total = sum(v or 0.0 for _, v, _ in parts)
    if abs(total-m.total) > 1e-6:
        raise ValueError("Lookup-path parts do not conserve the selected row")
    answer = groups["answer"][0] if groups["answer"] else None
    rows_index = {a.native_id: a.index for a in table.table.rows}
    columns_index = {a.native_id: a.index for a in table.table.columns}
    answer_row = next((rows_index[i] for i in ground.answer_row_ids if i in rows_index), None)
    answer_column = next((columns_index[i] for i in ground.answer_column_ids if i in columns_index), None)
    result = allocation(table, part, m, selection_label(parameters))
    observed = {"answer": group_mass(m.cells, groups["answer"]), "row_path": group_mass(m.cells, groups["row_path"]),
                "column_path": group_mass(m.cells, groups["column_path"]),
                "lookup": group_mass(m.cells, sorted(lookup_cells(table.table, ground)))}
    density = lambda bands: np.array([b.enrichment if b.enrichment is not None else np.nan for b in bands])
    statistics = [statistic("Answer cell mass", observed["answer"]),
                  statistic("Row-header path mass", observed["row_path"]), statistic("Column-header path mass", observed["column_path"]),
                  statistic("Lookup path mass", observed["lookup"]),
                  statistic("Answer cell rank among cells", rank(m.cells, answer) if answer is not None else None, "rank"),
                  statistic("Answer cell enrichment", result.cells[answer].enrichment if answer is not None else None, "ratio"),
                  statistic("Answer row rank (mass)", rank(m.rows, answer_row) if answer_row is not None else None, "rank"),
                  statistic("Answer row rank (enrichment)", rank(density(result.rows), answer_row) if answer_row is not None else None, "rank"),
                  statistic("Answer column rank (mass)", rank(m.columns, answer_column) if answer_column is not None else None, "rank"),
                  statistic("Answer column rank (enrichment)", rank(density(result.columns), answer_column) if answer_column is not None else None, "rank")]
    notices = []
    if answer is not None and not part.isolable[answer]:
        notices.append("The answer cell has no image position of its own: its mass is shared with neighbouring cells.")
    if answer is not None and not np.isfinite(m.cells[answer]):
        notices.append("No valid target position overlaps the answer cell, so its mass is unavailable, not zero.")
    tables = []
    if parameters.include_head_matrix:
        if context.values.size > 4_000_000:
            raise ValueError("The layer × head matrix exceeds the bounded analysis budget")
        all_rows, all_mask = selected_rows(context, parameters, heads=list(range(context.record.query_head_count)),
                                           layers=list(range(context.record.layer_count)))
        ownership = part.cells[:, sorted(lookup_cells(table.table, ground))].sum(axis=1)*all_mask
        matrix = (all_rows*ownership).sum(axis=-1)
        tables.append(result_table("Lookup path mass by layer and head", [f"Head {i}" for i in range(context.record.query_head_count)],
                                   [f"Layer {i}" for i in range(context.record.layer_count)], matrix.tolist()))
    swap, swap_notices = baseline(context, parameters, table, ground, part, observed)
    tables = swap+tables
    if swap:
        b = swap[0].values
        statistics += [statistic("Answer cell minus swap baseline", b[0][4]), statistic("Lookup path minus swap baseline", b[3][4])]
    if ground.binding_status == "needs_source_qa_review":
        notices.append("This question's binding is flagged for review and was included on request.")
    return Calculation(row=row, table_allocation=result, statistics=statistics,
        table=result_table("Lookup-path allocation", ["Mass"], [label if n is None else f"{label} ({n} cell{'' if n == 1 else 's'})" for label, _, n in parts],
                           [[v] for _, v, _ in parts], "mass"),
        tables=tables,
        plots=[plot("bar", "Lookup-path allocation", "Part", "Attention mass",
                    [{"label": label, "x": i, "y": v or 0.0, "group": label} for i, (label, v, _) in enumerate(parts)])],
        notices=[*notices, resolution_notice(part), *swap_notices,
                 "Parts are disjoint and assigned in the listed order, so they sum to the selected row's mass. "
                 "Allocation to the lookup path describes attention, not a reasoning sequence or reliance.", EVALUATION_NOTICE])


DEFINITION = define("table.lookup_path", "Lookup-path allocation",
    "Disjoint split of Σ_j A(s,j): answer cell → row-header path → column-header path → support → rest of answer row → "
    "rest of answer column → other headers → other body cells → table gap → caption and notes → rest of page → non-image. "
    "Swap baseline: mean over other questions' captures q' (whose lookup cells avoid the part) of m_q'(part).",
    Parameters, compute, requires=GROUNDING_REQUIRES,
    summary="How the selected prediction's attention divides between the question's answer cell, its row and column header paths "
            "and everything else, with a question-swap baseline from other questions on the same table.",
    limitations=["The swap baseline uses the saved captures supplied with the request; with few captures it is a coarse reference, "
                 "not a null distribution."])
