"""Matched-question contrast: two questions on the same table that share an answer row (or column) and
differ in the other axis. Does each question's attention favour its own answer cell and path?"""
import numpy as np
from .contracts import Calculation
from ._shared import statistic, selected_step
from ._table import (GroundingParameters, measure, allocation, selection_label, grounding, path_groups, lookup_cells, group_mass,
                     partition, masses, vector, matched_record, cell_label, result_table, resolution_notice, define,
                     EVALUATION_NOTICE, GROUNDING_REQUIRES)


def relation(table, a, b):
    rows = {x.native_id: x.index for x in table.rows}
    columns = {x.native_id: x.index for x in table.columns}
    ra, rb = {rows[i] for i in a.answer_row_ids if i in rows}, {rows[i] for i in b.answer_row_ids if i in rows}
    ca, cb = {columns[i] for i in a.answer_column_ids if i in columns}, {columns[i] for i in b.answer_column_ids if i in columns}
    if ra & rb and not ca & cb:
        return "same_row", f"Both questions use table row {min(ra & rb)+1}; the requested column changes from {min(ca)+1} to {min(cb)+1}."
    if ca & cb and not ra & rb:
        return "same_column", f"Both questions use table column {min(ca & cb)+1}; the requested row changes from {min(ra)+1} to {min(rb)+1}."
    return None, ""


def compute(context, parameters):
    if context.comparison is None:
        raise ValueError("Select a comparison capture of a matched question on the same table")
    table, part, m1, row, _ = measure(context, parameters)
    first = grounding(context, parameters)
    other = context.comparison
    matched_record(context.record, other.record, table, other.table)
    second = grounding(other, parameters)
    if other.table.case.question_id == table.case.question_id:
        raise ValueError("The comparison capture asks the same question; choose a different question on this table")
    kind, description = relation(table.table, first, second)
    if kind is None:
        raise ValueError("The two questions are not matched: they must share an answer row or an answer column, and differ in the other")
    weights, mask = vector(other, parameters)
    m2 = masses(partition(other.record, table, selected_step(other.record, parameters.step).target_count), weights, mask)
    g1, g2 = path_groups(table.table, first), path_groups(table.table, second)
    a1, a2 = g1["answer"], g2["answer"]
    p1, p2 = lookup_cells(table.table, first), lookup_cells(table.table, second)
    d1, d2 = sorted(p1-p2), sorted(p2-p1)
    groups = (("This question's answer cell", a1), ("Comparison question's answer cell", a2),
              ("Cells only this question's path uses", d1), ("Cells only the comparison path uses", d2))
    values = [[group_mass(m.cells, cells) for _, cells in groups] for m in (m1, m2)]
    def contrast(i, j):
        x = [values[0][i], values[1][i], values[1][j], values[0][j]]
        return None if any(v is None for v in x) else (x[0]-x[1])+(x[2]-x[3])
    own_first = None if None in (values[0][0], values[0][1]) else values[0][0]-values[0][1]
    own_second = None if None in (values[1][1], values[1][0]) else values[1][1]-values[1][0]
    cells = table.table.cells
    return Calculation(row=row, table_allocation=allocation(table, part, m1, selection_label(parameters)),
        statistics=[statistic("Answer-cell contrast", contrast(0, 1)), statistic("Distinct-path contrast", contrast(2, 3)),
                    statistic("This capture: own minus other answer cell", own_first),
                    statistic("Comparison capture: own minus other answer cell", own_second),
                    statistic("Cells only this path uses", len(d1), "cells"), statistic("Cells only the comparison path uses", len(d2), "cells")],
        table=result_table("Matched-question masses", [g[0] for g in groups],
                           [f"This capture · {table.case.question_id}", f"Comparison · {other.table.case.question_id}"], values, "mass"),
        tables=[result_table("Cells that differ between the two lookup paths", ["This capture", "Comparison capture", "Used by"],
                             [cell_label(cells[i]) for i in [*d1, *d2]],
                             [[None if not np.isfinite(m1.cells[i]) else float(m1.cells[i]), None if not np.isfinite(m2.cells[i]) else float(m2.cells[i]),
                               1.0 if i in d1 else 2.0] for i in [*d1, *d2]], "mass; mass; 1 = this question, 2 = comparison",
                             [cells[i].native_id for i in [*d1, *d2]])],
        notices=[description, "Contrast = [m₁(X₁) − m₂(X₁)] + [m₂(X₂) − m₁(X₂)]: positive when each question allocates more to its own "
                 "cells than the other question does. The page, table and resolution are identical, so only the question changes.",
                 "Each capture uses its own prediction at the selected step; their generated histories differ after the question.",
                 resolution_notice(part), EVALUATION_NOTICE])


DEFINITION = define("table.matched_contrast", "Matched-question contrast",
    "For questions 1 and 2 sharing an answer row or column: contrast(X) = [m₁(X₁) − m₂(X₁)] + [m₂(X₂) − m₁(X₂)] for the answer cells "
    "and for the cells only one lookup path uses.",
    GroundingParameters, compute, requires=GROUNDING_REQUIRES,
    summary="Two questions on the same page that share an answer row (or column) and differ in the other: does attention move to each "
            "question's own answer cell and header path?",
    limitations=["One pair is an observation, not a test; repeat across matched pairs and tables before generalising."])
