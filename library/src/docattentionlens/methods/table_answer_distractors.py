"""Answer cell versus distractors: the answer cell against body cells in its row, its column, its
neighbourhood and cells holding the same value."""
import re
import numpy as np
from .contracts import Calculation
from ._shared import statistic
from ._table import (GroundingParameters, measure, allocation, selection_label, grounding, header_cells, cell_index, cell_label,
                     rank, result_table, resolution_notice, define, EVALUATION_NOTICE, GROUNDING_REQUIRES)


def normalise(text):
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def compute(context, parameters):
    table, part, m, row, _ = measure(context, parameters)
    ground = grounding(context, parameters)
    cells = table.table.cells
    index = cell_index(table.table)
    if not ground.answer_cell_ids or ground.answer_cell_ids[0] not in index:
        raise ValueError("The answer cell is absent from the table revision")
    a = index[ground.answer_cell_ids[0]]
    answer = cells[a]
    result = allocation(table, part, m, selection_label(parameters))
    headers = header_cells(table.table)
    body = [i for i, c in enumerate(cells) if c.native_id not in headers and i != a]
    near = lambda c: (min(abs(r-s) for r in c.row_indices for s in answer.row_indices) <= 1
                      and min(abs(r-s) for r in c.column_indices for s in answer.column_indices) <= 1)
    targets = {normalise(t) for t in [*ground.accepted_answers, answer.text or answer.pdf_text or ""] if normalise(t)}
    pdf_used = []
    def same_value(i):
        c = cells[i]
        text = c.text or c.pdf_text
        if not c.text and c.pdf_text:
            pdf_used.append(i)
        return bool(text) and normalise(text) in targets
    sets = {"Same row": [i for i in body if set(cells[i].row_indices) & set(answer.row_indices)],
            "Same column": [i for i in body if set(cells[i].column_indices) & set(answer.column_indices)],
            "Neighbours": [i for i in body if near(cells[i])],
            "Same value": [i for i in body if same_value(i)],
            "All other body cells": body}
    mass_a = m.cells[a]
    enrich = np.array([c.enrichment if c.enrichment is not None else np.nan for c in result.cells])
    rows, listed = [], {}
    for name, members in sets.items():
        measured = [i for i in members if np.isfinite(m.cells[i])]
        strongest = max(measured, key=lambda i: (m.cells[i], -i)) if measured else None
        strongest_enriched = max((i for i in measured if np.isfinite(enrich[i])), key=lambda i: (enrich[i], -i), default=None)
        mean = float(np.mean([m.cells[i] for i in measured])) if measured else None
        rows.append([float(len(members)), float(len(measured)), float(sum(m.cells[i] for i in measured)) if measured else None,
                     float(m.cells[strongest]) if strongest is not None else None,
                     float(mass_a-m.cells[strongest]) if strongest is not None and np.isfinite(mass_a) else None,
                     float(enrich[a]-enrich[strongest_enriched]) if strongest_enriched is not None and np.isfinite(enrich[a]) else None,
                     float(mass_a/mean) if mean and np.isfinite(mass_a) else None])
        for i in members:
            listed.setdefault(i, []).append(name)
    ranked = sorted((i for i in listed if np.isfinite(m.cells[i])), key=lambda i: (-m.cells[i], i))[:30]
    population = [a, *body]
    values = np.array([m.cells[i] for i in population])
    enrichments = np.array([enrich[i] for i in population])
    notices = []
    if not part.isolable[a]:
        sharing = sorted({cells[j].native_id for p in np.nonzero(part.cells[:, a] > 0)[0] for j in np.nonzero(part.cells[p] > 0)[0] if j != a})
        notices.append(f"The answer cell has no image position of its own; it shares positions with {len(sharing)} other cells, "
                       "so a small margin may not separate it from them.")
    if not np.isfinite(mass_a):
        notices.append("No valid target position overlaps the answer cell, so its comparisons are unavailable.")
    if pdf_used:
        notices.append(f"Same-value matching read the PDF text layer for {len(set(pdf_used))} cells without a transcription "
                       "(evaluation only; never given to the model).")
    if not sets["Same value"]:
        notices.append("No other body cell holds the answer value.")
    return Calculation(row=row, table_allocation=result,
        statistics=[statistic("Answer cell mass", mass_a if np.isfinite(mass_a) else None),
                    statistic("Answer rank among body cells (mass)", rank(values, 0), "rank"),
                    statistic("Answer rank among body cells (enrichment)", rank(enrichments, 0), "rank"),
                    statistic("Body cells compared", len(population), "cells"),
                    statistic("Margin over strongest body cell", rows[-1][4]),
                    statistic("Cells holding the same value", len(sets["Same value"]), "cells")],
        table=result_table("Answer cell versus distractor sets",
                           ["Cells", "Measured", "Set mass", "Strongest cell mass", "Answer − strongest (mass)",
                            "Answer − strongest (enrichment)", "Answer ÷ set mean"],
                           list(sets), rows, "cells; cells; mass; mass; mass; ratio; ratio"),
        tables=[result_table("Strongest distractor cells", ["Mass", "Enrichment", "Answer − cell"],
                             [f"{cell_label(cells[i])} · {', '.join(n for n in listed[i] if n != 'All other body cells') or 'other'}" for i in ranked],
                             [[float(m.cells[i]), None if not np.isfinite(enrich[i]) else float(enrich[i]),
                               float(mass_a-m.cells[i]) if np.isfinite(mass_a) else None] for i in ranked],
                             "mass; ratio; mass", [cells[i].native_id for i in ranked])],
        notices=[*notices, resolution_notice(part), "Distractors are body cells; header cells belong to the lookup path. "
                 "A correct answer string and attention on the answer location are separate observations.", EVALUATION_NOTICE])


DEFINITION = define("table.answer_distractors", "Answer cell versus distractors",
    "m(answer) against body-cell sets D ∈ {same row, same column, 8-neighbourhood, same normalised value, all}: "
    "Σ_D m(c), max_D m(c), m(answer) − max_D m(c), enrichment margin and m(answer)/mean_D m(c).",
    GroundingParameters, compute, requires=GROUNDING_REQUIRES,
    summary="Whether the answer cell stands out from the plausible alternatives: cells in its row and column, its neighbours "
            "and other cells holding the same value.")
