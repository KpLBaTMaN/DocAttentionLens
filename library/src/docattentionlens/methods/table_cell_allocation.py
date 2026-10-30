"""Cell allocation: attention aggregated into the table's logical cells, row bands and column bands."""
import numpy as np
from .contracts import Calculation
from ._shared import statistic
from ._table import (Parameters, measure, allocation, selection_label, result_table, plot, resolution_notice, cell_label,
                     define, EVALUATION_NOTICE)


def compute(context, parameters):
    table, part, m, row, _ = measure(context, parameters)
    result = allocation(table, part, m, selection_label(parameters))
    cells = table.table.cells
    measured = [(i, float(v)) for i, v in enumerate(m.cells) if np.isfinite(v)]
    within = np.array([v for _, v in measured])
    total = float(within.sum())
    entropy = None
    if total > 0:
        p = within[within > 0]/total
        entropy = float(-(p*np.log2(p)).sum())
    ranked = sorted(measured, key=lambda item: (-item[1], item[0]))
    top = ranked[:25]
    band_points = lambda bands, name: [{"label": f"{name} {b.index+1}", "x": b.index+1, "y": b.mass or 0.0,
                                        "value": b.enrichment, "group": name} for b in bands]
    return Calculation(row=row, table_allocation=result,
        statistics=[statistic("Table mass", m.regions["table"]), statistic("Measured cells", len(measured), "cells"),
                    statistic("Top-5 cell share of table mass", sum(v for _, v in ranked[:5])/total if total > 0 else None, "fraction"),
                    statistic("Cell entropy within the table", entropy, "bits"),
                    statistic("Effective number of cells", 2**entropy if entropy is not None else None, "cells"),
                    statistic("Isolable cells", part.resolution.isolable_cells, "cells")],
        table=result_table("Cells by attention mass (top 25)", ["Mass", "Share of table mass", "Enrichment", "Isolable"],
                           [cell_label(cells[i]) for i, _ in top],
                           [[v, v/total if total > 0 else None, result.cells[i].enrichment, float(part.isolable[i])] for i, v in top],
                           "mass; fraction; ratio; 1 = isolable", [cells[i].native_id for i, _ in top]),
        plots=[plot("bar", "Table row profile", "Table row", "Attention mass", band_points(result.rows, "Row")),
               plot("bar", "Table column profile", "Table column", "Attention mass", band_points(result.columns, "Column"))],
        notices=[resolution_notice(part), "Rows and columns are the table's own bands, not rows of the image patch grid. "
                 "Concentration is calculated within the table (cell masses divided by table mass).", EVALUATION_NOTICE])


DEFINITION = define("table.cell_allocation", "Cell allocation",
    "m(c) = Σ_j A(s,j) w(c,j) with w(c,j) = area(b_j ∩ c)/area(b_j); row and column band masses use the native disjoint bands; "
    "entropy and top-k shares use m(c)/Σ_c m(c).", Parameters, compute,
    summary="Attention on each logical table cell as a heatmap that keeps merged cells, with table row and column profiles, "
            "area-adjusted enrichment and concentration.")
