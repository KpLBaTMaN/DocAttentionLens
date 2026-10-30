"""Mapping audit: how finely the captured image positions resolve the table's cells."""
from collections import defaultdict
from .contracts import Calculation
from ._shared import statistic
from ._table import Parameters, measure, allocation, selection_label, result_table, resolution_notice, cell_label, define, EVALUATION_NOTICE


def compute(context, parameters):
    table, part, m, row, _ = measure(context, parameters)
    cells = table.table.cells
    by_row, by_column = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for i, cell in enumerate(cells):
        for r in cell.row_indices:
            by_row[r][0] += 1; by_row[r][1] += bool(part.isolable[i])
        for c in cell.column_indices:
            by_column[c][0] += 1; by_column[c][1] += bool(part.isolable[i])
    # Cells sharing a position with this cell: the neighbours its mass cannot be separated from.
    touching = part.cells > 0
    shared = [int(touching[touching[:, i]].any(axis=0).sum())-1 if touching[:, i].any() else 0 for i in range(len(cells))]
    r = part.resolution
    return Calculation(row=row, table_allocation=allocation(table, part, m, selection_label(parameters)),
        statistics=[statistic("Isolable cells", r.isolable_cells, "cells"), statistic("Cells", r.cell_count, "cells"),
                    statistic("Isolable share", r.isolable_cells/r.cell_count if r.cell_count else None, "fraction"),
                    statistic("Image positions on the table", r.positions_on_table, "positions"),
                    statistic("Table rows per position", r.rows_per_position, "rows"),
                    statistic("Table columns per position", r.columns_per_position, "columns")],
        table=result_table("Cell resolution", ["Touching positions", "Largest position share", "Isolable", "Cells sharing its positions"],
                           [cell_label(c) for c in cells],
                           [[float(part.touching[i]), float(part.largest_share[i]), float(part.isolable[i]), float(shared[i])] for i in range(len(cells))],
                           "positions; fraction; 1 = isolable; cells", [c.native_id for c in cells]),
        tables=[result_table("Isolable cells by table row", ["Cells", "Isolable"], [f"Row {k+1}" for k in sorted(by_row)],
                             [[float(v[0]), float(v[1])] for _, v in sorted(by_row.items())], "cells"),
                result_table("Isolable cells by table column", ["Cells", "Isolable"], [f"Column {k+1}" for k in sorted(by_column)],
                             [[float(v[0]), float(v[1])] for _, v in sorted(by_column.items())], "cells")],
        notices=[resolution_notice(part), "A cell is isolable when some image position lies at least half inside it. "
                 "The audit uses position geometry only; the attention row is included so the page and heatmap can draw it.", EVALUATION_NOTICE])


DEFINITION = define("table.mapping_audit", "Table mapping audit",
    "For cell c: positions with area(b_j ∩ c) > 0, max_j area(b_j ∩ c)/area(b_j), and isolable when that maximum is at least 0.5.",
    Parameters, compute, units="positions; fraction",
    summary="Which table cells the captured image positions can separate, and which share every position with neighbours. "
            "Accompanies every cell-level result.")
