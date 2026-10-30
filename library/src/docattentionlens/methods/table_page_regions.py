"""Table and page budget: attention on the table, its caption and notes, the rest of the page and non-image positions."""
from .contracts import Calculation
from ._shared import statistic, selected_rows
from ._table import (Parameters, measure, allocation, selection_label, masses, result_table, plot,
                     resolution_notice, EVALUATION_NOTICE, define)


def compute(context, parameters):
    table, part, m, row, _ = measure(context, parameters)
    result = allocation(table, part, m, selection_label(parameters))
    values = [[r.mass, r.mass/m.image if m.image > 0 and not r.region.startswith("non_image") else None, r.area_share, r.enrichment]
              for r in result.regions]
    # Layer profile at the selected head or head mean: the same regions over every captured layer.
    heads = list(range(context.record.query_head_count)) if parameters.head_reduction == "head_mean" else [parameters.head]
    rows, mask = selected_rows(context, parameters, heads=heads, layers=list(range(context.record.layer_count)))
    points = []
    for layer in range(rows.shape[0]):
        lm = masses(part, rows[layer].mean(axis=0), mask)
        for key, label in (("table", "Table cells"), ("table_wrap", "Caption and notes"), ("rest_of_page", "Rest of the page")):
            points.append({"label": f"Layer {layer} · {label}", "x": layer, "y": lm.regions[key], "group": label})
        points.append({"label": f"Layer {layer} · Non-image", "x": layer, "y": float(sum(lm.non_image.values())), "group": "Non-image positions"})
    table_share = m.regions["table"]/m.image if m.image > 0 else None
    return Calculation(row=row, table_allocation=result,
        statistics=[statistic("Table mass", m.regions["table"]), statistic("Caption and notes mass", m.regions["table_wrap"]),
                    statistic("Rest-of-page mass", m.regions["rest_of_page"]), statistic("Non-image mass", sum(m.non_image.values())),
                    statistic("Image mass", m.image), statistic("Table share of image attention", table_share, "fraction"),
                    statistic("Table enrichment over its page area", next(r.enrichment for r in result.regions if r.region == "table"), "ratio")],
        table=result_table("Table and page regions", ["Mass", "Share of image attention", "Share of page area", "Enrichment"],
                           [r.label for r in result.regions], values, "mass; fraction; fraction; ratio"),
        plots=[plot("line", "Regions across layers", "Layer", "Attention mass", points)],
        notices=[resolution_notice(part), "Enrichment is the share of image attention over the share of page area; 1 is uniform over the page.",
                 "The rest of the page is unannotated: figures, body text and running heads are not separated.", EVALUATION_NOTICE])


DEFINITION = define("table.page_regions", "Table and page budget",
    "m(r) = Σ_j A(s,j) w(r,j) for r ∈ {table cells, table gap, caption and notes, rest of page} and each non-image role; "
    "enrichment(r) = (m(r)/m_image)/(area(r)/area(page)).", Parameters, compute,
    summary="How much of the selected prediction's attention goes to the table, its caption and notes, the rest of the page "
            "and non-image positions, with area-adjusted enrichment and a layer profile.")
