"""Unreduced head comparison for a retained prediction and selected layer."""
from .contracts import HeadParameters, Calculation, ComparisonRow
from ._shared import definition, selected_rows, selected_step, selection
from .element_allocation import allocate_row
from ._literature import VIG_2019_BERTVIZ


class Parameters(HeadParameters):
    pass


def compute(context, parameters):
    heads = selection(parameters.heads, context.record.query_head_count)
    rows, _ = selected_rows(context, parameters, heads=heads)
    step = selected_step(context.record, parameters.step)
    return Calculation(comparisons=[ComparisonRow(label=f"Head {head}", row=allocate_row(context.record, rows[0,i], step, parameters.step, parameters.layer, head)) for i, head in enumerate(heads)])


DEFINITION = definition(method_id="attention.head_by_head", name="Head-by-Head Attention", perspective="general", family="direct_attention",
    quantity="attention_weights", formula="Compare unreduced captured head rows at the same prediction and layer.",
    parameters=Parameters, calculate=compute, displays=["head_grid"],
    references=[VIG_2019_BERTVIZ],
    literature=("published_method", "Per-head inspection as demonstrated in BertViz (Vig 2019); heads are compared without reduction."))
