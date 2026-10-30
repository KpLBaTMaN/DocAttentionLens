"""Selected-head mean on the complete original captured causal domain."""
from .contracts import HeadParameters, Calculation
from ._shared import definition, selected_rows, selected_step
from .element_allocation import allocate_row
from ._literature import ABNAR_ZUIDEMA_2020


class Parameters(HeadParameters):
    pass


def compute(context, parameters):
    rows, _ = selected_rows(context, parameters, heads=parameters.heads)
    weights = rows[0].mean(axis=0)
    return Calculation(row=allocate_row(context.record, weights, selected_step(context.record, parameters.step), parameters.step, parameters.layer, -1))


DEFINITION = definition(method_id="attention.mean_heads", name="Mean-Head Attention", perspective="general", family="aggregation",
    quantity="attention_weights", formula="Arithmetic mean across the selected query heads; original causal support retained.",
    parameters=Parameters, calculate=compute,
    references=[ABNAR_ZUIDEMA_2020],
    literature=("standard_summary", "Equal-weight head mean: a generic reduction, like the maximum and minimum. Abnar & Zuidema (2020) use it with every head selected before rollout; it is not a distinct published method."))
