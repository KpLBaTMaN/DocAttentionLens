"""Target-wise minimum envelope, deliberately not normalized as a probability."""
from .contracts import HeadParameters, Calculation
from ._shared import definition, selected_rows, selected_step
from .element_allocation import allocate_row
from ._literature import VIT_EXPLAIN_SOFTWARE


class Parameters(HeadParameters):
    pass


def compute(context, parameters):
    rows, _ = selected_rows(context, parameters, heads=parameters.heads)
    row = allocate_row(context.record, rows[0].min(axis=0), selected_step(context.record, parameters.step), parameters.step, parameters.layer, -1, "head_minimum_envelope")
    return Calculation(row=row, normalization=row.normalization)


DEFINITION = definition(method_id="attention.min_heads", name="Min-Head Attention", perspective="general", family="aggregation",
    quantity="head_envelope", units="minimum head-weight envelope", normalization="No renormalization; envelope sum need not be one",
    formula="For each causal target, take the minimum captured weight across selected heads.", parameters=Parameters, calculate=compute,
    references=[VIT_EXPLAIN_SOFTWARE],
    literature=("standard_summary", "Elementwise minimum across heads, offered as a rollout head-fusion option in vit-explain; no originating paper verified. The envelope is not a probability distribution."))
