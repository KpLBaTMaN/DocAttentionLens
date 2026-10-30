"""Unreduced depth comparison, preserving captured layer identities."""
from typing import Annotated
from pydantic import Field
from .contracts import RowParameters, Calculation, ComparisonRow, MAX_LAYERS
from ._shared import definition, selected_rows, selected_step, selection
from .element_allocation import allocate_row
from ._literature import VIG_2019_BERTVIZ


class Parameters(RowParameters):
    layers: list[Annotated[int, Field(strict=True, ge=0, lt=MAX_LAYERS)]] = Field(default_factory=list, max_length=MAX_LAYERS, description="Empty selects all captured layers.")


def compute(context, parameters):
    layers = selection(parameters.layers, context.record.layer_count)
    rows, _ = selected_rows(context, parameters, layers=layers)
    step = selected_step(context.record, parameters.step)
    return Calculation(comparisons=[ComparisonRow(label=f"Layer {layer}", row=allocate_row(context.record, rows[i,0], step, parameters.step, layer, parameters.head)) for i, layer in enumerate(layers)])


DEFINITION = definition(method_id="attention.across_layers", name="Attention Across Layers", perspective="general", family="direct_attention",
    quantity="attention_weights", formula="Compare unreduced rows across captured layers for the same prediction and query head.",
    parameters=Parameters, calculate=compute, displays=["layer_series"],
    references=[VIG_2019_BERTVIZ],
    literature=("published_method", "Layer-by-layer inspection as in BertViz's multiscale views (Vig 2019); there is no propagation between layers."))
