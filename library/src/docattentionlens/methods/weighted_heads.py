"""Nonnegative normalized head weights, separate from display opacity."""
import numpy as np
from pydantic import Field
from .contracts import HeadParameters, Calculation, MethodDefinition, MAX_HEADS
from ._shared import definition, selected_rows, selected_step, selection
from .element_allocation import allocate_row


class Parameters(HeadParameters):
    weights: list[float] = Field(default_factory=list, max_length=MAX_HEADS,
                                description="One nonnegative weight per selected head; empty means equal weights.")


def prepare(parameters, record):
    heads = selection(parameters.heads, record.query_head_count)
    if not parameters.weights:
        parameters.weights = [1.0] * len(heads)
    w = np.asarray(parameters.weights)
    if len(w) != len(heads) or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("Weights must align with selected heads, be nonnegative and have positive total")
    return parameters


def compute(context, parameters):
    heads = selection(parameters.heads, context.record.query_head_count)
    weights = np.ones(len(heads)) if not parameters.weights else np.asarray(parameters.weights, dtype=np.float64)
    if weights.shape != (len(heads),) or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Head weights must align with selected heads, be nonnegative and have positive total")
    weights = weights / weights.max()
    rows, _ = selected_rows(context, parameters, heads=heads)
    row = allocate_row(context.record, (rows[0]*(weights/weights.sum())[:,None]).sum(axis=0),
        selected_step(context.record, parameters.step), parameters.step, parameters.layer, -1)
    return Calculation(row=row)


DEFINITION = definition(method_id="attention.weighted_heads", name="Weighted Head Aggregation", perspective="general", family="aggregation",
    quantity="attention_weights", formula="Weighted mean: sum_h w_h A_h / sum_h w_h; nonnegative head weights.", parameters=Parameters, calculate=compute,
    literature=("standard_summary", "Normalized weighted mean with researcher-chosen weights; not a learned or gradient-based head-importance method."))
DEFINITION = MethodDefinition(DEFINITION.descriptor, Parameters, compute, True, prepare)
