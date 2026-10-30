"""Top-k mass with deterministic original-index tie handling."""
import numpy as np
from pydantic import Field
from .contracts import RowParameters, Calculation
from ._shared import definition, selected_rows, statistic


class Parameters(RowParameters):
    k: int = Field(default=1, ge=1, le=1600, strict=True)


def compute(context, parameters):
    rows, mask = selected_rows(context, parameters)
    values = rows[0,0,mask]
    order = np.argsort(-values, kind="stable")
    return Calculation(statistics=[statistic(f"Top-{parameters.k} attention mass", values[order[:parameters.k]].sum(), "dimensionless mass"),
        statistic("Total captured mass", values.sum(), "dimensionless mass"), statistic("Valid targets", len(values), "positions")])


DEFINITION = definition(method_id="attention.topk_mass", name="Top-k Attention Mass", perspective="general", family="attention_statistics",
    quantity="topk_attention_mass", formula="Sum the k strongest valid causal target weights; ties use original position order.",
    parameters=Parameters, calculate=compute, displays=["scalar_statistic"],
    literature=("standard_summary", "Sum of the k largest weights; no originating attention-analysis paper verified."))
