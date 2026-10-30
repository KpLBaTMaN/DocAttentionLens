"""Residual attention rollout over an explicitly selected layer interval."""
from typing import Literal
from pydantic import Field
from .contracts import HeadParameters,Calculation
from ._attention_only import define
from ._propagation import rollout
from ._shared import selected_step,statistic
from .element_allocation import allocate_row
from ._literature import ABNAR_ZUIDEMA_2020

class Parameters(HeadParameters):
    start_layer: int = Field(default=0,ge=0,lt=128,strict=True)
    head_reduction: Literal["mean","max"] = "mean"
    residual_coefficient: float = Field(default=.5,ge=0,le=1)

def calculate(context,p):
    weights=rollout(context,p,p.start_layer,p.layer)
    step=selected_step(context.record,p.step)
    row=allocate_row(context.record,weights,step,p.step,p.layer,-1,quantity="partial_rollout_connectivity")
    return Calculation(row=row,source_indices=[step.source_index],
        statistics=[statistic("Start layer",p.start_layer,"layer"),statistic("End layer",p.layer,"layer"),statistic("Residual identity coefficient",p.residual_coefficient)],
        normalization=f"head {p.head_reduction}; row-normalized residual mixing; matrices composed from {p.start_layer} through {p.layer}",
        notices=["The distribution addresses inputs of the start layer, not necessarily original model inputs. Propagation is an approximate routing view."])

DEFINITION=define("attention.partial_rollout","Partial Attention Rollout","connectivity","partial_rollout_connectivity",
    "Compose head-reduced, residual-normalized attention from an explicit start layer through the chosen end layer.",
    Parameters,calculate,displays=["page_overlay","matrix"],
    references=[ABNAR_ZUIDEMA_2020],
    literature=("published_method","Abnar & Zuidema (2020) Eq. 1 defines rollout between any two layers. Matches with every head, mean reduction and residual coefficient 0.5; max reduction, head subsets and other residual coefficients are extensions."))
