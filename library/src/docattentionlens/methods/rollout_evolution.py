"""Rollout maps at explicit end layers, with one shared residual/head treatment."""
from dataclasses import replace
from typing import Literal
from pydantic import Field
from .contracts import HeadParameters,AxisIndex,Calculation,ComparisonRow,ResearchPlot,ResearchPoint
from ._attention_only import define
from ._propagation import rollouts
from ._shared import selected_step,selection,statistic
from .element_allocation import allocate_row
from ._literature import ABNAR_ZUIDEMA_2020

class Parameters(HeadParameters):
    start_layer: int = Field(default=0,ge=0,lt=128,strict=True)
    layers: list[AxisIndex] = Field(default_factory=list,max_length=32)
    head_reduction: Literal["mean","max"] = "mean"
    residual_coefficient: float = Field(default=.5,ge=0,le=1)

MAX_END_LAYERS=32

def default_ends(start_layer,layer_count):
    """Every end layer from the start, or, when that exceeds the bound, evenly spaced ends that keep the first and last."""
    count=layer_count-start_layer
    if count<=MAX_END_LAYERS: return list(range(start_layer,layer_count))
    return sorted({start_layer+round(i*(count-1)/(MAX_END_LAYERS-1)) for i in range(MAX_END_LAYERS)})

def prepare(p,record):
    # Explicit lists are validated to at most 32 layers, so a longer list is the resolved "all layers" default:
    # on a deeper model it becomes evenly spaced ends, saved as the layers actually compared.
    if len(p.layers)>MAX_END_LAYERS:
        p.layers=default_ends(p.start_layer,record.layer_count)
    return p

def calculate(context,p):
    ends=selection(p.layers,context.record.layer_count)
    if len(ends)>MAX_END_LAYERS: raise ValueError("Rollout evolution is bounded to 32 end layers")
    sampled=context.record.layer_count-p.start_layer>MAX_END_LAYERS and ends==default_ends(p.start_layer,context.record.layer_count)
    if any(end<p.start_layer for end in ends): raise ValueError("Every selected end layer must follow the start layer")
    step=selected_step(context.record,p.step)
    # One downward pass reduces each layer once; rows equal separate per-end rollouts exactly.
    rows=[allocate_row(context.record,weights,step,p.step,end,-1,quantity="rollout_evolution_connectivity")
          for end,weights in zip(ends,rollouts(context,p,p.start_layer,ends))]
    frames=[ComparisonRow(label=f"Rollout {p.start_layer}–{end}",row=row) for end,row in zip(ends,rows)]
    has_image=any(position.role=="image" for position in context.record.positions[:step.target_count])
    # Rollout rows sum to one, so image mass and the source's own entry are shares.
    points=[ResearchPoint(label=f"Rollout {p.start_layer}–{end} · image share",x=end,y=row.image_mass,group="Image share")
            for end,row in zip(ends,rows) if has_image]
    points+=[ResearchPoint(label=f"Rollout {p.start_layer}–{end} · source self share",x=end,y=row.values[step.source_index],group="Source self share")
             for end,row in zip(ends,rows)]
    return Calculation(comparisons=frames,source_indices=[step.source_index],
        statistics=[statistic("Compared rollout end layers",len(ends),"layers")],
        plots=[ResearchPlot(kind="line",title=f"Rollout shares by end layer (start layer {p.start_layer})",
            x_label="End layer",y_label="Share of rollout mass",points=points)],
        normalization=f"common head {p.head_reduction} and residual identity {p.residual_coefficient}; normalized square-matrix propagation",
        notices=["Each map is propagated rollout; this is distinct from comparing unpropagated raw layer rows.",
                 *([f"All {context.record.layer_count-p.start_layer} end layers exceed the bound of {MAX_END_LAYERS}, so {len(ends)} evenly spaced end layers are shown, including the first and last: {', '.join(map(str,ends))}. Choose end layers in the method settings to compare others."] if sampled else []),
                 *([] if has_image else ["This encoding has no image positions, so no image share is plotted."])])

DEFINITION=define("attention.rollout_evolution","Rollout Evolution","connectivity","rollout_evolution_connectivity",
    "Compare propagated attention maps at selected end layers, with a shared start layer, head reduction and residual treatment.",
    Parameters,calculate,displays=["layer_series","page_overlay"],
    references=[ABNAR_ZUIDEMA_2020],
    literature=("published_method","Rollout at successive end layers, as Abnar & Zuidema (2020) present rollout layer by layer. Matches from start layer 0 with every head, mean reduction and residual coefficient 0.5; the image and self-share plot is an addition."))
DEFINITION=replace(DEFINITION,prepare=prepare)
