"""Image-conditioned locality over explicit grid radii."""
from typing import Literal
from pydantic import Field
import numpy as np
from .contracts import RowParameters, ResearchPlot, ResearchPoint
from ._attention_only import define
from ._spatial import inputs,result
from ._literature import DOSOVITSKIY_2021_VIT, YEH_2023_ATTENTIONVIZ

class Parameters(RowParameters):
    source_scope: Literal["selected","all_image"] = "all_image"
    radius: int = Field(default=1,ge=0,le=64,strict=True)
    profile_radius: int = Field(default=8,ge=1,le=64,strict=True)

def calculate(context,p):
    data=inputs(context,p)
    indices,boxes,centres,grid,weights,uniform,measured,mass=data
    distance=np.abs(grid[:,None]-grid[None,:]).max(-1)
    local=(weights*(distance<=p.radius)).sum(-1)
    baseline=(uniform*(distance<=p.radius)).sum(-1)
    points=[ResearchPoint(label=f"Radius {r}",x=r,y=float((weights*(distance<=r)).sum(-1)[measured].mean()),group="Captured attention")
            for r in range(p.profile_radius+1)]
    points += [ResearchPoint(label=f"Uniform radius {r}",x=r,y=float((uniform*(distance<=r)).sum(-1)[measured].mean()),group="Valid-key uniform baseline") for r in range(p.profile_radius+1)]
    output=result(context,p,data,local,"Local attention mass",["Local mass","Uniform baseline","Local/uniform ratio","Self mass"],
        np.column_stack([local,baseline,np.divide(local,baseline,out=np.zeros_like(local),where=baseline>0),np.diag(weights)]),
        plots=[ResearchPlot(kind="line",title="Cumulative image attention by grid radius",x_label="Chebyshev radius (grid cells)",y_label="Image-conditional mass",points=points)])
    return output

DEFINITION=define("attention.local_global","Local / Global Attention","spatial_attention","local_attention_mass",
    "Attention mass within a declared Chebyshev grid radius, with cumulative radius profiles and valid-key uniform baselines.",
    Parameters,calculate,requires=["attention_weights","full_square_attention_matrices","complete_declared_support","image_mask"],displays=["patch_grid","layer_series"],
    references=[DOSOVITSKIY_2021_VIT,YEH_2023_ATTENTIONVIZ],
    literature=("adaptation","Local versus global attention follows observations in ViT (Dosovitskiy et al. 2021) and AttentionViz (Yeh et al. 2023); Chebyshev grid radii, cumulative mass and valid-key uniform baselines are DocAttentionLens choices."))
