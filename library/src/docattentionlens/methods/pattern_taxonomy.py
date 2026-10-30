"""Rule-based pattern labels with disclosed features and thresholds."""
from typing import Literal
from pydantic import Field
import numpy as np
from .contracts import RowParameters,AxisIndex,Calculation
from ._attention_only import define,table
from ._head_population import profiles
from ._shared import statistic
from ._literature import KOVALEVA_2019

class Parameters(RowParameters):
    layers: list[AxisIndex] = Field(default_factory=lambda:[0],max_length=128)
    heads: list[AxisIndex] = Field(default_factory=list,max_length=128)
    source_group: Literal["all","text","image","question"] = "all"
    max_sources: int = Field(default=64,ge=2,le=256,strict=True)
    sink_threshold: float = Field(default=.3,gt=0,lt=1)
    diagonal_threshold: float = Field(default=.5,ge=0,le=1)
    cross_modal_threshold: float = Field(default=.5,ge=0,le=1)
    block_threshold: float = Field(default=.9,ge=0,le=1)
    broad_threshold: float = Field(default=.8,ge=0,le=1)

def classify(features,p):
    sink,diagonal,visual,same,entropy=features
    tags=[]
    if sink>p.sink_threshold: tags.append("vertical")
    if diagonal>=p.diagonal_threshold: tags.append("diagonal")
    if visual is not None and visual>=p.cross_modal_threshold: tags.append("cross-modal")
    if same>=p.block_threshold: tags.append("block")
    if entropy>=p.broad_threshold: tags.append("broad")
    return ("vertical+diagonal" if "vertical" in tags and "diagonal" in tags else tags[0] if tags else "heterogeneous"),tags

def calculate(context,p):
    axes,rows,masks,sources=profiles(context,p)
    record=context.source_record or context.record
    image=np.array([x.role=="image" for x in record.positions[:rows.shape[-1]]])
    source_image=image[sources]; values=[]; labels=[]
    for (layer,head),heads in zip(axes,rows):
        sink=float(heads.mean(0).max())
        diagonal=np.zeros(len(sources))
        for offset in (-1,0,1):
            targets=np.array(sources)+offset; ok=(targets>=0)&(targets<heads.shape[-1])
            diagonal[ok]+=heads[np.flatnonzero(ok),targets[ok]]
        mass=heads[:,image].sum(-1)
        visual=float(mass[~source_image].mean()) if (~source_image).any() else None
        same=float(np.where(source_image,mass,1-mass).mean())
        positive=heads>0
        logs=np.log2(heads,out=np.zeros_like(heads),where=positive)
        entropy=-(heads*logs).sum(-1)
        denominator=np.log2(np.maximum(masks.sum(-1),1))
        normalized=np.divide(entropy,denominator,out=np.zeros_like(entropy),where=denominator>0)
        features=[sink,float(diagonal.mean()),visual,same,float(normalized.mean())]
        label,tags=classify(features,p)
        labels.append(f"L{layer} · H{head} · {label} ({', '.join(tags) or 'no threshold matched'})")
        values.append(features)
    return Calculation(source_indices=sources,
        statistics=[statistic("Compared heads",len(axes),"heads"),statistic("Sampled sources",len(sources),"sources")],
        table=table("Heuristic attention pattern labels",["Sink score","Previous/self/next mass","Text → image mass","Same-modality mass","Normalized entropy"],labels,values),
        normalization="source-wise normalized captured rows; normalized entropy uses the declared valid-key count including valid zeros",
        notices=[f"Thresholds: sink > {p.sink_threshold}; diagonal ≥ {p.diagonal_threshold}; cross-modal ≥ {p.cross_modal_threshold}; block ≥ {p.block_threshold}; broad ≥ {p.broad_threshold}.",
                 "Labels are threshold-based descriptions of this sample, not demonstrated functions of the heads. All satisfied tags are shown."])

DEFINITION=define("heads.pattern_taxonomy","Heuristic attention-pattern labels","head_and_representation","heuristic_head_patterns",
    "Describe captured heads with disclosed vertical, diagonal, cross-modal, block, broad and heterogeneous pattern rules.",
    Parameters,calculate,requires=["attention_weights","multiple_source_rows","complete_declared_support"],displays=["head_grid","head_table"],
    references=[KOVALEVA_2019],
    literature=("adaptation","Adapts the pattern classes of Kovaleva et al. (2019), who labelled maps by hand and trained a CNN classifier. Here hand-set thresholds assign labels, broad and cross-modal tags are added, and block means same-modality mass rather than sentence blocks."))
