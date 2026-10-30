"""Sink detection from mean incoming allocation over an explicit captured source population."""
from typing import Literal
from pydantic import Field
import numpy as np
from .contracts import RowParameters,Calculation
from ._attention_only import define,population,table
from ._shared import statistic
from ._literature import GU_2024_SINKS, XIAO_2023_SINKS

class Parameters(RowParameters):
    source_group: Literal["all","text","image","question"] = "all"
    threshold: float = Field(default=.3,gt=0,lt=1)
    top_k: int = Field(default=5,ge=1,le=32,strict=True)

def calculate(context,p):
    record=context.source_record or context.record
    rows,masks,sources=population(context,p.layer,p.head,p.source_group)
    if len(sources)<2: raise ValueError("Sink detection needs at least two captured source rows")
    alpha=rows.mean(0)
    candidates=np.flatnonzero(masks.any(0))
    order=sorted(candidates.tolist(),key=lambda i:(-alpha[i],i))[:p.top_k]
    top=order[0]
    head_values=[]
    for head in range(record.query_head_count):
        other,_,ids=population(context,p.layer,head,p.source_group)
        if ids!=sources: raise ValueError("Head source populations differ")
        head_values.append(float(other.mean(0).max()))
    special=[i for i,x in enumerate(record.positions[:len(alpha)]) if x.role=="special"]
    return Calculation(source_indices=sources,
        statistics=[statistic("Strongest sink score",alpha[top]),statistic("Selected head has a sink",float(alpha[top]>p.threshold)),
                    statistic("Layer sink-head rate",np.mean(np.array(head_values)>p.threshold)),
                    statistic("First-position score",alpha[0]),statistic("Special-role mean mass",alpha[special].sum()),
                    statistic("Captured sources",len(sources),"sources")],
        table=table("Mean received attention: strongest targets",["Position","Mean incoming weight","Exceeds threshold"],
            [f"{i} · {record.positions[i].piece}" for i in order],[[i,float(alpha[i]),float(alpha[i]>p.threshold)] for i in order]),
        tables=[table("Selected source allocation to strongest sink",["Weight"],
                      [f"{i} · {record.positions[i].piece}" for i in sources],[[float(row[top])] for row in rows]),
                table("Strongest sink score by head",["Score","Exceeds threshold"],
                      [f"Head {h}" for h in range(len(head_values))],[[v,float(v>p.threshold)] for v in head_values])],
        normalization="alpha_k is arithmetic mean original attention to target k across the explicitly captured sources; each source has equal weight",
        notices=[f"A sink is declared only when alpha_k > {p.threshold}; causal support and unobserved source coverage affect the score.",
                 "This is a captured-source sink detector, separate from selected-row role allocation and document Layout sinks."])

DEFINITION=define("attention.sink_detection","Attention Sink Detection","attention_statistics","captured_sink_score",
    "Detect sink targets from mean incoming attention over captured sources and report thresholded sink scores and the selected layer's sink-head rate.",
    Parameters,calculate,requires=["attention_weights","multiple_source_rows","complete_declared_support"],displays=["scalar_statistic","source_table"],
    references=[GU_2024_SINKS,XIAO_2023_SINKS],
    literature=("adaptation","Adapts the sink score of Gu et al. (2024) for the sinks named by Xiao et al. (2023). Gu averages a token's incoming attention over the queries that can attend to it and reports a rate over all layers and heads; this averages every captured source, takes the strongest target and reports one layer's rate. The 0.3 threshold follows Gu."))
