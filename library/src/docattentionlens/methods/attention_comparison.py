"""Aligned element differences plus separately labelled distribution and rank comparisons."""
import numpy as np
from pydantic import Field
from .contracts import RowParameters, Calculation
from ._document import aligned_masses,elements,table,define,REQUIRES
from ._shared import statistic


class Parameters(RowParameters):
    top_k: int = Field(default=5,ge=1,le=25760,strict=True)


def compute(context,parameters):
    a,b,ids,ma,mb,alignment=aligned_masses(context,parameters)
    x=np.array([ma[i] for i in ids]); y=np.array([mb[alignment[i]] for i in ids])
    divergence=None
    if x.sum()>0 and y.sum()>0:
        p,q=x/x.sum(),y/y.sum(); mean=(p+q)/2
        def kl(v):
            valid=v>0
            return float(np.sum(v[valid]*np.log2(v[valid]/mean[valid])))
        divergence=max(0.0,min(1.0,(kl(p)+kl(q))/2))
    k=min(parameters.top_k,len(ids))
    ta=set(sorted(ids,key=lambda i:(-ma[i],i))[:k]); tb=set(sorted(ids,key=lambda i:(-mb[alignment[i]],i))[:k])
    # Rows are baseline elements; aligned comparison elements are reached through the lineage.
    known=elements(context)
    return Calculation(table=table("Aligned element attention differences",["Baseline mass","Comparison mass","Comparison − baseline"],
        [f"{known[i].element_type} · {known[i].native_id}" for i in ids],
        [[ma[i],mb[alignment[i]],mb[alignment[i]]-ma[i]] for i in ids],"original-domain mass",ids,[known[i].page_id for i in ids]),
        statistics=[statistic("Jensen–Shannon divergence on aligned element support",divergence,"bits"),
            statistic("Top-k overlap fraction",len(ta&tb)/k if k else None,"fraction"),
            statistic("Baseline mass outside aligned elements",a.total_mass-float(x.sum()),"mass"),
            statistic("Comparison mass outside aligned elements",b.total_mass-float(y.sum()),"mass")],
        normalization="differences retain original-domain masses; Jensen–Shannon separately normalizes each aligned-element vector",
        notices=["Shared model/revision and consumed output history are required. Unaligned support remains outside the divergence domain."])


DEFINITION=define("attention.aligned_comparison","Aligned element attention comparison","Original-domain element differences, conditional aligned-support Jensen–Shannon divergence and deterministic top-k overlap.",Parameters,compute,
    requires=REQUIRES+["paired_capture"],units="mass difference; divergence bits; overlap fraction",signed=True)
