"""Pruned-capacity attention flow, a bounded lower-bound variant rather than exact dense flow."""
from typing import Literal,Annotated
from pydantic import Field
import numpy as np
from .contracts import HeadParameters,Calculation,AttentionGraph,AttentionGraphNode,AttentionGraphEdge
from ._attention_only import define,table
from ._propagation import reduced,rollout
from ._max_flow import FlowNetwork
from ._shared import statistic
from ._literature import ABNAR_ZUIDEMA_2020

class Parameters(HeadParameters):
    start_layer: int = Field(default=0,ge=0,lt=128,strict=True)
    head_reduction: Literal["mean","max"] = "mean"
    residual_coefficient: float = Field(default=.5,ge=0,le=1)
    mass_coverage: float = Field(default=.95,gt=0,le=1)
    max_edges_per_node: int = Field(default=32,ge=1,le=128,strict=True)
    max_targets: int = Field(default=16,ge=1,le=64,strict=True)
    target_indices: list[Annotated[int,Field(strict=True,ge=0,le=1535)]] = Field(default_factory=list,max_length=64)

def strongest(weights,active,coverage,limit):
    """Retained edges of one source row: positive active targets by descending weight, ties by
    position, until the running capacity reaches `coverage` or `limit` edges are kept.

    Returns targets, their weights and the retained capacity. `np.cumsum` adds sequentially,
    so the running capacity equals a left-to-right Python sum of the same weights.
    """
    values=weights[active]; positive=values>0
    targets=active[positive]; values=values[positive]
    order=np.lexsort((targets,-values)); targets=targets[order]; values=values[order]
    running=np.cumsum(values)
    reached=np.flatnonzero(running>=coverage-1e-12)
    count=min(len(targets),limit,int(reached[0])+1 if len(reached) else len(targets))
    return targets[:count].tolist(),values[:count].tolist(),float(running[count-1]) if count else 0.0

def calculate(context,p):
    record=context.record
    if not getattr(record,"encoding_attention",False) or context.read_array is None:
        raise ValueError("Attention flow needs complete saved attention matrices (a retriever encoding, or a generator capture with 'Save full attention matrices'); retained prediction rows are insufficient")
    if not 0<=p.start_layer<=p.layer<record.layer_count: raise ValueError("Choose a captured contiguous flow interval")
    source=record.steps[0].source_index; active=np.flatnonzero(record.attention_mask)
    edges=[]; reachable={source}; retained=[]; total_capacity=[]
    cache={}  # each layer is validated and reduced once, for the edges and for the rollout ranking
    for layer in range(p.layer,p.start_layer-1,-1):
        weights=reduced(context,p,layer,cache); next_sources=set()
        for position in sorted(reachable):
            selected,values,capacity=strongest(weights[position],active,p.mass_coverage,p.max_edges_per_node)
            retained.append(capacity); total_capacity.append(float(weights[position].sum()))
            for target,value in zip(selected,values):
                edges.append((f"L{layer+1}:P{position}",f"L{layer}:P{target}",value))
                next_sources.add(target)
        if len(edges)>500_000: raise ValueError("Flow edge budget exceeded; reduce layer interval or edges per node")
        reachable=next_sources
    routing=rollout(context,p,p.start_layer,p.layer,cache)
    if p.target_indices:
        if len(set(p.target_indices))!=len(p.target_indices) or not set(p.target_indices)<=set(active.tolist()):
            raise ValueError("Flow targets must be distinct active encoding positions")
        targets=p.target_indices
    else:
        targets=sorted(reachable,key=lambda i:(-routing[i],i))[:p.max_targets]
    if not targets: raise ValueError("The retained flow graph has no input targets")
    root=f"L{p.layer+1}:P{source}"
    network=FlowNetwork(edges)  # built once; each target solves on fresh residual capacities
    flows=[network.maximum_flow(root,f"L{p.start_layer}:P{target}") for target in targets]
    display=sorted(edges,key=lambda e:(-e[2],e[0],e[1]))[:5000]
    nodes={}
    for a,b,value in display:
        for identifier in (a,b):
            level,position=map(int,identifier.replace('L','').replace('P','').split(':'))
            nodes[identifier]=AttentionGraphNode(node_id=identifier,label=f"{position} · {record.positions[position].piece}",level=level,position=position,role=record.positions[position].role)
    graph=AttentionGraph(title="Retained residual-attention capacity graph",nodes=list(nodes.values()),
        edges=[AttentionGraphEdge(source=a,target=b,value=v,layer=int(a.split(':')[0][1:])-1) for a,b,v in display],
        units="residual-normalized edge capacity",total_edges=len(edges))
    return Calculation(graph=graph,source_indices=[source],
        statistics=[statistic("Calculated target flows",len(targets),"targets"),statistic("Retained graph edges",len(edges),"edges"),
                    statistic("Mean per-node retained capacity",np.mean(retained)),statistic("Minimum per-node retained capacity",min(retained)),
                    statistic("Node-weighted retained capacity fraction",sum(retained)/sum(total_capacity))],
        table=table("Independent input-target maximum flows",["Target position","Pruned maximum flow","Rollout mass"],
            [f"{t} · {record.positions[t].piece}" for t in targets],[[t,v,float(routing[t])] for t,v in zip(targets,flows)],"position; dimensionless capacity"),
        normalization="independent per-target maximum flows; not a probability distribution; edges use residual-normalized captured attention",
        notices=[f"Pruned variant: each reached node retains up to {p.max_edges_per_node} strongest edges toward {p.mass_coverage:.0%} capacity.",
                 "Pruning removes capacity, so measured flows are lower bounds of the corresponding dense graph's flows.",
                 "Only listed targets are measured; omitted targets are unmeasured, not zero. Automatic targets follow strongest rollout among reached inputs.",
                 f"The graph view retains the strongest {len(display)} of {len(edges)} calculation edges."])

DEFINITION=define("attention.flow","Attention Flow · pruned capacity","connectivity","pruned_maximum_flow",
    "Measure independent maximum flows through a bounded, explicitly pruned residual-attention graph and disclose retained capacity.",
    Parameters,calculate,displays=["attention_graph","target_table"],units="dimensionless capacity",
    limitations=["Independent flows do not sum to one. Target selection and edge pruning are material parameters."],
    references=[ABNAR_ZUIDEMA_2020],
    literature=("adaptation","Adapts attention flow (Abnar & Zuidema 2020): edges are pruned and only listed targets are solved, so flows are lower bounds on the dense graph's maximum flows."))
from dataclasses import replace
DEFINITION=replace(DEFINITION,descriptor=DEFINITION.descriptor.model_copy(update={"variant_id":"pruned_capacity_v1"}))
