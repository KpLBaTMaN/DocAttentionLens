"""Thresholded raw attention edges retaining every source, target, layer and head identity."""
from heapq import heappush,heapreplace
from pydantic import Field
import numpy as np
from .contracts import HeadParameters,AxisIndex,Calculation,AttentionGraph,AttentionGraphNode,AttentionGraphEdge
from ._attention_only import define,matrix,source_masks,table
from ._shared import selection,statistic
from ._literature import ABNAR_ZUIDEMA_2020

class Parameters(HeadParameters):
    layers: list[AxisIndex] = Field(default_factory=lambda:[0],max_length=128)
    threshold: float = Field(default=.1,gt=0,le=1)
    max_edges: int = Field(default=500,ge=1,le=5000,strict=True)

def calculate(context,p):
    record=context.record
    matrix(context,p.layer)
    layers=selection(p.layers,record.layer_count); heads=selection(p.heads,record.query_head_count)
    if len(layers)*len(heads)>64: raise ValueError("Graph scanning is bounded to 64 layer/head combinations")
    allowed=source_masks(record)&np.asarray(record.attention_mask,bool)[:,None]
    kept=[]; total=0
    for layer in layers:
        values=matrix(context,layer)
        for head in heads:
            selected=np.argwhere((values[head]>=p.threshold)&allowed)
            total+=len(selected)
            for source,target in selected:
                weight=float(values[head,source,target])
                item=(weight,-layer,-head,-int(source),-int(target))
                if len(kept)<p.max_edges: heappush(kept,item)
                elif item>kept[0]: heapreplace(kept,item)
    edges=[]; nodes={}
    for weight,layer,head,source,target in sorted(kept,reverse=True):
        layer,head,source,target=-layer,-head,-source,-target
        source_id=f"L{layer+1}:P{source}"; target_id=f"L{layer}:P{target}"
        for node_id,level,index in ((source_id,layer+1,source),(target_id,layer,target)):
            nodes[node_id]=AttentionGraphNode(node_id=node_id,label=f"{index} · {record.positions[index].piece}",level=level,position=index,role=record.positions[index].role)
        edges.append(AttentionGraphEdge(source=source_id,target=target_id,value=weight,layer=layer,head=head))
    graph=AttentionGraph(title="Thresholded raw attention graph",nodes=list(nodes.values()),edges=edges,units="original post-softmax weight",total_edges=total)
    return Calculation(graph=graph,source_indices=np.flatnonzero(record.attention_mask).tolist(),
        statistics=[statistic("Edges above threshold",total,"edges"),statistic("Retained strongest edges",len(edges),"edges"),statistic("Threshold",p.threshold)],
        table=table("Retained raw attention edges",["Layer","Head","Source","Target","Weight"],
            [f"{e.source} → {e.target}" for e in edges],
            [[e.layer,e.head,int(e.source.split('P')[1]),int(e.target.split('P')[1]),e.value] for e in edges],"indices; original weight"),
        normalization="raw captured weights; no propagation or renormalization",
        notices=[f"Retained the strongest {len(edges)} of {total} thresholded edges; ties follow layer/head/source/target order.",
                 "Parallel head edges remain separate. Graph edges represent allocation, not causal information flow."])

DEFINITION=define("attention.graph","Attention Graph","attention_graphs","raw_attention_edges",
    "Inspect strong raw source-to-target attention edges over selected layers and heads, with declared threshold and edge limits.",
    Parameters,calculate,displays=["attention_graph","edge_table"],units="original post-softmax weight",
    references=[ABNAR_ZUIDEMA_2020],
    literature=("adaptation","Attention graphs follow Abnar & Zuidema (2020); this keeps separate raw per-head edges above a threshold, capped in number, without their head averaging or residual terms."))
