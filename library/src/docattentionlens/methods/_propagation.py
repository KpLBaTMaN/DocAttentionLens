"""Residual/head treatment for saved square-matrix propagation variants."""
import numpy as np
from ._attention_only import matrix
from ._shared import selection

def reduced(context,p,layer,cache=None):
    """Head-reduced, residual-mixed, row-normalized layer. `cache` (a dict owned by one
    calculation with fixed parameters) keeps each layer so it is validated and reduced once."""
    if cache is not None and layer in cache: return cache[layer]
    record=context.record
    heads=selection(p.heads,record.query_head_count)
    values=matrix(context,layer)[heads].astype(np.float64)
    weights=getattr(np,p.head_reduction)(values,axis=0)
    active=np.asarray(record.attention_mask,bool)
    weights[:,~active]=0
    mixed=(1-p.residual_coefficient)*weights+p.residual_coefficient*np.diag(active.astype(float))
    sums=mixed.sum(-1,keepdims=True)
    result=np.divide(mixed,sums,out=np.zeros_like(mixed),where=sums>0)
    if cache is not None: cache[layer]=result
    return result

def rollouts(context,p,start,ends,cache=None):
    """Source rollout row for every end layer, reducing each layer once.

    Layers are visited once from the highest end downward. Each row starts at its own end
    layer and is multiplied by the same matrices in the same order as a separate `rollout`
    call, so the rows are bit-identical to separate calls. Without `cache`, only one reduced
    layer is held at a time.
    """
    if any(not 0<=start<=end<context.record.layer_count for end in ends):
        raise ValueError("Choose a captured contiguous start/end layer interval")
    if not ends: return []
    source=context.record.steps[0].source_index
    rows={}
    for layer in range(max(ends),start-1,-1):
        if layer in ends:
            rows[layer]=np.zeros(len(context.record.positions)); rows[layer][source]=1
        weights=reduced(context,p,layer,cache)
        for end in rows: rows[end]=rows[end]@weights
    return [rows[end] for end in ends]

def rollout(context,p,start,end,cache=None):
    return rollouts(context,p,start,[end],cache)[0]
