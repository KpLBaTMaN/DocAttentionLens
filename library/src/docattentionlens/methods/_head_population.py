"""Bounded, explicitly sampled source rows shared by head analyses."""
import numpy as np
from ._attention_only import population
from ._shared import selection

def profiles(context,p):
    record=context.record
    axes=[(l,h) for l in selection(p.layers,record.layer_count) for h in selection(p.heads,record.query_head_count)]
    if len(axes)>64: raise ValueError("Head analysis is bounded to 64 profiles; choose fewer layers or heads")
    blocks=[population(context,l,h,p.source_group) for l,h in axes]
    indices=blocks[0][2]
    if any(b[2]!=indices for b in blocks): raise ValueError("Head profiles have different source identities")
    if len(indices)>p.max_sources:
        reference=context.source_record or record
        text=[i for i in indices if reference.positions[i].role!="image"]
        image=[i for i in indices if reference.positions[i].role=="image"]
        if len(text)>p.max_sources:
            selected=[text[int(i)] for i in np.linspace(0,len(text)-1,p.max_sources)]
        else:
            left=p.max_sources-len(text)
            selected=text+([image[int(i)] for i in np.linspace(0,len(image)-1,min(left,len(image)))] if left and image else [])
        indices=sorted(set(selected))
    picked=[blocks[0][2].index(i) for i in indices]
    rows=np.stack([b[0][picked] for b in blocks])
    masks=blocks[0][1][picked]
    rows=rows/rows.sum(-1,keepdims=True)
    return axes,rows,masks,indices

def divergences(rows):
    n=len(rows); result=np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            p,q=rows[i],rows[j]; middle=(p+q)/2
            def term(x):
                positive=x>0
                return np.sum(x[positive]*np.log2(x[positive]/middle[positive]))
            result[i,j]=result[j,i]=max(0,min(1,.5*(term(p)+term(q))/len(p)))
    return result
