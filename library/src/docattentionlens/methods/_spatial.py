"""Verified image geometry and result presentation for spatial attention methods."""
import numpy as np
from .contracts import Calculation, ResearchPlot, ResearchPoint
from ._attention_only import matrix, source_masks, table
from ._shared import statistic

def inputs(context,p):
    record=context.record
    payload=matrix(context,p.layer)
    images=[x for x in record.positions if x.role=="image" and x.bbox and record.attention_mask[x.index]]
    if not images: raise ValueError("Spatial analysis needs geometrically mapped image patches")
    pages={x.page_id for x in images}
    if len(pages)!=1: raise ValueError("Spatial matrix analysis requires one aligned page")
    indices=np.array([x.index for x in images],dtype=int)
    boxes=np.array([x.bbox for x in images],dtype=float)
    if not np.isfinite(boxes).all() or np.any(boxes[:,2:]<=boxes[:,:2]):
        raise ValueError("Image geometry is invalid")
    centres=(boxes[:,:2]+boxes[:,2:])/2
    xs=np.unique(centres[:,0]); ys=np.unique(centres[:,1])
    grid=np.array([[np.searchsorted(ys,y),np.searchsorted(xs,x)] for x,y in centres])
    support=source_masks(record)[np.ix_(indices,indices)]
    weights=payload[p.head][np.ix_(indices,indices)].astype(float)
    mass=weights.sum(-1)
    measured=mass>0
    if p.source_scope=="selected":
        source=record.steps[0].source_index
        if source not in indices: raise ValueError("Select a mapped image-patch source")
        measured &= indices==source
    if not measured.any(): raise ValueError("No measured image-to-image source rows")
    conditional=np.divide(weights,mass[:,None],out=np.zeros_like(weights),where=mass[:,None]>0)
    uniform=support/support.sum(-1,keepdims=True)
    return indices,boxes,centres,grid,conditional,uniform,measured,mass

def result(context,p,data,values,title,columns,extras,*,units="image-conditional mass",plots=None):
    indices,boxes,centres,grid,conditional,uniform,measured,mass=data
    lo=boxes[:,:2].min(0); extent=boxes[:,2:].max(0)-lo
    points=[ResearchPoint(label=f"Source patch {indices[i]}",x=float((centres[i,0]-lo[0])/extent[0]),
        y=float((centres[i,1]-lo[1])/extent[1]),value=float(values[i]),page_id=context.record.positions[indices[i]].page_id)
        for i in np.flatnonzero(measured)]
    return Calculation(source_indices=indices[measured].tolist(),
        statistics=[statistic(title,float(values[measured].mean()),units),statistic("Measured image sources",int(measured.sum()),"sources"),
                    statistic("Mean original image mass",float(mass[measured].mean()))],
        table=table(title+" by source",columns,[f"Patch {indices[i]}" for i in np.flatnonzero(measured)],
                    [[float(v) for v in extras[i]] for i in np.flatnonzero(measured)],units),
        plots=[ResearchPlot(kind="grid",title=title+" by image source",x_label="Page x (fraction)",y_label="Page y (fraction)",points=points),*(plots or [])],
        normalization="each source conditioned on its eligible image mass; equal weight over measured sources; valid zero keys retained in baselines",
        notices=["Patch centres and grid positions describe the language encoding's image tokens, not vision-tower attention.",
                 "Non-image attention is excluded from spatial quantities; mean original image mass is reported separately."])
