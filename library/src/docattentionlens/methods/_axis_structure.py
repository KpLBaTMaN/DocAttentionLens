"""Common presentation for row/column geometry; owner modules select the measured axis."""
import numpy as np
from .contracts import ResearchPlot,ResearchPoint
from ._spatial import inputs,result

def structure(context,p,axis,name):
    data=inputs(context,p)
    indices,boxes,centres,grid,weights,uniform,measured,mass=data
    offset=grid[None,:,axis]-grid[:,None,axis]
    other=~np.eye(len(indices),dtype=bool)
    same=(offset==0)&other
    values=(weights*same).sum(-1); baseline=(uniform*same).sum(-1)
    points=[ResearchPoint(label=f"{name} offset {int(d)}",x=float(d),y=float((weights*((offset==d)&other)).sum(-1)[measured].mean()))
            for d in np.unique(offset)]
    return result(context,p,data,values,f"Same-{name.lower()} attention mass",
        ["Same-axis mass (self excluded)","Uniform baseline","Self mass"],
        np.column_stack([values,baseline,np.diag(weights)]),
        plots=[ResearchPlot(kind="bar",title=f"{name} offset allocation (self excluded)",x_label=f"{name} offset (grid cells)",y_label="Image-conditional mass",points=points)])
