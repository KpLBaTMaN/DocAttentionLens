"""Document-balanced unit-page allocation over an explicit saved sample."""
import numpy as np
from pydantic import Field
from .contracts import RowParameters, Calculation
from ._document import document, row, sample_contexts, table, plot, define, REQUIRES


class Parameters(RowParameters):
    grid_size: int = Field(default=12,ge=2,le=32,strict=True)


def compute(context, parameters):
    size=parameters.grid_size; docs={}; support_docs={}
    for sample in sample_contexts(context):
        doc=document(sample); selected,mask=row(sample,parameters)
        if selected.image_mass<=0: raise ValueError("Spatial profiles require positive mapped image mass")
        pages={p.page_id:p for p in doc.pages}; grid=np.zeros((size,size)); evidence=np.zeros_like(grid)
        for i,p in enumerate(sample.record.positions[:len(mask)]):
            if not mask[i] or p.role!="image" or not p.bbox: continue
            page=pages[p.page_id]; x1,y1,x2,y2=p.bbox
            x1,x2=x1/page.width,x2/page.width; y1,y2=y1/page.height,y2/page.height
            area=(x2-x1)*(y2-y1)
            if area<=0 or x1<0 or y1<0 or x2>1+1e-8 or y2>1+1e-8: raise ValueError("Patch lies outside its mapped unit page")
            for r in range(max(0,int(y1*size)),min(size,int(np.ceil(y2*size)))):
                for c in range(max(0,int(x1*size)),min(size,int(np.ceil(x2*size)))):
                    overlap=max(0,min(x2,(c+1)/size)-max(x1,c/size))*max(0,min(y2,(r+1)/size)-max(y1,r/size))
                    grid[r,c]+=selected.values[i]*overlap/area/selected.image_mass
        supplied=set(sample.record.context.selected_page_ids)
        for e in doc.elements:
            if e.element_id in doc.case.support_element_ids and e.page_id in supplied and not e.geometry_flags:
                page=pages[e.page_id]; b=e.bbox
                c=min(size-1,int((b.x1+b.x2)/2/page.width*size)); r=min(size-1,int((b.y1+b.y2)/2/page.height*size))
                evidence[r,c]+=1
        if not np.isclose(grid.sum(),1,atol=1e-8): raise ValueError("Spatial rasterization did not conserve image mass")
        if evidence.sum(): evidence/=evidence.sum()
        docs.setdefault(doc.case.document_id,[]).append(grid)
        if evidence.sum(): support_docs.setdefault(doc.case.document_id,[]).append(evidence)
    mean=np.mean([np.mean(v,axis=0) for v in docs.values()],axis=0)
    support=np.mean([np.mean(v,axis=0) for v in support_docs.values()],axis=0) if support_docs else None
    points=[dict(label=f"Cell {r+1},{c+1}",x=(c+.5)/size,y=(r+.5)/size,value=float(mean[r,c])) for r in range(size) for c in range(size)]
    return Calculation(table=table("Unit-page spatial allocation",["x centre","y centre","Attention share","Support-centre share"],
        [p["label"] for p in points],[[p["x"],p["y"],p["value"],float(support[r,c]) if support is not None else None] for p,(r,c) in zip(points,[(r,c) for r in range(size) for c in range(size)],strict=True)],"normalized page coordinates; fraction"),
        plots=[plot("grid","Document-balanced spatial attention","Normalized page x (0–1)","Normalized page y (0–1)",points)],
        normalization="per-capture image-only share, then mean within document, then equal-document mean",
        notices=[f"{len(docs)} documents, {len(sample_contexts(context))} captures; supplied pages only. Support histogram uses element centres.","Position allocation does not isolate position bias from document content or aspect ratio."])


DEFINITION=define("attention.spatial_prior_map","Spatial prior map","Rasterize normalized image allocation across saved captures with equal-document weighting.",Parameters,compute,
    requires=REQUIRES+["corpus_sample_manifest"],normalization="per-capture image-only share, then within-document mean and equal-document mean")
