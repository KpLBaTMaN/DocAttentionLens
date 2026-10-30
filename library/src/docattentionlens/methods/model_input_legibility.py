"""Consumed element dimensions are size proxies, not a measured readability judgement."""
from .contracts import RowParameters, Calculation
from ._document import masses, table, plot, define, REQUIRES


class Parameters(RowParameters):
    pass


def compute(context, parameters):
    row, measured, known, _ = masses(context, parameters)
    pages={p.page_id:p for p in context.record.context.pages}
    ids=sorted(i for i in measured if known[i].page_id in pages and not known[i].geometry_flags)
    values=[]; points=[]
    for i in ids:
        e=known[i]; p=pages[e.page_id]; b=e.bbox
        w=(b.x2-b.x1)*p.resized_width/p.source_width
        h=(b.y2-b.y1)*p.resized_height/p.source_height
        if w<=0 or h<=0: raise ValueError("Element has no valid processed area")
        values.append([w,h,w*h,measured[i],measured[i]/(w*h)])
        points.append(dict(label=f"{e.element_type} · {e.native_id}",x=h,y=measured[i],group=e.element_type,element_id=i,page_id=e.page_id))
    return Calculation(row=row,table=table("Consumed-input element size",["Width (px)","Height (px)","Area (px²)","Attention mass","Mass / px²"],
        [f"{known[i].element_type} · {known[i].native_id}" for i in ids],values,"consumed pixels; mass; mass/px²",ids,[known[i].page_id for i in ids]),
        plots=[plot("scatter","Processed element height versus attention","Consumed element height (px)","Original-domain attention mass",points)],
        notices=["Element height and area are legibility proxies. Text glyph height, OCR accuracy and actual readability are not measured."])


DEFINITION=define("attention.model_input_legibility","Model-input size and attention","Compare processed element dimensions with original-domain attention mass.",Parameters,compute,requires=REQUIRES+["image_mask"])
