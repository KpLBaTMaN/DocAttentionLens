"""Annotated header/footer/page-number allocation and explicitly bounded margin cells."""
from pydantic import Field
from .contracts import RowParameters, Calculation
from ._document import masses, document, table, define


class Parameters(RowParameters):
    margin_fraction: float = Field(default=0.05, ge=0, le=0.25)


def compute(context, parameters):
    row, measured, known, mask = masses(context, parameters)
    groups=["pageheader","pagefooter","pagenumber"]
    pages={p.page_id:p for p in document(context).pages}
    values=[]
    for kind in groups:
        selected=[i for i in measured if known[i].element_type.casefold()==kind]
        # A group with no measured element is unavailable, not a measured zero.
        values.append([sum(measured[i] for i in selected) if selected else None,float(len(selected))])
    margin=0.0
    for i,p in enumerate(context.record.positions[:len(mask)]):
        if mask[i] and p.role=="image" and p.bbox and p.page_id in pages:
            page=pages[p.page_id]; x1,y1,x2,y2=p.bbox; f=parameters.margin_fraction
            # Background fraction only: annotated margin content stays in its type group.
            if ((x1+x2)/2 < f*page.width or (x1+x2)/2 > (1-f)*page.width
                or (y1+y2)/2 < f*page.height or (y1+y2)/2 > (1-f)*page.height):
                margin+=row.values[i]*p.ownership.get("background",0)
    values.append([margin,None])
    return Calculation(row=row,table=table("Declared layout sink groups",["Attention mass","Measured elements"],groups+["Background in declared margin"],values,"mass; elements"),
        notices=["Margins use image-cell centres and only background ownership; groups are descriptive allocations, not persistent or causal sinks.",
                 "A group with no measured annotated element has unavailable mass, not zero."])


DEFINITION=define("attention.layout_sinks","Layout sinks","Allocation to annotated page headers, footers, page numbers and background cells within an explicit margin.",Parameters,compute)
