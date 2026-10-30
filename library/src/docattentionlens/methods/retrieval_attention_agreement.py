"""Rank agreement retains signed retrieval contributions and original attention units."""
import numpy as np
from .contracts import RowParameters, Calculation
from ._document import masses, table, plot, define, REQUIRES
from ._shared import statistic


class Parameters(RowParameters):
    pass


def ranks(values):
    values=np.asarray(values); result=np.empty(len(values),dtype=float)
    for value in np.unique(values):
        indices=np.flatnonzero(values==value)
        result[indices]=float(np.sum(values<value))+(len(indices)+1)/2
    return result


def spearman(a,b):
    if len(a)<2: return None
    x,y=ranks(a),ranks(b)
    if np.ptp(x)==0 or np.ptp(y)==0: return None
    return float(np.corrcoef(x,y)[0,1])


def compute(context,parameters):
    _, measured, known, _=masses(context,parameters)
    retrieval=context.retrieval
    if retrieval is None or context.record.retrieval_capture_id!=retrieval.capture_id:
        raise ValueError("A retrieval capture with matching generator lineage is required")
    # An element that owns retrieval patches but wins no query token is measured at exactly zero.
    # Elements without retrieval ownership remain missing, never zero.
    owned={i for p in retrieval.pages for o in p.ownership for i,f in o.items() if f>0}
    contributions={i:sum(p.contributions.get(i,0.0) for p in retrieval.pages) for i in measured if i in owned}
    ids=sorted(contributions); x=[contributions[i] for i in ids]; y=[measured[i] for i in ids]
    return Calculation(statistics=[statistic("Spearman rank correlation",spearman(x,y)),statistic("Aligned measured elements",len(ids),"elements")],
        table=table("Retrieval versus generator allocation",["Signed MaxSim contribution","Attention mass"],
            [f"{known[i].element_type} · {known[i].native_id}" for i in ids],[[contributions[i],measured[i]] for i in ids],"signed score; mass",ids,[known[i].page_id for i in ids]),
        plots=[plot("scatter","Retrieval–attention agreement","Signed MaxSim contribution (score)","Generator attention mass",
            [dict(label=f"{known[i].element_type} · {known[i].native_id}",x=contributions[i],y=measured[i],element_id=i,page_id=known[i].page_id) for i in ids])],
        notices=["Only elements measured in both stages participate. An element that owns retrieval patches but wins no query token contributes exactly zero; elements outside retrieval ownership are excluded.",
                 "Constant or insufficient rank populations have unavailable correlation."])


DEFINITION=define("attention.retrieval_agreement","Retrieval–attention agreement","Spearman correlation and paired values on jointly mapped elements; signed retrieval scores remain signed.",Parameters,compute,
    requires=REQUIRES+["retrieval_evidence"],units="rank correlation; signed score; mass",signed=True)
