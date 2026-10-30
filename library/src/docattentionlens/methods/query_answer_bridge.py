"""Two distinct edge populations: signed query winner contributions and answer attention."""
from .contracts import RowParameters, Calculation, ResearchEdge
from ._document import masses, table, define, REQUIRES


class Parameters(RowParameters):
    pass


def compute(context,parameters):
    _, measured, known, _=masses(context,parameters)
    retrieval=context.retrieval
    if retrieval is None or context.record.retrieval_capture_id!=retrieval.capture_id:
        raise ValueError("Matching retrieval lineage is required for the query-to-answer bridge")
    eligible=[p for p in retrieval.query_positions if p.eligible]
    edges=[]
    for page in retrieval.pages:
        if len(page.winner_indices)!=len(eligible) or len(page.query_maxima)!=len(eligible):
            raise ValueError("MaxSim winners differ from the eligible query population")
        for q,winner,score in zip(eligible,page.winner_indices,page.query_maxima,strict=True):
            if not 0<=winner<len(page.ownership): raise ValueError("MaxSim winner is outside saved positions")
            for i,fraction in page.ownership[winner].items():
                if i in known:
                    edges.append(ResearchEdge(source=f"Query {q.index}: {q.piece}",target=i,value=score*fraction,signal="retrieval",units="signed MaxSim contribution"))
    selected=next(s for s in context.record.steps if s.step==parameters.step)
    edges += [ResearchEdge(source=i,target=f"Prediction {parameters.step}: {selected.predicted_piece}",value=v,signal="attention",units="original-domain attention mass") for i,v in measured.items()]
    return Calculation(edges=edges,table=table("Element bridge",["Signed query-winner contribution","Answer-prediction attention mass"],
        [f"{known[i].element_type} · {known[i].native_id}" for i in sorted(measured)],
        [[sum(e.value for e in edges if e.signal=="retrieval" and e.target==i),measured[i]] for i in sorted(measured)],"signed score; mass",sorted(measured),[known[i].page_id for i in sorted(measured)]),
        notices=["Retrieval edges and attention edges have different units and are never summed into a flow score. One captured answer prediction is selected."])


DEFINITION=define("attention.query_answer_bridge","Query-to-answer bridge","Link actual query MaxSim winner ownership to annotated elements and the selected captured answer prediction.",Parameters,compute,
    requires=REQUIRES+["retrieval_evidence"],units="signed MaxSim contribution; attention mass",signed=True)
