"""Explicit pipeline inclusion counts; answer agreement is not evidence grounding."""
from pydantic import Field
from .contracts import RowParameters, Calculation
from ._document import masses,document,sample_contexts,table,define,REQUIRES
from ..evaluation.analysis import assess_answer


class Parameters(RowParameters):
    attention_threshold: float = Field(default=0.01,ge=0,le=1)
    retrieval_k: int = Field(default=2,ge=1,le=8,strict=True)


def compute(context,parameters):
    labels=[]; values=[]
    for sample in sample_contexts(context):
        _, measured, known, _=masses(sample,parameters); case=document(sample).case
        support=set(case.support_element_ids)
        retrieval=sample.retrieval
        candidate={p.page_id for p in retrieval.pages} if retrieval else None
        top={p.page_id for p in sorted(retrieval.pages,key=lambda p:(-p.score,p.page_number,p.page_id))[:parameters.retrieval_k]} if retrieval else None
        supplied=set(sample.record.context.selected_page_ids)
        count=lambda pages: float(sum(i in known and known[i].page_id in pages for i in support)) if pages is not None else None
        verdict=None
        if document(sample).question_type in ("short","true_or_false"):
            verdict=float(assess_answer(sample.record.answer,case.reference_answer,"boolean_strict_v1" if document(sample).question_type=="true_or_false" else "normalized_exact_v1")["verdict"]=="correct")
        labels.append(case.query)
        values.append([float(len(support)),count(candidate),count(top),count(supplied),float(len(support&set(measured))),
            float(sum(i in measured and measured[i]>=parameters.attention_threshold for i in support)),verdict])
    return Calculation(table=table("Designated-evidence pipeline measurements",["Designated elements","In candidate scope","On top-k pages","On supplied pages","Mapped in attention","At/above attention threshold","Answer rubric agreement"],labels,values,"element counts; answer agreement 0/1"),
        notices=["Candidate-scope retrieval only. Attention threshold is descriptive; it does not measure whether evidence was used.","Answer agreement is separate from grounding. Long answers require human adjudication and remain unavailable here."])


DEFINITION=define("attention.evidence_funnel","Evidence funnel","Track designated element inclusion through bounded retrieval, context, attention mapping and separately measured answer agreement.",Parameters,compute,
    requires=REQUIRES+["designated_support_annotations"],units="element counts; rubric agreement")
