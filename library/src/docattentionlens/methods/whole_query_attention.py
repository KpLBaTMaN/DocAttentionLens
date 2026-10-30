"""Whole-query image-conditioned aggregation, retaining every source's original image mass."""
from typing import Literal
import numpy as np
from .contracts import HeadParameters, Calculation
from ._shared import selection, selected_step, statistic
from ._attention_only import define, population, table
from .element_allocation import allocate_row
from ._literature import AFLALO_2022_VL_INTERPRET

class Parameters(HeadParameters):
    head_reduction: Literal["mean","max","min"] = "mean"
    token_reduction: Literal["mean","max"] = "mean"
    source_group: Literal["question","text"] = "question"

def calculate(context, p):
    record=context.record
    source_record=context.source_record or record
    group=p.source_group
    fallback=group=="question" and getattr(record,"encoding_attention",False) and not any(x.role=="query" for x in record.positions)
    if fallback: group="text"
    heads=selection(p.heads,record.query_head_count)
    blocks=[population(context,p.layer,h,group) for h in heads]
    indices=blocks[0][2]
    if any(b[2]!=indices for b in blocks): raise ValueError("Head source axes differ")
    reduced=getattr(np,p.head_reduction if p.head_reduction!="mean" else "mean")(np.stack([b[0] for b in blocks]),axis=0)
    image=[x.index for x in source_record.positions if x.role=="image"]
    if not image: raise ValueError("Whole-query page allocation requires captured image targets")
    masses=reduced[:,image].sum(-1)
    keep=masses>0
    if not keep.any(): raise ValueError("No selected source assigns attention to the image")
    conditioned=reduced[keep][:,image]/masses[keep,None]
    combined=getattr(np,p.token_reduction)(conditioned,axis=0)
    step=selected_step(record,p.step)
    weights=np.zeros(step.target_count)
    if max(image)>=step.target_count: raise ValueError("Selected source domain does not contain every image target")
    weights[image]=combined
    row=allocate_row(record,weights,step,p.step,p.layer,-1,quantity="whole_query_image_allocation")
    row=row.model_copy(update={"source_index":None,"predicted_index":None})
    return Calculation(row=row, source_indices=indices,
        statistics=[statistic("Captured sources",len(indices),"sources"),statistic("Sources with positive image mass",int(keep.sum()),"sources"),
                    statistic("Mean original image mass",masses.mean()),statistic("Combined map mass",weights.sum())],
        table=table("Original image attention by source",["Image mass","Included"],[f"{i} · {source_record.positions[i].piece}" for i in indices],
                    [[float(m),float(k)] for m,k in zip(masses,keep)],"original-domain mass"),
        normalization=f"heads: {p.head_reduction}; each positive-image source conditioned on its image mass; sources: {p.token_reduction}",
        notices=[*(['No question tokens occur in this document encoding; captured document-prompt/special text sources are aggregated instead. This is not question-conditioned retrieval attention.'] if fallback else []),
                 "Every source's image mass is retained in the table; zero-image rows are excluded.",
                 *(['Max across sources is an envelope whose sum may exceed one.'] if p.token_reduction=="max" else [])])

DEFINITION=define("attention.whole_query","Whole-Query Attention","aggregation","whole_query_image_allocation",
    "Reduce selected heads, condition each captured question/text source on its image attention, then combine source distributions by mean or max.",
    Parameters,calculate,requires=["attention_weights","question_source_rows","image_mask","complete_declared_support"],
    displays=["page_overlay","source_table"],limitations=["Only captured question sources are used; document encoding falls back explicitly to its own text prompt."],
    references=[AFLALO_2022_VL_INTERPRET],
    literature=("adaptation","Builds on text-to-image attention inspection (VL-InterpreT). Conditioning each source on its image mass before mean or max is a DocAttentionLens recipe with no verified published original; low- and high-image-mass sources can receive equal weight."))
