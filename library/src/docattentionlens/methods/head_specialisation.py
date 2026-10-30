"""Corpus head preferences on a declared sample; stable behaviour needs held-out replication."""
import numpy as np
from pydantic import Field
from .contracts import AxisIndex,RowParameters,Calculation
from ._document import sample_contexts,masses,document,table,define,REQUIRES
from ._shared import selection


class Parameters(RowParameters):
    layers: list[AxisIndex] = Field(default_factory=lambda:[0],max_length=128)
    heads: list[AxisIndex] = Field(default_factory=list,max_length=128)


def profile(context,parameters,vocabulary):
    row,measured,known,_=masses(context,parameters)
    if row.image_mass<=0: raise ValueError("Type profiles require positive image mass")
    values=[sum(v for i,v in measured.items() if known[i].element_type==kind)/row.image_mass for kind in vocabulary]
    return np.array(values+[max(0,1-sum(values))],dtype=float)


def vocabulary_of(samples):
    """Element types on pages each sample actually supplied to the model.

    Types found only on unsupplied pages were never measured, so they must not become
    columns of zeros.
    """
    kinds=set()
    for c in samples:
        supplied=set(c.record.context.selected_page_ids)
        kinds.update(e.element_type for e in document(c).elements if e.page_id in supplied)
    return sorted(kinds)


def document_balanced_profile(samples,step,layer,head,vocabulary):
    """The declared head profile: equal mean over documents of within-document mean type profiles."""
    docs={}
    selected=RowParameters(step=step,layer=layer,head=head)
    for sample in samples:
        docs.setdefault(document(sample).case.document_id,[]).append(profile(sample,selected,vocabulary))
    return np.mean([np.mean(v,axis=0) for v in docs.values()],axis=0)


def compute(context,parameters):
    samples=sample_contexts(context)
    vocabulary=vocabulary_of(samples)
    labels=[]; values=[]
    for layer in selection(parameters.layers,context.record.layer_count):
        for head in selection(parameters.heads,context.record.query_head_count):
            mean=document_balanced_profile(samples,parameters.step,layer,head,vocabulary)
            preferred=int(np.argmax(mean)); sorted_values=np.sort(mean)
            name=(vocabulary+["background/unreviewed image"])[preferred]
            labels.append(f"Layer {layer} · head {head} · preferred {name}")
            values.append([float(layer),float(head),*mean.tolist(),float(sorted_values[-1]-sorted_values[-2]) if len(mean)>1 else float(mean[0])])
    return Calculation(table=table("Document-balanced head preference catalogue",["Layer","Head",*vocabulary,"Background/unreviewed image","Top-share gap"],labels,values,"indices; image-attention fractions"),
        normalization="per-row image-attention profile; equal mean of document means",
        notices=[f"{len(samples)} saved captures; preferences describe this declared sample. They are not validated stable specialisation labels."])


DEFINITION=define("attention.head_specialisation","Head preference catalogue","Document-balanced element-type preferences and selectivity gap for selected captured heads and layers.",Parameters,compute,
    requires=REQUIRES+["corpus_sample_manifest"],units="image-attention fraction",
    normalization="per-row image-only shares; equal-document mean of within-document profile means")
