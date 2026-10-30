"""Deterministic PCA of comparable saved question profiles, with recorded feature columns."""
import numpy as np
from .contracts import RowParameters,Calculation
from ._document import sample_contexts,document,table,plot,define,REQUIRES
from .head_specialisation import profile,vocabulary_of


class Parameters(RowParameters):
    pass


def compute(context,parameters):
    samples=sample_contexts(context)
    if len(samples)<2: raise ValueError("Question atlas requires at least two distinct question captures")
    def question_identity(c):
        case=document(c).case
        return tuple(getattr(case,k,None) for k in ("dataset_id","dataset_version","document_id","question_id","query_variant","query"))
    if len({question_identity(c) for c in samples})!=len(samples):
        raise ValueError("Question atlas requires distinct questions, not repeated captures of one case")
    vocabulary=vocabulary_of(samples)
    matrix=np.stack([profile(c,parameters,vocabulary) for c in samples]); centred=matrix-matrix.mean(axis=0)
    _,s,axes=np.linalg.svd(centred,full_matrices=False)
    for axis in axes:
        pivot=int(np.argmax(np.abs(axis)))
        if axis[pivot]<0: axis*=-1
    coordinates=np.zeros((len(samples),2)); count=min(2,len(axes))
    coordinates[:,:count]=centred@axes[:count].T
    points=[dict(label=document(c).case.query,x=float(coordinates[i,0]),y=float(coordinates[i,1]),
                 group=document(c).question_type,capture_id=c.record.capture_id) for i,c in enumerate(samples)]
    variance=s*s; explained=variance[:2].sum()/variance.sum() if variance.sum() else 0
    return Calculation(table=table("Question attention profile features",vocabulary+["Background/unreviewed image"],
            [document(c).case.query for c in samples],matrix.tolist(),"image-attention fractions"),
        normalization="per-row image-attention fractions; features centred over the explicit question sample before PCA",
        plots=[plot("projection","Question attention atlas · PCA","Principal component 1 (profile units)","Principal component 2 (profile units)",points)],
        notices=[f"Deterministic centred PCA; two-axis explained variance {explained:.3%}. Features, model, source step, layer and head are shared.","Question proximity describes allocation-profile similarity, not answer correctness or a causal cluster."])


DEFINITION=define("attention.question_atlas","Question atlas","Deterministic principal-component projection of same-model saved question element-type profiles.",Parameters,compute,
    requires=REQUIRES+["corpus_sample_manifest"],units="profile fractions; PCA coordinates",
    normalization="per-row image-only shares; centred PCA over the explicit question sample")
