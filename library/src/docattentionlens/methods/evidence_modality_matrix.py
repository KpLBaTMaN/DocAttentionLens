"""Designated evidence category versus attended document category; fractional mixed evidence."""
from .contracts import RowParameters, Calculation
from ._document import masses, document, sample_contexts, table, define


class Parameters(RowParameters):
    pass


def compute(context, parameters):
    labels=sorted({e.element_category for c in sample_contexts(context) for e in document(c).elements}|{v for c in sample_contexts(context) for v in (document(c).support_categories or {}).values()})
    matrix={k:{j:0.0 for j in labels} for k in labels}; counts={k:0.0 for k in labels}
    omitted=0
    for sample in sample_contexts(context):
        row, measured, known, _=masses(sample,parameters)
        categories=document(sample).support_categories or {i:e.element_category for i,e in known.items()}
        support=[categories[i] for i in document(sample).case.support_element_ids if i in categories]
        if not support or not row.image_mass:
            omitted+=1; continue
        for evidence in set(support):
            fraction=support.count(evidence)/len(support)
            counts[evidence]+=fraction
            for attended in labels:
                matrix[evidence][attended]+=fraction*sum(v for i,v in measured.items() if known[i].element_category==attended)/row.image_mass
    return Calculation(table=table("Evidence category × attended category",labels+["Fractional question count"],labels,
        [[matrix[k][j]/counts[k] if counts[k] else None for j in labels]+[counts[k]] for k in labels],"mean image-attention fraction; fractional questions"),
        normalization="per-capture image-attention shares; mixed evidence categories weighted by support-element fraction, then category means",
        notices=["Mixed designated evidence contributes in proportion to its annotated element counts; unmapped/background image attention remains outside category cells.",
                 f"{omitted} captures omitted because support categories or image mass are unavailable. Dataset category is not model input modality."])


DEFINITION=define("attention.evidence_modality_matrix","Evidence-modality matrix","Fractional designated evidence categories versus original image-domain attended element categories.",Parameters,compute,
    requires=["attention_weights","generator_prediction_row_capture","complete_declared_support","validated_ownership_map","mapped_elements","designated_support_annotations","element_type_annotations"],
    normalization="per-capture image-only attention shares and fractional evidence-category means")
