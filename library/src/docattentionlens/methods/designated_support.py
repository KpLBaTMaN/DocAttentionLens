"""Evaluation-only designated support mass, ranks, precision and mapping coverage."""
from .contracts import MethodDefinition, MethodDescriptor

DESCRIPTOR = MethodDescriptor.model_validate({'method_id': 'attention.designated_support',
 'variant_id': 'annotated_support_mass_v1',
 'descriptor_version': '0.1.0',
 'status': 'planned',
 'implementation_version': None,
 'perspective': 'document_aware',
 'family': 'document_structure',
 'question': "How much attention does the selected row allocate to the dataset's designated supporting elements?",
 'evidence_kind': 'attention_allocation',
 'quantity': 'designated_support_mass',
 'definition': 'Sum of element attention mass over designated supporting elements; other elements and '
               'residual buckets reported separately.',
 'source_references': ['design/attention-tool-ui.md'],
 'requires': ['attention_weights', 'validated_ownership_map', 'mapped_elements', 'designated_support_annotations'],
 'execution_requirement': 'saved_capture',
 'target_requirement': 'None',
 'parameters_schema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
 'output': {'schema': 'support_mass',
            'units': 'dimensionless mass',
            'signed': False,
            'support': 'designated supporting elements, other elements and residual buckets',
            'normalization': 'original captured domain'},
 'displays': ['page_overlay', 'element_allocation'],
 'comparison_constraints': ['Same support annotation version, ownership policy and normalization.'],
 'limitations': ['Designated support is an evaluation annotation: it may omit relevant regions and is never '
                 'supplied to the model.',
                 'Attention allocated to support does not establish that the model used that evidence.'],
 'validation_evidence': [],
 'name': 'Designated Supporting Elements',
 'summary': 'Measures attention allocated to annotated supporting elements. These annotations identify '
            'designated evidence and may not include every relevant region.'})

from pydantic import Field
from .contracts import RowParameters, Calculation
from ._document import masses, document, table, ownership_weights
from ._shared import statistic


class Parameters(RowParameters):
    top_k: int = Field(default=5, ge=1, le=25760, strict=True)
    include_head_matrix: bool = True


def compute(context, parameters):
    row, measured, known, _ = masses(context, parameters)
    support = set(document(context).case.support_element_ids)
    if not support:
        raise ValueError("This case has no designated support annotations")
    ordered = sorted(measured, key=lambda i:(-measured[i], i))
    top = ordered[:parameters.top_k]
    ranked = [i for i in ordered if i in support]
    # Support outside every valid mapped target is missing, never a measured zero.
    support_mass = sum(measured[i] for i in ranked) if ranked else None
    matrices=[]; notices=[]
    if not ranked:
        notices.append("No designated support element is mapped to a valid target of this source, so support mass, "
                       "precision and the layer-by-head support matrix are unavailable, not zero.")
    elif parameters.include_head_matrix:
        from ._shared import selected_rows
        if context.values.size>4_000_000: raise ValueError("The full support matrix exceeds the bounded analysis budget")
        rows,mask=selected_rows(context,parameters,heads=list(range(context.record.query_head_count)),layers=list(range(context.record.layer_count)))
        ownership=ownership_weights(context.record,mask,sorted(support)).sum(axis=1)
        matrix=(rows*ownership).sum(axis=-1)
        matrices=[table("Designated support mass by layer and head",[f"Head {i}" for i in range(context.record.query_head_count)],
            [f"Layer {i}" for i in range(context.record.layer_count)],matrix.tolist())]
    return Calculation(row=row, statistics=[statistic("Designated support mass", support_mass, "dimensionless mass"),
        statistic("Other element mass", sum(measured.values())-(support_mass or 0.0), "dimensionless mass"),
        statistic("Residual mass", row.total_mass-sum(measured.values()), "dimensionless mass"),
        statistic("Measured support elements", len(ranked), "elements"),
        statistic("Unmeasured support elements", len(support-set(measured)), "elements"),
        statistic("Precision at selected k", sum(i in support for i in top)/len(top) if top and ranked else None, "fraction"),
        statistic("Evaluated top-k population",len(top),"elements")],
        table=table("Designated support ranks",["Attention mass","Rank"],[f"{known[i].element_type} · {known[i].native_id}" for i in ranked],
                    [[measured[i],float(ordered.index(i)+1)] for i in ranked], "mass; rank",ranked,[known[i].page_id for i in ranked]),
        tables=matrices,
        notices=[*notices,"Support outside the captured mapping is unmeasured; support annotations are evaluation-only and may be incomplete."])


DESCRIPTOR = DESCRIPTOR.model_copy(update={"status":"implemented", "implementation_version":"document_saved_rows_v1",
    "descriptor_version":"0.2.0", "parameters_schema":Parameters.model_json_schema(),
    "validation_evidence":["library/tests/methods/test_document_methods.py: support conservation, ranks and missing support"]})
DEFINITION = MethodDefinition(DESCRIPTOR, Parameters, compute, True)
