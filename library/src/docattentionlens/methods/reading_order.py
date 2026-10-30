"""Captured element allocation ordered by annotated page and reading position."""
from .contracts import MethodDefinition, MethodDescriptor

DESCRIPTOR = MethodDescriptor.model_validate({'method_id': 'attention.reading_order',
 'variant_id': 'annotated_order_profile_v1',
 'descriptor_version': '0.1.0',
 'status': 'planned',
 'implementation_version': None,
 'perspective': 'document_aware',
 'family': 'document_structure',
 'question': 'How does element attention mass relate to the annotated reading order of the document?',
 'evidence_kind': 'attention_summary',
 'quantity': 'reading_order_attention_profile',
 'definition': 'Element attention mass ordered by annotated reading position, with unordered elements '
               'and residual buckets reported separately.',
 'source_references': ['design/attention-tool-ui.md'],
 'requires': ['attention_weights', 'validated_ownership_map', 'mapped_elements', 'reading_order_annotations'],
 'execution_requirement': 'saved_capture',
 'target_requirement': 'None',
 'parameters_schema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
 'output': {'schema': 'reading_order_profile',
            'units': 'dimensionless mass by reading position',
            'signed': False,
            'support': 'elements with annotated reading order; unordered elements and residuals separate',
            'normalization': 'original captured domain'},
 'displays': ['element_allocation', 'reading_order_ribbon'],
 'comparison_constraints': ['Same reading-order annotation version and ownership policy.'],
 'limitations': ['Requires annotated reading order; box position or native depth alone does not establish '
                 'order or section relations.'],
 'validation_evidence': [],
 'name': 'Reading-Order Attention',
 'summary': 'Relate element attention mass to the annotated reading order, such as mass by position in '
            'the sequence and between adjacent elements.'})

from .contracts import RowParameters, Calculation
from ._document import masses, document, table, plot


class Parameters(RowParameters):
    pass


def compute(context, parameters):
    row, measured, known, _ = masses(context, parameters)
    pages = {p.page_id:p.page_number for p in document(context).pages}
    ordered = sorted((i for i in measured if known[i].reading_order is not None),
                     key=lambda i:(pages[known[i].page_id],known[i].reading_order,i))
    if not ordered:
        raise ValueError("No measured element has annotated reading order")
    for pid in {known[i].page_id for i in ordered}:
        values=[known[i].reading_order for i in ordered if known[i].page_id==pid]
        if len(values)!=len(set(values)):
            raise ValueError("Reading-order annotations contain duplicate page positions")
    labels=[f"Page {pages[known[i].page_id]} · order {known[i].reading_order} · {known[i].element_type}" for i in ordered]
    unordered=[i for i in measured if i not in ordered]
    return Calculation(row=row, table=table("Reading-order attention",["Page","Annotated order","Attention mass"],labels,
        [[float(pages[known[i].page_id]),float(known[i].reading_order),measured[i]] for i in ordered],"page; order; mass",ordered,[known[i].page_id for i in ordered]),
        plots=[plot("ribbon","Attention along annotated reading order","Ordered element position","Original-domain attention mass",
            [dict(label=labels[j],x=float(j+1),y=measured[i],element_id=i,page_id=known[i].page_id) for j,i in enumerate(ordered)])],
        tables=[table("Unordered and residual support",["Mass"],["Unordered elements","Residuals"],
            # No measured unordered element: unavailable, not a measured zero.
            [[sum(measured[i] for i in unordered) if unordered else None],[row.total_mass-sum(measured.values())]])],
        notices=["Annotated sequence orders the displayed allocations; it does not establish an attention path or reading behaviour."])


DESCRIPTOR = DESCRIPTOR.model_copy(update={"status":"implemented", "implementation_version":"document_saved_rows_v1",
    "descriptor_version":"0.2.0", "parameters_schema":Parameters.model_json_schema(),
    "validation_evidence":["library/tests/methods/test_document_methods.py: page ordering and unordered residuals"]})
DEFINITION = MethodDefinition(DESCRIPTOR, Parameters, compute, True)
