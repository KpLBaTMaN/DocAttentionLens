"""Element-type mass and explicitly mapped-cell area adjustment."""
from .contracts import MethodDefinition, MethodDescriptor

DESCRIPTOR = MethodDescriptor.model_validate({'method_id': 'attention.element_type_profile',
 'variant_id': 'type_mass_area_v1',
 'descriptor_version': '0.1.0',
 'status': 'planned',
 'implementation_version': None,
 'perspective': 'document_aware',
 'family': 'document_structure',
 'question': 'How is attention allocated across annotated element types, and is any type attended '
             'disproportionately to the area it occupies?',
 'evidence_kind': 'attention_allocation',
 'quantity': 'element_type_mass_and_area_adjusted_attention',
 'definition': 'Per element type: total element attention mass and, where mapped area is defined, '
               'mass share divided by area share (area-adjusted attention).',
 'source_references': ['design/attention-tool-ui.md'],
 'requires': ['attention_weights', 'validated_ownership_map', 'mapped_elements', 'element_type_annotations'],
 'execution_requirement': 'saved_capture',
 'target_requirement': 'None',
 'parameters_schema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
 'output': {'schema': 'element_type_profile',
            'units': 'dimensionless mass; dimensionless area-adjusted ratio',
            'signed': False,
            'support': 'annotated element types and residual buckets',
            'normalization': 'mass in the original captured domain; area adjustment divides by a declared area share'},
 'displays': ['element_allocation', 'table'],
 'comparison_constraints': ['Same type vocabulary, ownership policy, area basis and normalization.'],
 'limitations': ['Area adjustment needs a declared area basis (source page or mapped patch area) and an '
                 'overlap policy.',
                 'Total mass and area-adjusted attention are different measurements and are never '
                 'displayed as interchangeable.',
                 'Type labels come from the annotation source, whose provenance is reported.'],
 'validation_evidence': [],
 'name': 'Element-Type Profile',
 'summary': 'Compare attention allocated to headings, tables, figures, body text and other annotated '
            'element types, as total mass and, where defined, area-adjusted attention.'})

from .contracts import RowParameters, Calculation
from ._document import masses, table, plot


class Parameters(RowParameters):
    pass


def compute(context, parameters):
    row, measured, known, mask = masses(context, parameters)
    types = sorted({known[i].element_type for i in measured})
    # Area uses valid mapped processor cells, with exactly the saved fractional
    # ownership. It never double counts overlapping raw bounding rectangles.
    areas = {kind: 0.0 for kind in types}
    image_count = 0
    for i, position in enumerate(context.record.positions[:len(mask)]):
        if mask[i] and position.role == "image":
            image_count += 1
            for bucket, fraction in position.ownership.items():
                if bucket in measured:
                    areas[known[bucket].element_type] += fraction
    totals = {kind: sum(value for i,value in measured.items() if known[i].element_type == kind) for kind in types}
    values = []
    for kind in types:
        area_share = areas[kind] / image_count if image_count else None
        attention_share = totals[kind] / row.image_mass if row.image_mass else None
        enrichment = attention_share / area_share if area_share and attention_share is not None else None
        values.append([totals[kind], attention_share, area_share, enrichment])
    residual = row.total_mass - sum(totals.values())
    return Calculation(row=row, table=table("Element-type allocation", ["Original-domain mass", "Image attention share", "Mapped-cell area share", "Area-adjusted ratio"], types, values,
        "mass; fraction; fraction; dimensionless ratio"),
        tables=[table("Unallocated-to-type support", ["Mass"], ["Background, unreviewed geometry and non-image"], [[residual]])],
        plots=[plot("bar", "Attention by element type", "Element type (category index)", "Original-domain attention mass",
                    [dict(label=k,x=float(i),y=totals[k]) for i,k in enumerate(types)])],
        normalization="mass retains original captured domain; area-adjusted ratio divides image attention share by fractional valid image-cell share",
        notices=["Area basis: mapped processor cells, not raw source-box area. Absent area ratios are unavailable."])


DESCRIPTOR = DESCRIPTOR.model_copy(update={"status":"implemented", "implementation_version":"document_saved_rows_v1",
    "descriptor_version":"0.2.0", "parameters_schema":Parameters.model_json_schema(),
    "validation_evidence":["library/tests/methods/test_document_methods.py: overlap, residual and zero-support fixtures"]})
DEFINITION = MethodDefinition(DESCRIPTOR, Parameters, compute, True)
