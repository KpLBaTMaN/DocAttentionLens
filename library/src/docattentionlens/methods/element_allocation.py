"""Element attention aggregation: canonical definition and calculation ownership."""
from .contracts import MethodDefinition, MethodDescriptor

DESCRIPTOR = MethodDescriptor.model_validate({'method_id': 'attention.element_allocation',
 'variant_id': 'ownership_mass_v1',
 'descriptor_version': '0.6.0',
 'status': 'implemented',
 'implementation_version': 'qwen25vl_prediction_rows_v1',
 'perspective': 'document_aware',
 'family': 'aggregation',
 'question': 'How much attention is allocated to each document element?',
 'evidence_kind': 'attention_allocation',
 'quantity': 'element_attention_mass',
 'definition': 'm(e|s) = sum_j A(s,j) w(e,j); retain residual buckets.',
 'source_references': ['design/capture-and-mapping.md'],
 'requires': ['attention_weights',
              'validated_ownership_map',
              'mapped_elements',
              'source_target_position_axes',
              'coverage',
              'generator_prediction_row_capture'],
 'execution_requirement': 'saved_capture',
 'target_requirement': 'None',
 'parameters_schema': {'type': 'object',
                       'properties': {'step': {'type': 'integer', 'minimum': 0, 'maximum': 63},
                                      'layer': {'type': 'integer', 'minimum': 0, 'maximum': 35},
                                      'head': {'type': 'integer', 'minimum': 0, 'maximum': 15},
                                      'source_reduction': {'const': 'single_row'},
                                      'normalization': {'const': 'original_capture_domain'}},
                       'additionalProperties': False},
 'output': {'schema': 'element_mass',
            'units': 'dimensionless mass',
            'signed': False,
            'support': 'element and residual buckets',
            'normalization': 'original captured domain'},
 'displays': ['page_overlay', 'element_allocation', 'matrix'],
 'comparison_constraints': ['Same ownership policy, source reduction and normalization.'],
 'limitations': ['Overlapping/invalid geometry needs a declared policy; density is separate.',
                 'Ordinary captures keep the first eight free-running prediction rows; full-attention '
                 'captures keep every generated step and every prefill row. All language layers/query '
                 'heads. Vision attention is unavailable.',
                 'Validation covers the pinned adapter and tested pages; no answer-quality or '
                 'causal-attribution claim.'],
 'validation_evidence': ['scripts/generation_smoke.py: actual two-page pinned Qwen2.5-VL capture, '
                         'processor/window order, causal prediction alignment, capture-on/off tokens '
                         'and logits',
                         'library/tests/test_generation.py: full-domain allocation, page/non-image '
                         'residuals and preview inversion',
                         'frontend/tests/rendering/page-overlay-methods.spec.ts and '
                         'frontend/unit/page-canvas.test.tsx: element regions carry known-answer '
                         'masses; attention-pixels.spec.ts: on-screen region colours'],
 'name': 'Element attention aggregation',
 'summary': 'Allocate one captured prediction attention row to document elements and explicit '
            'background, geometry and non-image buckets using a saved ownership map.'})


from typing import Literal
import numpy as np
from ..generation.contracts import AttentionRow, Allocation
from .contracts import RowParameters


class Parameters(RowParameters):
    source_reduction: Literal["single_row"] = "single_row"
    normalization: Literal["original_capture_domain"] = "original_capture_domain"


def allocate_row(record, weights, selected, step, layer, head, quantity="post_softmax_attention_weight"):
    """Conserve weighted ownership and explicit residuals for this target domain."""
    ledger = {}
    image_mass = 0.0
    input_mask=getattr(getattr(record,"context",None),"input_attention_mask",[])
    for index, (position, weight) in enumerate(zip(record.positions[:selected.target_count], weights, strict=True)):
        if position.index != index:
            raise ValueError("Target position identities are not contiguous")
        if any(not np.isfinite(v) or v < 0 for v in position.ownership.values()) or abs(sum(position.ownership.values())-1) > 1e-8:
            raise ValueError("Ownership does not cover the original target domain")
        if getattr(record,"encoding_attention",False):
            valid=bool(record.attention_mask[index]) and (not getattr(record,"causal",False) or index<=selected.source_index)
            if not valid:
                if abs(weight)>1e-8: raise ValueError("Masked encoding positions carry attention")
                continue
        elif index<len(input_mask) and not input_mask[index]:
            if abs(weight)>1e-8: raise ValueError("Masked input positions carry attention")
            continue
        if position.role == "image":
            image_mass += float(weight)
        for bucket, fraction in position.ownership.items():
            key = (bucket, position.page_id)
            ledger[key] = ledger.get(key, 0.0) + float(weight)*fraction
    total = float(weights.sum())
    error = abs(sum(ledger.values())-total)
    if error > 1e-8:
        raise ValueError("Attention allocation failed conservation")
    normalizations={
        "whole_query_image_allocation":"per-source image conditioning followed by the declared head/source reductions; source image totals reported separately",
        "modality_block_attention":"original weights within the declared source/target modality block; sum equals block mass",
        "rollout_connectivity":"row-stochastic residual/head-normalized approximate connectivity",
        "partial_rollout_connectivity":"row-stochastic approximate connectivity over the declared partial layer interval",
        "rollout_evolution_connectivity":"row-stochastic approximate connectivity at the declared end layer"}
    return AttentionRow(capture_id=record.capture_id, step=step, layer=layer, head=head,
        source_index=selected.source_index, predicted_index=selected.predicted_index,
        target_count=selected.target_count, values=weights.tolist(), total_mass=total,
        image_mass=image_mass, allocation_error=error, display_max=float(weights.max()), quantity=quantity,
        normalization=normalizations.get(quantity, (("causal encoding prefix; active earlier keys only; no renormalization" if getattr(record, "causal", False) else "full encoding prefix; active keys only; no renormalization") if getattr(record, "encoding_attention", False) else "original causal target domain; no renormalization") if quantity == "post_softmax_attention_weight" else "target-wise head envelope; sums need not equal one; no renormalization"),
        allocations=[Allocation(bucket=b, page_id=p, element_id=b if b.startswith("element_") else None, mass=v)
                     for (b,p),v in sorted(ledger.items(), key=lambda item:-item[1])])


def compute(context, parameters):
    from .raw_attention import compute as raw_compute
    return raw_compute(context, parameters)


DESCRIPTOR = DESCRIPTOR.model_copy(update={"parameters_schema": Parameters.model_json_schema()})
DEFINITION = MethodDefinition(DESCRIPTOR, Parameters, compute, True)
