"""Raw attention: canonical definition and calculation ownership."""
from .contracts import MethodDefinition, MethodDescriptor

DESCRIPTOR = MethodDescriptor.model_validate({'method_id': 'attention.row',
 'variant_id': 'captured_weights_v1',
 'descriptor_version': '0.6.0',
 'status': 'implemented',
 'implementation_version': 'qwen25vl_prediction_rows_v1',
 'perspective': 'general',
 'family': 'direct_attention',
 'question': 'What targets does this captured source row allocate attention to?',
 'evidence_kind': 'attention_allocation',
 'quantity': 'attention_weights',
 'definition': 'Captured normalized weights at declared component/layer/head/source.',
 'source_references': ['https://arxiv.org/abs/1706.03762'],
 'literature_relation': {'kind': 'published_method',
                         'note': 'Post-softmax attention weights as defined by Vaswani et al. (2017), read at a '
                                 'captured layer, head and source. Inspection only; their experiments are not replicated.'},
 'requires': ['attention_weights',
              'source_target_position_axes',
              'masks',
              'normalization_domain',
              'generator_prediction_row_capture'],
 'execution_requirement': 'saved_capture',
 'target_requirement': 'None',
 'parameters_schema': {'type': 'object',
                       'properties': {'step': {'type': 'integer', 'minimum': 0, 'maximum': 63},
                                      'layer': {'type': 'integer', 'minimum': 0, 'maximum': 35},
                                      'head': {'type': 'integer', 'minimum': 0, 'maximum': 15}},
                       'additionalProperties': False},
 'output': {'schema': 'attention_row',
            'units': 'dimensionless mass',
            'signed': False,
            'support': 'captured target domain',
            'normalization': 'original captured domain'},
 'displays': ['page_overlay', 'matrix'],
 'comparison_constraints': ['Same compatible component/axes and normalization.'],
 'limitations': ['Not output importance; masked and uncaptured positions differ.',
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
                         'frontend/tests/rendering/page-overlay-methods.spec.ts: the workspace page '
                         "draws the known-answer row at each patch's source box",
                         'frontend/tests/rendering/attention-pixels.spec.ts and '
                         'frontend/unit/attention-layer.test.tsx: on-screen overlay and heatmap '
                         'colours; alignment through zoom, pan and resize'],
 'name': 'Raw attention',
 'summary': 'Inspect real language attention from a captured answer prediction step, at a chosen layer '
            'and query head, over supplied page patches and non-image positions.'})


import numpy as np
from .contracts import RowParameters, Calculation
from ._shared import validate_payload, selected_rows


class Parameters(RowParameters):
    """One retained prediction, layer and query head."""


def attention_row(record, values, step, layer, head):
    if not 0 <= layer < record.layer_count or not 0 <= head < record.query_head_count:
        raise ValueError("Selection is outside captured steps, layers or query heads")
    selected = validate_payload(record, values, step)
    weights = np.asarray(values)[layer, head].astype(np.float64)
    if abs(float(weights.sum())-1) > .02:
        raise ValueError("Attention allocation or softmax normalization failed")
    from .element_allocation import allocate_row
    return allocate_row(record, weights, selected, step, layer, head)


def compute(context, parameters):
    selected_rows(context, parameters)
    return Calculation(row=attention_row(context.record, context.values, parameters.step, parameters.layer, parameters.head))


DESCRIPTOR = DESCRIPTOR.model_copy(update={"parameters_schema": Parameters.model_json_schema()})
DEFINITION = MethodDefinition(DESCRIPTOR, Parameters, compute, True)
