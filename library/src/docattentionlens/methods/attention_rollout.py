"""Layer attention rollout: canonical definition and calculation ownership.

Abnar & Zuidema (2020), "Quantifying Attention Flow in Transformers", eq. 1: each
layer's head-mean attention gains a residual identity term and is row-normalized,
then layers are composed by matrix multiplication from the first layer upward.
Only complete square matrices over one aligned position domain qualify, so saved
retriever encodings are supported and selected generator prediction rows are not.
"""
import numpy as np
from pydantic import Field
from .contracts import Calculation, MethodDefinition, MethodDescriptor, Parameters as AxisParameters
from ._shared import parameter_schema, selected_step, statistic, validate_payload, valid_support
from .element_allocation import allocate_row

class RolloutParameters(AxisParameters):
    """`layer` is the last layer composed; rollout always starts at layer 0 and averages every query head."""
    residual_coefficient: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)


DESCRIPTOR = MethodDescriptor.model_validate({'method_id': 'attention.rollout',
 'variant_id': 'residual_mean_square_v1',
 'descriptor_version': '0.2.0',
 'status': 'implemented',
 'implementation_version': 'square_matrix_rollout_v1',
 'perspective': 'general',
 'family': 'connectivity',
 'question': 'What connectivity does this compatible layer-chain approximation imply?',
 'evidence_kind': 'attention_summary',
 'quantity': 'rollout_connectivity',
 'definition': 'Compose compatible head-mean square matrices with declared residual identity '
               'coefficient and row normalization: Â_l = r·I + (1−r)·mean_h A_l, rows renormalized '
               'over the valid key domain; rollout through layer L is Â_L·…·Â_0, read at the selected source row.',
 'source_references': ['https://arxiv.org/abs/2005.00928',
                       'paper-workspace/research/interpretability methods/069-quantifying-attention-flow-in-transformers.pdf'],
 'literature_relation': {'kind': 'published_method',
                         'note': 'Abnar & Zuidema (2020) rollout. Matches at residual coefficient 0.5, with every head '
                                 'averaged over complete aligned matrices; other residual coefficients are extensions.'},
 'requires': ['full_square_attention_matrices', 'aligned_layer_axes', 'residual_policy'],
 'execution_requirement': 'saved_capture',
 'target_requirement': 'None',
 'parameters_schema': parameter_schema(RolloutParameters),
 'output': {'schema': 'connectivity_matrix',
            'units': 'dimensionless approximate allocation',
            'signed': False,
            'support': 'aligned layer position domain',
            'normalization': 'declared row normalization after residual addition'},
 'displays': ['matrix', 'page_overlay'],
 'comparison_constraints': ['Same topology, head reduction, residual rule and layer interval.'],
 'limitations': ['Selected rows/rectangular cross-attention are incompatible; not complete information '
                 'flow.',
                 'All query heads are averaged with equal weight; value vectors, MLPs and layer norms are '
                 'ignored, so rollout is a routing proxy, not attribution or causation.',
                 'Retriever encodings cover the language transformer only; vision-tower attention is not '
                 'part of the chain.'],
 'validation_evidence': ['library/tests/methods/test_planned_methods.py: identity, single-layer, '
                         'two-layer hand-composed and causal fixtures against an independent '
                         'full-matrix product; row sums and masked keys checked',
                         'frontend/tests/rendering/page-overlay-methods.spec.ts: the workspace page '
                         "draws this method's own known-answer row"],
 'name': 'Layer attention rollout',
 'summary': 'Combine compatible attention matrices across layers into an approximate connectivity view, '
            'using explicit head reduction and residual treatment.'})


def compose(matrices, source, mask, coefficient):
    """Rollout row of `source` through every supplied layer, earliest first.

    Each item is a (heads, n, n) matrix. Composition reads left to right as
    e_source · Â_L · … · Â_0, so only one row vector is carried between layers.
    """
    domain = np.flatnonzero(mask)
    if source not in set(domain.tolist()):
        raise ValueError("The selected source is outside the valid key domain")
    vector = np.zeros(len(domain))
    vector[domain.tolist().index(source)] = 1.0
    reduced = []
    for matrix in matrices:
        mean = np.asarray(matrix, dtype=np.float64).mean(axis=0)[np.ix_(domain, domain)]
        mixed = coefficient*np.eye(len(domain)) + (1-coefficient)*mean
        sums = mixed.sum(axis=1)
        if np.any(sums <= 0):
            raise ValueError("A valid source row has no attention mass to normalize")
        reduced.append(mixed/sums[:, None])
    for layer in reversed(reduced):
        vector = vector @ layer
    row = np.zeros(len(mask))
    row[domain] = vector
    return row


def calculate(context, parameters):
    record = context.record
    if context.read_array is None or not getattr(record, "encoding_attention", False):
        raise ValueError("Rollout needs complete square attention matrices for every composed layer "
                         "(a retriever encoding, or a generator capture with 'Save full attention matrices'); retained prediction rows are insufficient")
    selected = selected_step(record, parameters.step)
    validate_payload(record, context.values, parameters.step)
    if not 0 <= parameters.layer < record.layer_count:
        raise ValueError("End layer is outside captured coverage")
    mask = valid_support(record, parameters.step)
    active = np.asarray(record.attention_mask, dtype=bool)
    n = selected.target_count
    matrices = []
    for layer in range(parameters.layer+1):
        matrix = np.asarray(context.read_array(layer))
        if matrix.shape != (record.query_head_count, n, n) or not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError("Encoding attention shard axes differ")
        rows = matrix[:, active, :]
        if np.any(rows[..., ~active] > 1e-8) or np.any(np.abs(rows.sum(axis=-1)-1) > .02):
            raise ValueError("Captured softmax normalization or masked support differs")
        matrices.append(matrix)
    # The rollout domain is every active key; a causal encoder's later positions get no
    # mass because each composed matrix is lower-triangular.
    row = compose(matrices, selected.source_index, active, parameters.residual_coefficient)
    if np.any(row[~mask] > 1e-8):
        raise ValueError("Rollout assigned mass outside the selected source's support")
    row[~mask] = 0.0
    result = allocate_row(record, row, selected, parameters.step, parameters.layer, -1, quantity="rollout_connectivity")
    image = [p.index for p in record.positions[:n] if p.role == "image"]
    return Calculation(row=result.model_copy(update={"normalization": "row-stochastic rollout over the valid key domain; sums to one"}),
        statistics=[statistic("Layers composed", parameters.layer+1, "layers"),
                    statistic("Residual identity coefficient", parameters.residual_coefficient),
                    statistic("Heads averaged", record.query_head_count, "query heads"),
                    statistic("Image share", float(row[image].sum()) if image else None),
                    statistic("Source self share", float(row[selected.source_index])),
                    statistic("Maximum target share", float(row.max()))],
        normalization="Â_l = r·I + (1−r)·head mean, rows renormalized over valid keys; product of row-stochastic matrices sums to one",
        notices=["Rollout approximates routing through attention only; it ignores value vectors, MLPs and layer norms."])


DEFINITION = MethodDefinition(DESCRIPTOR, RolloutParameters, calculate, True)
