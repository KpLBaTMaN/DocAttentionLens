"""Retrieval score decomposition: canonical definition and calculation ownership."""
from .contracts import MethodDefinition, MethodDescriptor

DESCRIPTOR = MethodDescriptor.model_validate({'method_id': 'retrieval.maxsim',
 'variant_id': 'actual_scorer_v1',
 'descriptor_version': '0.4.0',
 'status': 'implemented',
 'implementation_version': 'colpali_transformers_v1',
 'perspective': 'general',
 'family': 'similarity_and_retrieval',
 'question': 'Which scored document positions supply each query maximum and the ranking score?',
 'evidence_kind': 'retrieval_scoring',
 'quantity': 'maxsim_contributions',
 'definition': 'sum_i max_j q_i dot d_j using the exact scorer positions, masks and normalization.',
 'source_references': ['https://huggingface.co/vidore/colpali-v1.3-hf',
                       'https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/colpali/processing_colpali.py'],
 'requires': ['query_document_vectors', 'scorer_configuration', 'scored_masks', 'position_identities'],
 'execution_requirement': 'saved_capture',
 'target_requirement': 'Declared retrieval score',
 'parameters_schema': {'type': 'object', 'properties': {}, 'additionalProperties': False},
 'output': {'schema': 'winner_contribution_ledger',
            'units': 'scorer dot-product units',
            'signed': True,
            'support': 'eligible scored query/document positions',
            'normalization': 'actual scorer vector normalization'},
 'displays': ['winner_ledger', 'page_overlay', 'paired_pages'],
 'comparison_constraints': ['Same query positions, scorer/masks and normalization.'],
 'limitations': ['Not attention; image-only display support may differ from scored support.',
                 'Validation covers the pinned checkpoint and bounded candidate pages, not corpus-wide '
                 'retrieval or general attention.'],
 'validation_evidence': ['library/tests/test_retrieval.py',
                         'scripts/retrieval_smoke.py: real portrait/landscape/facing-page scorer and '
                         'ownership checks',
                         'frontend/tests/rendering/page-overlay-methods.spec.ts and '
                         'frontend/unit/page-canvas.test.tsx: signed known-answer similarity '
                         'patches and the winner outline on the page'],
 'name': 'Retrieval score decomposition',
 'summary': 'Inspect query-to-page vector similarities, winning positions and their contributions to '
            'the late-interaction retrieval score. These values are not attention weights.'})

DEFINITION = MethodDefinition(DESCRIPTOR, executable=True)

import numpy as np


def maxsim(query, document, query_mask=None, document_mask=None):
    q, d = np.asarray(query, dtype=np.float32), np.asarray(document, dtype=np.float32)
    if q.ndim != 2 or d.ndim != 2 or q.shape[1] != d.shape[1] or not np.isfinite(q).all() or not np.isfinite(d).all():
        raise ValueError("Expected finite aligned two-dimensional vectors")
    qm=np.ones(len(q),dtype=bool) if query_mask is None else np.asarray(query_mask)
    dm=np.ones(len(d),dtype=bool) if document_mask is None else np.asarray(document_mask)
    if qm.shape!=(len(q),) or dm.shape!=(len(d),) or qm.dtype!=bool or dm.dtype!=bool:
        raise ValueError("Eligibility masks must be aligned boolean vectors")
    qi = np.flatnonzero(qm)
    di = np.flatnonzero(dm)
    if not len(qi) or not len(di):
        raise ValueError("MaxSim requires nonempty eligible query and document domains")
    similarities = q[qi] @ d[di].T
    local = similarities.argmax(axis=1)  # First eligible index breaks exact ties.
    maxima = similarities[np.arange(len(qi)), local]
    return float(maxima.sum(dtype=np.float32)), di[local].tolist(), maxima.tolist(), similarities


def contributions(winners, maxima, mappings):
    ledger = {}
    for winner, value in zip(winners, maxima, strict=True):
        for bucket, weight in mappings[winner].items():
            ledger[bucket] = ledger.get(bucket, 0.0) + value*weight
    if abs(sum(ledger.values())-sum(maxima)) > 1e-6:
        raise ValueError("Contribution ledger failed conservation")
    return ledger
