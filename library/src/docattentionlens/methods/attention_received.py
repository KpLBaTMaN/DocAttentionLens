"""Attention received: the weight one target receives from each captured source row.

This reads a column of the attention matrix across the captured source rows.
Each value is a post-softmax weight from a different normalized row, so the
column does not sum to one and is not renormalized. Sources whose support does
not include the target (a causal row before the target, or a masked key) are
omitted rather than reported as zero.
"""
import numpy as np
from .contracts import ReceivedAttention, ReceivedSource


def received(capture_id, target_index, layer, head, rows):
    """`rows` yields (source_index, kind, label, weights) with each source row's full weight vector."""
    sources = []
    for source_index, kind, label, weights in rows:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0) or abs(float(weights.sum()) - 1) > .02:
            raise ValueError("Source row is not a normalized captured attention row")
        if 0 <= target_index < len(weights):
            sources.append(ReceivedSource(source_index=source_index, kind=kind, label=label, value=float(weights[target_index])))
    if not sources:
        raise ValueError("No captured source row includes this target")
    return ReceivedAttention(capture_id=capture_id, target_index=target_index, layer=layer, head=head,
        sources=sources, maximum=max(s.value for s in sources))
