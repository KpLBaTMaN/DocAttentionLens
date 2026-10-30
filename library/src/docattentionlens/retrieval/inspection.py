"""Saved-vector inspection; no model execution or HTTP dependencies."""
from .contracts import RemovalResult
from .scoring import maxsim


def similarity_row(record, page, q, d, query_index):
    from ..methods.query_patch_similarity import similarity_row as calculate
    return calculate(record, page, q, d, query_index)


def similarity_column(record, page, q, d, position_index):
    from ..methods.query_patch_similarity import similarity_column as calculate
    return calculate(record, page, q, d, position_index)


def fixed_vector_removal(record,page,q,d,removed_indices):
    removed=set(removed_indices)
    if not removed or len(removed)!=len(removed_indices) or min(removed)<0 or max(removed)>=len(page.positions):
        raise ValueError("Removal positions must be distinct valid captured indices")
    mask=[p.eligible and p.index not in removed for p in page.positions]
    if not any(mask):
        raise ValueError("Cannot remove every eligible page position")
    score,winners,_,_=maxsim(q,d,[p.eligible for p in record.query_positions],mask)
    return RemovalResult(capture_id=record.capture_id,page_id=page.page_id,removed_position_indices=sorted(removed),baseline_score=page.score,score=score,score_delta=score-page.score,winner_indices=winners)
