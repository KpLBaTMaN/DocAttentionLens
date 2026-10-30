"""Saved query-to-patch dot products used by the actual MaxSim scorer."""
from ..retrieval.contracts import SimilarityRow, SimilarityColumn
from .retrieval_score_decomposition import maxsim

from .contracts import MethodDefinition, MethodDescriptor

DESCRIPTOR = MethodDescriptor(method_id="retrieval.query_patch_similarity", variant_id="saved_dot_products_v1",
    descriptor_version="0.9.0", status="implemented", implementation_version="colpali_saved_vectors_v1", perspective="general",
    family="similarity_and_retrieval", name="Query ↔ Patch Similarity", summary="Inspect saved query-to-page dot products and the actual MaxSim winner.",
    question="Which saved page vectors match a selected query vector?", evidence_kind="embedding_similarity",
    quantity="embedding_dot_product", definition="q_i @ d_j.T on recorded eligible positions; exact first-index MaxSim tie policy.",
    source_references=["definition:embedding dot product under the recorded MaxSim scorer"], requires=["query_document_vectors", "scored_masks", "position_identities"],
    execution_requirement="saved_capture", target_requirement="Selected retrieval query position",
    parameters_schema={"type":"object", "properties":{"query_index":{"type":"integer", "minimum":0}}, "additionalProperties":False},
    output={"schema":"similarity_row", "units":"signed dot-product score", "signed":True, "support":"eligible saved page vectors", "normalization":"actual recorded scorer; no softmax"},
    displays=["page_overlay"], comparison_constraints=["Same scorer, masks and saved vector identities."],
    limitations=["Embedding similarity is not attention or output attribution."], validation_evidence=["library/tests/test_retrieval.py",
        "frontend/tests/rendering/page-overlay-methods.spec.ts: a picked patch is outlined and its query-token column is read"])
DEFINITION = MethodDefinition(DESCRIPTOR, executable=True)


def similarity_row(record,page,q,d,query_index):
    if not 0<=query_index<len(record.query_positions) or not record.query_positions[query_index].eligible:
        raise ValueError("Query position is absent or ineligible")
    mask=[p.eligible for p in page.positions]
    _,winners,maxima,_=maxsim(q[query_index:query_index+1],d,document_mask=mask)
    similarities=q[query_index:query_index+1]@d.T
    values=[float(value) if eligible else None for value,eligible in zip(similarities[0],mask,strict=True)]
    return SimilarityRow(capture_id=record.capture_id,page_id=page.page_id,query_index=query_index,similarities=values,winner_index=winners[0],maximum=maxima[0])




def similarity_column(record,page,q,d,position_index):
    """The same dot products read for one page vector across every query vector."""
    if not 0<=position_index<len(page.positions) or not page.positions[position_index].eligible:
        raise ValueError("Page position is absent or ineligible")
    eligible=[p.eligible for p in record.query_positions]
    similarities=q@d[position_index]
    values=[float(value) if ok else None for value,ok in zip(similarities,eligible,strict=True)]
    # Winner indices are stored in eligible query order by the recorded scorer.
    order=[p.index for p in record.query_positions if p.eligible]
    if len(order)!=len(page.winner_indices):
        raise ValueError("Recorded winners differ from eligible query positions")
    winning=[index for index,winner in zip(order,page.winner_indices,strict=True) if winner==position_index]
    return SimilarityColumn(capture_id=record.capture_id,page_id=page.page_id,position_index=position_index,
        similarities=values,winning_query_indices=winning,maximum=max(v for v in values if v is not None))
