"""Verified retriever encoding rows; formulas stay in each individual method file."""
from types import SimpleNamespace
import numpy as np
from .contracts import RetrievalAttentionRequest
from ..methods.contracts import Context, Calculation
from ..methods.attention_received import received
from ..methods.registry import list_methods, get_definition, prepare_parameters

SUPPORTED = {"attention.row", "attention.element_allocation", "attention.mean_heads",
    "attention.max_heads", "attention.min_heads", "attention.weighted_heads",
    "attention.entropy", "attention.concentration", "attention.topk_mass",
    "attention.head_by_head", "attention.across_layers", "attention.sinks",
    "attention.modality_ratio", "heads.distribution_similarity", "attention.element_type_profile",
    "attention.designated_support", "attention.reading_order", "attention.layout_sinks", "attention.section_rollup", "attention.modality_budget_depth",
    "attention.rollout", "attention.spatial_distance"}
SUPPORTED.update({"attention.whole_query", "attention.text_to_text", "attention.text_to_vision",
    "attention.vision_to_text", "attention.vision_to_vision", "attention.spatial_direction", "attention.local_global",
    "attention.row_structure", "attention.column_structure", "attention.sink_detection", "heads.clustering",
    "heads.pattern_taxonomy", "attention.partial_rollout", "attention.rollout_evolution", "attention.graph", "attention.flow"})
# One element across every captured layer and head, and table-cell allocation (optionally over a rollout
# window) of the diagnostic page-plus-prompt encodings. Contrasts and swap baselines need saved comparison
# or sample captures, which this path does not supply, so table.matched_contrast stays generator-only.
SUPPORTED.update({"attention.element_profile", "table.cell_allocation", "table.lookup_path", "table.page_regions",
    "table.mapping_audit"})
# Methods that read every captured layer of the source row rather than one selected layer.
ALL_LAYERS = {"attention.element_profile", "table.page_regions"}


# Generator summaries speak of answer predictions; a retriever row is an encoding source row.
_ENCODING_WORDS = (("a captured answer prediction step", "a saved encoding source row"),
                   ("one captured prediction attention row", "one saved encoding attention row"),
                   ("one retained prediction", "one saved encoding source row"),
                   ("the same prediction", "the same encoding source row"))


def _encoding_wording(text: str) -> str:
    for old, new in _ENCODING_WORDS:
        text = text.replace(old, new)
    return text


def methods():
    """Explicit encoding variants; never imply a prediction or vision-tower capture."""
    results = []
    for method in list_methods():
        if method.method_id not in SUPPORTED:
            continue
        schema = dict(method.parameters_schema)
        schema["properties"] = {k:v for k,v in schema.get("properties", {}).items() if k != "step"}
        results.append(method.model_copy(update={
            "variant_id":"retriever_encoding_v1", "implementation_version":"retriever_encoding_rows_v1",
            "name":method.name.replace("captured prediction", "encoding row"),
            "summary":_encoding_wording(method.summary) + " Uses saved retriever language-encoding attention.",
            "question":"How does the selected retrieval-model encoding row distribute attention?",
            "definition":method.definition.replace("causal", "encoding").replace("prediction", "source row"),
            "requires":["retriever_encoding_attention", "complete_declared_support",
                *(["validated_ownership_map", "mapped_elements"] if method.perspective == "document_aware" else []),
                *[r for r in method.requires if r.endswith("_annotations") or r in
                    {"image_mask", "question_source_rows", "multiple_source_rows", "captured_text_sources", "captured_image_sources", "full_square_attention_matrices"}]],
            "parameters_schema":schema,
            "output":method.output.model_copy(update={"support":"active-key encoding prefix; earlier keys only for causal encoders",
                "normalization":method.output.normalization.replace("causal", "encoding")}),
            "limitations":[*method.limitations, "Language transformer inside the retriever; vision-tower attention is not captured.",
                "ColQwen2 encodings are causal: page patches precede the prompt and cannot attend to it.",
                "Query and document encodings are separate. This is not query-to-document cross-attention.",
                "Descriptive attention does not establish importance or causation."],
            "comparison_constraints":["Same encoding, source, key support and normalization."],
            "validation_evidence":["Encoding masks, axes and row sums checked at capture; saved mathematical fixtures."]}))
    return results


def row_context(capture, encoding_id, source_index):
    encoding = next((e for e in capture.attention_encodings if e.encoding_id == encoding_id), None)
    if encoding is None:
        raise ValueError("This encoding has no captured retriever attention; acquire a new capture")
    page = next((p for p in capture.pages if p.page_id == encoding.page_id), None)
    original = encoding.positions or (capture.query_positions if encoding.pass_kind == "query" else page.positions if page else [])
    ownership = encoding.ownership or (page.ownership if page else [])
    n = encoding.position_count
    if (encoding.layer_count != len(encoding.layer_arrays) or len(encoding.attention_mask) != n
        or len(original) != n or any(v not in (0,1) for v in encoding.attention_mask)
        or any(p.index != i or bool(encoding.attention_mask[i]) != p.eligible for i,p in enumerate(original))
        or not 0 <= source_index < n or not encoding.attention_mask[source_index]):
        raise ValueError("Encoding axes, positions or source selection differ from the saved mask")
    # A reference answer added to a diagnostic joint prompt is prompt text, not a generated token.
    roles = {"visual_patch":"image", "query_text":"query", "visual_prompt":"prompt", "answer_text":"prompt",
             "augmentation":"special", "prefix_or_special":"special", "bos":"special", "newline":"special"}
    positions = [SimpleNamespace(**{**p.model_dump(), "role":roles[p.role], "page_id":encoding.page_id,
        "ownership":ownership[i] if ownership else {"non_page":1.0}}) for i,p in enumerate(original)]
    record = SimpleNamespace(capture_id=capture.capture_id, encoding_attention=True, causal=encoding.causal,
        layer_count=encoding.layer_count, query_head_count=encoding.query_head_count,
        attention_mask=encoding.attention_mask, positions=positions,
        steps=[SimpleNamespace(step=0, source_index=source_index, predicted_index=None, target_count=n)],
        # The encoded page is the only supplied page; element profiles order elements by it.
        context=SimpleNamespace(selected_page_ids=[encoding.page_id] if encoding.page_id else []))
    return encoding, record


def prepare(capture, body):
    if body.method_id not in SUPPORTED:
        raise ValueError("Method is unavailable for captured retriever encoding rows")
    encoding, record = row_context(capture, body.encoding_id, body.source_index)
    descriptor = next(m for m in list_methods() if m.method_id == body.method_id)
    if "step" in body.parameters:
        raise ValueError("Retriever encodings have source positions, not prediction steps")
    parameters = prepare_parameters(body.method_id, descriptor.variant_id, {**body.parameters, "step":0}, record)
    return encoding, record, get_definition(body.method_id, descriptor.variant_id), parameters


def mapping_gap(encoding, record):
    """Why element methods cannot use this encoding, or None. Missing structure is never zero mass."""
    if encoding.pass_kind == "query":
        return "Document-aware methods require a document encoding with a valid element mapping; the query encoding has no mapped document positions"
    if not any(bucket.startswith("element_") for p in record.positions for bucket in p.ownership):
        return "Document-aware methods require annotated elements mapped to this page's patches; none are recorded"
    from ..methods._shared import valid_support
    mask=valid_support(record,0)
    if not any(mask[i] and any(bucket.startswith("element_") and fraction>0 for bucket,fraction in p.ownership.items()) for i,p in enumerate(record.positions)):
        return "No annotated element is visible from this causal source position; select a later document position"
    return None


def calculate(capture, body: RetrievalAttentionRequest, read_array, document=None, table=None) -> Calculation:
    encoding, record, definition, parameters = prepare(capture, body)
    # Checked at execution only: a saved view may legitimately select an unavailable method.
    if definition.descriptor.perspective == "document_aware" and (gap := mapping_gap(encoding, record)):
        raise ValueError(gap)
    # Read only layers used by the method. Each shard has its own verified hash.
    layers = getattr(parameters, "layers", None)
    if layers is None:
        layers = [parameters.layer] if hasattr(parameters, "layer") else list(range(encoding.layer_count))
    if getattr(parameters,"include_head_matrix",False) or body.method_id in ALL_LAYERS:
        layers=list(range(encoding.layer_count))
    if definition.descriptor.perspective == "table_aware" and table is None:
        raise ValueError("Table-aware methods need a table grounding case with verified table annotations")
    from functools import lru_cache
    def read_layer(layer):
        """A complete verified (heads, sources, targets) layer, for methods that compose or scan whole layers."""
        if not 0 <= layer < encoding.layer_count:
            raise ValueError("Layer is outside the captured encoding")
        matrix = read_array(encoding.layer_arrays[layer])
        if matrix.shape != (encoding.query_head_count, encoding.position_count, encoding.position_count):
            raise ValueError("Encoding attention shard axes differ")
        return matrix
    layer_matrix = lru_cache(maxsize=32)(read_layer)
    values = np.zeros((encoding.layer_count, encoding.query_head_count, encoding.position_count), dtype=np.float32)
    for layer in layers:
        # Whole-depth reads keep only the source row, not 36 complete layers, in memory.
        values[layer] = (read_layer if len(layers) > 8 else layer_matrix)(layer)[:,body.source_index,:]
    result = definition.calculate(Context(record, values, document, read_array=layer_matrix, table=table), parameters)
    if len(result.model_dump_json().encode()) > 4*1024**2:
        raise ValueError("Analysis result exceeds four MiB; reduce compared axes, frames or edges")
    # Generator wording names a causal target domain; a bidirectional encoder has none.
    return result if encoding.causal else result.model_copy(update={"normalization":result.normalization.replace("causal", "encoding")})


def received_column(capture, body, read_array):
    """Weight the selected target receives from every active source row of one encoding."""
    encoding, record = row_context(capture, body.encoding_id, body.target_index)
    if not 0 <= body.layer < encoding.layer_count or not 0 <= body.head < encoding.query_head_count:
        raise ValueError("Layer or head is outside the captured encoding")
    matrix = read_array(encoding.layer_arrays[body.layer])
    if matrix.shape != (encoding.query_head_count, encoding.position_count, encoding.position_count):
        raise ValueError("Encoding attention shard axes differ")
    # In a causal encoding only the target itself and later sources can see it.
    rows = ((p.index, "encoding", p.piece, matrix[body.head, p.index]) for p in record.positions
            if encoding.attention_mask[p.index] and (not encoding.causal or p.index >= body.target_index))
    return received(capture.capture_id, body.target_index, body.layer, body.head, rows)
