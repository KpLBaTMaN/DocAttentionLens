"""Single registry of method modules; metadata never implies execution."""
from functools import lru_cache
from importlib import import_module
from dataclasses import replace
from .contracts import PreparedMethod
from ._full_matrix import FULL_MATRIX_REQUIREMENTS


REQUIREMENT_LABELS = {
    "attention_weights": "captured attention weights",
    "source_target_position_axes": "captured source and target identities",
    "masks": "the captured causal support",
    "normalization_domain": "a declared normalization domain",
    "generator_prediction_row_capture": "a saved generator prediction capture",
    "validated_ownership_map": "a validated document ownership mapping",
    "coverage": "declared capture coverage",
    "normalized_probability_row": "normalized attention rows",
    "eligible_mask": "valid-key masks",
    "complete_declared_support": "complete declared causal support",
    "full_square_attention_matrices": "full attention matrices: run a new generator capture with 'Save full attention matrices'",
    "aligned_layer_axes": "aligned layer matrices",
    "residual_policy": "a declared residual treatment",
    "validated_source_target_geometry": "mapped source and target geometry",
    "image_mask": "mapped image targets",
    "document_relation_profiles": "declared document behaviour profiles",
    "corpus_sample_manifest": "a saved-capture sample chosen in ⋯ › Experiments › Document studies",
    "profile_definition": "a declared profile definition",
    "differentiable_input": "differentiable model inputs",
    "gradient_capable_adapter": "a gradient-capable model adapter",
    "baseline": "an attribution baseline",
    "scalar_output_target": "a named scalar model output",
    "query_document_vectors": "saved query and page vectors",
    "scorer_configuration": "the recorded retrieval scorer",
    "scored_masks": "recorded scored positions",
    "position_identities": "captured position identities",
    "retriever_encoding_attention": "saved retriever encoding attention",
    "mapped_elements": "annotated element boxes mapped to captured patches",
    "designated_support_annotations": "designated supporting-element annotations",
    "element_type_annotations": "element-type annotations",
    "reading_order_annotations": "reading-order annotations",
    "retrieval_evidence": "a saved retrieval capture with matching generation lineage",
    "paired_capture": "a comparison capture selected in Document studies",
    "experiment_outcomes": "a completed document experiment selected in Document studies",
    "element_lineage": "verified experiment element lineage",
    "question_source_rows": "retained question-token source rows",
    "multiple_source_rows": "at least two captured source rows",
    "captured_text_sources": "captured textual source rows",
    "captured_image_sources": "captured image-patch source rows",
    "attention_intervention_outcomes": "a completed attention-intervention experiment",
    "table_structure_annotations": "a table located on the case's page (table grounding datasets)",
    "table_grounding_annotations": "the question's table grounding labels",
}


def prepare_methods(capabilities: set[str], capture=None) -> list[PreparedMethod]:
    """Describe compatibility without running a calculation or acquiring data.

    The caller supplies verified capture metadata, never a client assertion of
    capabilities. Array integrity is still checked when the result is read.
    """
    prepared = []
    for method in list_methods():
        reasons = []
        try:
            require_executable(method.method_id, method.variant_id)
        except NotImplementedError:
            reasons.append("Planned: this method variant is not implemented in DocAttentionLens.")
        missing = [REQUIREMENT_LABELS.get(r, r.replace("_", " ")) for r in method.requires if r not in capabilities]
        if missing:
            reasons.append("Needs " + "; ".join(missing) + ".")
        bounds = {}
        full = getattr(capture, "full_attention", None) if capture is not None else None
        matrix_view = full is not None and bool(FULL_MATRIX_REQUIREMENTS & set(method.requires))
        if capture is not None and ("generator_prediction_row_capture" in method.requires or matrix_view):
            bounds = {"steps": [s.step for s in capture.steps], "layer_count": capture.layer_count,
                      "head_count": capture.query_head_count, "target_counts": {str(s.step): s.target_count for s in capture.steps}}
        if matrix_view:
            # Any source below the square limit can be selected; later rows were never captured.
            bounds["full_attention"] = {"position_count": full.position_count, "square_limit": full.square_limit}
        prepared.append(PreparedMethod(method=method, available=not reasons, reasons=reasons, control_bounds=bounds))
    return prepared


def prediction_capabilities(capture) -> set[str]:
    """Only declare capabilities retained by the bounded prediction adapter."""
    if capture is None or not capture.steps:
        return set()
    capabilities = {"source_target_position_axes", "generator_prediction_row_capture", "coverage"}
    sources = {s.source_index for s in [*capture.steps, *getattr(capture, "source_rows", [])] if hasattr(s, "source_index")}
    if len(sources) >= 2:
        capabilities.add("multiple_source_rows")
    if getattr(capture, "source_rows", []):
        capabilities.add("question_source_rows")
    if any(capture.positions[i].role != "image" for i in sources):
        capabilities.add("captured_text_sources")
    if any(capture.positions[i].role == "image" for i in sources):
        capabilities.add("captured_image_sources")
    if capture.coverage.get("quantity", "").startswith("post_softmax"):
        capabilities.update({"attention_weights", "normalization_domain", "normalized_probability_row"})
    if capture.coverage.get("all_causal_targets"):
        capabilities.update({"masks", "eligible_mask", "complete_declared_support"})
    full = getattr(capture, "full_attention", None)
    if full is not None:
        # Saved prefill triangles plus retained step rows give every position below the square
        # limit a complete causal row over one aligned position domain per layer
        # (methods/_full_matrix.py). The causal mask is declared by construction.
        capabilities.update({"full_square_attention_matrices", "aligned_layer_axes", "residual_policy",
                             "masks", "eligible_mask", "complete_declared_support"})
        if any(p.role == "image" for p in capture.positions[:full.square_limit]):
            capabilities.add("captured_image_sources")
        # `multiple_source_rows` keeps the rule above: methods requiring it read retained rows,
        # not the full-matrix view, so the squares do not enlarge their population.
    if capture.coverage.get("ownership") and any(p.ownership for p in capture.positions):
        capabilities.add("validated_ownership_map")
        # Without an annotated element in the mapping, element allocation would report
        # residuals only, which reads as zero element attention rather than missing structure.
        if any(bucket.startswith("element_") for p in capture.positions for bucket in p.ownership):
            capabilities.add("mapped_elements")
    if any(p.role == "image" and p.bbox for p in capture.positions):
        capabilities.add("image_mask")
    # Without an opt-in full-attention capture, dense prefill was only transient and was not
    # retained. Never infer full matrices or visual source geometry from a prediction row.
    return capabilities


UPLOAD_ADAPTERS = {"uploaded_page_images_v1", "uploaded_page_images_v2"}


def document_structure(case, snapshot):
    """Verified annotations on the case's input pages and where they came from.

    Returns the capabilities document-aware methods may require and a summary for display.
    Structure is never inferred here: a document without annotations reports zero counts.
    """
    from .contracts import DocumentStructure
    pages = set(case.page_ids)
    elements = [e for e in snapshot.elements if e.document_id == case.document_id and e.page_id in pages]
    summary = snapshot.summary
    imported = summary.adapter_version in UPLOAD_ADAPTERS
    structure = DocumentStructure(
        provenance="imported" if imported else "dataset",
        source=f"{summary.name} {'import' if imported else 'dataset annotations'}",
        adapter_version=summary.adapter_version, annotation_version=case.annotation_version,
        element_count=len(elements), boxed_element_count=sum(e.display_bbox is not None for e in elements),
        flagged_element_count=sum(bool(e.geometry_flags) for e in elements),
        reading_order_count=sum(e.reading_order is not None for e in elements),
        element_types=sorted({e.element_type for e in elements if e.element_type}),
        support_element_count=len(case.support_element_ids))
    capabilities = set()
    if structure.support_element_count:
        capabilities.add("designated_support_annotations")
    if structure.element_types:
        capabilities.add("element_type_annotations")
    if structure.reading_order_count:
        capabilities.add("reading_order_annotations")
    return capabilities, structure


def table_structure(case, snapshot):
    """The verified table a table-grounding case reads, for table-aware methods. None for layout cases."""
    from .contracts import TableStructure
    ground = getattr(case, "table_grounding", None)
    if ground is None:
        return set(), None
    table = next((t for t in snapshot.tables if t.table_id == ground.table_id), None)
    if table is None or table.table_sha256 != ground.table_sha256:
        return set(), None
    structure = TableStructure(source=f"{snapshot.summary.name} table annotations", adapter_version=snapshot.summary.adapter_version,
        table_id=table.table_id, title=table.title, revision=table.revision, cell_count=len(table.cells), row_count=table.row_count,
        column_count=table.column_count, quality_status=table.quality_status, grounding=True, binding_status=ground.binding_status)
    return {"table_structure_annotations", "table_grounding_annotations"}, structure


# Each listed module owns one method variant as its `DEFINITION`. Registering a method means adding
# its module here, once. This order is the order of `list_methods()`, the catalogue export and Toolbox.
_METHOD_MODULES = (
    "raw_attention", "element_allocation", "mean_heads", "max_heads", "min_heads", "weighted_heads",
    "attention_entropy", "attention_concentration", "topk_mass", "attention_sinks", "modality_ratio",
    "head_by_head", "attention_across_layers", "head_similarity", "attention_distance", "attention_rollout",
    "head_profile_comparison", "integrated_gradients", "retrieval_score_decomposition",
    "query_patch_similarity", "layout_move_group", "designated_support", "element_type_profile", "reading_order",
    "layout_sinks", "section_rollup", "spatial_prior_map", "model_input_legibility", "modality_budget_depth",
    "evidence_modality_matrix", "evidence_funnel", "retrieval_attention_agreement", "query_answer_bridge",
    "head_specialisation", "question_atlas", "attention_comparison", "image_text_contrast", "element_masking",
    "element_attention_steering",
    # Attention-only release.
    "whole_query_attention", "text_to_text", "text_to_vision", "vision_to_text", "vision_to_vision",
    "spatial_direction", "local_global", "row_structure", "column_structure", "sink_detection",
    "head_clustering", "pattern_taxonomy", "partial_rollout", "rollout_evolution", "attention_graph", "attention_flow",
    # Direct attention interventions.
    "head_ablation", "layer_ablation", "edge_ablation", "modality_ablation", "region_attention_masking",
    # Table-aware methods.
    "table_page_regions", "table_mapping_audit", "table_cell_allocation", "table_lookup_path", "table_answer_distractors",
    "table_matched_contrast",
    "element_profile",
)


@lru_cache
def _definitions():
    modules = [import_module(f"{__package__}.{name}") for name in _METHOD_MODULES]
    definitions = {}
    for module in modules:
        definition = module.DEFINITION
        references = definition.descriptor.source_references
        if any(reference.startswith("design/") for reference in references):
            # Preserve source fingerprints used by saved experiment plans. Publish
            # a live implementation reference in place of archived design notes.
            descriptor = definition.descriptor.model_copy(update={"source_references": [
                *(reference for reference in references if not reference.startswith("design/")),
                f"library/src/{module.__name__.replace('.', '/')}.py",
            ]})
            definition = replace(definition, descriptor=descriptor)
        definitions[definition.descriptor.method_id, definition.descriptor.variant_id] = definition
    if len(definitions) != len(modules):
        raise ValueError("Duplicate method variant in registry")
    return definitions


def list_methods():
    return [m.descriptor.model_copy(deep=True) for m in _definitions().values()]


def get_definition(method_id, variant_id):
    try:
        return _definitions()[method_id, variant_id]
    except KeyError:
        raise KeyError(method_id) from None


def get_method(method_id, variant_id=None):
    matches = [m for m in list_methods() if m.method_id == method_id and (variant_id is None or m.variant_id == variant_id)]
    if not matches:
        raise KeyError(method_id)
    if len(matches) != 1:
        raise ValueError("Select an exact method variant")
    return matches[0]


def require_executable(method_id, variant_id):
    definition = get_definition(method_id, variant_id)
    if definition.descriptor.status == "planned" or not definition.executable or not definition.descriptor.implementation_version:
        raise NotImplementedError("This method variant is planned and cannot execute.")


def prepare_parameters(method_id, variant_id, parameters, record):
    from ._shared import validate_parameters, selection
    definition = get_definition(method_id, variant_id)
    require_executable(method_id, variant_id)
    if definition.calculate is None or definition.parameters is None:
        raise NotImplementedError("Use the method's compatible retrieval or experiment workflow.")
    parsed = definition.parameters.model_validate(parameters)
    validate_parameters(parsed, record)
    # Persist resolved defaults rather than an implicit changing population.
    for name, count in (("heads", record.query_head_count), ("layers", record.layer_count)):
        if hasattr(parsed, name):
            setattr(parsed, name, selection(getattr(parsed, name), count))
    if definition.prepare:
        parsed = definition.prepare(parsed, record)
    return parsed


def execute(method_id, variant_id, parameters, context):
    parsed = prepare_parameters(method_id, variant_id, parameters, context.record)
    return get_definition(method_id, variant_id).calculate(context, parsed)


