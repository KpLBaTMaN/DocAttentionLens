"""Analysis perspectives are classified by what each calculation needs, not by display."""
from types import SimpleNamespace as NS
from docattentionlens.methods.registry import (document_structure, list_methods, prepare_methods,
    require_executable)
from docattentionlens.retrieval.attention import methods as retriever_methods

# Capabilities that only document structure can provide.
STRUCTURE = {"mapped_elements", "designated_support_annotations", "element_type_annotations",
             "reading_order_annotations", "document_relation_profiles", "reviewed_group_relation", "element_lineage"}
# Capabilities that only table structure can provide.
TABLE = {"table_structure_annotations", "table_grounding_annotations"}


def by_id():
    return {m.method_id: m for m in list_methods()}


def test_every_descriptor_declares_a_perspective_that_matches_its_requirements():
    for method in [*list_methods(), *retriever_methods()]:
        assert method.perspective in {"general", "document_aware", "table_aware"}
        needs_structure = bool(STRUCTURE & set(method.requires))
        assert needs_structure == (method.perspective == "document_aware"), method.method_id
        assert bool(TABLE & set(method.requires)) == (method.perspective == "table_aware"), method.method_id


def test_raw_attention_stays_general_although_boxes_can_be_overlaid():
    methods = by_id()
    # General: model positions, coordinates of patches or captured token roles only.
    for identifier in ("attention.row", "attention.mean_heads", "attention.entropy", "attention.head_by_head",
                       "attention.modality_ratio", "attention.sinks", "attention.spatial_distance", "attention.rollout"):
        assert methods[identifier].perspective == "general", identifier
    # Document-aware: aggregated or compared using document regions and relations.
    for identifier in ("attention.element_allocation", "attention.designated_support",
                       "attention.element_type_profile", "attention.reading_order"):
        assert methods[identifier].perspective == "document_aware", identifier
    assert "mapped_elements" in methods["attention.element_allocation"].requires
    assert methods["attention.element_allocation"].quantity == "element_attention_mass"


def test_document_methods_are_executable_but_require_verified_annotations():
    methods = by_id()
    assert methods["attention.designated_support"].summary == (
        "Measures attention allocated to annotated supporting elements. These annotations identify designated "
        "evidence and may not include every relevant region.")
    capture_only = {"attention_weights", "validated_ownership_map", "mapped_elements"}
    prepared = {p.method.method_id: p for p in prepare_methods(capture_only)}
    for identifier, missing in (("attention.designated_support", "designated supporting-element annotations"),
                                ("attention.element_type_profile", "element-type annotations"),
                                ("attention.reading_order", "reading-order annotations")):
        assert methods[identifier].status == "implemented"
        require_executable(methods[identifier].method_id, methods[identifier].variant_id)
        assert not prepared[identifier].available
        assert not any(r.startswith("Planned") for r in prepared[identifier].reasons)
        assert any(missing in r for r in prepared[identifier].reasons), identifier
    everything = {r for m in list_methods() for r in m.requires}
    assert not any(p.available for p in prepare_methods(everything) if p.method.status == "planned")


def element(page, *, box=True, order=1, kind="Table", flags=()):
    return NS(document_id="doc", page_id=page, element_type=kind, reading_order=order,
              display_bbox=object() if box else None, geometry_flags=list(flags))


def test_document_structure_reports_annotations_and_their_provenance():
    case = NS(page_ids=["p1"], document_id="doc", annotation_version="v2", support_element_ids=["element_1"])
    snapshot = NS(summary=NS(name="DocLayout benchmark", adapter_version="doclayout_mm_rag_bench_v1"),
                  elements=[element("p1"), element("p1", box=False, order=None, kind="Text", flags=["degenerate"]),
                            element("p2", kind="Figure")])
    capabilities, structure = document_structure(case, snapshot)
    assert capabilities == {"designated_support_annotations", "element_type_annotations", "reading_order_annotations"}
    assert structure.provenance == "dataset"
    assert (structure.element_count, structure.boxed_element_count, structure.flagged_element_count,
            structure.reading_order_count, structure.support_element_count) == (2, 1, 1, 1, 1)
    assert structure.element_types == ["Table", "Text"]
    imported = NS(summary=NS(name="scan.pdf", adapter_version="uploaded_page_images_v2"), elements=[])
    capabilities, structure = document_structure(NS(**{**vars(case), "support_element_ids": []}), imported)
    # Imported page images carry no annotations: absent, never zero-valued.
    assert capabilities == set() and structure.provenance == "imported" and structure.element_count == 0
