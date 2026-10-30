"""General Analysis methods state how each calculation relates to its cited literature."""
import json
from pathlib import Path
from docattentionlens.methods.registry import list_methods
from docattentionlens.retrieval.attention import methods as retriever_methods

AUDIT = Path(__file__).resolve().parents[3] / "tests/fixtures/methods/literature-relations.json"


def analysis_picker(methods):
    # The General perspective of the Analysis picker (AttentionPanel.tsx `isAttention`).
    return [m for m in methods if m.perspective == "general"
            and (m.evidence_kind.startswith("attention_") or m.family == "head_and_representation")]


def test_every_general_analysis_method_declares_its_literature_relation():
    for method in analysis_picker(list_methods()):
        relation = method.literature_relation
        assert relation is not None and relation.note.strip(), method.method_id
        # Published methods and adaptations name the work they match or depart from.
        if relation.kind != "standard_summary":
            assert any(r.startswith("https://") for r in method.source_references), method.method_id


def test_descriptor_relations_agree_with_the_literature_audit():
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))["methods"]
    expected = {m["method_id"]: m["origin_class"] if m["origin_class"] in ("adaptation", "standard_summary") else "published_method"
                for m in audit}
    assert {m.method_id: m.literature_relation.kind for m in analysis_picker(list_methods())} == expected


def test_selected_role_allocation_is_not_presented_as_sink_detection():
    methods = {m.method_id: m for m in list_methods()}
    assert methods["attention.sinks"].literature_relation.kind == "standard_summary"
    assert "does not detect sinks" in methods["attention.sinks"].literature_relation.note
    assert methods["attention.sink_detection"].literature_relation.kind == "adaptation"


def test_retriever_variants_keep_the_relation_of_their_calculation():
    canonical = {m.method_id: m for m in analysis_picker(list_methods())}
    for variant in analysis_picker(retriever_methods()):
        assert variant.literature_relation == canonical[variant.method_id].literature_relation, variant.method_id
        assert variant.source_references == canonical[variant.method_id].source_references, variant.method_id
