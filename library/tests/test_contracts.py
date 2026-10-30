import hashlib
from pathlib import Path

import pytest

from docattentionlens.core.integrity import IntegrityInput, check_integrity
from docattentionlens.methods.registry import get_method, list_methods, require_executable


def test_method_availability_respects_capture_support_and_implementation_independently():
    from types import SimpleNamespace
    from docattentionlens.methods.registry import prediction_capabilities, prepare_methods

    assert not any(p.available for p in prepare_methods(prediction_capabilities(None)))
    # Metadata fixture: a retained prediction row, not full layer matrices.
    capture = SimpleNamespace(steps=[0], coverage={"quantity": "post_softmax; dropout disabled", "all_causal_targets": True, "ownership": "exact cell partition"},
                              positions=[SimpleNamespace(role="image", bbox=[0, 0, 1, 1], ownership={"background": 1})])
    capabilities = prediction_capabilities(capture)
    assert "full_square_attention_matrices" not in capabilities
    assert "validated_source_target_geometry" not in capabilities
    prepared = {p.method.method_id: p for p in prepare_methods(capabilities)}
    # A mapping with no annotated element is missing structure, not zero element attention.
    assert sum(p.available for p in prepared.values()) == 14
    assert any("annotated element boxes mapped to captured patches" in r for r in prepared["attention.element_allocation"].reasons)
    capture.positions.append(SimpleNamespace(role="image", bbox=[1, 0, 2, 1], ownership={"element_a": .5, "background": .5}))
    capabilities = prediction_capabilities(capture)
    prepared = {p.method.method_id: p for p in prepare_methods(capabilities)}
    assert sum(p.available for p in prepared.values()) == 15
    assert prepared["attention.element_allocation"].available
    assert prepared["attention.entropy"].available
    assert any("Save full attention matrices" in r for r in prepared["attention.rollout"].reasons)
    capture.coverage.pop("ownership")
    prepared = {p.method.method_id: p for p in prepare_methods(prediction_capabilities(capture))}
    assert prepared["attention.row"].available
    assert not prepared["attention.element_allocation"].available
    # Preparation checks declared prerequisites independently of actual execution.
    all_requirements = {r for m in list_methods() for r in m.requires}
    assert sum(p.available for p in prepare_methods(all_requirements)) == 67


def test_registry_retains_evidence_distinctions_and_planned_gate():
    methods = list_methods()
    assert len(methods) == 67
    for descriptor in (m for m in methods if m.status == "planned"):
        assert descriptor.status == "planned"
        assert descriptor.implementation_version is None
        assert descriptor.validation_evidence == []
        with pytest.raises(NotImplementedError):
            require_executable(descriptor.method_id, descriptor.variant_id)
    for identifier in ("attention.row","attention.element_allocation"):
        implemented=get_method(identifier)
        assert implemented.status=="implemented" and implemented.validation_evidence
        require_executable(implemented.method_id,implemented.variant_id)
    retrieval = get_method("retrieval.maxsim")
    assert retrieval.status == "implemented"
    require_executable(retrieval.method_id, retrieval.variant_id)
    assert retrieval.evidence_kind == "retrieval_scoring"
    assert retrieval.output.signed
    assert get_method("attention.row").evidence_kind == "attention_allocation"
    assert get_method("retrieval.query_patch_similarity").evidence_kind == "embedding_similarity"
    assert get_method("heads.distribution_similarity").quantity != get_method("heads.profile_similarity").quantity
    methods[0].requires.clear()
    assert get_method("attention.row").requires


def test_integrity_operation_measures_utf8_bytes():
    text = "Page α / 頁"
    result = check_integrity(IntegrityInput(text=text))
    assert result.byte_count == len(text.encode("utf-8"))
    assert result.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert "no model" in result.interpretation


def test_method_references_resolve_without_the_design_archive():
    root = Path(__file__).resolve().parents[2]
    for method in list_methods():
        for reference in method.source_references:
            assert not reference.startswith("design/"), method.method_id
            if reference.startswith("library/"):
                assert (root / reference.partition(":")[0]).is_file(), reference
