import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from docattentionlens.datasets.adapter import identity
from docattentionlens.datasets.contracts import (
    DisplaySettings, ElementRecord, PageRecord, QuestionRecord, ViewState,
)

EXAMPLES = Path(__file__).resolve().parents[2] / "tests/fixtures/datasets"


def test_old_display_settings_gain_safe_defaults_and_new_preferences_round_trip():
    old = DisplaySettings.model_validate({"schema_version": "0.2.0", "zoom": 2})
    assert old.input_view == "original"
    assert old.attention_scale_mode == "auto"
    assert not old.evidence_dock_open
    assert old.rail_tab == "pages"
    assert not old.rail_collapsed and not old.inspector_collapsed
    assert old.canvas_fit == "manual"
    assert old.attention_opacity == old.similarity_opacity == 0.65
    assert old.attention_display_mode == "overlay"
    assert not old.attention_show_patches and not old.attention_smooth
    assert old.attention_colormap == "inferno" and old.attention_range_floor == "zero"
    assert ViewState.model_validate({"attention_layer": 35, "attention_head": 15}).attention_method_id == "attention.row"
    assert ViewState().attention_variant_id == "captured_weights_v1"
    new = DisplaySettings.model_validate({"input_view": "split", "attention_scale_mode": "fixed",
        "attention_scale_max": 0.01, "rail_width": 236, "inspector_width": 336, "dock_height": 196,
        "rail_collapsed": True, "inspector_collapsed": True, "rail_tab": "elements",
        "evidence_dock_open": True, "canvas_fit": "width", "attention_opacity": 0.35, "similarity_opacity": 0,
        "attention_display_mode": "blend", "attention_show_patches": True, "attention_smooth": True,
        "attention_colormap": "viridis", "attention_range_floor": "minimum"})
    assert DisplaySettings.model_validate(new.model_dump()) == new
    assert DisplaySettings(attention_scale_mode="causal").attention_scale_mode == "causal"
    for invalid in ({"attention_scale_max": 0}, {"retrieval_scale_max": -1}, {"dock_height": 999},
                    {"attention_opacity": -0.01}, {"similarity_opacity": 1.01}, {"attention_opacity": float("nan")},
                    {"attention_display_mode": "glow"}, {"attention_colormap": "jet"}, {"attention_range_floor": "mean"}):
        with pytest.raises(ValidationError):
            DisplaySettings.model_validate(invalid)

    selected = ViewState(attention_method_id="attention.element_allocation", attention_variant_id="ownership_mass_v1", attention_layer=35, attention_head=15)
    assert ViewState.model_validate(selected.model_dump()) == selected
    # Registry-backed method/variant validation belongs to save/preparation, not
    # an obsolete two-value transport enum.
    advanced = ViewState(attention_method_id="attention.mean_heads", attention_variant_id="captured_prediction_v1", attention_parameters={"heads":[0,1]})
    assert ViewState.model_validate(advanced.model_dump()) == advanced
    with pytest.raises(ValidationError):
        ViewState(attention_method_id="")


def example(name):
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


def test_resource_identity_binds_version_domain_document_and_native_type():
    base = identity("element", "version", "annual_reports", 1, 5)
    alternatives = [
        ("next-version", "annual_reports", 1, 5),
        ("version", "professional_services_reports", 1, 5),
        ("version", "annual_reports", 2, 5),
        ("version", "annual_reports", 1, "5"),
    ]
    assert len({base, *(identity("element", *parts) for parts in alternatives)}) == 5


def test_actual_flagged_box_and_empty_ocr_are_preserved():
    flagged = example("flagged-geometry")
    element = ElementRecord.model_validate(flagged["element"])
    page = PageRecord.model_validate(flagged["page"])
    assert element.geometry_flags
    assert element.bbox.model_dump() == flagged["element"]["bbox"]
    assert element.display_geometry_policy != "raw"
    if element.display_bbox:
        assert 0 <= element.display_bbox.x1 < element.display_bbox.x2 <= page.width
        assert 0 <= element.display_bbox.y1 < element.display_bbox.y2 <= page.height
    empty = ElementRecord.model_validate(example("empty-ocr")["element"])
    assert empty.ocr.status == "empty"
    assert empty.ocr.text is not None and not empty.ocr.text.strip()


def test_real_mult_page_support_references_remain_evaluation_annotations():
    value = example("multi-page-support")
    question = QuestionRecord.model_validate(value["question"])
    assert len(question.support_page_ids) > 1
    assert len(question.support_element_refs) == len(question.support_element_ids)
    assert len(question.support_page_refs) == len(question.support_page_ids)
    assert question.support_role == "evaluation_annotation"
    assert all(ref.document_id == question.ref.document_id for ref in question.support_element_refs)
    assert {page["page_id"] for page in value["pages"]} == set(question.support_page_ids)


def test_display_state_rejects_scientific_parameters_and_nonfinite_pan():
    with pytest.raises(ValidationError):
        DisplaySettings(attention_normalization="row")
    with pytest.raises(ValidationError):
        DisplaySettings(center_x=float("inf"))
