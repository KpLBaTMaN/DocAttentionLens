import pytest
from pydantic import ValidationError
from docattentionlens.datasets.contracts import DisplaySettings


def test_old_settings_keep_defaults_and_new_settings_round_trip():
    old = DisplaySettings.model_validate({"schema_version": "0.2.0"})
    assert old.attention_grouping == old.retrieval_grouping == "positions"
    assert old.attention_element_view == "elements"
    new = old.model_copy(update={"attention_grouping": "elements", "retrieval_grouping": "elements",
                                 "attention_element_view": "patches", "attention_grouped_comparison_index": 2})
    assert DisplaySettings.model_validate_json(new.model_dump_json()) == new


@pytest.mark.parametrize("values", [{"attention_grouping": "types"}, {"attention_element_view": "raw"},
                                   {"attention_grouped_comparison_index": -1}, {"attention_grouped_comparison_index": 128}])
def test_rejects_invalid_grouping_settings(values):
    with pytest.raises(ValidationError):
        DisplaySettings.model_validate(values)
