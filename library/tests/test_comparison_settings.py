import pytest
from pydantic import ValidationError
from docattentionlens.datasets.contracts import DisplaySettings


def test_legacy_display_settings_gain_comparison_defaults():
    settings = DisplaySettings.model_validate({"schema_version": "0.2.0", "zoom": 2})
    assert settings.comparison_columns == 2
    assert settings.comparison_zoom == 1
    assert settings.comparison_center_x is None
    assert settings.zoom == 2


def test_comparison_view_round_trips_and_rejects_invalid_bounds():
    settings = DisplaySettings(comparison_columns=4, comparison_zoom=3,
                               comparison_center_x=480, comparison_center_y=620)
    assert DisplaySettings.model_validate_json(settings.model_dump_json()) == settings
    for value in ({"comparison_columns": 3}, {"comparison_zoom": 0.5}, {"comparison_center_x": -1}):
        with pytest.raises(ValidationError):
            DisplaySettings.model_validate(value)
