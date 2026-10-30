"""Parameter errors read as one sentence per parameter."""
import pytest
from pydantic import ValidationError
from docattentionlens_api.errors import readable
from docattentionlens.methods.attention_sinks import Parameters as Sinks
from docattentionlens.methods.contracts import RowParameters


def error(model, values):
    with pytest.raises(ValidationError) as caught:
        model.model_validate(values)
    return readable(caught.value)


def test_validation_errors_name_each_parameter_without_pydantic_internals():
    assert error(RowParameters, {"layer": 128}) == "Parameter layer: Input should be less than 128."
    assert error(RowParameters, {"head": "3"}) == "Parameter head: Input should be a valid integer."
    assert error(RowParameters, {"bandwidth": 1}) == "Unknown parameter bandwidth."
    assert error(Sinks, {"roles": ["image", "image"]}) == "Parameter roles: Target roles must be distinct."
    # Steps reach 63: full-attention captures keep every generated step (up to 64 tokens).
    assert error(RowParameters, {"layer": -1, "step": 64}) == ("Parameter step: Input should be less than or equal to 63. "
                                                             "Parameter layer: Input should be greater than or equal to 0.")
    assert "validation error" not in error(RowParameters, {"layer": 128}) and "[type=" not in error(RowParameters, {"layer": 128})


def test_plain_value_errors_keep_their_message():
    assert readable(ValueError("Top-k exceeds valid causal support")) == "Top-k exceeds valid causal support"
