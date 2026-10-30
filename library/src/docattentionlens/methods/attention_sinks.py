"""Selected-row allocation to declared position roles; not persistent sink detection."""
from typing import Literal
from pydantic import Field, field_validator
from .contracts import RowParameters, Calculation
from ._shared import definition, selected_rows, statistic
from ._literature import CLARK_2019, XIAO_2023_SINKS


class Parameters(RowParameters):
    roles: list[Literal["image", "document_text", "query", "prompt", "special", "generated"]] = Field(default=["special"], min_length=1, max_length=6)

    @field_validator("roles")
    @classmethod
    def distinct(cls, values):
        if len(set(values)) != len(values):
            raise ValueError("Target roles must be distinct")
        return values


def compute(context, parameters):
    rows, mask = selected_rows(context, parameters)
    indices = [i for i,p in enumerate(context.record.positions[:len(mask)]) if mask[i] and p.role in parameters.roles]
    weights = rows[0,0]
    return Calculation(statistics=[statistic("Declared-group attention mass", weights[indices].sum(), "dimensionless mass"),
        statistic("Group valid targets", len(indices), "positions"), statistic("Total captured mass", weights.sum(), "dimensionless mass")])


DEFINITION = definition(method_id="attention.sinks", name="Selected-role attention mass", perspective="general", family="attention_statistics",
    quantity="declared_group_mass", formula="Sum attention on explicitly selected target roles for one retained prediction.",
    parameters=Parameters, calculate=compute, displays=["scalar_statistic"],
    limitations=["This group allocation does not demonstrate persistent sink behaviour across sources or examples."],
    references=[CLARK_2019, XIAO_2023_SINKS],
    literature=("standard_summary", "Attention mass on chosen token roles in one row. Clark et al. (2019) report attention to special tokens and Xiao et al. (2023) name attention sinks, but this sum does not detect sinks."))
