"""Concentration over the valid support, including valid zero weights."""
import numpy as np
from .contracts import RowParameters, Calculation
from ._shared import definition, probability_row, statistic, STAT_NORMALIZATION
from ._literature import HURLEY_RICKARD_2009, SIMPSON_1949, HILL_1973


class Parameters(RowParameters):
    pass


def compute(context, parameters):
    p, mask = probability_row(context, parameters)
    n = len(p)
    ordered = np.sort(p)
    gini = float(np.dot(2*np.arange(1,n+1)-n-1, ordered)/n)
    return Calculation(statistics=[statistic("Maximum probability", p.max()),
        statistic("Gini concentration", gini), statistic("Inverse Simpson effective support", 1/np.dot(p,p), "positions"),
        statistic("Valid targets", mask.sum(), "positions")], normalization=STAT_NORMALIZATION)


DEFINITION = definition(method_id="attention.concentration", name="Attention Concentration", perspective="general", family="attention_statistics",
    quantity="concentration", units="dimensionless concentration; effective support in positions", formula="Maximum probability, Gini concentration and inverse Simpson effective support on declared valid targets.",
    parameters=Parameters, calculate=compute, displays=["scalar_statistic"], normalization=STAT_NORMALIZATION,
    references=[HURLEY_RICKARD_2009, SIMPSON_1949, HILL_1973],
    literature=("standard_summary", "Maximum weight, Gini index (Hurley & Rickard) and inverse Simpson effective number (Simpson 1949; Hill 1973) of one row: standard statistics, not one published attention method."))
