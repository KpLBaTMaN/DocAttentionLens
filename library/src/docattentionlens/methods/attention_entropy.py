"""Shannon entropy of a retained prediction over declared valid support."""
from typing import Literal
import numpy as np
from .contracts import RowParameters, Calculation
from ._shared import definition, probability_row, statistic, STAT_NORMALIZATION
from ._literature import SHANNON_1948, CLARK_2019


class Parameters(RowParameters):
    log_base: Literal[2] = 2


def compute(context, parameters):
    p, mask = probability_row(context, parameters)
    positive = p[p > 0]
    entropy = float(-(positive*np.log2(positive)).sum())
    n = int(mask.sum())
    return Calculation(statistics=[statistic("Shannon entropy", entropy, "bits"),
        statistic("Normalized entropy", entropy/np.log2(n) if n > 1 else 0),
        statistic("Valid targets", n, "positions")], normalization=STAT_NORMALIZATION)


DEFINITION = definition(method_id="attention.entropy", variant_id="shannon_row_v1",
    name="Attention entropy", perspective="general", family="attention_statistics", quantity="entropy", units="bits",
    formula="H = -sum p log2 p; normalize by log2 of valid target count, including valid zero weights.",
    parameters=Parameters, calculate=compute, displays=["scalar_statistic"], normalization=STAT_NORMALIZATION,
    references=[SHANNON_1948, CLARK_2019],
    literature=("published_method", "Shannon (1948) entropy in bits of one attention row, as Clark et al. (2019) apply entropy to attention. Normalized entropy and the one-target zero convention are additions; Clark's corpus averages are not reproduced."))
