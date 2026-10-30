"""Distribution similarity = 1 minus base-2 Jensen–Shannon divergence."""
import numpy as np
from .contracts import HeadParameters, Calculation, ResultTable, ResearchPlot, ResearchPoint
from ._shared import definition, selected_rows, selection, STAT_NORMALIZATION
from ._literature import CLARK_2019, LIN_1991_JSD


class Parameters(HeadParameters):
    pass


def compute(context, parameters):
    heads = selection(parameters.heads, context.record.query_head_count)
    rows, mask = selected_rows(context, parameters, heads=heads)
    distributions = rows[0][:,mask]
    distributions = distributions/distributions.sum(axis=1)[:,None]
    values = []
    for p in distributions:
        line = []
        for q in distributions:
            m = (p+q)/2
            pi, qi = p>0, q>0
            divergence = .5*(np.sum(p[pi]*np.log2(p[pi]/m[pi]))+np.sum(q[qi]*np.log2(q[qi]/m[qi])))
            line.append(float(np.clip(1-divergence, 0, 1)))
        values.append(line)
    labels = [f"Head {h}" for h in heads]
    # Grid cells use the head_profile_comparison layout: row i is the head, column j the compared head.
    last = max(len(heads)-1, 1)
    points = [ResearchPoint(label=f"{labels[i]} vs {labels[j]}", x=j/last, y=i/last, value=values[i][j])
              for i in range(len(heads)) for j in range(len(heads))]
    return Calculation(table=ResultTable(columns=labels, labels=labels, values=values, units="1 − Jensen–Shannon divergence (base 2)"),
        plots=[ResearchPlot(kind="grid", title=f"Head distribution similarity · layer {parameters.layer}", x_label="Compared head", y_label="Head", points=points)],
        normalization=STAT_NORMALIZATION)


DEFINITION = definition(method_id="heads.distribution_similarity", name="Head Similarity · captured prediction", perspective="general", family="head_and_representation",
    quantity="head_distribution_similarity", formula="1 minus Jensen–Shannon divergence between head probability rows on identical valid causal support.",
    parameters=Parameters, calculate=compute, displays=["head_grid"], units="dimensionless similarity in [0,1]",
    normalization=STAT_NORMALIZATION, limitations=["Only selected retained predictions; no corpus population or stable head-behaviour claim."],
    references=[CLARK_2019, LIN_1991_JSD],
    literature=("adaptation", "Uses Jensen-Shannon divergence (Lin 1991) as Clark et al. (2019) do, but for one source row and reported as 1 - JSD (base 2), not as distances summed over a dataset."))
