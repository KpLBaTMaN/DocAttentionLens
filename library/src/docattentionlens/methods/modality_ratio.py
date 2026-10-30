"""Role allocations and attention enrichment relative to valid token shares."""
from .contracts import RowParameters, Calculation, ResultTable
from ._shared import definition, selected_rows
from ._literature import CHEN_2024_FASTV


class Parameters(RowParameters):
    pass


def compute(context, parameters):
    rows, mask = selected_rows(context, parameters)
    weights = rows[0,0]
    total = float(weights.sum())
    roles = ["image", "query", "prompt", "special", "generated"]
    if any(p.role=="document_text" for p in context.record.positions[:len(mask)]): roles.insert(1,"document_text")
    table = []
    for role in roles:
        indices = [i for i,p in enumerate(context.record.positions[:len(mask)]) if mask[i] and p.role == role]
        mass = float(weights[indices].sum())
        share = len(indices)/int(mask.sum())
        table.append([mass, len(indices), share, mass/total/share if indices else None])
    return Calculation(table=ResultTable(columns=["Attention mass", "Valid targets", "Token share", "Enrichment"],
        labels=roles, values=table, units="mass; positions; fraction; dimensionless enrichment"),
        normalization="mass retains original domain; enrichment divides mass by total captured mass and valid token share")


DEFINITION = definition(method_id="attention.modality_ratio", name="Token-role share and enrichment", perspective="general", family="attention_statistics",
    quantity="modality_allocation_and_enrichment", units="mass and dimensionless enrichment", formula="Role mass and (role mass / total mass) / valid-token share; absent groups are unavailable.",
    parameters=Parameters, calculate=compute, displays=["table"],
    references=[CHEN_2024_FASTV],
    literature=("adaptation", "Adapts FastV attention allocation and efficiency by token type (Chen et al. 2024): one source and head instead of response tokens averaged over heads and samples, DocAttentionLens token roles, and efficiency rescaled by the valid-token count so that 1 means uniform."))
