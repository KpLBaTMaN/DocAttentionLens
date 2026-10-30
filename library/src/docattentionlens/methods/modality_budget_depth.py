"""Original-domain token-role budgets across captured decoder layers."""
from pydantic import Field
from .contracts import AxisIndex,RowParameters, Calculation
from ._shared import selected_rows, selection, definition
from ._document import table, plot
from ._literature import CHEN_2024_FASTV


class Parameters(RowParameters):
    layers: list[AxisIndex] = Field(default_factory=list, max_length=128)


def compute(context, parameters):
    layers=selection(parameters.layers,context.record.layer_count)
    rows, mask=selected_rows(context,parameters,layers=layers)
    roles=["image","document_text","query","prompt","special","generated"]
    values=[]; points=[]
    for j,layer in enumerate(layers):
        values.append([float(rows[j,0,[i for i,p in enumerate(context.record.positions[:len(mask)]) if mask[i] and p.role==role]].sum()) for role in roles])
        for role,mass in zip(roles,values[-1],strict=True): points.append(dict(label=f"Layer {layer} · {role}",x=float(layer),y=mass,group=role))
    return Calculation(table=table("Modality budget over decoder depth",roles,[f"Layer {i}" for i in layers],values),
        plots=[plot("line","Attention budget by decoder layer","Decoder layer (index)","Original-domain attention mass",points)],
        notices=["One captured source and selected query head; decoder attention only, not vision-encoder attention."])


DEFINITION=definition(method_id="attention.modality_budget_depth",name="Modality budget over depth",perspective="general",family="attention_statistics",
    quantity="modality_budget_depth",formula="Sum original-domain attention by recorded token role independently at each selected decoder layer.",parameters=Parameters,calculate=compute,displays=["layer_series","table"],
    references=[CHEN_2024_FASTV],
    literature=("adaptation", "Adapts FastV per-layer attention allocation by token type (Chen et al. 2024) to one source and head, with DocAttentionLens token roles."))
