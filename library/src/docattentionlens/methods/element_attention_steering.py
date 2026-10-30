"""Element-owned target logit bias in live decoder attention; full valid-domain softmax."""
from pydantic import Field
from .contracts import RowParameters,Calculation
from ._document import table,plot,define,REQUIRES


class Parameters(RowParameters):
    element_ids: list[str] = Field(default_factory=list,max_length=20)
    strengths: list[float] = Field(default_factory=lambda:[-2.0,0.0,2.0],min_length=1,max_length=7)


def compute(context,parameters):
    if context.experiment is None: raise ValueError("Run and select a saved attention-steering experiment")
    outcomes=context.experiment.outcomes; baseline=outcomes[0]
    points=[dict(label=o.name,x=o.strength if o.strength is not None else 0,y=o.selected_element_mass,group="matched control" if o.name.startswith("control_") else "selected element") for o in outcomes if getattr(o,"selected_element_mass",None) is not None]
    return Calculation(table=table("Element steering dose response",["Logit bias strength","Answer text changed","Generated tokens","Selected element mass"],
        [o.name for o in outcomes],[[o.strength,float(o.answer!=baseline.answer),float(len(o.output_token_ids)),getattr(o,"selected_element_mass",None)] for o in outcomes],"logit bias; 0/1; tokens; attention mass"),
        plots=[plot("line","Element steering dose response","Logit bias strength","Selected element original-domain mass",points)] if points else [],
        notices=[f"{o.name}: {o.answer}" for o in outcomes]+["The selected decoder layer/head receives strength × target ownership before softmax. Causal masking and full-domain normalization are retained."])


DEFINITION=define("document.element_steering","Element attention steering","Bias live decoder target logits by selected element ownership and measure fixed-evidence generation across strengths and a matched-area control.",Parameters,compute,
    requires=REQUIRES+["experiment_outcomes"],units="logit bias; answer change",
    updates={"execution_requirement":"paired_execution","evidence_kind":"internal_intervention_effect"})
