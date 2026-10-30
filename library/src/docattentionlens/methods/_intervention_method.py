"""Canonical descriptors for direct attention interventions; never gradient attribution."""
from dataclasses import replace
from ._attention_only import define

def result(context,p):
    if context.experiment is None or getattr(context.experiment,"result",None) is None:
        raise ValueError("Prepare and run this method in Attention interventions, then open its saved experiment result")
    return context.experiment.result

def definition(identifier,name,formula,parameters):
    method=define(identifier,name,"attention_interventions","internal_attention_intervention_effect",
        formula,parameters,result,requires=["attention_intervention_outcomes"],displays=["experiment_comparison"],
        units="model output and score changes",limitations=["Requires a new model run; it cannot be calculated by editing a saved attention map."])
    return replace(method,descriptor=method.descriptor.model_copy(update={
        "evidence_kind":"internal_intervention_effect","execution_requirement":"model_execution",
        "target_requirement":"Named pinned-model capture and exact frozen input",
        "output":method.descriptor.output.model_copy(update={"signed":True,"normalization":"baseline versus intervention; original inputs unchanged"})}))
