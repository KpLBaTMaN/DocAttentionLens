"""Fixed-evidence image masking, re-encoding and measured generation effects."""
from typing import Literal
from pydantic import Field
from .contracts import RowParameters,Calculation
from ._document import table,define,REQUIRES


class Parameters(RowParameters):
    element_ids: list[str] = Field(default_factory=list,max_length=20)
    replacement: Literal["white","blur"] = "white"


def compute(context,parameters):
    if context.experiment is None: raise ValueError("Run and select a saved element-masking experiment")
    outcomes=context.experiment.outcomes; baseline=outcomes[0]
    return Calculation(table=table("Masking outcomes",["Answer text changed","Generated tokens"],[o.name for o in outcomes],
        [[float(o.answer!=baseline.answer),float(len(o.output_token_ids))] for o in outcomes],"0/1; tokens"),
        notices=[f"{o.name}: {o.answer}" for o in outcomes]+["Masks are input perturbations and each image is re-encoded. Effects are conditional on the declared replacement and fixed evidence."])


DEFINITION=define("document.element_masking","Element masking test","Compare fixed-evidence generation after replacing selected element pixels, with a matched-area non-support control.",Parameters,compute,
    requires=REQUIRES+["experiment_outcomes"],units="answer change; output tokens",
    updates={"execution_requirement":"paired_execution","evidence_kind":"input_perturbation_effect"})
