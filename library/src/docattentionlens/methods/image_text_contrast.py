"""Controlled generation representation experiment and measured paired outcomes."""
from typing import Literal
from pydantic import Field
from .contracts import RowParameters,Calculation
from ._document import table,define,REQUIRES


class Parameters(RowParameters):
    representations: list[Literal["image","ocr","structured"]] = Field(default_factory=lambda:["image","ocr","structured"],min_length=2,max_length=3)


def compute(context,parameters):
    if context.experiment is None: raise ValueError("Run and select a saved document representation experiment")
    outcomes=context.experiment.outcomes
    baseline=outcomes[0]
    return Calculation(table=table("Representation outcomes",["Answer text changed","Input positions","Generated tokens"],
        [o.name for o in outcomes],[[float(o.answer!=baseline.answer),float(o.input_positions),float(len(o.output_token_ids))] for o in outcomes],"0/1; positions; tokens"),
        notices=[f"{o.name}: {o.answer}" for o in outcomes]+["Attention must be compared on aligned elements and compatible consumed histories; image and OCR token axes are not interchangeable."])


DEFINITION=define("document.image_text_contrast","Image versus structured-text contrast","Measure generation under fixed evidence rendered as images, OCR text and structured element text.",Parameters,compute,
    requires=REQUIRES+["experiment_outcomes"],units="answer change; input/output positions",
    updates={"execution_requirement":"paired_execution","evidence_kind":"input_perturbation_effect"})
