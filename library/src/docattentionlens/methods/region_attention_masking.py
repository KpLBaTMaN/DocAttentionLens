"""Remove incoming edges to patch centres in a declared unit-page rectangle."""
from typing import Annotated
from pydantic import Field
from .contracts import RowParameters
from ._intervention_method import definition
class Parameters(RowParameters):
    region: list[Annotated[float,Field(ge=0,le=1)]] = Field(default_factory=lambda:[.25,.25,.75,.75],min_length=4,max_length=4,
        description="Unit-page rectangle x1,y1,x2,y2; targets are patch centres inside it.")
    all_layers: bool = False
    all_heads: bool = False
DEFINITION=definition("attention.region_masking","Region Attention Masking",
    "Remove attention from sources outside a declared page rectangle to image-patch centres inside it, leaving source pixels unchanged.",Parameters)
