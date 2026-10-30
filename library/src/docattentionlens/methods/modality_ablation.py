"""Remove one declared modality block of attention edges."""
from typing import Literal
from .contracts import RowParameters
from ._intervention_method import definition
class Parameters(RowParameters):
    block: Literal["text_to_vision","vision_to_text","text_to_text","vision_to_vision"] = "text_to_vision"
    all_layers: bool = True
    all_heads: bool = True
DEFINITION=definition("attention.modality_ablation","Modality-Specific Attention Ablation",
    "Remove one non-image/image source-target block before softmax across selected language layers/heads; native masks remain and all-masked rows are rejected.",Parameters)
