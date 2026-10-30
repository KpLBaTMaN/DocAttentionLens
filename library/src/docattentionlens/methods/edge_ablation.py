"""Exact frozen source-target logit masking."""
from pydantic import Field
from .contracts import RowParameters
from ._intervention_method import definition
class Parameters(RowParameters):
    source_index: int = Field(default=0,ge=0,le=1535,strict=True)
    target_index: int = Field(default=1,ge=0,le=1535,strict=True)
    all_layers: bool = False
    all_heads: bool = False
DEFINITION=definition("attention.edge_ablation","Attention-Edge Ablation",
    "Remove a declared frozen input source-to-target edge before softmax, at selected or all language layers/heads, and measure model behaviour.",Parameters)
