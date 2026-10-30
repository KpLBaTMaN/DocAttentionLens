"""One head's context output is zeroed before output projection."""
from .contracts import RowParameters
from ._intervention_method import definition
DEFINITION=definition("attention.head_ablation","Head Ablation",
    "Zero one language-attention head's context output at a chosen layer across native sources, then measure model behaviour.",RowParameters)
