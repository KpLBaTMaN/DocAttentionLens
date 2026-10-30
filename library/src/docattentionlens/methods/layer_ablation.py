"""All head context outputs of one attention layer are zeroed; residual and MLP remain."""
from .contracts import Parameters
from ._intervention_method import definition
DEFINITION=definition("attention.layer_ablation","Layer Attention Ablation",
    "Zero all head context outputs of one attention sublayer before projection, preserving the residual/MLP path, then measure model behaviour.",Parameters)
