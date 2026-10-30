"""Row Structure: same-axis allocation excluding self, with a declared uniform baseline."""
from typing import Literal
from .contracts import RowParameters
from ._attention_only import define
from ._axis_structure import structure
from ._literature import YEH_2023_ATTENTIONVIZ

class Parameters(RowParameters):
    source_scope: Literal["selected","all_image"] = "all_image"

def calculate(context,p):
    return structure(context,p,0,"Row")

DEFINITION=define("attention.row_structure","Row Structure","spatial_attention","axis_attention_mass",
    "Measure same-row image attention excluding self, with offset distributions and valid-key uniform baselines.",
    Parameters,calculate,requires=["attention_weights","full_square_attention_matrices","complete_declared_support","image_mask"],displays=["patch_grid","source_table"],
    references=[YEH_2023_ATTENTIONVIZ],
    literature=("adaptation","AttentionViz (Yeh et al. 2023) observes heads attending along the same image row; self-excluded mass, the offset profile and the uniform baseline quantify that observation and are additions."))
