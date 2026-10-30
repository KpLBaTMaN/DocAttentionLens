"""Text → Vision Attention: declared source/target block, original weights and conditional shares."""
from .contracts import RowParameters
from ._attention_only import define
from ._modality_block import block
from ._literature import AFLALO_2022_VL_INTERPRET

def calculate(context,p):
    return block(context,p,False,True)

DEFINITION=define("attention.text_to_vision","Text → Vision Attention","multimodal_attention","modality_block_attention",
    "Inspect non-image source attention on image-patch targets, preserving original-domain block mass and separately reporting within-block shares.",
    RowParameters,calculate,requires=["attention_weights","complete_declared_support","captured_text_sources","image_mask"],
    displays=["page_overlay","token_table"],
    references=[AFLALO_2022_VL_INTERPRET],
    literature=("published_method","Text-to-image block of the four language/vision attention blocks that VL-InterpreT separates (Aflalo et al. 2022). Text means every non-image position; image positions are the language model's image tokens."))
