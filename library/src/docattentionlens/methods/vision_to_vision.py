"""Vision → Vision Attention: declared source/target block, original weights and conditional shares."""
from .contracts import RowParameters
from ._attention_only import define
from ._modality_block import block
from ._literature import AFLALO_2022_VL_INTERPRET

def calculate(context,p):
    return block(context,p,True,True)

DEFINITION=define("attention.vision_to_vision","Vision → Vision Attention","multimodal_attention","modality_block_attention",
    "Inspect image-patch source attention on image-patch targets, preserving original-domain block mass and separately reporting within-block shares.",
    RowParameters,calculate,requires=["attention_weights","complete_declared_support","captured_image_sources","image_mask"],
    displays=["page_overlay","token_table"],
    limitations=["Generator captures: image patches precede the question, so a patch row attends only to earlier template tokens and patches, never to the question or the answer."],
    references=[AFLALO_2022_VL_INTERPRET],
    literature=("published_method","Image-to-image block of the four language/vision attention blocks that VL-InterpreT separates (Aflalo et al. 2022). Image positions are the language model's image tokens, not vision-tower attention."))
