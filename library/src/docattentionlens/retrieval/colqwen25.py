"""ColQwen2.5 v0.2 retriever: the ColQwen adapter on the Qwen2.5-VL-3B backbone.

The checkpoint is a LoRA adapter over vidore/colqwen2.5-base; `colqwen.load_colqwen25` merges the pinned
pair at load. Pixel mapping, causal masks, scoring and attention capture are the ColQwen adapter's, plus a
per-page check that the vision tower's window permutation is inverted before the language model.
"""
from .colqwen import COLQWEN25_REGISTRATION_ID, capture as _colqwen_capture, release_runtime

__all__ = ["capture", "release_runtime"]


def capture(case, inputs, cache, interrupted=lambda: False, **options):
    return _colqwen_capture(case, inputs, cache, interrupted, registration=COLQWEN25_REGISTRATION_ID, **options)
