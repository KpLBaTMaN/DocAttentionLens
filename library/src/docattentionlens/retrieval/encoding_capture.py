"""Bounded retriever language-attention extraction and runtime equivalence checks.

ColPali's PaliGemma encoder attends over the full active prefix; ColQwen2's Qwen2-VL
encoder is causal. Both are verified from the actual runtime, never assumed.

Transformers' eager kernel rounds attention logits to the model dtype. Qwen2-VL logits
reach about 9,000 in layer 0 and 1,150 in layer 16, where one BF16 step is 64 and 8, so
BF16 eager encodings drift far from the SDPA kernels, which keep logits in FP32. ColQwen2
therefore extracts with FLOAT32_EAGER: the eager formula with FP32 logits, softmax and
value mixing from the BF16 projections, as PyTorch's reference SDPA computes it. Only
the attention output returns to BF16. ColPali's logits are small; it keeps stock eager.
"""
import numpy as np
from .contracts import EncodingAttention, QUERY_POSITION_LIMIT, PAGE_POSITION_LIMIT

FLOAT32_EAGER = "eager_float32"


def eager_float32_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Eager attention in FP32; the returned post-softmax weights are never rounded to BF16."""
    if dropout:
        raise ValueError("Attention extraction runs in evaluation mode without dropout")
    # Grouped key/value heads repeat in place, as Transformers' repeat_kv does.
    key = key.repeat_interleave(module.num_key_value_groups, dim=1).float()
    value = value.repeat_interleave(module.num_key_value_groups, dim=1).float()
    logits = query.float() @ key.transpose(2, 3) * scaling
    if attention_mask is not None:
        logits = logits + attention_mask[:, :, :, :key.shape[-2]].float()
    weights = logits.softmax(dim=-1)
    return (weights @ value).to(query.dtype).transpose(1, 2).contiguous(), weights


def register_float32_eager():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, eager_mask
    AttentionInterface.register(FLOAT32_EAGER, eager_float32_attention)
    # Without its own mask entry Transformers would build no causal mask for this kernel.
    AttentionMaskInterface.register(FLOAT32_EAGER, eager_mask)


def capture_encoding(torch, model, batch, encoding_id, pass_kind, arrays, page_id=None, *, causal=False, float32_attention=False, rope_positions=None):
    """rope_positions, when known independently: the (axes, positions) rotary indices the language model must receive."""
    language = model.vlm.model.language_model
    n = batch["input_ids"].shape[1]
    if n > (QUERY_POSITION_LIMIT if pass_kind == "query" else PAGE_POSITION_LIMIT):
        raise ValueError("Encoding exceeds the bounded attention plan; no truncation applied")
    mask = batch["attention_mask"][0].bool()
    observed = []

    def check_mask(module, args, kwargs):
        actual = kwargs.get("attention_mask")
        if causal:
            # Qwen2-VL receives the padding mask and builds its causal mask internally;
            # causality is verified on the returned weights below.
            if actual is None or tuple(actual.shape) != (1, n) or not torch.equal(actual.bool(), mask[None]):
                raise ValueError("Retriever runtime padding mask differs from the encoded positions")
        else:
            expected = mask[None,None,None,:].expand(1,1,n,n)
            if actual is None or actual.shape != expected.shape or not torch.equal(actual == 0, expected):
                raise ValueError("Retriever runtime mask differs from the full active-key prefix")
        if rope_positions is not None:
            actual = kwargs.get("position_ids")
            if actual is None or not np.array_equal(actual[:, 0].cpu().numpy(), rope_positions):
                raise ValueError("Retriever runtime position ids differ from the independent rotary reconstruction")
        observed.append(True)

    hook = language.register_forward_pre_hook(check_mask, with_kwargs=True)
    original = language.config._attn_implementation
    try:
        if float32_attention:
            register_float32_eager()
        language.set_attn_implementation(FLOAT32_EAGER if float32_attention else "eager")
        with torch.inference_mode():
            captured = model(**batch, output_attentions=True, use_cache=False)
            attention = captured.attentions
            embeddings = captured.embeddings[0].float().cpu().numpy()
            if attention is None or len(attention) != model.config.vlm_config.text_config.num_hidden_layers:
                raise ValueError("Retriever language attention was not returned by every layer")
            names = []
            max_mass_error = 0.0
            heads = model.config.vlm_config.text_config.num_attention_heads
            future = torch.triu(torch.ones(n, n, dtype=torch.bool, device=mask.device), 1)
            for layer, value in enumerate(attention):
                if value.shape != (1,heads,n,n) or not torch.isfinite(value).all() or (value < 0).any():
                    raise ValueError("Retriever attention axes or finite weights differ")
                if (~mask).any() and (value[...,~mask] != 0).any():
                    raise ValueError("Retriever attention includes masked keys")
                if causal and (value[0][:, future] != 0).any():
                    raise ValueError("Causal retriever attention includes future keys")
                error = float((value.float().sum(dim=-1)-1).abs().max())
                if error > .02:
                    raise ValueError("Retriever attention normalization failed")
                max_mass_error = max(max_mass_error,error)
                name = f"attn_{encoding_id}_l{layer}"
                arrays[name] = value[0].float().cpu().numpy()
                names.append(name)
            del attention, captured
            plain = model(**batch, output_attentions=False, use_cache=False).embeddings[0].float().cpu().numpy()
            if not np.array_equal(embeddings, plain):
                raise ValueError("Retriever capture-on/off embeddings differ")
        if len(observed) != 2:
            raise ValueError("Retriever mask validation did not cover both forwards")
    finally:
        hook.remove()
        language.set_attn_implementation(original)
    return embeddings, EncodingAttention(encoding_id=encoding_id, pass_kind=pass_kind, page_id=page_id,
        layer_count=len(names), query_head_count=heads, position_count=n,
        attention_mask=mask.int().tolist(), layer_arrays=names, causal=causal,
        **({"normalization": "causal encoding prefix; each source sees itself and earlier active keys"} if causal else {})), max_mass_error
