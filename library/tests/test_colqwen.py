"""ColQwen2 adapter pieces that run without a GPU: geometry, token roles and causal encodings.

Mathematical fixtures, never model evidence. The real adapter is validated at capture time.
"""
import math
from types import SimpleNamespace as NS
import numpy as np
import pytest
from docattentionlens.retrieval.colqwen import merged_boxes, token_role
from docattentionlens.retrieval.contracts import (EncodingAttention, Position, RetrievalAttentionRequest,
    RetrievalReceivedRequest, RETRIEVER_PROFILES, retriever_for_adapter)
from docattentionlens.retrieval.attention import calculate, received_column


def test_merged_cells_tile_the_page_row_major():
    boxes = np.asarray(merged_boxes([1, 52, 38], 2480, 3509))
    assert len(boxes) == 26 * 19
    assert boxes[1].tolist() == pytest.approx([2480 / 19, 0, 2 * 2480 / 19, 3509 / 26])  # x changes fastest
    assert boxes[19].tolist() == pytest.approx([0, 3509 / 26, 2480 / 19, 2 * 3509 / 26])
    assert ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sum() == pytest.approx(2480 * 3509)
    assert boxes[-1].tolist() == pytest.approx([2480 * 18 / 19, 3509 * 25 / 26, 2480, 3509])
    for grid in ([1, 3, 4], [2, 4, 4]):
        with pytest.raises(ValueError):
            merged_boxes(grid, 100, 100)


def test_token_roles_separate_patches_controls_and_prompt_text():
    special = {151644, 151652}
    assert token_role(151655, "<|image_pad|>", 151655, special) == "visual_patch"
    assert token_role(151644, "<|im_start|>", 151655, special) == "prefix_or_special"
    assert token_role(198, "Ċ", 151655, special) == "newline"
    assert token_role(74785, "Describe", 151655, special) == "visual_prompt"


def test_profiles_map_adapters_to_their_own_pinned_models():
    colqwen = retriever_for_adapter("colqwen2_transformers_v1")
    assert colqwen["registration_id"] == "builtin-colqwen2" and colqwen["module"] == "colqwen" and colqwen["causal"]
    assert (colqwen["layers"], colqwen["query_heads"], colqwen["image_positions_per_page"]) == (28, 12, 768)
    assert retriever_for_adapter("colpali_transformers_v1")["registration_id"] == "builtin-retrieval"
    assert retriever_for_adapter("unknown") is None
    # The attention budget covers one query and one document encoding at FP32.
    n = 800
    assert RETRIEVER_PROFILES["builtin-colqwen2"]["attention_bytes"] >= 28 * 12 * n * n * 4
    # ColQwen2.5 shares the adapter module family but pins both its LoRA checkpoint and that checkpoint's base.
    colqwen25 = retriever_for_adapter("colqwen2_5_transformers_v1")
    assert colqwen25["registration_id"] == "builtin-colqwen2.5" and colqwen25["module"] == "colqwen25" and colqwen25["causal"]
    assert (colqwen25["model_id"], colqwen25["base_model_id"]) == ("vidore/colqwen2.5-v0.2", "vidore/colqwen2.5-base")
    assert all(len(colqwen25[k]) == 40 for k in ("revision", "base_revision"))
    assert colqwen25["attention_bytes"] >= 36 * 16 * (n * n + 256 * 256) * 4


def lora_fixture(torch, layers=2, hidden=6, rank=2, seed=0):
    from docattentionlens.retrieval.colqwen import LORA_TARGETS
    rng = torch.Generator().manual_seed(seed)
    modules = [f"model.layers.{i}.{part}.{name}" for i in range(layers)
               for part, names in (("self_attn", LORA_TARGETS[:4]), ("mlp", LORA_TARGETS[4:])) for name in names] + ["custom_text_proj"]
    base = {m + ".weight": torch.randn(hidden, hidden, generator=rng).to(torch.bfloat16) for m in modules}
    adapter = {}
    for m in modules:
        adapter[f"base_model.model.{m}.lora_A.weight"] = torch.randn(rank, hidden, generator=rng)
        adapter[f"base_model.model.{m}.lora_B.weight"] = torch.randn(hidden, rank, generator=rng)
    return base, adapter, modules


def test_lora_merge_folds_every_adapted_matrix_in_fp32():
    torch = pytest.importorskip("torch")
    from docattentionlens.retrieval.colqwen import merge_lora
    base, adapter, modules = lora_fixture(torch)
    original = {k: v.clone() for k, v in base.items()}
    count, projection = merge_lora(base, adapter, 2.0, layers=2)
    assert count == len(modules) == 2 * 7 + 1
    for m in modules:
        a, b = adapter[f"base_model.model.{m}.lora_A.weight"], adapter[f"base_model.model.{m}.lora_B.weight"]
        expected = original[m + ".weight"].float() + 2.0 * (b @ a)
        # Merged in FP32, then cast back to the base matrix's own dtype.
        assert base[m + ".weight"].dtype == torch.bfloat16 and torch.equal(base[m + ".weight"], expected.to(torch.bfloat16))
    # The projection is returned before its BF16 cast, for comparison with the checkpoint's FP32 export.
    assert projection.dtype == torch.float32 and torch.allclose(projection, original["custom_text_proj.weight"].float()
        + 2.0 * adapter["base_model.model.custom_text_proj.lora_B.weight"] @ adapter["base_model.model.custom_text_proj.lora_A.weight"])


@pytest.mark.parametrize("damage", ["missing pair", "extra module", "unpaired matrix", "wrong rank"])
def test_lora_merge_refuses_adapters_that_do_not_cover_exactly_the_language_projections(damage):
    torch = pytest.importorskip("torch")
    from docattentionlens.retrieval.colqwen import merge_lora
    base, adapter, _ = lora_fixture(torch)
    if damage == "missing pair":
        del adapter["base_model.model.model.layers.1.mlp.up_proj.lora_A.weight"], adapter["base_model.model.model.layers.1.mlp.up_proj.lora_B.weight"]
    elif damage == "extra module":
        adapter["base_model.model.visual.blocks.0.attn.qkv.lora_A.weight"] = torch.zeros(2, 6)
        adapter["base_model.model.visual.blocks.0.attn.qkv.lora_B.weight"] = torch.zeros(6, 2)
    elif damage == "unpaired matrix":
        del adapter["base_model.model.custom_text_proj.lora_B.weight"]
    else:
        adapter["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"] = torch.zeros(6, 3)
    with pytest.raises(RuntimeError, match="incomplete or incompatible"):
        merge_lora(base, adapter, 1.0, layers=2)


def test_colqwen25_checkpoint_names_map_onto_transformers_modules():
    from docattentionlens.retrieval.colqwen import colqwen25_name
    assert colqwen25_name("model.layers.3.mlp.up_proj.weight") == "vlm.model.language_model.layers.3.mlp.up_proj.weight"
    assert colqwen25_name("model.embed_tokens.weight") == "vlm.model.language_model.embed_tokens.weight"
    assert colqwen25_name("model.norm.weight") == "vlm.model.language_model.norm.weight"
    assert colqwen25_name("visual.blocks.0.attn.qkv.weight") == "vlm.model.visual.blocks.0.attn.qkv.weight"
    assert colqwen25_name("custom_text_proj.bias") == "embedding_proj_layer.bias"
    with pytest.raises(RuntimeError):
        colqwen25_name("lm_head.weight")


def causal_fixture(seed=0, n=8, layers=2, heads=3):
    rng = np.random.default_rng(seed)
    mask = np.ones(n, dtype=int)
    mask[6] = 0
    matrices = rng.dirichlet(np.full(n, .5), size=(layers, heads, n))
    matrices *= np.tril(np.ones((n, n)))          # no future keys
    matrices[..., mask == 0] = 0
    matrices = (matrices / matrices.sum(-1, keepdims=True)).astype(np.float32)
    encoding = EncodingAttention(encoding_id="document", pass_kind="document", page_id="page_a", layer_count=layers,
                                 query_head_count=heads, position_count=n, attention_mask=mask.tolist(),
                                 layer_arrays=[f"l{i}" for i in range(layers)], causal=True)
    roles = ["prefix_or_special", "visual_patch", "visual_patch", "visual_patch", "visual_patch", "visual_prompt", "visual_prompt", "prefix_or_special"]
    positions = [Position(index=i, token_id=i, piece=str(i), role=roles[i], eligible=bool(mask[i]),
                          bbox=[i, 0, i + 1, 1] if roles[i] == "visual_patch" else None) for i in range(n)]
    ownership = [{"element_a": 1.0} if p.bbox else {"non_page": 1.0} for p in positions]
    capture = NS(capture_id="numeric_fixture", attention_encodings=[encoding], query_positions=[],
                 pages=[NS(page_id="page_a", positions=positions, ownership=ownership)])
    return capture, {f"l{i}": matrices[i] for i in range(layers)}, mask.astype(bool)


def encoded(capture, arrays, method, source, **parameters):
    return calculate(capture, RetrievalAttentionRequest(encoding_id="document", source_index=source, method_id=method,
                                                         parameters=parameters), arrays.__getitem__)


def test_causal_statistics_count_only_earlier_active_keys():
    capture, arrays, active = causal_fixture()
    for source in (0, 3, 7):
        a = arrays["l1"][2, source].astype(np.float64)
        support = active & (np.arange(len(a)) <= source)
        p = a[support] / a[support].sum()
        entropy = {s.name: s.value for s in encoded(capture, arrays, "attention.entropy", source, layer=1, head=2).statistics}
        assert entropy["Valid targets"] == support.sum()
        assert entropy["Shannon entropy"] == pytest.approx(-(p[p > 0] * np.log2(p[p > 0])).sum(), abs=1e-9)
        if support.sum() > 1:
            assert entropy["Normalized entropy"] == pytest.approx(entropy["Shannon entropy"] / math.log2(support.sum()), abs=1e-9)
        with pytest.raises(ValueError, match="Top-k"):
            encoded(capture, arrays, "attention.topk_mass", source, layer=1, head=2, k=int(support.sum()) + 1)
    # The first position sees only itself.
    assert encoded(capture, arrays, "attention.row", 0, layer=0, head=0).row.values[0] == pytest.approx(1)
    row = encoded(capture, arrays, "attention.row", 5, layer=0, head=0).row
    assert "causal encoding prefix" in row.normalization and "encoding encoding" not in row.normalization


def test_causal_rows_with_future_mass_are_refused():
    capture, arrays, _ = causal_fixture()
    arrays["l0"][0, 2] = [.5, 0, 0, .5, 0, 0, 0, 0]   # source 2 attending to later key 3
    with pytest.raises(ValueError, match="masked support"):
        encoded(capture, arrays, "attention.row", 2, layer=0, head=0)


def test_causal_columns_omit_sources_that_cannot_see_the_target():
    capture, arrays, active = causal_fixture()
    column = received_column(capture, RetrievalReceivedRequest(encoding_id="document", target_index=3, layer=1, head=0), arrays.__getitem__)
    expected = [i for i in np.flatnonzero(active) if i >= 3]
    assert [s.source_index for s in column.sources] == expected
    assert [s.value for s in column.sources] == pytest.approx(arrays["l1"][0, expected, 3].tolist(), abs=1e-9)
    # Bidirectional encodings keep every active source.
    capture.attention_encodings[0].causal = False
    arrays["l1"][:] = 1 / active.sum() * active
    full = received_column(capture, RetrievalReceivedRequest(encoding_id="document", target_index=3, layer=1, head=0), arrays.__getitem__)
    assert [s.source_index for s in full.sources] == np.flatnonzero(active).tolist()


def fake_language(monkeypatch):
    from docattentionlens.retrieval import colqwen
    calls=[]
    language=NS(config=NS(_attn_implementation="sdpa"))
    def set_mode(mode):
        calls.append(mode); language.config._attn_implementation=mode
    language.set_attn_implementation=set_mode
    model=NS(vlm=NS(model=NS(language_model=language)))
    monkeypatch.setattr(colqwen,"runtime",lambda cache,registration=colqwen.REGISTRATION_ID:(None,None,model))
    return colqwen,language,calls


def test_joint_capture_scores_with_the_sdpa_runtime(monkeypatch):
    colqwen,language,calls=fake_language(monkeypatch)
    def capture(*args,**kwargs):
        assert language.config._attn_implementation=="sdpa" and kwargs["capture_joint_attention"]
        return "record",{}
    monkeypatch.setattr(colqwen,"_capture",capture)
    assert colqwen.capture(None,[],None,capture_attention=True,capture_joint_attention=True)==("record",{})
    assert calls==[]


@pytest.mark.parametrize("failure", [False, True])
def test_controlled_baseline_uses_float32_eager_kernel_and_restores_it(monkeypatch, failure):
    from docattentionlens.retrieval import encoding_capture
    colqwen,language,calls=fake_language(monkeypatch)
    monkeypatch.setattr(encoding_capture,"register_float32_eager",lambda:calls.append("registered"))
    def capture(*args,**kwargs):
        assert language.config._attn_implementation==encoding_capture.FLOAT32_EAGER
        if failure: raise ValueError("fixture capture failure")
        class Record:
            provenance={}; validation={}
            query_positions=[NS(eligible=True)]*30+[NS(eligible=False)]*4
            def model_copy(self,update):return NS(**update)
        return Record(),{}
    monkeypatch.setattr(colqwen,"_capture",capture)
    if failure:
        with pytest.raises(ValueError,match="fixture capture failure"):
            colqwen.capture(None,[],None,force_eager=True)
    else:
        record,_=colqwen.capture(None,[],None,force_eager=True)
        assert record.provenance["attention_implementation"]=="FP32-arithmetic eager controlled baseline/intervention"
        # The worker compares the baseline with the SDPA source scores within this declared tolerance.
        assert record.validation["eager_score_kernel_tolerance"]==pytest.approx(30*colqwen.SCORE_KERNEL_TOLERANCES[colqwen.REGISTRATION_ID])
    assert calls==["registered",encoding_capture.FLOAT32_EAGER,"sdpa"]
    assert language.config._attn_implementation=="sdpa"


def test_rotary_positions_follow_qwen2_vl_text_and_image_rules():
    from docattentionlens.retrieval.colqwen import rope_positions
    assert rope_positions([5, 6, 7], 99).tolist() == [[0, 1, 2]] * 3
    # Two text tokens, a 2 x 3 merged image (grid 1 x 4 x 6), then two text tokens.
    tokens = [5, 6, *[99] * 6, 7, 8]
    assert rope_positions(tokens, 99, [1, 4, 6]).tolist() == [
        [0, 1, 2, 2, 2, 2, 2, 2, 5, 6],   # temporal: one index for the still image
        [0, 1, 2, 2, 2, 3, 3, 3, 5, 6],   # height: image row
        [0, 1, 2, 3, 4, 2, 3, 4, 5, 6]]   # width: image column; text resumes past the largest index
    tall = rope_positions([5, *[99] * 6, 7], 99, [1, 6, 4])   # 3 rows x 2 columns
    assert tall[:, -1].tolist() == [4, 4, 4] and tall[1, 1:7].tolist() == [1, 1, 2, 2, 3, 3]
    for tokens, grid in (([5, 99, 99, 99, 7, 99, 99, 99], [1, 4, 6]),   # split image cells
                         ([5, *[99] * 6, 7], [2, 4, 6])):              # video-like temporal grid
        with pytest.raises(ValueError):
            rope_positions(tokens, 99, grid)


def test_kernel_difference_averages_eligible_position_distances():
    from docattentionlens.retrieval.colqwen import kernel_difference
    baseline = np.zeros((3, 4))
    eager = np.array([[.3, .4, 0, 0], [0, 0, 0, .1], [9, 9, 9, 9]])
    assert kernel_difference(eager, baseline, [True, True, False]) == pytest.approx((.3, .4))


def test_every_colqwen_checkpoint_declares_its_own_kernel_tolerances():
    from docattentionlens.retrieval import colqwen
    assert set(colqwen.EMBEDDING_KERNEL_TOLERANCES) == set(colqwen.SCORE_KERNEL_TOLERANCES) == set(colqwen.BACKBONES)
    # Calibrated noise maxima stay inside; the smallest masking errors the check must catch stay outside.
    assert .024 < colqwen.EMBEDDING_KERNEL_TOLERANCES[colqwen.REGISTRATION_ID] < .089
    assert .059 < colqwen.EMBEDDING_KERNEL_TOLERANCES[colqwen.COLQWEN25_REGISTRATION_ID] < .14
    assert .019 < colqwen.SCORE_KERNEL_TOLERANCES[colqwen.COLQWEN25_REGISTRATION_ID]


def test_float32_eager_attention_matches_reference_softmax_without_rounding():
    torch = pytest.importorskip("torch")
    from docattentionlens.retrieval.encoding_capture import eager_float32_attention
    generator = torch.Generator().manual_seed(0)
    query = (torch.randn(1, 4, 5, 8, generator=generator) * 40).bfloat16()   # large logits, as in Qwen2-VL
    key, value = (torch.randn(1, 2, 5, 8, generator=generator).bfloat16() for _ in range(2))
    causal = torch.full((5, 5), torch.finfo(torch.bfloat16).min).triu(1).bfloat16()[None, None]
    output, weights = eager_float32_attention(NS(num_key_value_groups=2), query, key, value, causal, scaling=8 ** -.5)
    k, v = key.float().repeat_interleave(2, 1), value.float().repeat_interleave(2, 1)
    expected = torch.softmax(query.float() @ k.transpose(2, 3) * 8 ** -.5 + causal.float(), -1)
    assert weights.dtype == torch.float32 and torch.equal(weights, expected)
    assert (weights[..., torch.ones(5, 5, dtype=torch.bool).triu(1)] == 0).all()
    assert torch.equal(output, (expected @ v).bfloat16().transpose(1, 2))
