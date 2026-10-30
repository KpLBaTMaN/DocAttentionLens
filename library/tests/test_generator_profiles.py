"""Generator adapter profiles: identities, geometry and architecture matching."""
import pytest
from docattentionlens.generation import contracts
from docattentionlens.generation.contracts import GENERATORS, QWEN25_VL_3B, QWEN2_VL_2B, QWEN3_VL_8B, compatible_profile


def pinned_config(profile):
    """The fields each pinned checkpoint's config.json declares, in that file's layout."""
    text = dict(profile.text)
    config = {"model_type": profile.model_type, "architectures": [profile.model_class], "vision_config": dict(profile.vision)}
    return ({**config, "text_config": text} if profile is QWEN3_VL_8B else {**config, **text}), \
        {"patch_size": profile.patch_size, "merge_size": profile.merge_size, "temporal_patch_size": profile.temporal_patch_size}


def test_the_default_generator_keeps_its_original_identity_and_geometry():
    # Saved captures and run specifications name these values; they must not move.
    assert GENERATORS["builtin"] is QWEN25_VL_3B
    assert (QWEN25_VL_3B.model_id, QWEN25_VL_3B.revision, QWEN25_VL_3B.adapter_version) == (contracts.MODEL_ID, contracts.MODEL_REVISION, contracts.ADAPTER_VERSION)
    assert QWEN25_VL_3B.question_rows_version == contracts.QUESTION_ROWS_VERSION
    assert QWEN25_VL_3B.max_pixels == contracts.MAX_PIXELS and QWEN25_VL_3B.required_free_gib == 12
    assert list(GENERATORS)[0] == "builtin"


def test_profiles_have_distinct_identities_and_their_published_architectures():
    profiles = list(GENERATORS.values())
    for field in ("registration_id", "adapter_version", "question_rows_version", "model_id"):
        assert len({getattr(p, field) for p in profiles}) == len(profiles), field
    assert (QWEN2_VL_2B.layers, QWEN2_VL_2B.query_heads, QWEN2_VL_2B.kv_heads, QWEN2_VL_2B.cell) == (28, 12, 2, 28)
    assert (QWEN3_VL_8B.layers, QWEN3_VL_8B.query_heads, QWEN3_VL_8B.kv_heads, QWEN3_VL_8B.cell) == (36, 32, 8, 32)
    assert QWEN3_VL_8B.deepstack_visual_indexes == (8, 16, 24) and not QWEN3_VL_8B.window_permutation
    assert QWEN25_VL_3B.window_permutation and not QWEN2_VL_2B.window_permutation
    for p in profiles:
        assert p.max_pixels == 512 * p.cell ** 2 and p.min_pixels == 4 * p.cell ** 2
        assert len(p.revision) == 40 and all(c in "0123456789abcdef" for c in p.revision)


@pytest.mark.parametrize("profile", list(GENERATORS.values()), ids=lambda p: p.registration_id)
def test_each_architecture_selects_only_its_own_adapter(profile):
    config, processor = pinned_config(profile)
    assert compatible_profile(config, processor) is profile
    assert [p for p in GENERATORS.values() if p.matches(config, processor)] == [profile]


@pytest.mark.parametrize("change", [
    lambda c, p: c.update(quantization_config={"bits": 4}),
    lambda c, p: c.update(auto_map={"AutoModel": "remote.py"}),
    lambda c, p: c.get("text_config", c).update(use_sliding_window=True),
    lambda c, p: c.get("text_config", c).update(num_attention_heads=c.get("text_config", c)["num_attention_heads"] * 2),
    lambda c, p: c["vision_config"].update(patch_size=c["vision_config"]["patch_size"] + 2),
    lambda c, p: p.update(merge_size=4),
    lambda c, p: c.update(model_type="llava"),
], ids=["quantized", "remote-code", "sliding-window", "heads", "vision-patch", "processor-merge", "model-type"])
def test_any_architecture_difference_leaves_a_checkpoint_metadata_only(change):
    for profile in GENERATORS.values():
        config, processor = pinned_config(profile)
        change(config, processor)
        assert compatible_profile(config, processor) is None
