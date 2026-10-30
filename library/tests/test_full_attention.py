import numpy as np
import pytest

from docattentionlens.generation import full_attention as fa


def causal_square(heads, n, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(heads, n, n)).astype(np.float32)
    logits[:, ~np.tri(n, dtype=bool)] = -np.inf
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    # Truncate to BF16-exact values, as the eager decoder returns them.
    return (np.asarray(weights, np.float32).view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


def test_round_trip_is_exact_and_half_size():
    sq = causal_square(3, 7)
    packed = fa.pack(sq)
    assert packed.dtype == np.uint16 and packed.shape == (3, fa.triangle_length(7))
    assert np.array_equal(fa.unpack(packed, 7), sq)


def test_prefix_square_matches_leading_block():
    sq = causal_square(2, 9, seed=1)
    packed = fa.pack(sq)
    assert np.array_equal(fa.unpack(packed, 9, 4), sq[:, :4, :4])
    assert np.array_equal(fa.square(packed, 9, source=3), sq[:, :4, :4])


def test_decode_rows_extend_the_square():
    n, heads = 6, 2
    full = causal_square(heads, n + 2, seed=2)
    packed = fa.pack(full[:, :n, :n])
    decode = [full[:, n, :n + 1], full[:, n + 1, :n + 2]]
    assert np.array_equal(fa.square(packed, n, source=n + 1, decode_rows=decode), full)
    with pytest.raises(ValueError):
        fa.square(packed, n, source=n + 2, decode_rows=decode)


def test_refuses_lossy_values_and_future_keys():
    with pytest.raises(ValueError):
        fa.bf16_bits(np.array([0.1], np.float32))
    sq = causal_square(1, 4)
    sq[0, 0, 3] = 0.5
    with pytest.raises(ValueError):
        fa.pack(sq)


def test_unpack_checks_axes():
    packed = fa.pack(causal_square(1, 5))
    with pytest.raises(ValueError):
        fa.unpack(packed, 6)
    with pytest.raises(ValueError):
        fa.unpack(packed.astype(np.float32), 5)


def test_prefill_row_is_exact_and_names_round_trip():
    sq = causal_square(3, 8, seed=4)
    packed = fa.pack(sq)
    for s in range(8):
        assert np.array_equal(fa.row(packed, 8, s), sq[:, s, :s + 1])
    with pytest.raises(ValueError):
        fa.row(packed, 8, 8)
    assert fa.prefill_row_index(fa.prefill_row_name(17)) == 17
    assert fa.prefill_row_index("question_17") is None and fa.prefill_row_index("prefill_row_x") is None
