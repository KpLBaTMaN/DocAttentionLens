"""Opt-in full prefill attention for generator captures: contract, packing in the step-0 hook, and exact checks.

Torch is not installed for CPU tests, so the hook's tensors are stand-ins with the same interface
(`dtype`, `shape`, indexing, `cpu()`, `float()`, `numpy()`).
"""
from datetime import UTC, datetime
from io import BytesIO
import numpy as np
import pytest
from pydantic import ValidationError

from docattentionlens.generation import full_attention as fa
from docattentionlens.generation.contracts import (CAPTURE_STEPS, MAX_NEW_TOKENS, ContextManifest, FullAttention,
    GenerationCapture, GenerationRequest, PredictionStep)
from docattentionlens.generation.qwen import (check_retained_steps, full_attention_record, npy_bytes, pack_prefill,
    retained_step_limit, verify_decode_rows, verify_full_attention)

BF16 = "torch.bfloat16"


class Tensor:
    def __init__(self, values, dtype=BF16):
        self.values = np.asarray(values, dtype=np.float32)
        self.dtype = dtype

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, index):
        return Tensor(self.values[index], self.dtype)

    def cpu(self):
        return self

    def float(self):
        return Tensor(self.values, "torch.float32")

    def numpy(self):
        return self.values


def causal_square(heads, n, seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(heads, n, n)).astype(np.float32)
    logits[:, ~np.tri(n, dtype=bool)] = -np.inf
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return (np.asarray(weights, np.float32).view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)


LAYERS, HEADS, N, QUESTION = 3, 2, 7, [4, 5]


def hook_outputs():
    """Dense prefill matrices as the eager decoder returns them, and the rows the hook copies separately."""
    squares = [causal_square(HEADS, N, seed) for seed in range(LAYERS)]
    attentions = [Tensor(s[None]) for s in squares]
    arrays = {"step_0": np.stack([s[:, -1, :] for s in squares])}
    for i in QUESTION:
        arrays[f"question_{i}"] = np.ascontiguousarray(np.stack([s[:, i, :i+1] for s in squares]))
    return squares, attentions, arrays


def test_hook_packs_every_layer_losslessly_and_round_trip_matches_saved_rows():
    squares, attentions, arrays = hook_outputs()
    packed, mass = pack_prefill(attentions, BF16, HEADS, N)
    assert list(packed) == ["full_l0", "full_l1", "full_l2"] and mass < .02
    for layer, square in enumerate(squares):
        assert packed[f"full_l{layer}"].dtype == np.uint16 and packed[f"full_l{layer}"].shape == (HEADS, fa.triangle_length(N))
        assert np.array_equal(fa.unpack(packed[f"full_l{layer}"], N), square)
    arrays.update(packed)
    verify_full_attention(arrays, N, LAYERS, QUESTION)
    record = full_attention_record(arrays, N, LAYERS, HEADS, step_count=8)
    assert record.square_limit == N + 7 and record.layer_arrays == ["full_l0", "full_l1", "full_l2"] and record.query_head_count == HEADS
    saved = 0
    for name in record.layer_arrays:
        stream = BytesIO(); np.save(stream, arrays[name], allow_pickle=False); saved += len(stream.getvalue())
    assert record.bytes == saved == sum(npy_bytes(arrays[name]) for name in record.layer_arrays)


def test_hook_refuses_non_bf16_wrong_shape_lossy_future_or_invalid_weights():
    squares, attentions, _ = hook_outputs()
    with pytest.raises(ValueError, match="BF16"):
        pack_prefill([Tensor(squares[0][None], "torch.float16")], BF16, HEADS, N)
    with pytest.raises(ValueError, match="BF16"):
        pack_prefill([Tensor(squares[0][None, :, :-1, :-1])], BF16, HEADS, N)
    lossy = squares[0].copy(); lossy[0, 3, 1] += np.float32(1e-6)
    with pytest.raises(ValueError, match="not exact BF16"):
        pack_prefill([Tensor(lossy[None])], BF16, HEADS, N)
    future = squares[0].copy(); future[1, 2, 5] = 0.5
    with pytest.raises(ValueError, match="future keys"):
        pack_prefill([Tensor(future[None])], BF16, HEADS, N)
    invalid = squares[0].copy(); invalid[0, 6, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pack_prefill([Tensor(invalid[None])], BF16, HEADS, N)


def test_any_single_bit_difference_from_saved_rows_fails_the_capture():
    _, attentions, arrays = hook_outputs()
    packed, _ = pack_prefill(attentions, BF16, HEADS, N)
    for name, index in (("step_0", (2, 1, 3)), ("question_5", (0, 0, 2))):
        changed = {**arrays, **packed, name: arrays[name].copy()}
        changed[name][index] = np.nextafter(changed[name][index], np.float32(1))
        with pytest.raises(ValueError, match="differs from the retained prefill rows"):
            verify_full_attention(changed, N, LAYERS, QUESTION)
    missing = {**arrays, **packed}; del missing["full_l2"]
    with pytest.raises(ValueError, match="decoder layers"):
        verify_full_attention(missing, N, LAYERS, QUESTION)
    extra = {**arrays, **packed, "full_l3": packed["full_l0"]}
    with pytest.raises(ValueError, match="decoder layers"):
        verify_full_attention(extra, N, LAYERS, QUESTION)


def manifest(schema_version=None, full=None, steps=2, positions=N, layers=LAYERS, outputs=None, step_records=None):
    if step_records is None:
        step_records = [PredictionStep(step=k, pass_kind="prefill" if k == 0 else "decode", source_index=positions+k-1, consumed_token_id=0,
            consumed_piece="x", predicted_index=positions+k, predicted_token_id=0, predicted_piece="x", target_count=positions+k, array=f"step_{k}")
            for k in range(steps)]
    return dict(**({"schema_version": schema_version} if schema_version else {}), capture_id="generation_"+"c"*32, case_id="fixture",
        created_at=datetime.now(UTC).isoformat(), producer_version="test", dependencies=[],
        context=ContextManifest(retrieved_page_ids=[], selected_page_ids=[], omitted_page_ids=[], query="q", prompt="p", pages=[],
            input_token_ids=list(range(positions)), input_attention_mask=[1]*positions).model_dump(),
        positions=[], output_token_ids=[1] * max(1, len(step_records)) if outputs is None else outputs, answer="a", stop_reason="eos",
        steps=[s.model_dump() for s in step_records],
        layer_count=layers, query_head_count=HEADS, kv_head_count=1, provenance={}, coverage={}, validation={},
        **({"full_attention": full} if full is not None else {}))


def declaration(**changes):
    return {"layout": fa.LAYOUT, "position_count": N, "square_limit": N + 1, "layer_arrays": ["full_l0", "full_l1", "full_l2"],
            "query_head_count": HEADS, "bytes": 3 * 1000, **changes}


def test_older_manifests_stay_readable_and_new_captures_default_to_0_8_0():
    for version in ("0.4.0", "0.5.0", "0.6.0", "0.7.0"):
        record = GenerationCapture.model_validate(manifest(version))
        assert record.schema_version == version and record.full_attention is None
    assert GenerationCapture.model_validate(manifest()).schema_version == "0.8.0"
    assert "full_attention" not in GenerationCapture.model_validate(manifest("0.6.0")).model_dump(exclude_none=True)


def test_full_attention_declaration_round_trips_and_must_match_the_capture():
    record = GenerationCapture.model_validate(manifest("0.7.0", declaration()))
    assert isinstance(record.full_attention, FullAttention)
    assert GenerationCapture.model_validate_json(record.model_dump_json()) == record
    for invalid in (manifest("0.6.0", declaration()), manifest("0.7.0", declaration(square_limit=N)),
                    manifest("0.7.0", declaration(layer_arrays=["full_l0", "full_l1"])), manifest("0.7.0", declaration(query_head_count=3)),
                    manifest("0.7.0", declaration(position_count=N - 1, square_limit=N)), manifest("0.7.0", declaration(layer_arrays=["full_l0", "full_l2", "full_l1"])),
                    manifest("0.7.0", declaration(layout="bf16_dense"))):
        with pytest.raises(ValidationError):
            GenerationCapture.model_validate(invalid)
    # One retained step leaves no decode rows: the square stops at the prefill.
    assert GenerationCapture.model_validate(manifest("0.7.0", declaration(square_limit=N), steps=1)).full_attention.square_limit == N
    assert GenerationCapture.model_validate(manifest("0.8.0", declaration(square_limit=N), steps=1)).full_attention.square_limit == N


def test_full_attention_captures_from_0_8_0_keep_every_generated_step():
    # A 12-token answer: every step is retained and the squares reach the last consumed position n+10.
    record = GenerationCapture.model_validate(manifest(full=declaration(square_limit=N + 11), steps=12, outputs=list(range(12))))
    assert record.schema_version == "0.8.0" and [s.step for s in record.steps] == list(range(12))
    assert (record.steps[-1].source_index, record.steps[-1].target_count, record.full_attention.square_limit) == (N + 10, N + 11, N + 11)
    # Earlier 0.7.0 full-attention captures (8 steps of a longer answer) stay valid.
    legacy = GenerationCapture.model_validate(manifest("0.7.0", declaration(square_limit=N + 7), steps=8, outputs=list(range(36))))
    assert len(legacy.steps) == CAPTURE_STEPS and legacy.full_attention.square_limit == N + 7
    invalid = [
        manifest(steps=12, outputs=list(range(12))),                                             # more than 8 without full attention
        manifest("0.7.0", declaration(square_limit=N + 11), steps=12, outputs=list(range(12))),  # 0.7.0 keeps at most 8
        manifest(full=declaration(square_limit=N + 7), steps=8, outputs=list(range(12))),        # 0.8.0 full capture misses steps
        manifest(full=declaration(square_limit=N + 11), steps=12, outputs=list(range(13))),
        manifest(steps=3, outputs=[1, 2]),                                                       # more steps than generated tokens
        manifest(full=declaration(square_limit=N + MAX_NEW_TOKENS), steps=MAX_NEW_TOKENS + 1, outputs=list(range(MAX_NEW_TOKENS + 1)))]
    for values in invalid:
        with pytest.raises(ValidationError):
            GenerationCapture.model_validate(values)


def step_record(k, n=N, **changes):
    return PredictionStep(**{"step": k, "pass_kind": "prefill" if k == 0 else "decode", "source_index": n + k - 1, "consumed_token_id": 0,
        "consumed_piece": "x", "predicted_index": n + k, "predicted_token_id": 0, "predicted_piece": "x", "target_count": n + k,
        "array": f"step_{k}", **changes})


@pytest.mark.parametrize("version", ["0.4.0", "0.7.0", "0.8.0"])
def test_prediction_steps_must_be_contiguous_rows_of_consumed_positions(version):
    assert GenerationCapture.model_validate(manifest(version, step_records=[step_record(0), step_record(1), step_record(2)]))
    for records in ([step_record(0), step_record(2)], [step_record(1)], [step_record(0), step_record(1, source_index=N + 1)],
                    [step_record(0), step_record(1, target_count=N + 2)], [step_record(0), step_record(1, array="step_7")],
                    [step_record(0), step_record(1, predicted_index=N)], [step_record(0, pass_kind="decode")]):
        with pytest.raises(ValidationError, match="contiguous"):
            GenerationCapture.model_validate(manifest(version, step_records=records))


def test_both_passes_must_retain_the_same_steps_and_full_captures_compare_every_step():
    rng = np.random.default_rng(3)
    logits = [rng.normal(size=11).astype(np.float32) for _ in range(12)]
    rows = [object()] * 12
    assert retained_step_limit(False) == CAPTURE_STEPS and retained_step_limit(True) == MAX_NEW_TOKENS
    assert check_retained_steps(True, 12, logits, [x.copy() for x in logits], rows) == 0.0
    assert check_retained_steps(False, 12, logits[:8], [x.copy() for x in logits[:8]], rows[:8]) == 0.0
    # Ordinary captures keep eight steps of a 12-token answer; a full capture must keep all twelve in both passes.
    for full, plain, sampled, copied in ((True, logits[:8], logits[:8], rows[:8]), (False, logits, logits, rows),
                                         (True, logits[:11], logits, rows), (True, logits, logits, rows[:11])):
        with pytest.raises(ValueError, match="coverage equivalence"):
            check_retained_steps(full, 12, plain, sampled, copied)
    # A difference at step 10 is compared, and fails, only when every step is retained.
    changed = [x.copy() for x in logits]
    changed[10][4] += np.float32(1e-3)
    with pytest.raises(ValueError, match="logits equivalence"):
        check_retained_steps(True, 12, logits, changed, rows)
    assert check_retained_steps(False, 12, logits[:8], changed[:8], rows[:8]) == 0.0
    # The tolerance is unchanged.
    tiny = [x.copy() for x in logits]
    tiny[11][0] += np.float32(5e-6)
    assert 0 < check_retained_steps(True, 12, logits, tiny, rows) <= 1e-5


def decode_fixture(count=12, n=N):
    """Every prediction step of a `count`-token answer, from one dense causal BF16 square per layer."""
    dense = np.stack([causal_square(HEADS, n + count - 1, seed) for seed in range(LAYERS)])
    arrays = {f"step_{k}": np.ascontiguousarray(dense[:, :, n + k - 1, :n + k]) for k in range(count)}
    return dense, arrays, [step_record(k, n) for k in range(count)]


def test_every_decode_row_must_be_a_normalized_exact_bf16_causal_row():
    dense, arrays, steps = decode_fixture()
    verify_decode_rows(arrays, steps, N, LAYERS, HEADS, 12)
    # The rows complete every square up to the last consumed position, exactly.
    for layer in range(LAYERS):
        rows = [arrays[f"step_{k}"][layer] for k in range(1, 12)]
        assert np.array_equal(fa.square(fa.pack(dense[layer, :, :N, :N]), N, N + 10, rows), dense[layer])
    with pytest.raises(ValueError, match="missing prediction steps"):
        verify_decode_rows(arrays, steps[:11], N, LAYERS, HEADS, 12)
    with pytest.raises(ValueError, match="missing prediction steps"):
        verify_decode_rows(arrays, [*steps[:5], *steps[6:]], N, LAYERS, HEADS, 12)
    with pytest.raises(ValueError, match="axes"):
        verify_decode_rows({**arrays, "step_9": arrays["step_9"][:, :, :-1]}, steps, N, LAYERS, HEADS, 12)
    with pytest.raises(ValueError, match="axes"):
        verify_decode_rows({k: v for k, v in arrays.items() if k != "step_11"}, steps, N, LAYERS, HEADS, 12)
    with pytest.raises(ValueError, match="normalized"):
        verify_decode_rows({**arrays, "step_11": arrays["step_11"] * np.float32(1.5)}, steps, N, LAYERS, HEADS, 12)
    lossy = {**arrays, "step_4": arrays["step_4"].copy()}
    lossy["step_4"][0, 0, 0] += np.float32(1e-6)
    with pytest.raises(ValueError, match="not exact BF16"):
        verify_decode_rows(lossy, steps, N, LAYERS, HEADS, 12)


def test_request_defaults_to_no_full_attention():
    base = {"case_id": "case", "page_ids": ["p"], "idempotency_key": "k"}
    assert GenerationRequest.model_validate(base).capture_full_attention is False
    assert GenerationRequest.model_validate({**base, "capture_full_attention": True}).capture_full_attention is True
    with pytest.raises(ValidationError):
        GenerationRequest.model_validate({**base, "capture_full_attention": "maybe"})


def test_step_selections_reach_every_generated_step():
    from docattentionlens.datasets.contracts import ViewState
    from docattentionlens.methods.contracts import HeadParameters, RowParameters
    from docattentionlens.methods.integrated_gradients import IGParameters
    last = MAX_NEW_TOKENS - 1
    for model, field in ((ViewState, "attention_step"), (RowParameters, "step"), (HeadParameters, "step"), (IGParameters, "step")):
        assert getattr(model.model_validate({field: 11}), field) == 11 and getattr(model.model_validate({field: last}), field) == last
        with pytest.raises(ValidationError):
            model.model_validate({field: MAX_NEW_TOKENS})
    # Saved views from before keep loading unchanged.
    assert ViewState.model_validate({"attention_step": 7}).attention_step == 7
