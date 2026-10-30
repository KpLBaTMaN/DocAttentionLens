"""Prediction-step retention in the real `qwen.capture` hook, driven by a synthetic decoder (CPU, needs Torch).

The stand-in model returns dense causal BF16 attention taken from one known (layers, heads, m, m) square per
layer, so every retained row and the rebuilt full-attention squares can be compared exactly. A document-text
representation avoids the vision tower; the hooks, both generate passes and every check are the real code.
"""
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace as NS
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from docattentionlens.generation import full_attention as fa, qwen
from docattentionlens.generation.contracts import CAPTURE_STEPS, GenerationCapture
from PIL import Image

LAYERS, HEADS, KV_HEADS, VOCAB = 2, 2, 1, 32
QUERY = "Which value?"


def dense_attention(m, seed):
    """Normalized causal BF16 weights, as float32, for every position of prompt and answer."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=2, size=(HEADS, m, m))
    logits[:, ~np.tri(m, dtype=bool)] = -np.inf
    weights = np.exp(logits - logits.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return torch.tensor(weights, dtype=torch.float32).to(torch.bfloat16).float().numpy()


class Tokenizer:
    """One token per character; offsets are character spans."""
    all_special_ids = []

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return NS(input_ids=[ord(c) for c in text], offset_mapping=[(i, i + 1) for i in range(len(text))])

    def convert_ids_to_tokens(self, token):
        return chr(token)

    def decode(self, ids, **_):
        return "".join(chr(i) for i in ids)


class Batch(dict):
    __getattr__ = dict.__getitem__

    def to(self, _device):
        return self


class Processor:
    image_token = "<|image_pad|>"

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.image_processor = NS(size={"shortest_edge": 1, "longest_edge": 1}, min_pixels=1, max_pixels=1, image_mean=[.5] * 3,
                                  image_std=[.5] * 3, rescale_factor=1 / 255, resample=3)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        system, user = messages
        return "S:" + system["content"] + "|U:" + "".join(part["text"] for part in user["content"]) + "|A:"

    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor([[ord(c) for c in text[0]]])
        return Batch(input_ids=ids, attention_mask=torch.ones_like(ids))


class Visual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.merger = torch.nn.Identity()


class Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = Visual()
        self.language_model = NS(layers=[NS(self_attn=NS(num_key_value_groups=HEADS // KV_HEADS))])
        self.rope_deltas = None


class Decoder(torch.nn.Module):
    """Greedy decoding of a fixed answer; attention rows come from the known dense squares.

    `perturb` adds to the logits of one step in the plain (uninstrumented) pass only.
    """

    def __init__(self, answer, dense, perturb=None):
        super().__init__()
        self.model = Inner()
        self.answer, self.dense, self.perturb = answer, dense, perturb
        self.config = NS(image_token_id=1, text_config=NS(num_hidden_layers=LAYERS, num_attention_heads=HEADS, num_key_value_heads=KV_HEADS))
        self.generation_config = NS(eos_token_id=[3])
        self.logits = np.random.default_rng(1).normal(size=(len(answer), VOCAB)).astype(np.float32)

    def forward(self, input_ids, position_ids, cache_position, output_attentions, step):
        q = input_ids.shape[1]
        logits = torch.zeros((1, q, VOCAB))
        logits[0, -1] = torch.tensor(self.logits[step])
        if self.perturb and self.perturb[0] == step and not output_attentions:
            logits[0, -1, 0] += self.perturb[1]
        attentions = None
        if output_attentions:
            end = int(cache_position[-1]) + 1
            attentions = tuple(torch.tensor(self.dense[layer][:, end - q:end, :end]).to(torch.bfloat16)[None] for layer in range(LAYERS))
        return NS(logits=logits, attentions=attentions)

    def generate(self, input_ids, attention_mask, output_attentions, max_new_tokens, **_options):
        n = input_ids.shape[1]
        sequence = input_ids[0].tolist()
        for step in range(min(max_new_tokens, len(self.answer))):
            start = 0 if step == 0 else n + step - 1
            ids = torch.tensor([sequence[start:]])
            positions = torch.arange(start, n + step)[None, None].repeat(4, 1, 1)
            self(input_ids=ids, position_ids=positions, cache_position=torch.arange(start, n + step),
                 output_attentions=output_attentions, step=step)
            sequence.append(self.answer[step])
        self.model.rope_deltas = torch.zeros((1, 1), dtype=torch.long)
        return NS(sequences=torch.tensor([sequence]))


def run(monkeypatch, full_attention, answer_length=12, perturb=None):
    raw = BytesIO(); Image.new("RGB", (8, 8), "white").save(raw, format="PNG"); raw = raw.getvalue()
    digest = sha256(raw).hexdigest()
    page = NS(page_id="page_a", image_sha256=digest, width=8, height=8, page_number=1, image_url="/fixture.png")
    case = NS(case_id="case_fixture", page_ids=["page_a"], query=QUERY, page_image_sha256={"page_a": digest})
    processor = Processor()
    prompt = processor.apply_chat_template([{"content": "S"}, {"content": [{"text": "Document page 1:\n"}, {"text": QUERY}]}])
    n = len(prompt)
    answer = [ord("a") + k for k in range(answer_length - 1)] + [3]
    dense = [dense_attention(n + answer_length - 1, seed) for seed in range(LAYERS)]
    model = Decoder(answer, dense, perturb)
    monkeypatch.setattr(qwen, "runtime", lambda *_: (torch, processor, model))
    for name, value in (("reset_peak_memory_stats", lambda: None), ("max_memory_allocated", lambda: 0), ("get_device_name", lambda: "fixture")):
        monkeypatch.setattr(torch.cuda, name, value)
    record, arrays = qwen.capture(case, None, [(page, raw, [])], None, system_prompt="S", representation="ocr", full_attention=full_attention)
    return record, arrays, dense, n


def test_full_attention_keeps_every_generated_step_exactly(monkeypatch):
    record, arrays, dense, n = run(monkeypatch, True)
    assert record.schema_version == "0.8.0" and len(record.steps) == len(record.output_token_ids) == 12
    assert [s.array for s in record.steps] == [f"step_{k}" for k in range(12)]
    assert sorted(k for k in arrays if k.startswith("step_")) == sorted(f"step_{k}" for k in range(12))
    for k, step in enumerate(record.steps):
        assert (step.source_index, step.target_count, step.predicted_index) == (n + k - 1, n + k, n + k)
        assert np.array_equal(arrays[f"step_{k}"], np.stack([d[:, n + k - 1, :n + k] for d in dense]))
    assert record.full_attention.square_limit == n + 11
    # The triangle plus every decode row rebuild the dense square up to the last consumed position.
    for layer in range(LAYERS):
        rows = [arrays[f"step_{k}"][layer] for k in range(1, 12)]
        assert np.array_equal(fa.square(arrays[fa.array_name(layer)], n, n + 10, rows), dense[layer])
    assert record.coverage["retained_prediction_steps"] == record.coverage["captured_steps"] == 12
    assert record.coverage["full_attention"]["prediction_steps"] == "all"
    assert record.coverage["full_attention"]["prediction_row_bytes"] == sum(qwen.npy_bytes(arrays[f"step_{k}"]) for k in range(12))
    assert "every generated token's row" in record.coverage["source_rows"]
    assert "every generated step's logit vector" in record.validation["equivalence_scope"]
    assert "every generated step retained" in record.validation["full_attention_decode_rows"]
    assert GenerationCapture.model_validate_json(record.model_dump_json()) == record


def test_ordinary_captures_keep_the_first_eight_steps_unchanged(monkeypatch):
    record, arrays, dense, n = run(monkeypatch, False)
    assert record.full_attention is None and len(record.steps) == CAPTURE_STEPS and len(record.output_token_ids) == 12
    assert sorted(k for k in arrays if k.startswith("step_")) == sorted(f"step_{k}" for k in range(CAPTURE_STEPS))
    assert not any(k.startswith("full_l") for k in arrays) and "retained_prediction_steps" not in record.coverage
    assert record.coverage["source_rows"] == "last consumed position for first eight free-running prediction steps; every question-token prefill row"
    assert record.coverage["uncaptured"] == "prefill source rows other than question tokens; later output steps; vision attention; pre-softmax scores"
    assert record.validation["equivalence_scope"] == ("identical eager decoder kernels and declared representation/intervention; "
                                                      "first eight logit vectors and complete greedy output token sequence")
    assert "full_attention_decode_rows" not in record.validation


def test_logits_of_every_retained_step_are_compared(monkeypatch):
    # Step 10 is retained, and compared, only by a full-attention capture.
    with pytest.raises(ValueError, match="logits equivalence"):
        run(monkeypatch, True, perturb=(10, 1e-3))
    assert len(run(monkeypatch, False, perturb=(10, 1e-3))[0].steps) == CAPTURE_STEPS
    for full_attention in (True, False):
        with pytest.raises(ValueError, match="logits equivalence"):
            run(monkeypatch, full_attention, perturb=(3, 1e-3))


def test_short_answers_keep_all_their_steps_either_way(monkeypatch):
    full, _, _, n = run(monkeypatch, True, answer_length=5)
    ordinary = run(monkeypatch, False, answer_length=5)[0]
    assert len(full.steps) == len(ordinary.steps) == 5 and full.full_attention.square_limit == n + 4
