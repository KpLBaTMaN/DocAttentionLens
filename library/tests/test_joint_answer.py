"""Diagnostic joint passes with and without the reference answer: token roles, prompt text and refusals.

A word-level fake processor stands in for the pinned ColQwen2 processor, so the alignment logic runs
without a model or GPU; the encoding forward itself is replaced by a stub.
"""
import re
from types import SimpleNamespace as NS

import pytest

from docattentionlens.retrieval import joint_attention
from docattentionlens.retrieval.contracts import EncodingAttention, Position

IMAGE = "<img>"
TOKEN = re.compile(r"<img>|<\|[a-z_]+\|>|\w+|[^\w\s]|\n")


class Ids(list):
    def tolist(self):
        return list(self)


class Batch(dict):
    def to(self, _device):
        return self


class Tokenizer:
    def __init__(self):
        self.vocab = {}

    def _id(self, piece):
        return self.vocab.setdefault(piece, len(self.vocab) + 1)

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        found = list(TOKEN.finditer(text))
        return {"input_ids": [self._id(m.group()) for m in found], "offset_mapping": [(m.start(), m.end()) for m in found]}

    def convert_ids_to_tokens(self, token_id):
        return next(piece for piece, i in self.vocab.items() if i == token_id)


class Processor:
    image_token = IMAGE

    def __init__(self, patches):
        self.patches = patches
        self.tokenizer = Tokenizer()
        self.visual_prompt_prefix = f"<|im_start|>user {IMAGE} Describe the image.<|im_end|>"

    def __call__(self, images, return_tensors):
        expanded = self.visual_prompt_prefix.replace(IMAGE, IMAGE * self.patches)
        ids = self.tokenizer(expanded)["input_ids"]
        return Batch(input_ids=[Ids(ids)], attention_mask=[Ids([1] * len(ids))])


def run(monkeypatch, answer=None, encoding_id="joint"):
    patches = 3
    processor = Processor(patches)
    positions = [Position(index=i, token_id=1, piece=IMAGE, role="visual_patch", eligible=True, bbox=[i, 0, i + 1, 1]) for i in range(patches)]
    ownership = [{"element_a": 1.0} for _ in range(patches)]

    def encode(_torch, _model, batch, enc_id, pass_kind, _arrays, page_id, **_):
        n = len(batch["input_ids"][0])
        return None, EncodingAttention(encoding_id=enc_id, pass_kind=pass_kind, page_id=page_id, layer_count=1, query_head_count=1,
                                       position_count=n, attention_mask=[1] * n, layer_arrays=[f"attn_{enc_id}_l0"], causal=True), 0.0

    monkeypatch.setattr(joint_attention, "capture_encoding", encode)
    model = NS(config=NS(vlm_config=NS(image_token_id=processor.tokenizer._id(IMAGE))))
    question = "What was net income in 2011?"
    return joint_attention.capture_joint(None, model, processor, None, question, positions, ownership, {}, "page", True,
                                         answer=answer, encoding_id=encoding_id)[0]


def roles(encoding, role):
    return [p.piece for p in encoding.positions if p.role == role]


def test_question_only_pass_is_unchanged(monkeypatch):
    encoding = run(monkeypatch)
    assert encoding.encoding_id == "joint" and encoding.answer_source is None and encoding.answer_text is None
    assert roles(encoding, "query_text") == ["What", "was", "net", "income", "in", "2011", "?"]
    assert roles(encoding, "answer_text") == []
    assert encoding.prompt_text == "<|im_start|>user <img> What was net income in 2011?<|im_end|>"


def test_answer_pass_marks_the_reference_answer_and_records_its_source(monkeypatch):
    encoding = run(monkeypatch, answer=" $4,130,000 ", encoding_id="joint_answer")
    assert encoding.encoding_id == "joint_answer" and encoding.pass_kind == "joint"
    assert encoding.answer_source == "reference" and encoding.answer_text == "$4,130,000"
    assert encoding.prompt_text == "<|im_start|>user <img> What was net income in 2011?\nAnswer: $4,130,000<|im_end|>"
    # The question keeps its role; "Answer" and ":" are prompt text; only the answer span is answer_text.
    assert roles(encoding, "query_text") == ["What", "was", "net", "income", "in", "2011", "?"]
    assert roles(encoding, "answer_text") == ["$", "4", ",", "130", ",", "000"]
    assert "Answer" in roles(encoding, "visual_prompt")
    assert len(roles(encoding, "visual_patch")) == 3


def test_an_empty_reference_answer_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="no reference answer"):
        run(monkeypatch, answer="   ", encoding_id="joint_answer")
