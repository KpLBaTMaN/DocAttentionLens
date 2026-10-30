from dataclasses import dataclass
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from docattentionlens.retrieval.contracts import ArrayRef

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
ADAPTER_VERSION = "qwen25vl_prediction_rows_v1"
MAX_PIXELS = 512 * 28 * 28
MAX_INPUT_POSITIONS = 1536
# Merged image positions per page: the standard plan, and a finer plan for a single supplied page
# (tables need it: at 512 most cells share every image position with their neighbours).
STANDARD_IMAGE_POSITIONS = 512
SINGLE_PAGE_IMAGE_POSITIONS = 1024
IMAGE_POSITION_LIMITS = (STANDARD_IMAGE_POSITIONS, SINGLE_PAGE_IMAGE_POSITIONS)
MAX_NEW_TOKENS = 64
# Prediction steps whose rows an ordinary capture keeps. Full-attention captures (schema 0.8.0)
# keep every generated step, up to MAX_NEW_TOKENS.
CAPTURE_STEPS = 8
# Question-token prefill rows retained per capture; the dense prefill pass computes them anyway.
QUESTION_ROWS_LIMIT = 64
QUESTION_ROWS_VERSION = "qwen25vl_question_rows_v1"
SYSTEM_PROMPT = "Answer the question using only the supplied document page images. If the images do not provide enough evidence, say so. Give a concise answer."


@dataclass(frozen=True, eq=False)
class GeneratorProfile:
    """One pinned generator checkpoint and the adapter that captures its language attention.

    Every profile shares the capture contract (prediction and question rows, causal
    support, page ownership); the fields below are what differs between checkpoints.
    """
    registration_id: str
    name: str
    model_id: str
    revision: str
    adapter_version: str
    question_rows_version: str
    model_type: str
    model_class: str
    text: dict
    vision: dict
    patch_size: int
    required_free_gib: int
    merge_size: int = 2
    temporal_patch_size: int = 2
    # Qwen2.5-VL permutes vision windows and inverts the permutation before the decoder.
    window_permutation: bool = False
    # Qwen3-VL adds these vision layers' features to image positions after decoder layers 0..n-1.
    deepstack_visual_indexes: tuple[int, ...] = ()

    @property
    def cell(self) -> int:
        """Source-image pixels covered by one merged image position (per side, after resizing)."""
        return self.patch_size * self.merge_size

    @property
    def max_pixels(self) -> int:
        return self.pixels_for(STANDARD_IMAGE_POSITIONS)  # at most 512 merged image positions per page

    def pixels_for(self, image_positions: int) -> int:
        """Resize ceiling for at most `image_positions` merged image positions per page."""
        if image_positions not in IMAGE_POSITION_LIMITS:
            raise ValueError("Unsupported image position plan")
        return image_positions * self.cell ** 2

    @property
    def min_pixels(self) -> int:
        return 4 * self.cell ** 2

    @property
    def layers(self) -> int:
        return self.text["num_hidden_layers"]

    @property
    def query_heads(self) -> int:
        return self.text["num_attention_heads"]

    @property
    def kv_heads(self) -> int:
        return self.text["num_key_value_heads"]

    def matches(self, config: dict, processor: dict) -> bool:
        """An uploaded checkpoint can use this adapter only with the identical architecture."""
        text = config.get("text_config", config)
        vision = config.get("vision_config", {})
        return (isinstance(text, dict) and isinstance(vision, dict) and isinstance(processor, dict)
            and config.get("model_type") == self.model_type and not config.get("quantization_config") and not config.get("auto_map")
            and not text.get("use_sliding_window") and all(text.get(k) == v for k, v in self.text.items())
            and all(vision.get(k) == v for k, v in self.vision.items())
            and all(processor.get(k) == v for k, v in {"patch_size": self.patch_size, "merge_size": self.merge_size,
                                                        "temporal_patch_size": self.temporal_patch_size}.items()))


# Architecture values come from each pinned checkpoint's config.json.
QWEN25_VL_3B = GeneratorProfile(registration_id="builtin", name="Qwen2.5-VL-3B-Instruct", model_id=MODEL_ID, revision=MODEL_REVISION,
    adapter_version=ADAPTER_VERSION, question_rows_version=QUESTION_ROWS_VERSION, model_type="qwen2_5_vl",
    model_class="Qwen2_5_VLForConditionalGeneration", required_free_gib=12, patch_size=14, window_permutation=True,
    text={"num_hidden_layers": 36, "num_attention_heads": 16, "num_key_value_heads": 2, "hidden_size": 2048, "intermediate_size": 11008, "vocab_size": 151936},
    vision={"depth": 32, "hidden_size": 1280, "num_heads": 16, "out_hidden_size": 2048, "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2})
QWEN2_VL_2B = GeneratorProfile(registration_id="builtin-qwen2-vl-2b", name="Qwen2-VL-2B-Instruct", model_id="Qwen/Qwen2-VL-2B-Instruct",
    revision="895c3a49bc3fa70a340399125c650a463535e71c", adapter_version="qwen2vl_prediction_rows_v1",
    question_rows_version="qwen2vl_question_rows_v1", model_type="qwen2_vl", model_class="Qwen2VLForConditionalGeneration",
    required_free_gib=8, patch_size=14,
    text={"num_hidden_layers": 28, "num_attention_heads": 12, "num_key_value_heads": 2, "hidden_size": 1536, "intermediate_size": 8960, "vocab_size": 151936},
    vision={"depth": 32, "embed_dim": 1280, "hidden_size": 1536, "num_heads": 16, "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2})
QWEN3_VL_8B = GeneratorProfile(registration_id="builtin-qwen3-vl-8b", name="Qwen3-VL-8B-Instruct", model_id="Qwen/Qwen3-VL-8B-Instruct",
    revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", adapter_version="qwen3vl_prediction_rows_v1",
    question_rows_version="qwen3vl_question_rows_v1", model_type="qwen3_vl", model_class="Qwen3VLForConditionalGeneration",
    required_free_gib=21, patch_size=16, deepstack_visual_indexes=(8, 16, 24),
    text={"num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096, "intermediate_size": 12288, "vocab_size": 151936},
    vision={"depth": 27, "hidden_size": 1152, "num_heads": 16, "out_hidden_size": 4096, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2})
GENERATORS = {p.registration_id: p for p in (QWEN25_VL_3B, QWEN2_VL_2B, QWEN3_VL_8B)}
ADAPTER_VERSIONS = [p.adapter_version for p in GENERATORS.values()]


def compatible_profile(config: dict, processor: dict) -> GeneratorProfile | None:
    return next((p for p in GENERATORS.values() if p.matches(config, processor)), None)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GenerationRequest(Contract):
    retrieval_capture_id: str | None = None
    case_id: str | None = None
    model_registration_id: str = "builtin"
    system_prompt: str = Field(default=SYSTEM_PROMPT, min_length=1, max_length=4000)
    page_ids: list[str] = Field(min_length=1, max_length=2)
    idempotency_key: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    # Opt-in: keep every layer's complete causal prefill attention (lossless BF16 lower triangle).
    capture_full_attention: bool = False
    # Merged image positions per page. 1,024 is offered for a single supplied page only.
    image_token_limit: Literal[512, 1024] = STANDARD_IMAGE_POSITIONS

    @model_validator(mode="after")
    def input_lineage(self):
        if bool(self.case_id) == bool(self.retrieval_capture_id):
            raise ValueError("Choose either a direct case or a retrieval capture")
        if self.image_token_limit != STANDARD_IMAGE_POSITIONS and len(self.page_ids) != 1:
            raise ValueError("The 1,024-position plan needs exactly one supplied page")
        return self


class ContextPage(Contract):
    page_id: str
    page_number: int
    image_sha256: str
    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    grid_thw: list[int]
    position_indices: list[int]
    pixel_array: str | None
    preview_url: str


class GenerationPosition(Contract):
    index: int
    token_id: int
    piece: str
    role: Literal["image", "document_text", "query", "prompt", "special", "generated"]
    page_id: str | None = None
    bbox: list[float] | None = None
    ownership: dict[str, float]


class PredictionStep(Contract):
    step: int
    pass_kind: Literal["prefill", "decode"]
    source_index: int
    consumed_token_id: int
    consumed_piece: str
    predicted_index: int
    predicted_token_id: int
    predicted_piece: str
    target_count: int
    array: str


class SourceRow(Contract):
    """A retained prefill row whose source is a question token.

    The row is the attention of that position when the model reads the next
    input token, so the predicted token is the actual next prompt token.
    """
    source_index: int
    token_id: int
    piece: str
    predicted_index: int
    predicted_token_id: int
    predicted_piece: str
    target_count: int
    array: str


class ContextManifest(Contract):
    construction_version: Literal["selected_page_images_v1", "selected_document_representation_v2"] = "selected_page_images_v1"
    representation: Literal["image", "ocr", "structured"] = "image"
    retrieved_page_ids: list[str]
    selected_page_ids: list[str]
    omitted_page_ids: list[str]
    query: str
    prompt: str
    system_prompt: str = SYSTEM_PROMPT
    input_source: Literal["retrieved_pages", "supplied_pages"] = "retrieved_pages"
    pages: list[ContextPage]
    input_token_ids: list[int]
    input_attention_mask: list[int]
    truncation: Literal["none; over-budget input rejected"] = "none; over-budget input rejected"
    evaluation_labels_consumed: Literal[False] = False
    ocr_consumed: bool = False
    # Most merged image positions per page in this capture; earlier captures used 512.
    image_token_limit: int = STANDARD_IMAGE_POSITIONS


class FullAttention(Contract):
    """Complete prefill attention kept by an opt-in capture (see generation/full_attention.py).

    Each layer array holds the raw BF16 bit patterns of the causal lower triangle, rows in order.
    Decode rows n..square_limit-1 come from the retained prediction-step arrays. From schema 0.8.0
    every generated step is retained, so the squares reach the last consumed (generated) position.
    """
    layout: Literal["bf16_lower_triangle_rows_v1"]
    position_count: int          # n, prefill positions (rows of the triangle)
    square_limit: int            # positions with complete rows: n + max(0, len(steps) - 1)
    layer_arrays: list[str]      # ["full_l0", ...], one per decoder layer, in order
    query_head_count: int
    bytes: int                   # total stored bytes of the layer arrays

    @model_validator(mode="after")
    def consistent(self):
        if (self.position_count < 1 or self.square_limit < self.position_count or self.query_head_count < 1 or self.bytes < 0
                or not self.layer_arrays or self.layer_arrays != [f"full_l{i}" for i in range(len(self.layer_arrays))]):
            raise ValueError("Full attention declaration is inconsistent")
        return self


FULL_ATTENTION_VERSIONS = ("0.7.0", "0.8.0")
# Full-attention captures from this version keep every generated prediction step.
ALL_STEP_VERSIONS = ("0.8.0",)


class GenerationCapture(Contract):
    schema_version: Literal["0.4.0", "0.5.0", "0.6.0", "0.7.0", "0.8.0"] = "0.8.0"
    record_kind: Literal["generation_capture"] = "generation_capture"
    capture_id: str
    case_id: str
    retrieval_capture_id: str | None = None
    created_at: str
    producer_version: str
    dependencies: list[str]
    context: ContextManifest
    positions: list[GenerationPosition]
    output_token_ids: list[int]
    answer: str
    stop_reason: Literal["eos", "token_limit"]
    steps: list[PredictionStep]
    source_rows: list[SourceRow] = Field(default_factory=list)
    layer_count: int
    query_head_count: int
    kv_head_count: int
    arrays: list[ArrayRef] = Field(default_factory=list)
    full_attention: FullAttention | None = None
    provenance: dict[str, Any]
    coverage: dict[str, Any]
    validation: dict[str, Any]
    content_sha256: str | None = None

    @model_validator(mode="after")
    def full_attention_axes(self):
        full = self.full_attention
        if full is not None and (self.schema_version not in FULL_ATTENTION_VERSIONS or len(full.layer_arrays) != self.layer_count
                or full.query_head_count != self.query_head_count or full.position_count != len(self.context.input_token_ids)
                or full.square_limit != full.position_count + max(0, len(self.steps) - 1)):
            raise ValueError("Full attention differs from the capture's layers, heads, positions or retained steps")
        return self

    @model_validator(mode="after")
    def retained_steps(self):
        """Step k is the row of position n+k-1 over n+k causal targets, for k = 0..len(steps)-1.

        Ordinary captures keep at most CAPTURE_STEPS steps. Only full-attention captures from schema
        0.8.0 keep more, and they keep every generated step, so their squares (prefill triangle plus
        decode rows) reach the last consumed position.
        """
        n, count = len(self.context.input_token_ids), len(self.steps)
        for k, s in enumerate(self.steps):
            if (s.step, s.pass_kind, s.source_index, s.predicted_index, s.target_count, s.array) != (
                    k, "prefill" if k == 0 else "decode", n + k - 1, n + k, n + k, f"step_{k}"):
                raise ValueError("Prediction steps must be contiguous from 0, each the row of position n+k-1 over n+k targets")
        all_steps = self.full_attention is not None and self.schema_version in ALL_STEP_VERSIONS
        if count > MAX_NEW_TOKENS or (count > CAPTURE_STEPS and not all_steps):
            raise ValueError(f"Only full-attention captures (generation schema {ALL_STEP_VERSIONS[0]}) retain more than "
                             f"{CAPTURE_STEPS} prediction steps")
        if self.schema_version in ALL_STEP_VERSIONS:
            if count > len(self.output_token_ids):
                raise ValueError("A capture cannot retain more prediction steps than it generated tokens")
            if all_steps and count != len(self.output_token_ids):
                raise ValueError("A full-attention capture keeps the row of every generated prediction step")
        return self


class Allocation(Contract):
    bucket: str
    page_id: str | None = None
    element_id: str | None = None
    mass: float


class AttentionRow(Contract):
    capture_id: str
    step: int
    layer: int
    head: int
    source_index: int | None
    predicted_index: int | None
    target_count: int
    values: list[float]
    quantity: Literal["post_softmax_attention_weight", "head_maximum_envelope", "head_minimum_envelope", "rollout_connectivity",
        "whole_query_image_allocation", "modality_block_attention", "partial_rollout_connectivity", "rollout_evolution_connectivity"] = "post_softmax_attention_weight"
    normalization: str = "original causal target domain; no renormalization"
    total_mass: float
    image_mass: float
    allocations: list[Allocation]
    allocation_error: float
    display_max: float
