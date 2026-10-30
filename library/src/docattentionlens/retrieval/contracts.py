from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

MODEL_ID = "vidore/colpali-v1.3-hf"
MODEL_REVISION = "133a9eb02947310513f52f8f0d39d622e0eab8dc"
ADAPTER_VERSION = "colpali_transformers_v1"
ATTENTION_ADAPTER_VERSION = "colpali_language_attention_v1"

# Second built-in retriever: Qwen2-VL-2B late interaction with a causal language model.
COLQWEN_MODEL_ID = "vidore/colqwen2-v1.0-hf"
COLQWEN_MODEL_REVISION = "ddc07d2317c80f75fc742b7362ee9ad1912908f9"
COLQWEN_ADAPTER_VERSION = "colqwen2_transformers_v1"
# v2: FP32 attention arithmetic in the eager extraction pass (v1 extracted with BF16 eager).
COLQWEN_ATTENTION_ADAPTER_VERSION = "colqwen2_language_attention_v2"

# Third built-in retriever: ColQwen2.5 v0.2, a LoRA adapter (language layers and projection) over its own
# Qwen2.5-VL-3B base checkpoint. Both repositories are pinned; the ColQwen adapter merges them at load.
COLQWEN25_MODEL_ID = "vidore/colqwen2.5-v0.2"
COLQWEN25_MODEL_REVISION = "dcbe8d9cede518bce830488364ba0e40c873645b"
COLQWEN25_BASE_MODEL_ID = "vidore/colqwen2.5-base"
COLQWEN25_BASE_MODEL_REVISION = "92908120384b7a2110c5beda3ab29cbdb2c08e49"
COLQWEN25_ADAPTER_VERSION = "colqwen2_5_transformers_v1"
COLQWEN25_ATTENTION_ADAPTER_VERSION = "colqwen2_5_language_attention_v1"

# Built-in retrievers by registration. `module` names the adapter under retrieval/;
# attention budgets bound one query plus one document encoding at FP32.
RETRIEVER_PROFILES = {
    "builtin-retrieval": {"name": "ColPali v1.3", "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "adapter_version": ADAPTER_VERSION, "attention_adapter_version": ATTENTION_ADAPTER_VERSION, "module": "colpali",
        "layers": 18, "query_heads": 8, "image_positions_per_page": 1024, "causal": False,
        "attention_bytes": 735_000_000, "attention_free_bytes": 800*1024**2, "required_free_mib": 8192},
    "builtin-colqwen2": {"name": "ColQwen2 v1.0", "model_id": COLQWEN_MODEL_ID, "revision": COLQWEN_MODEL_REVISION,
        "adapter_version": COLQWEN_ADAPTER_VERSION, "attention_adapter_version": COLQWEN_ATTENTION_ADAPTER_VERSION, "module": "colqwen",
        "layers": 28, "query_heads": 12, "image_positions_per_page": 768, "causal": True,
        "attention_bytes": 950_000_000, "attention_free_bytes": 1100*1024**2, "required_free_mib": 8192},
    # 36 layers x 16 heads hold 1.7x ColQwen2's attention; BF16 weights take 7.5 GB.
    "builtin-colqwen2.5": {"name": "ColQwen2.5 v0.2", "model_id": COLQWEN25_MODEL_ID, "revision": COLQWEN25_MODEL_REVISION,
        "base_model_id": COLQWEN25_BASE_MODEL_ID, "base_revision": COLQWEN25_BASE_MODEL_REVISION,
        "adapter_version": COLQWEN25_ADAPTER_VERSION, "attention_adapter_version": COLQWEN25_ATTENTION_ADAPTER_VERSION, "module": "colqwen25",
        "layers": 36, "query_heads": 16, "image_positions_per_page": 768, "causal": True,
        "attention_bytes": 1_630_000_000, "attention_free_bytes": 1900*1024**2, "required_free_mib": 12288},
}


QUERY_POSITION_LIMIT = 256
PAGE_POSITION_LIMIT = 1100


def attention_capture_budget(profile, capture_joint_attention=False, joint_answer=False):
    """Worst-case retained FP32 matrices; each forward releases GPU matrices before the next.

    Each optional joint pass (question; question with the reference answer) adds a full
    page-sized matrix at every language layer. Artifact headroom also covers vectors,
    manifests and atomic publication.
    """
    page_passes = 1 + bool(capture_joint_attention) + bool(capture_joint_attention and joint_answer)
    retained = profile["layers"] * profile["query_heads"] * 4 * (
        QUERY_POSITION_LIMIT**2 + page_passes * PAGE_POSITION_LIMIT**2)
    required_disk = max(profile["attention_free_bytes"], retained + 128 * 1024**2)
    return retained, required_disk


def retriever_for_adapter(adapter_version):
    return next((dict(p, registration_id=k) for k, p in RETRIEVER_PROFILES.items() if p["adapter_version"] == adapter_version), None)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RetrievalRequest(Contract):
    case_id: str
    model_registration_id: str = "builtin-retrieval"
    page_ids: list[str] = Field(min_length=1, max_length=8)
    idempotency_key: str = Field(min_length=8, max_length=128)
    capture_attention: bool = False
    attention_page_id: str | None = None
    capture_joint_attention: bool = False
    # An additional diagnostic joint pass whose prompt also carries the case's reference answer.
    joint_answer_source: Literal["reference"] | None = None


class Position(Contract):
    index: int
    token_id: int
    piece: str
    role: str
    eligible: bool
    bbox: list[float] | None = None
    offsets: list[int] | None = None


class ArrayRef(Contract):
    name: str
    sha256: str
    shape: list[int]
    axes: list[str]
    # uint16 only for raw BF16 bit-pattern arrays (generator full attention); everything else is float32.
    dtype: Literal["float32", "uint16"] = "float32"
    bytes: int


class PageEvidence(Contract):
    page_id: str
    image_sha256: str
    width: int
    height: int
    page_number: int
    score: float
    reference_score: float
    reconstruction_error: float
    positions: list[Position]
    vector_array: str
    ownership: list[dict[str, float]]
    contributions: dict[str, float]
    winner_indices: list[int]
    query_maxima: list[float]


class EncodingAttention(Contract):
    encoding_id: str
    pass_kind: Literal["query", "document", "joint"]
    component: Literal["retriever_language"] = "retriever_language"
    page_id: str | None = None
    layer_count: int
    query_head_count: int
    position_count: int
    attention_mask: list[int]
    layer_arrays: list[str]
    normalization: str = "full encoding prefix; active keys only; no causal prediction mask"
    # Causal encoders (ColQwen2) let each source see only itself and earlier active keys.
    causal: bool = False
    positions: list[Position] = Field(default_factory=list)
    ownership: list[dict[str, float]] = Field(default_factory=list)
    # Joint passes only: the exact text that replaced the page prompt's caption, and, when an
    # answer was appended, where it came from. A reference answer is a case annotation.
    prompt_text: str | None = None
    answer_source: Literal["reference"] | None = None
    answer_text: str | None = None


class RetrievalAttentionRequest(Contract):
    encoding_id: str
    source_index: int = Field(default=0, ge=0, strict=True)
    method_id: str = "attention.row"
    parameters: dict[str, Any] = Field(default_factory=dict)


class RetrievalReceivedRequest(Contract):
    """Read one target's column from a saved encoding at one layer and query head."""
    encoding_id: str
    target_index: int = Field(ge=0, strict=True)
    layer: int = Field(default=0, ge=0, strict=True)
    head: int = Field(default=0, ge=0, strict=True)


class SimilarityColumn(Contract):
    """One page vector's dot product with every query vector under the recorded scorer."""
    capture_id: str
    page_id: str
    position_index: int
    similarities: list[float | None]
    winning_query_indices: list[int]
    maximum: float
    units: str = "dot product of captured vectors"
    normalization: str = "none after encoder L2 normalization; signed, not probability"


class RetrievalCapture(Contract):
    schema_version: Literal["0.3.0", "0.9.0", "0.10.0", "0.11.0", "0.12.0"] = "0.3.0"
    record_kind: Literal["retrieval_capture"] = "retrieval_capture"
    capture_id: str
    case_id: str
    created_at: str
    producer_version: str
    dependencies: list[str]
    dataset_version: str
    annotation_version: str
    query: str
    query_positions: list[Position]
    query_array: str
    pages: list[PageEvidence]
    arrays: list[ArrayRef] = []
    provenance: dict[str, Any]
    scorer: dict[str, Any]
    geometry: dict[str, Any]
    validation: dict[str, Any]
    content_sha256: str | None = None
    attention_encodings: list[EncodingAttention] = Field(default_factory=list)


class SimilarityRow(Contract):
    capture_id: str
    page_id: str
    query_index: int
    similarities: list[float | None]
    winner_index: int
    maximum: float
    units: str = "dot product of captured vectors"
    normalization: str = "none after encoder L2 normalization; signed, not probability"


class RemovalRequest(Contract):
    page_id: str
    removed_position_indices: list[int] = Field(min_length=1, max_length=1029)


class RemovalResult(Contract):
    operation: Literal["fixed_vector_removal"] = "fixed_vector_removal"
    capture_id: str
    page_id: str
    removed_position_indices: list[int]
    baseline_score: float
    score: float
    score_delta: float
    winner_indices: list[int]
    interpretation: str = "Rescoring frozen vectors; inputs were not re-encoded. This is not a causal intervention."
