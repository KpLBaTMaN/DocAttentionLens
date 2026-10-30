"""Shared method types. Scientific definitions live in individual method modules."""
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from ..generation.contracts import AttentionRow, MAX_NEW_TOKENS


class OutputContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, serialize_by_alias=True)
    result_schema: str = Field(alias="schema")
    units: str
    signed: bool
    support: str
    normalization: str


Perspective = Literal["general", "document_aware", "table_aware"]


class LiteratureRelation(BaseModel):
    """How the calculation relates to its cited sources. A published method matches only at the
    settings its note states; an adaptation's note names every departure from the cited work."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["published_method", "standard_summary", "adaptation"]
    note: str


class MethodDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    method_id: str
    variant_id: str
    descriptor_version: str
    status: Literal["planned", "implemented", "validated"]
    implementation_version: str | None
    # Classified by what the calculation needs, never by folder or display. General methods
    # use model positions only; document-aware methods aggregate or compare attention using
    # document structure (element boxes, designated support, reading order). Boxes drawn
    # only as an orientation overlay do not make a method document-aware.
    perspective: Perspective
    family: str
    name: str
    summary: str
    question: str
    evidence_kind: str
    quantity: str
    definition: str
    source_references: list[str]
    literature_relation: LiteratureRelation | None = None
    requires: list[str]
    execution_requirement: str
    target_requirement: str
    parameters_schema: dict[str, Any]
    output: OutputContract
    displays: list[str]
    comparison_constraints: list[str]
    limitations: list[str]
    validation_evidence: list[str]


# Schema ceilings only. Each request is validated against its capture's actual
# layer and query-head counts, which differ between model adapters.
MAX_LAYERS = 128
MAX_HEADS = 128
AxisIndex = Annotated[int, Field(strict=True, ge=0, lt=128)]


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    # The first eight steps on ordinary captures; every generated step on full-attention captures.
    # Each request is checked against the capture's retained steps.
    step: int = Field(default=0, ge=0, le=MAX_NEW_TOKENS - 1, strict=True)
    layer: int = Field(default=0, ge=0, lt=MAX_LAYERS, strict=True)


class RowParameters(Parameters):
    head: int = Field(default=0, ge=0, lt=MAX_HEADS, strict=True)


class HeadParameters(Parameters):
    heads: list[Annotated[int, Field(strict=True, ge=0, lt=MAX_HEADS)]] = Field(default_factory=list, max_length=MAX_HEADS,
                             description="Empty selects all captured query heads.")


class Statistic(BaseModel):
    name: str
    value: float | None
    units: str


class ComparisonRow(BaseModel):
    label: str
    row: AttentionRow


class ResultTable(BaseModel):
    columns: list[str]
    labels: list[str]
    values: list[list[float | None]]
    units: str
    title: str = ""
    element_ids: list[str | None] = Field(default_factory=list)
    page_ids: list[str | None] = Field(default_factory=list)
    # Table-aware results: the native cell ID each row describes, so it links to the heatmap and page.
    cell_ids: list[str | None] = Field(default_factory=list)


class ResearchPoint(BaseModel):
    label: str
    x: float
    y: float
    value: float | None = None
    group: str = ""
    element_id: str | None = None
    page_id: str | None = None
    capture_id: str | None = None


class ResearchPlot(BaseModel):
    kind: Literal["bar", "line", "scatter", "ribbon", "grid", "projection"]
    title: str
    x_label: str
    y_label: str
    points: list[ResearchPoint]


class ResearchEdge(BaseModel):
    source: str
    target: str
    value: float
    signal: Literal["retrieval", "attention", "section"]
    units: str


class AttentionGraphNode(BaseModel):
    node_id: str
    label: str
    level: int
    position: int
    role: str


class AttentionGraphEdge(BaseModel):
    source: str
    target: str
    value: float
    layer: int
    head: int | None = None


class AttentionGraph(BaseModel):
    title: str
    nodes: list[AttentionGraphNode]
    edges: list[AttentionGraphEdge]
    units: str
    total_edges: int


class TableCellMass(BaseModel):
    """One logical cell. `mass` is None when no valid target position overlaps the cell (unmeasured,
    not zero). `enrichment` compares the cell's share of image attention with its share of page area."""
    cell_id: str
    row_indices: list[int]
    column_indices: list[int]
    mass: float | None
    area_share: float
    enrichment: float | None
    isolable: bool
    touching_positions: int
    largest_position_share: float


class TableBandMass(BaseModel):
    """A table row or column band (native disjoint bands), never a row of the image patch grid."""
    band_id: str
    index: int
    mass: float | None
    area_share: float
    enrichment: float | None


class TableRegionMass(BaseModel):
    region: str
    label: str
    mass: float
    area_share: float | None = None
    enrichment: float | None = None


class TableResolution(BaseModel):
    """How finely the captured image positions resolve this table. A cell is isolable when some
    position lies at least half inside it; other cells share all their positions with neighbours."""
    image_positions: int
    positions_on_table: int
    cell_count: int
    isolable_cells: int
    rows_per_position: float
    columns_per_position: float


class TableAllocation(BaseModel):
    """Attention aggregated into one table's structure at analysis time (table_cell_partition_v1)."""
    table_id: str
    native_table_id: str
    policy: Literal["table_cell_partition_v1"] = "table_cell_partition_v1"
    selection: str
    denominator: str = "original captured domain; image and table shares are labelled conditional quantities"
    total_mass: float
    image_mass: float
    table_mass: float
    cells: list[TableCellMass]
    rows: list[TableBandMass]
    columns: list[TableBandMass]
    regions: list[TableRegionMass]
    resolution: TableResolution


class ProfileElement(BaseModel):
    """One mapped element's original-domain mass from the selected source, indexed [layer][head]."""
    element_id: str
    page_id: str
    label: str
    reading_order: int | None = None
    # Fraction of valid image positions (every supplied page) the element owns; None without image positions.
    area_share: float | None
    mass: list[list[float]]


class ElementProfile(BaseModel):
    """Element attention profile: every captured layer and query head for one source row.

    Mass keeps the original captured domain. Image mass is the denominator of the within-image
    share; the area-relative ratio divides that share by the element's area share.
    """
    policy: Literal["element_layer_head_mass_v1"] = "element_layer_head_mass_v1"
    layer_count: int
    head_count: int
    kv_head_count: int | None = None
    # Decoder layers whose image keys and values already include DeepStack vision features.
    deepstack_layers: list[int] = Field(default_factory=list)
    image_positions: int
    image_mass: list[list[float]]
    elements: list[ProfileElement]
    # Annotated elements on supplied pages with no mapped valid position: unmeasured, never zero.
    unmeasured_element_ids: list[str] = Field(default_factory=list)
    significant_digits: int = 7
    denominator: str = "original captured domain; within-image share divides by image mass; area ratio divides that share by area share"


class Calculation(BaseModel):
    row: AttentionRow | None = None
    statistics: list[Statistic] = Field(default_factory=list)
    comparisons: list[ComparisonRow] = Field(default_factory=list)
    table: ResultTable | None = None
    normalization: str = "original causal target domain; no renormalization"
    tables: list[ResultTable] = Field(default_factory=list)
    plots: list[ResearchPlot] = Field(default_factory=list)
    edges: list[ResearchEdge] = Field(default_factory=list)
    notices: list[str] = Field(default_factory=list)
    graph: AttentionGraph | None = None
    source_indices: list[int] = Field(default_factory=list)
    table_allocation: TableAllocation | None = None
    element_profile: ElementProfile | None = None


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    generation_capture_id: str = Field(pattern=r"^generation_[a-f0-9]{32}$")
    method_id: str = Field(min_length=1, max_length=128)
    variant_id: str = Field(min_length=1, max_length=128)
    parameters: dict[str, Any] = Field(default_factory=dict)
    # A retained question-token row instead of a prediction step; the step parameter is then unused.
    source_index: int | None = Field(default=None, ge=0, le=1535)
    comparison_capture_id: str | None = Field(default=None, pattern=r"^generation_[a-f0-9]{32}$")
    sample_capture_ids: list[str] = Field(default_factory=list, max_length=64)
    experiment_result_id: str | None = None


class AnalysisResult(Calculation):
    schema_version: Literal["0.8.0", "0.9.0", "0.10.0"] = "0.10.0"
    analysis_id: str
    content_sha256: str | None = None
    method_id: str
    variant_id: str
    implementation_version: str
    generation_capture_id: str
    capture_sha256: str
    array_sha256: str
    parameters: dict[str, Any]
    source_index: int
    predicted_index: int
    target_count: int
    valid_target_indices: list[int]
    quantity: str
    units: str
    coverage: dict[str, Any]
    input_manifest: dict[str, Any] = Field(default_factory=dict)


class ReceivedSource(BaseModel):
    source_index: int
    kind: Literal["prediction", "question", "encoding"]
    label: str
    value: float


class ReceivedAttention(BaseModel):
    """One target's weight in each captured source row: a column of the attention matrix."""
    capture_id: str
    target_index: int
    layer: int
    head: int
    sources: list[ReceivedSource]
    maximum: float
    quantity: Literal["attention_received"] = "attention_received"
    units: str = "post-softmax weight within each source row"
    normalization: str = "each value is the target's share of one normalized source row; values across sources do not sum to one"


class PreparedMethod(BaseModel):
    method: MethodDescriptor
    available: bool
    reasons: list[str]
    control_bounds: dict[str, Any] = Field(default_factory=dict)


class DocumentStructure(BaseModel):
    """Document annotations available to document-aware methods, with their origin.

    Counts cover the case's input pages. A zero count means the annotation is absent,
    which is different from an element that receives zero attention.
    """
    provenance: Literal["dataset", "imported", "inferred"]
    source: str
    adapter_version: str
    annotation_version: str
    element_count: int
    boxed_element_count: int
    flagged_element_count: int
    reading_order_count: int
    element_types: list[str]
    support_element_count: int


@dataclass(frozen=True)
class DocumentInputs:
    case: Any
    elements: tuple[Any, ...]
    pages: tuple[Any, ...]
    annotation_sha256: str
    provenance: str = "dataset"
    question_type: str = "unknown"
    support_categories: dict[str,str] | None = None


@dataclass(frozen=True)
class TableInputs:
    """A table case's verified structure, resolved from its frozen table revision and hash.
    `grounding` holds evaluation-only labels that never reach the model."""
    case: Any
    table: Any
    page: Any
    grounding: Any
    annotation_sha256: str
    provenance: str = "dataset"


class TableStructure(BaseModel):
    """Table annotations available to table-aware methods. Zero counts mean the annotation is absent."""
    provenance: Literal["dataset"] = "dataset"
    source: str
    adapter_version: str
    table_id: str
    title: str
    revision: str
    cell_count: int
    row_count: int
    column_count: int
    quality_status: str
    grounding: bool
    binding_status: str | None = None


@dataclass(frozen=True)
class Context:
    record: Any
    values: np.ndarray
    document: DocumentInputs | None = None
    retrieval: Any = None
    comparison: Any = None
    samples: tuple[Any, ...] = ()
    alignment: dict[str, str] | None = None
    experiment: Any = None
    read_array: Callable | None = None
    read_source: Callable | None = None
    source_record: Any = None
    table: TableInputs | None = None


@dataclass(frozen=True)
class MethodDefinition:
    descriptor: MethodDescriptor
    parameters: type[BaseModel] | None = None
    calculate: Callable[[Context, Any], Calculation] | None = None
    executable: bool = False
    prepare: Callable[[Any, Any], Any] | None = None
