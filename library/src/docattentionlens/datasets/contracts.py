from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_serializer

NativeID = StrictInt | StrictStr


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdditiveRecord(Record):
    """A record that gained optional fields after records were stored and hashed.

    Fields named in `_additive` are left out of every dump while they hold their default, so
    existing records (and snapshots, cases and API responses built from them) keep their bytes.
    """
    _additive: ClassVar[tuple[str, ...]] = ()

    @model_serializer(mode="wrap")
    def _omit_additive_defaults(self, handler):
        data = handler(self)
        for name in self._additive:
            if name in data and getattr(self, name) == type(self).model_fields[name].get_default(call_default_factory=True):
                data.pop(name)
        return data


class SourceRef(AdditiveRecord):
    _additive: ClassVar[tuple[str, ...]] = ("table_id",)
    dataset_id: str
    dataset_version: str
    domain: str
    document_id: NativeID
    page_number: int | None = None
    element_id: NativeID | None = None
    question_id: NativeID | None = None
    table_id: NativeID | None = None


class Box(Record):
    x1: int | float
    y1: int | float
    x2: int | float
    y2: int | float


class CoordinateSpace(Record):
    units: Literal["source_image_pixels"] = "source_image_pixels"
    origin: Literal["top_left"] = "top_left"
    bbox_format: Literal["x1_y1_x2_y2"] = "x1_y1_x2_y2"
    x_direction: Literal["right"] = "right"
    y_direction: Literal["down"] = "down"


class OCRValue(Record):
    status: Literal["present", "empty", "missing"]
    text: str | None


class PageRecord(Record):
    key: str
    page_id: str
    document_id: str
    ref: SourceRef
    page_number: int
    width: int
    height: int
    image_sha256: str
    image_url: str
    coordinate_space: CoordinateSpace = Field(default_factory=CoordinateSpace)
    orientation: Literal["portrait", "landscape", "square"]
    spread_status: Literal["not_annotated"] = "not_annotated"


class ElementRecord(Record):
    key: str
    element_id: str
    page_id: str
    document_id: str
    ref: SourceRef
    native_id: NativeID
    page_number: int
    element_type: str
    element_category: str
    bbox: Box
    level: int | None
    reading_order: int | None
    ocr: OCRValue
    geometry_flags: list[Literal["out_of_bounds", "degenerate"]]
    display_bbox: Box | None
    display_geometry_policy: Literal["raw", "clipped_to_page", "unavailable"]


class TableRole(Record):
    """A structural role and where it came from: `source_supplied` (PubTables), `proposed` or `reviewed`."""
    role: str
    status: str
    method: str


class TableAxisRecord(Record):
    """A table row or column band. Boxes are in full-page image pixels unless named crop."""
    native_id: str
    index: int
    bbox: Box
    crop_bbox: Box


class TableCellRecord(Record):
    """One logical cell. `bbox` is the projected logical box that partitions the table; `pdf_bboxes`
    are the native PubTables PDF cell boxes, kept for display and provenance only."""
    native_id: str
    row_ids: list[str]
    column_ids: list[str]
    row_indices: list[int]
    column_indices: list[int]
    bbox: Box
    crop_bbox: Box
    pdf_bboxes: list[Box]
    # Transcribed text from the table annotation (often absent), and the PDF's own text layer.
    text: str | None
    text_method: str
    text_status: str
    pdf_text: str | None
    roles: list[TableRole]
    native_flags: list[Literal["column_header", "projected_row_header"]]


class TableQualityIssue(Record):
    code: str
    severity: str
    detail: str
    cell_ids: list[str] = Field(default_factory=list)


class TableRecord(Record):
    """A table located on a registered page. Geometry is in that page's image pixels."""
    key: str
    table_id: str
    native_id: str
    document_id: str
    page_id: str
    ref: SourceRef
    title: str
    revision: str
    table_sha256: str
    grounding_revision: str
    grounding_sha256: str
    geometry_frame: Literal["page_image_pixels"] = "page_image_pixels"
    bbox: Box
    # The table with its caption and notes (PubTables pdf_table_wrap_bbox).
    wrap_bbox: Box
    crop_width: int
    crop_height: int
    crop_sha256: str
    crop_to_page: list[list[float]]
    registration_correlation: float
    geometry_verification: str
    row_count: int
    column_count: int
    rows: list[TableAxisRecord]
    columns: list[TableAxisRecord]
    cells: list[TableCellRecord]
    quality_status: str
    publishable: bool
    quality_issues: list[TableQualityIssue]
    source: dict[str, Any]


class TableSummary(Record):
    """A table without its cells, for listing."""
    table_id: str
    native_id: str
    document_id: str
    page_id: str
    title: str
    source_page_number: int
    source_page_count: int
    row_count: int
    column_count: int
    cell_count: int
    question_count: int
    questions_needing_review: int
    quality_status: str


class TableGrounding(Record):
    """A table question's answer location and lookup path. Evaluation labels: never model input."""
    table_id: str
    native_table_id: str
    table_revision: str
    table_sha256: str
    answer_cell_ids: list[str]
    answer_row_ids: list[str]
    answer_column_ids: list[str]
    row_header_path: list[str]
    column_header_path: list[str]
    support_cell_ids: list[str]
    accepted_answers: list[str]
    value_type: str
    unit: str | None
    rationale: str
    # needs_source_qa_review: the saved answer differs from the text in its bound PDF cell.
    binding_status: Literal["text_matched_not_human_approved", "needs_source_qa_review"]
    pdf_answer_text: str | None
    authored_input: str
    review_status: str
    support_role: Literal["evaluation_annotation"] = "evaluation_annotation"


class QuestionRecord(AdditiveRecord):
    _additive: ClassVar[tuple[str, ...]] = ("table_grounding",)
    key: str
    question_id: str
    document_id: str
    ref: SourceRef
    native_id: NativeID
    question: str
    document_scoped_question: str
    collection_scoped_question: str
    answer: str
    question_type: str
    native_element_ids: list[NativeID]
    support_element_ids: list[str]
    support_page_ids: list[str]
    support_element_refs: list[SourceRef]
    support_page_refs: list[SourceRef]
    support_role: Literal["evaluation_annotation"] = "evaluation_annotation"
    table_grounding: TableGrounding | None = None


class DocumentRecord(AdditiveRecord):
    _additive: ClassVar[tuple[str, ...]] = ("title", "table_ids")
    key: str
    document_id: str
    ref: SourceRef
    native_id: NativeID
    filename: str
    total_pages: int
    page_ids: list[str]
    question_count: int
    element_count: int
    annotation_version: str
    source_pdf_sha256: str
    ocr_model: str
    ocr_status: str
    title: str | None = None
    table_ids: list[str] = Field(default_factory=list)


class QualityIssue(Record):
    code: str
    element_id: str | None = None
    page_id: str | None = None
    document_id: str
    ref: SourceRef
    raw_bbox: Box | None = None
    detail: str


StructureKind = Literal["document_layout", "table_grounding"]


class DatasetSummary(AdditiveRecord):
    _additive: ClassVar[tuple[str, ...]] = ("structure_kind",)
    dataset_id: str
    name: str
    dataset_version: str
    adapter_version: str = "doclayout_mm_rag_bench_v1"
    counts: dict[str, int]
    domain_counts: dict[str, dict[str, int]]
    domain_aliases: dict[str, str]
    element_types: list[str]
    element_categories: list[str]
    question_types: list[str]
    quality_counts: dict[str, int]
    quality_issues: list[QualityIssue]
    source_manifest: dict[str, Any]
    source_file_count: int
    json_snapshot_sha256: str
    coordinate_convention: str = "Source image pixels, top-left origin; one-based page numbers."
    geometry_validation: str = "Source boxes preserved; display alignment requires representative visual validation."
    # What the annotations describe: layout elements, or table cells, rows and columns on a page.
    structure_kind: StructureKind = "document_layout"


class Snapshot(AdditiveRecord):
    _additive: ClassVar[tuple[str, ...]] = ("tables",)
    # 0.3.0 adds tables; layout snapshots stay 0.2.0 and byte-identical.
    schema_version: Literal["0.2.0", "0.3.0"] = "0.2.0"
    summary: DatasetSummary
    documents: list[DocumentRecord]
    pages: list[PageRecord]
    elements: list[ElementRecord]
    questions: list[QuestionRecord]
    file_sha256: dict[str, str]
    image_files: dict[str, str]
    source_records: dict[str, Any]
    tables: list[TableRecord] = Field(default_factory=list)


class CaseRequest(Record):
    question_id: str
    query_variant: Literal["original", "document_scoped", "collection_scoped"] = "collection_scoped"
    # Table questions have one wording and a table scope.
    task_scope: Literal["document", "collection", "table"] = "collection"
    page_ids: list[str] | None = None


class CaseRecord(AdditiveRecord):
    _additive: ClassVar[tuple[str, ...]] = ("table_grounding",)
    key: str
    schema_version: Literal["0.2.0"] = "0.2.0"
    case_id: str
    dataset_id: str
    dataset_version: str
    document_id: str
    question_id: str
    query_variant: str
    task_scope: str
    query: str
    page_ids: list[str]
    page_image_sha256: dict[str, str]
    annotation_version: str
    support_page_ids: list[str]
    support_element_ids: list[str]
    reference_answer: str
    label_exposure: Literal["evaluation_only"] = "evaluation_only"
    producer_version: str
    record_kind: Literal["case"] = "case"
    created_at: str | None = None
    content_sha256: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    # Table cases: the answer cell and lookup path, frozen with the case. Evaluation only.
    table_grounding: TableGrounding | None = None


class DisplaySettings(Record):
    schema_version: Literal["0.2.0"] = "0.2.0"
    zoom: float = Field(default=1, ge=0.25, le=8)
    center_x: float | None = Field(default=None, ge=0, le=100000)
    center_y: float | None = Field(default=None, ge=0, le=100000)
    show_boxes: bool = True
    show_support: bool = False
    show_reading_order: bool = True
    show_geometry_flags: bool = True
    input_view: Literal["original", "consumed", "split"] = "original"
    retrieval_scale_mode: Literal["auto", "fixed"] = "auto"
    retrieval_scale_max: float = Field(default=1, gt=0, le=100000)
    attention_scale_mode: Literal["auto", "causal", "fixed"] = "auto"
    attention_scale_max: float = Field(default=1, gt=0, le=100000)
    attention_opacity: float = Field(default=0.65, ge=0, le=1)
    # Display only: how page attention is drawn above the original image. Blend lays the
    # colour map over the whole page at the attention opacity.
    attention_display_mode: Literal["original", "heatmap", "overlay", "blend"] = "overlay"
    attention_show_patches: bool = False
    attention_smooth: bool = False
    # Display only: colour map for Heatmap, Blend and element fills, and where its scale starts
    # (zero, or stretched from the shown map's minimum). Stored values never change.
    attention_colormap: Literal["inferno", "magma", "viridis", "plasma"] = "inferno"
    attention_range_floor: Literal["zero", "minimum"] = "zero"
    # Document grouping is a display of the selected method's saved allocations.
    attention_grouping: Literal["positions", "elements"] = "positions"
    retrieval_grouping: Literal["positions", "elements"] = "positions"
    attention_element_view: Literal["elements", "patches", "selected_element"] = "elements"
    attention_grouped_comparison_index: int = Field(default=0, ge=0, le=127)
    comparison_columns: Literal[2, 4] = 2
    comparison_zoom: float = Field(default=1, ge=1, le=4)
    comparison_center_x: float | None = Field(default=None, ge=0, le=100000)
    comparison_center_y: float | None = Field(default=None, ge=0, le=100000)
    similarity_opacity: float = Field(default=0.65, ge=0, le=1)
    rail_width: int = Field(default=220, ge=160, le=360)
    inspector_width: int = Field(default=360, ge=240, le=480)
    dock_height: int = Field(default=180, ge=100, le=320)
    rail_collapsed: bool = False
    inspector_collapsed: bool = False
    rail_tab: Literal["pages", "elements"] = "pages"
    evidence_dock_open: bool = False
    canvas_fit: Literal["page", "width", "manual"] = "manual"


class ViewState(Record):
    retrieval_attention: bool = True
    retrieval_encoding_id: str = "document"
    retrieval_source_index: int = Field(default=0, ge=0, le=1099)
    retrieval_attention_method: str = "attention.row"
    retrieval_attention_parameters: dict[str, Any] = Field(default_factory=dict)
    # Layer/head ceilings match methods.contracts; the saved capture sets the real bounds.
    retrieval_comparison_index: int = Field(default=0, ge=0, le=127)
    investigation_model_role: Literal["generation", "retrieval"] = "generation"
    attention_lens: Literal["generation", "context"] = "generation"
    investigation_focus: Literal["generation", "context", "both"] = "both"
    page_id: str | None = None
    selected_element_id: str | None = None
    source_element_id: str | None = None
    target_element_id: str | None = None
    retrieval_capture_id: str | None = None
    query_position: int = Field(default=0, ge=0, le=255)
    similarity_overlay: bool = True
    generation_capture_id: str | None = None
    evidence_signal: Literal["retrieval", "attention", "document"] = "retrieval"
    # Prediction step: 0-7 on ordinary captures, every generated step (up to generation.MAX_NEW_TOKENS - 1)
    # on full-attention captures. Saved views are checked against the capture's retained steps.
    attention_step: int = Field(default=0, ge=0, le=63)
    attention_layer: int = Field(default=0, ge=0, le=127)
    attention_head: int = Field(default=0, ge=0, le=127)
    attention_overlay: bool = True
    attention_method_id: str = Field(default="attention.row", min_length=1, max_length=128)
    attention_variant_id: str = Field(default="captured_weights_v1", min_length=1, max_length=128)
    attention_parameters: dict[str, Any] = Field(default_factory=dict)
    attention_parameter_drafts: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=128)
    # Last method chosen in each analysis perspective, keyed "<role>:<perspective>" with a
    # "<method_id>/<variant_id>" value. Compatible General results may additionally be
    # grouped into elements using DisplaySettings, without changing the canonical method.
    perspective_methods: dict[Literal["generation:general", "generation:document_aware", "generation:table_aware", "retrieval:general",
                                      "retrieval:document_aware", "retrieval:table_aware"], str] = Field(default_factory=dict)
    attention_analysis_id: str | None = Field(default=None, pattern=r"^analysis_[a-f0-9]{64}$")
    # Token/patch exploration. A question-token source replaces the prediction step when set.
    attention_source_index: int | None = Field(default=None, ge=0, le=1535)
    # Row: what the source attends to. Column: what attends to the selected target.
    attention_direction: Literal["row", "column"] = "row"
    attention_target_index: int | None = Field(default=None, ge=0, le=1599)
    similarity_direction: Literal["row", "column"] = "row"
    similarity_target_index: int | None = Field(default=None, ge=0, le=1099)


class InvestigationCreate(Record):
    name: str = Field(min_length=1, max_length=200)
    case_id: str
    notes: str = Field(default="", max_length=20000)
    view_state: ViewState = Field(default_factory=ViewState)
    display_settings: DisplaySettings = Field(default_factory=DisplaySettings)


class InvestigationUpdate(InvestigationCreate):
    expected_revision: int = Field(ge=1)


class InvestigationRecord(InvestigationCreate):
    key: str
    revision_id: str
    schema_version: Literal["0.2.0", "0.3.0", "0.4.0"] = "0.4.0"
    investigation_id: str
    revision: int
    created_at: str
    content_sha256: str
    producer_version: str
    record_kind: Literal["investigation"] = "investigation"
    dependencies: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
