from typing import Any, Literal
from pydantic import Field, model_validator
from docattentionlens.datasets.contracts import Record, CaseRecord, PageRecord, ElementRecord

VERSION = "intact_group_translation_v1"
Stage = Literal["retrieval_only", "generation_fixed_evidence", "end_to_end"]


class ExperimentDraftRequest(Record):
    case_id: str
    page_id: str
    element_ids: list[str] = Field(min_length=2, max_length=50)
    non_support_element_ids: list[str] = Field(min_length=1, max_length=50)
    candidate_page_ids: list[str] = Field(min_length=1, max_length=8)
    fixed_context_page_ids: list[str] = Field(min_length=1, max_length=2)
    stage: Stage
    name: str = Field(min_length=1, max_length=200)


class Review(Record):
    reviewer: str = Field(min_length=1, max_length=200)
    notes: str = Field(min_length=10, max_length=5000)
    relationship_confirmed: bool
    content_and_meaning_preserved: bool
    readable: bool
    answer_policy: Literal["unchanged_after_review"]
    controls_and_resize_limitations_accepted: bool

    @model_validator(mode="after")
    def completed(self):
        if not all((self.relationship_confirmed, self.content_and_meaning_preserved,
                    self.readable, self.controls_and_resize_limitations_accepted)):
            raise ValueError("All review requirements must be completed before execution")
        if not self.reviewer.strip() or len(self.notes.strip()) < 10:
            raise ValueError("Review identity and substantive notes are required")
        return self


class ArmInput(Record):
    name: str
    case: CaseRecord
    page: PageRecord
    elements: list[ElementRecord]
    asset_sha256: str
    pixel_sha256: str
    element_lineage: dict[str, str]
    translation: list[int]
    crop_box: list[int] | None = None


class ExperimentRecord(Record):
    schema_version: Literal["0.5.0"] = "0.5.0"
    record_kind: Literal["experiment_draft", "experiment_protocol", "experiment_result"]
    experiment_id: str
    source_case_id: str
    name: str
    created_at: str
    producer_version: str
    dependencies: list[str]
    provenance: dict[str, Any]
    content_sha256: str | None = None
    protocol: ExperimentDraftRequest
    arms: list[ArmInput]
    validation: dict[str, Any]
    review: Review | None = None
    outcomes: list[dict[str, Any]] = Field(default_factory=list)
    comparisons: list[dict[str, Any]] = Field(default_factory=list)


class ExperimentRunRequest(Record):
    idempotency_key: str = Field(min_length=8, max_length=128)


class FixedContextSelection(Record):
    """Candidate identity carrier, explicitly not a similarity capture or rescore."""
    capture_id: str
    case_id: str
    query: str
    pages: list[PageRecord]
