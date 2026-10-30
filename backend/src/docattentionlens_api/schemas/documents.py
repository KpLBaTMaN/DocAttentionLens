from pydantic import Field

from typing import Literal

from docattentionlens_api.services.dataset_sources import DOCLAYOUT, SOURCE_PATTERN
from docattentionlens.datasets.contracts import (
    Record, DatasetSummary, DocumentRecord, QuestionRecord, ElementRecord, InvestigationRecord, TableRecord, TableSummary,
)


class RegistrationRequest(Record):
    source: str = Field(default=DOCLAYOUT, pattern=SOURCE_PATTERN)
    idempotency_key: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")


class DatasetSourceRecord(Record):
    """A dataset folder DocAttentionLens can register, and whether it is present under the data root."""
    source: str
    label: str
    description: str
    structure_kind: Literal["document_layout", "table_grounding"]
    present: bool


class DatasetSourcePage(Record):
    items: list[DatasetSourceRecord]


class TablePage(Record):
    items: list[TableRecord]
    total: int
    coordinate_convention: str = "Page image pixels with top-left origin."


class TableSummaryPage(Record):
    items: list[TableSummary]


class DatasetPage(Record):
    items: list[DatasetSummary]


class DocumentPage(Record):
    items: list[DocumentRecord]
    next_cursor: str | None
    total: int


class QuestionPage(Record):
    items: list[QuestionRecord]
    next_cursor: str | None
    total: int


class ElementPage(Record):
    items: list[ElementRecord]
    total: int
    coordinate_convention: str = "Source image pixels with top-left origin."


class InvestigationPage(Record):
    items: list[InvestigationRecord]
    next_cursor: str | None
