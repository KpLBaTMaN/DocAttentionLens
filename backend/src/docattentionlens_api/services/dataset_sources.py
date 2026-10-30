"""Dataset folders DocAttentionLens can register, each read by its own library adapter.

A source is a folder under the data root. Registration names one of these; the worker reads it with
the matching adapter. Nothing else under the data root is registered. The two example sources are
small exports committed with the code, so a fresh clone has something to register.
"""
from dataclasses import dataclass
from pathlib import Path

DOCLAYOUT = "doclayout_mm_rag_bench"
TABLE_GROUNDING = "TableGroundQA-Pilot-v1"
DOCLAYOUT_EXAMPLES = "doclayout_mm_rag_examples"
TABLE_EXAMPLES = "TableGroundQA-Examples-v1"
DOCLAYOUT_OPERATION = "doclayout_mm_rag_bench_v1"
TABLE_OPERATION = "table_ground_qa_v1"


@dataclass(frozen=True)
class DatasetSource:
    source: str
    operation_version: str
    structure_kind: str
    label: str
    description: str


SOURCES = {s.source: s for s in (
    DatasetSource(DOCLAYOUT, DOCLAYOUT_OPERATION, "document_layout", "Document layout benchmark",
                  "Multi-page documents with layout elements, reading order and designated support."),
    DatasetSource(TABLE_GROUNDING, TABLE_OPERATION, "table_grounding", "Table grounding pilot",
                  "Tables on full document pages with cells, rows, columns, header paths and answer cells."),
    DatasetSource(DOCLAYOUT_EXAMPLES, DOCLAYOUT_OPERATION, "document_layout", "Document layout example",
                  "One annual report from the document layout benchmark, with its layout elements and questions."),
    DatasetSource(TABLE_EXAMPLES, TABLE_OPERATION, "table_grounding", "Table grounding examples",
                  "Three tables of different sizes from the example annual report, with cells, headers and questions."),
)}
SOURCE_PATTERN = "^(" + "|".join(s.replace(".", r"\.").replace("-", r"\-") for s in SOURCES) + ")$"


def read_source(data_root: Path, source: str, operation_version: str):
    """Read a registered source with its adapter; the operation version must match the source."""
    known = SOURCES.get(source)
    if known is None or known.operation_version != operation_version:
        raise ValueError("Unsupported dataset operation")
    if known.structure_kind == "document_layout":
        from docattentionlens.datasets.adapter import read_dataset
        return read_dataset(data_root / source, name=source)
    from docattentionlens.datasets.table_grounding import read_table_dataset
    return read_table_dataset(data_root / source)


def source_root(data_root: Path, summary) -> Path:
    """Folder holding a registered snapshot's page images."""
    if summary.name in SOURCES:
        return data_root / summary.name
    if summary.adapter_version == TABLE_OPERATION:
        raise ValueError("Registered table export is not a known source")
    return data_root / DOCLAYOUT
