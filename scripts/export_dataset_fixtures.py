"""Generate versioned transport fixtures from the real read-only source."""
import json
from pathlib import Path

from docattentionlens.datasets.adapter import read_dataset


ROOT = Path(__file__).resolve().parents[1]
snapshot = read_dataset(ROOT / "data/doclayout_mm_rag_bench")
destination = ROOT / "tests/fixtures/datasets"
destination.mkdir(parents=True, exist_ok=True)


def write(name, value):
    (destination / f"{name}.json").write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


write("dataset-summary", snapshot.summary.model_dump(mode="json"))
page_map = {page.page_id: page for page in snapshot.pages}
document_map = {doc.document_id: doc for doc in snapshot.documents}
examples = {
    "portrait": next(page for page in snapshot.pages if page.orientation == "portrait"),
    "landscape": next(page for page in snapshot.pages if page.orientation == "landscape"
        and not (page.ref.domain == "annual_reports" and page.ref.document_id == 1)),
    # Native source does not annotate spreads; preserve that uncertainty. This
    # source page was visually inspected as a facing-page image. This is evidence
    # for representative display checks, not a new native spread annotation.
    "spread-candidate": next(page for page in snapshot.pages if page.ref.domain == "annual_reports"
        and page.ref.document_id == 1 and page.page_number == 2),
}
for name, page in examples.items():
    elements = [element.model_dump(mode="json") for element in snapshot.elements if element.page_id == page.page_id]
    write(name, {"page": page.model_dump(mode="json"),
        "document": document_map[page.document_id].model_dump(mode="json"),
        "elements": {"items": elements, "total": len(elements),
            "coordinate_convention": "Source image pixels with top-left origin."}})
for name, element in (
    ("flagged-geometry", next(element for element in snapshot.elements if element.geometry_flags)),
    ("empty-ocr", next(element for element in snapshot.elements if element.ocr.status == "empty")),
):
    write(name, {"element": element.model_dump(mode="json"),
        "page": page_map[element.page_id].model_dump(mode="json")})
question = next(question for question in snapshot.questions if len(question.support_page_ids) > 1)
write("multi-page-support", {"question": question.model_dump(mode="json"),
    "pages": [page_map[pid].model_dump(mode="json") for pid in question.support_page_ids]})
audit = json.loads((destination / "dataset-audit.json").read_text(encoding="utf-8-sig"))
for field in ("documents", "pages", "questions", "elements"):
    assert snapshot.summary.counts[field] == audit["totals"][field]
for field in ("empty_ocr_texts", "multi_page_support_questions"):
    assert snapshot.summary.quality_counts[field] == audit["totals"][field]
assert snapshot.summary.quality_counts["bbox_outside_or_degenerate"] == audit["issues"]["bbox_outside_or_degenerate"]["count"]
assert snapshot.summary.json_snapshot_sha256 == audit["json_snapshot_sha256"]
assert len({element.key for element in snapshot.elements}) == len(snapshot.elements)
assert len({question.key for question in snapshot.questions}) == len(snapshot.questions)
write("provenance", {"kind": "real_source_transport_fixture", "schema_version": "0.2.0",
    "generator": "scripts/export_dataset_fixtures.py", "dataset_version": snapshot.summary.dataset_version,
    "source_file_sha256": snapshot.file_sha256, "synthetic_model_results": False,
    "spread_note": "Annual reports document 1 page 2 visually inspected as a facing-page image; source supplies no spread annotation.",
    "verification": "Counts, quality and full JSON hash matched tests/fixtures/datasets/dataset-audit.json."})
print(json.dumps({"counts": snapshot.summary.counts, "quality": snapshot.summary.quality_counts,
    "version": snapshot.summary.dataset_version, "fixtures": str(destination)}, indent=2))
