"""Layout elements a researcher marks on uploaded pages.

A draft is mutable and revisioned. Freezing it publishes a new immutable snapshot that reuses the
upload's verified page images and adds the marked elements, so cases, captures and methods read it
like any other dataset. The upload itself never changes.
"""
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import math
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from psycopg.types.json import Jsonb
from pydantic import Field

from docattentionlens_api.errors import fail
from docattentionlens_api.services.imports import PDF_LOCK, digest_file
from docattentionlens.datasets.adapter import canonical
from docattentionlens.datasets.contracts import (
    Box, DatasetSummary, DocumentRecord, ElementRecord, Record, Snapshot, SourceRef,
)

MARKED_ADAPTER = "uploaded_page_elements_v1"
UPLOAD_ADAPTERS = {"uploaded_page_images_v1", "uploaded_page_images_v2"}
ElementType = Literal["title", "heading", "text", "listitem", "table", "figure", "caption", "footnote", "formula",
                      "pageheader", "pagefooter", "pagenumber"]
MAX_ELEMENTS = 2000
# Fitting keeps pixels darker than this luminance (0–255) and a small margin around them.
INK_THRESHOLD = 235
MARGIN = 3


def category(element_type):
    """The benchmark's rule: tables are TABLE, figures IMAGE, every other type TEXT."""
    return "TABLE" if element_type == "table" else "IMAGE" if element_type == "figure" else "TEXT"


class MarkedElement(Record):
    key: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    page_id: str
    element_type: ElementType
    bbox: Box
    reading_order: int = Field(ge=1)
    text: str | None = Field(default=None, max_length=20000)
    text_source: Literal["pdf_text_layer", "typed", "none"] = "none"


class MarkingPage(Record):
    page_id: str
    page_number: int
    width: int
    height: int
    image_url: str
    text_layer: bool


class MarkedVersion(Record):
    document_id: str
    annotation_version: str
    element_count: int


class MarkingDraft(Record):
    """The working elements of one uploaded document; `frozen` is the last version saved for investigation."""
    document_id: str
    filename: str
    revision: int
    pages: list[MarkingPage]
    elements: list[MarkedElement]
    reviewed_page_ids: list[str]
    manual_order_page_ids: list[str]
    frozen: MarkedVersion | None = None


class MarkingSave(Record):
    expected_revision: int = Field(ge=0)
    elements: list[MarkedElement] = Field(max_length=MAX_ELEMENTS)
    reviewed_page_ids: list[str] = Field(default_factory=list)
    manual_order_page_ids: list[str] = Field(default_factory=list)


class FitRequest(Record):
    page_id: str
    bbox: Box
    tighten: bool = True


class FitResult(Record):
    bbox: Box
    tightened: bool
    text: str | None
    text_source: Literal["pdf_text_layer", "none"]


class FreezeRequest(Record):
    expected_revision: int = Field(ge=1)


class FreezeResult(Record):
    document: DocumentRecord
    draft: MarkingDraft


def locate(documents, document_id):
    """Uploads and marked versions name their dataset after their document, so one snapshot read finds them;
    a general lookup reads every registered snapshot, which takes seconds once the benchmark is registered."""
    dataset_id = "dataset_" + document_id.removeprefix("document_")
    with documents.jobs.connect() as conn:
        known = conn.execute("SELECT 1 FROM datasets WHERE dataset_id=%s", (dataset_id,)).fetchone()
    if known:
        snapshot = documents.snapshot(dataset_id)
        document = next((d for d in snapshot.documents if d.document_id == document_id), None)
        if document:
            return snapshot, document
    return documents.resource("documents", document_id)


def upload(documents, document_id):
    """The uploaded document behind `document_id`, which may name the upload or a marked version of it."""
    snapshot, document = locate(documents, document_id)
    if snapshot.summary.adapter_version == MARKED_ADAPTER:
        snapshot, document = locate(documents, snapshot.summary.source_manifest["base_document_id"])
    if snapshot.summary.adapter_version not in UPLOAD_ADAPTERS:
        fail("unsupported_document", "Elements can be marked on uploaded documents only")
    pages = {p.page_id: p for p in snapshot.pages}
    return snapshot, document, [pages[pid] for pid in document.page_ids]


def page_of(pages, page_id):
    page = next((p for p in pages if p.page_id == page_id), None)
    if page is None:
        fail("invalid_reference", "The page is not part of this document")
    return page


@lru_cache(maxsize=64)
def upload_file(root, filename, digest):
    """The kept upload file whose bytes match the import manifest; imports keep their files."""
    for folder in sorted(Path(root).glob("upload_*")):
        candidate = folder / "files" / filename
        if candidate.is_file() and digest_file(candidate) == digest:
            return str(candidate)
    return None


def pdf_source(settings, snapshot, page_id):
    """The PDF file and zero-based page index an uploaded page was rendered from, or None for images."""
    manifest = snapshot.summary.source_manifest
    source = next((s for s in manifest.get("page_sources", []) if s.get("page_id") == page_id), None)
    if not source or not source["filename"].lower().endswith(".pdf"):
        return None
    digest = next((f["sha256"] for f in manifest.get("files", []) if f["filename"] == source["filename"]), None)
    path = digest and upload_file(str(settings.artifact_root / "imports"), source["filename"], digest)
    return (path, source["original_page_number"] - 1) if path else None


@lru_cache(maxsize=16)
def pdf_words(path, index, width, height):
    """Words of the PDF text layer in page-image pixels, in the PDF's block, line and word sequence."""
    import pymupdf
    with PDF_LOCK, pymupdf.open(path) as pdf:
        page = pdf[index]
        # Import rendered the displayed (rotated) page with one uniform scale.
        sx, sy, matrix = width / page.rect.width, height / page.rect.height, page.rotation_matrix
        words = []
        for x0, y0, x1, y1, word, block, line, _ in page.get_text("words"):
            r = pymupdf.Rect(x0, y0, x1, y1) * matrix
            words.append((r.x0 * sx, r.y0 * sy, r.x1 * sx, r.y1 * sy, word, block, line))
    return tuple(words)


def text_in(words, box):
    """Words whose centre lies in the box, one output line per PDF line."""
    lines = {}
    for x0, y0, x1, y1, word, block, line in words:
        if box.x1 <= (x0 + x1) / 2 <= box.x2 and box.y1 <= (y0 + y1) / 2 <= box.y2:
            lines.setdefault((block, line), []).append(word)
    return "\n".join(" ".join(line) for line in lines.values())


_LUMINANCE = {}


def luminance(documents, snapshot, page):
    """The page image as greyscale, keeping the last few pages for repeated fits."""
    if page.image_sha256 not in _LUMINANCE:
        if len(_LUMINANCE) >= 4:
            _LUMINANCE.pop(next(iter(_LUMINANCE)))
        _LUMINANCE[page.image_sha256] = np.asarray(Image.open(BytesIO(documents.image_bytes(snapshot, page))).convert("L"))
    return _LUMINANCE[page.image_sha256]


def fit(documents, document_id, request):
    """Tighten a drawn box to the content inside it (never beyond it) and read its PDF text.

    Content is any pixel darker than INK_THRESHOLD, so figures and scanned pages fit as well as text.
    Text comes from the PDF text layer only; image uploads have none.
    """
    snapshot, _, pages = upload(documents, document_id)
    page = page_of(pages, request.page_id)
    b = request.bbox
    box = Box(x1=max(0, min(b.x1, b.x2)), y1=max(0, min(b.y1, b.y2)), x2=min(page.width, max(b.x1, b.x2)), y2=min(page.height, max(b.y1, b.y2)))
    if box.x2 - box.x1 < 1 or box.y2 - box.y1 < 1:
        fail("invalid_geometry", "Draw a box with an area inside the page")
    tightened = False
    if request.tighten:
        left, top = math.floor(box.x1), math.floor(box.y1)
        ink = luminance(documents, snapshot, page)[top:math.ceil(box.y2), left:math.ceil(box.x2)] < INK_THRESHOLD
        rows, columns = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
        if rows.size:
            fitted = Box(x1=max(box.x1, left + int(columns[0]) - MARGIN), y1=max(box.y1, top + int(rows[0]) - MARGIN),
                         x2=min(box.x2, left + int(columns[-1]) + 1 + MARGIN), y2=min(box.y2, top + int(rows[-1]) + 1 + MARGIN))
            tightened, box = fitted != box, fitted
    source = pdf_source(documents.jobs.settings, snapshot, page.page_id)
    words = pdf_words(*source, page.width, page.height) if source else None
    return FitResult(bbox=box, tightened=tightened, text=None if words is None else text_in(words, box),
                     text_source="none" if words is None else "pdf_text_layer")


def validate(pages, elements, reviewed=(), manual=()):
    known = {p.page_id: p for p in pages}
    if len({e.key for e in elements}) != len(elements):
        fail("invalid_input", "Each element needs its own key")
    if not set(reviewed) <= set(known) or not set(manual) <= set(known):
        fail("invalid_reference", "A page list names a page outside this document")
    for e in elements:
        page = known.get(e.page_id)
        if page is None:
            fail("invalid_reference", "An element names a page outside this document")
        b, where = e.bbox, f"Element {e.reading_order} on page {page.page_number}"
        if not all(math.isfinite(v) for v in (b.x1, b.y1, b.x2, b.y2)) or b.x2 - b.x1 < 2 or b.y2 - b.y1 < 2:
            fail("invalid_geometry", f"{where} has no area")
        if b.x1 < 0 or b.y1 < 0 or b.x2 > page.width or b.y2 > page.height:
            fail("invalid_geometry", f"{where} extends beyond the page")
    for page in pages:
        orders = sorted(e.reading_order for e in elements if e.page_id == page.page_id)
        if orders != list(range(1, len(orders) + 1)):
            fail("invalid_input", f"Reading order on page {page.page_number} must run 1, 2, 3… without gaps")


EMPTY = {"elements": [], "reviewed_page_ids": [], "manual_order_page_ids": [], "frozen": None}


def seed(documents, requested_id, document):
    """A first draft: empty for the upload, or the elements of the marked version that was opened."""
    if requested_id == document.document_id:
        return EMPTY
    snapshot, marked = locate(documents, requested_id)
    return {**EMPTY, "elements": snapshot.source_records["marked_elements"],
            "frozen": {"document_id": marked.document_id, "annotation_version": marked.annotation_version, "element_count": marked.element_count}}


def assemble(documents, snapshot, document, pages, revision, content):
    settings = documents.jobs.settings
    return MarkingDraft(document_id=document.document_id, filename=document.filename, revision=revision,
        pages=[MarkingPage(page_id=p.page_id, page_number=p.page_number, width=p.width, height=p.height, image_url=p.image_url,
                           text_layer=pdf_source(settings, snapshot, p.page_id) is not None) for p in pages],
        elements=content["elements"], reviewed_page_ids=content["reviewed_page_ids"],
        manual_order_page_ids=content["manual_order_page_ids"], frozen=content["frozen"])


def draft(documents, document_id):
    snapshot, document, pages = upload(documents, document_id)
    with documents.jobs.connect() as conn:
        row = conn.execute("SELECT revision,content FROM element_marking_drafts WHERE document_id=%s", (document.document_id,)).fetchone()
    content = row["content"] if row else seed(documents, document_id, document)
    return assemble(documents, snapshot, document, pages, row["revision"] if row else 0, content)


def stale(revision):
    fail("revision_conflict", "These elements were changed in another window. Reload to continue.", 409, current_revision=revision)


def save(documents, document_id, body):
    snapshot, document, pages = upload(documents, document_id)
    validate(pages, body.elements, body.reviewed_page_ids, body.manual_order_page_ids)
    with documents.jobs.connect() as conn:
        row = conn.execute("SELECT revision,content FROM element_marking_drafts WHERE document_id=%s FOR UPDATE", (document.document_id,)).fetchone()
        current = row["revision"] if row else 0
        if current != body.expected_revision:
            stale(current)
        content = {"elements": [e.model_dump(mode="json") for e in body.elements], "reviewed_page_ids": body.reviewed_page_ids,
                   "manual_order_page_ids": body.manual_order_page_ids,
                   "frozen": (row["content"] if row else seed(documents, document_id, document))["frozen"]}
        if row:
            conn.execute("UPDATE element_marking_drafts SET revision=%s,content=%s,updated_at=clock_timestamp() WHERE document_id=%s",
                (current + 1, Jsonb(content), document.document_id))
        elif not conn.execute("INSERT INTO element_marking_drafts(document_id,revision,content) VALUES (%s,1,%s) ON CONFLICT DO NOTHING",
                (document.document_id, Jsonb(content))).rowcount:
            stale(1)
    return assemble(documents, snapshot, document, pages, current + 1, content)


def freeze(documents, document_id, body):
    """Publish the saved draft as an immutable marked version; the same elements give the same version."""
    snapshot, document, pages = upload(documents, document_id)
    with documents.jobs.connect() as conn:
        row = conn.execute("SELECT revision,content FROM element_marking_drafts WHERE document_id=%s FOR UPDATE", (document.document_id,)).fetchone()
        if not row or row["revision"] != body.expected_revision:
            stale(row["revision"] if row else 0)
        elements = [MarkedElement.model_validate(e) for e in row["content"]["elements"]]
        if not elements:
            fail("invalid_input", "Mark at least one element before saving a version")
        validate(pages, elements)
        marked = build(snapshot, document, pages, elements)
        checksum = documents.store.put(marked)
        conn.execute("INSERT INTO datasets(dataset_id,content_hash,summary) VALUES (%s,%s,%s) ON CONFLICT(dataset_id) DO NOTHING",
            (marked.summary.dataset_id, checksum, Jsonb(marked.summary.model_dump(mode="json"))))
        record = marked.documents[0]
        content = {**row["content"], "frozen": {"document_id": record.document_id, "annotation_version": record.annotation_version, "element_count": record.element_count}}
        conn.execute("UPDATE element_marking_drafts SET revision=%s,content=%s,updated_at=clock_timestamp() WHERE document_id=%s",
            (row["revision"] + 1, Jsonb(content), document.document_id))
    return FreezeResult(document=record, draft=assemble(documents, snapshot, document, pages, row["revision"] + 1, content))


def build(snapshot, document, pages, elements):
    """The marked snapshot: the upload's page images under new identities, with one element per mark."""
    number = {p.page_id: i for i, p in enumerate(pages, 1)}
    ordered = sorted(elements, key=lambda e: (number[e.page_id], e.reading_order))
    boxes = [Box(**{k: round(float(getattr(e.bbox, k)), 2) for k in ("x1", "y1", "x2", "y2")}) for e in ordered]
    annotation_version = sha256(canonical({"base_document_id": document.document_id, "elements": [
        {"page_number": number[e.page_id], "element_type": e.element_type, "bbox": b.model_dump(), "reading_order": e.reading_order,
         "text": e.text, "text_source": e.text_source} for e, b in zip(ordered, boxes)]})).hexdigest()
    version = sha256(canonical({"adapter": MARKED_ADAPTER, "base_dataset_version": snapshot.summary.dataset_version,
                                "annotation_version": annotation_version})).hexdigest()
    dataset_id, document_id = "dataset_" + version, "document_" + version
    ref = SourceRef(dataset_id=dataset_id, dataset_version=version, domain="uploaded", document_id=document_id)
    page_ids = {p.page_id: "page_" + sha256(f"{version}:{number[p.page_id]}".encode()).hexdigest() for p in pages}
    new_pages = [p.model_copy(update={"key": page_ids[p.page_id], "page_id": page_ids[p.page_id], "document_id": document_id,
        "ref": ref.model_copy(update={"page_number": number[p.page_id]}), "image_url": f"/api/v1/pages/{page_ids[p.page_id]}/image"}) for p in pages]
    image_files = {page_ids[p.page_id]: snapshot.image_files[p.page_id] for p in pages}
    records = []
    for native, (e, box) in enumerate(zip(ordered, boxes), 1):
        eid = "element_" + sha256(f"{version}:{native}".encode()).hexdigest()
        records.append(ElementRecord(key=eid, element_id=eid, page_id=page_ids[e.page_id], document_id=document_id,
            ref=ref.model_copy(update={"page_number": number[e.page_id], "element_id": native}), native_id=native,
            page_number=number[e.page_id], element_type=e.element_type, element_category=category(e.element_type), bbox=box,
            level=None, reading_order=e.reading_order,
            ocr={"status": "missing" if e.text is None else "present" if e.text.strip() else "empty", "text": e.text},
            geometry_flags=[], display_bbox=box, display_geometry_policy="raw"))
    text_layer = any(e.text_source == "pdf_text_layer" for e in ordered)
    record = document.model_copy(update={"key": document_id, "document_id": document_id, "ref": ref, "native_id": document_id,
        "page_ids": [page_ids[p.page_id] for p in pages], "element_count": len(records), "annotation_version": annotation_version,
        "ocr_model": "pdf_text_layer" if text_layer else "none", "ocr_status": "pdf_text_layer" if text_layer else "not_available"})
    base = snapshot.summary.source_manifest
    manifest = {"adapter": MARKED_ADAPTER, "base_dataset_id": snapshot.summary.dataset_id, "base_dataset_version": snapshot.summary.dataset_version,
        "base_document_id": document.document_id, "annotation_version": annotation_version,
        "layout_annotations": "marked by the researcher on the registered page images", "files": base.get("files", []),
        "page_sources": [{**s, "page_id": page_ids[s["page_id"]]} for s in base.get("page_sources", []) if s.get("page_id") in page_ids],
        "ocr": "pdf_text_layer" if text_layer else False}
    summary = DatasetSummary(dataset_id=dataset_id, name=document.filename, dataset_version=version, adapter_version=MARKED_ADAPTER,
        counts={"documents": 1, "pages": len(pages), "questions": 0, "elements": len(records)},
        domain_counts={"uploaded": {"documents": 1, "pages": len(pages), "elements": len(records)}}, domain_aliases={},
        element_types=sorted({r.element_type for r in records}), element_categories=sorted({r.element_category for r in records}),
        question_types=[], quality_issues=[], source_manifest=manifest,
        quality_counts={"bbox_outside_or_degenerate": 0, "empty_ocr_texts": sum(r.ocr.status == "empty" for r in records), "missing_ocr_joins": 0,
                        "maximum_elements_per_page": max(sum(r.page_id == p for r in records) for p in page_ids.values()), "multi_page_support_questions": 0},
        source_file_count=snapshot.summary.source_file_count, json_snapshot_sha256=annotation_version,
        geometry_validation="Boxes marked by the researcher on the registered page images; no geometry repairs.")
    return Snapshot(summary=summary, documents=[record], pages=new_pages, elements=records, questions=[],
        file_sha256={path: snapshot.file_sha256[path] for path in image_files.values()}, image_files=image_files,
        source_records={"marked_elements": [e.model_dump(mode="json") for e in ordered]})
