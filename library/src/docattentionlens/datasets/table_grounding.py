"""Read-only adapter for TableGroundQA exports: table questions grounded on full document pages.

Registers each export page (the full-page PNG) as a page, each table as a TableRecord whose
cells, rows and columns are located in that page's image pixels, and each question with its
answer cell and lookup path as evaluation-only grounding. No layout elements are created: the
export annotates the table, not the rest of the page. No model calculations.

A page may hold several tables and a document several pages; such a page record binds each table
in `table_bindings`, while a single-table page (the pilot) carries its binding at the page level.
"""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath

from PIL import Image

from .adapter import canonical, identity
from .contracts import (
    Box, DatasetSummary, DocumentRecord, PageRecord, QualityIssue, QuestionRecord, Snapshot, SourceRef,
    TableAxisRecord, TableCellRecord, TableGrounding, TableQualityIssue, TableRecord, TableRole,
)

ADAPTER_VERSION = "table_ground_qa_v1"
EXPORT_FORMAT = "table-qa-pilot-export"
# The pilot's domain and frames; another export names its own in the manifest.
DOMAIN = "pubtables_1m"
COORDINATE_CONVENTION = "Full-page PNG pixels (150 DPI), top-left origin. Table geometry is projected from crop pixels by the export's verified transforms."
GEOMETRY_VALIDATION = "Crop-to-page transforms verified by the export: registration correlation, table-edge residuals and checked overlays. Independent human review pending."
# The table box projected through the dataset's transforms must land on its recorded page box.
TRANSFORM_TOLERANCE_PIXELS = 2.0
NATIVE_FLAGS = {"is_column_header": "column_header", "is_projected_row_header": "projected_row_header"}


def _matrix_apply(matrix, x, y):
    return (matrix[0][0]*x + matrix[0][1]*y + matrix[0][2], matrix[1][0]*x + matrix[1][1]*y + matrix[1][2])


def _compose(a, b):
    """The affine map that applies b, then a."""
    return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _box(values) -> Box:
    x1, y1, x2, y2 = (float(v) for v in values)
    if not (x1 < x2 and y1 < y2):
        raise ValueError("Table geometry has no area")
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def _project(matrix, values) -> Box:
    x1, y1 = _matrix_apply(matrix, values[0], values[1])
    x2, y2 = _matrix_apply(matrix, values[2], values[3])
    return _box([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])


def _binding(page_meta, table_id):
    """How one table is bound to its page: its grounding file, transform and verification status."""
    bindings = page_meta.get("table_bindings")
    if bindings is None:
        return page_meta
    found = [b for b in bindings if b["table_id"] == table_id]
    if len(found) != 1:
        raise ValueError("Page record does not bind this table exactly once")
    return found[0]


def read_table_dataset(root: Path) -> Snapshot:
    root = root.resolve(strict=True)
    listed = {}
    for line in (root / "checks/SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.strip().split(None, 1)
            listed[PurePosixPath(relative.strip()).as_posix()] = digest.lower()
    hashes, source_records = {}, {}

    def confined(relative: str) -> Path:
        relative = PurePosixPath(relative).as_posix()
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Source reference escapes the export or is not a file")
        return path

    def read_bytes(relative: str) -> bytes:
        """Every source file the snapshot uses must be in the export's checksum list and match it."""
        relative = PurePosixPath(relative).as_posix()
        raw = confined(relative).read_bytes()
        digest = sha256(raw).hexdigest()
        if listed.get(relative) != digest:
            raise ValueError(f"Export file differs from checks/SHA256SUMS.txt: {relative}")
        hashes[relative] = digest
        return raw

    def load(relative: str, keep=False):
        value = json.loads(read_bytes(relative).decode("utf-8-sig"))
        if keep:
            source_records[PurePosixPath(relative).as_posix()] = value
        return value

    def jsonl(relative: str):
        return [json.loads(line) for line in read_bytes(relative).decode("utf-8-sig").splitlines() if line.strip()]

    # Verify the whole checksum list, then bind the content version to it.
    for relative in listed:
        read_bytes(relative)
    checksum_list = (root / "checks/SHA256SUMS.txt").read_bytes()
    version = sha256(canonical({"files": dict(sorted(listed.items())), "checksum_list": sha256(checksum_list).hexdigest()})).hexdigest()
    manifest = load("manifest.json", keep=True)
    if manifest.get("export_format") != EXPORT_FORMAT or manifest.get("export_id") != root.name:
        raise ValueError("Not a TableGroundQA export, or the folder name differs from its export_id")
    name = manifest["export_id"]
    domain = manifest.get("domain", DOMAIN)
    dataset_id = identity("dataset", name, version)
    base = SourceRef(dataset_id=dataset_id, dataset_version=version, domain=domain, document_id="")
    links = {r["question_id"]: r for r in jsonl("qa/page_grounding.jsonl")}
    questions_by_table = {}
    for record in jsonl("qa/questions.jsonl"):
        questions_by_table.setdefault(record["scope"]["table_id"], []).append(record)

    pages, tables, questions, issues = [], [], [], []
    image_files, page_sources, held_documents = {}, [], {}
    quality = Counter()
    for entry in manifest["tables"]:
        native_table, native_page, native_document = entry["table_id"], entry["page_id"], entry["document_id"]
        table_bytes = read_bytes(entry["table_path"])
        if sha256(table_bytes).hexdigest() != entry["table_sha256"]:
            raise ValueError("Table snapshot differs from its manifest pin")
        table = json.loads(table_bytes.decode("utf-8-sig"))
        page_meta = load(entry["page_path"], keep=True)
        document_meta = load(f"documents/{native_document}/document.json", keep=True)
        binding = _binding(page_meta, native_table)
        grounding_path = binding["grounding_path"]
        grounding_bytes = read_bytes(grounding_path)
        if sha256(grounding_bytes).hexdigest() != binding["grounding_sha256"]:
            raise ValueError("Page grounding differs from its page record")
        grounding = json.loads(grounding_bytes.decode("utf-8-sig"))
        native = load(grounding["source_annotation"]["path"])
        if hashes[grounding["source_annotation"]["path"]] != grounding["source_annotation"]["sha256"]:
            raise ValueError("Native PDF annotation differs from its grounding record")
        if (grounding["table_id"], grounding["table_revision"], grounding["table_sha256"], grounding["page_id"]) != (native_table, entry["table_revision"], entry["table_sha256"], native_page):
            raise ValueError("Page grounding is bound to a different table or page")
        if binding["binding_status"] != "page_grounding_verified" or not binding.get("crop_to_page_transform_verified") or not grounding["verification"].get("automatic_passed"):
            raise ValueError("Page grounding is not verified by the export")
        if page_meta["document_id"] != native_document or native_page not in document_meta["page_ids"]:
            raise ValueError("Page and document records disagree")

        image_path = page_meta["image_path"]
        image_hash = sha256(read_bytes(image_path)).hexdigest()
        if image_hash != page_meta["image_sha256"] or image_hash != grounding["page_image_sha256"]:
            raise ValueError("Page image differs from its page and grounding records")
        with Image.open(confined(image_path)) as image:
            width, height = image.size
        if not (width, height) == (page_meta["image_width"], page_meta["image_height"]) == (grounding["page_image_width"], grounding["page_image_height"]):
            raise ValueError("Page image dimensions differ from the grounding frame")
        crop_hash = sha256(read_bytes(entry["crop_path"])).hexdigest()

        did = identity("document", version, domain, native_document)
        pid = identity("page", version, domain, native_document, native_page)
        tid = identity("table", version, domain, native_table)
        page_number = int(page_meta["source_pdf_page_number"])
        doc_ref = base.model_copy(update={"document_id": native_document})
        page_ref = doc_ref.model_copy(update={"page_number": page_number})
        source_name = grounding["source_annotation"]["source_pdf_file_name"]
        if pid not in image_files:
            pages.append(PageRecord(key=pid, page_id=pid, document_id=did, ref=page_ref, page_number=page_number,
                width=width, height=height, image_sha256=image_hash, image_url=f"/api/v1/pages/{pid}/image",
                orientation="portrait" if height > width else "landscape" if width > height else "square"))
            image_files[pid] = image_path
            page_sources.append({"page_id": pid, "filename": source_name, "original_page_number": page_number,
                                 "original_page_count": int(page_meta["source_pdf_page_count"])})
        elif image_files[pid] != image_path:
            raise ValueError("Two tables on one page name different page images")

        # Geometry: projected logical boxes (the ownership partition) and native PDF boxes (display).
        transforms = grounding["transforms"]
        to_page = _compose(transforms["page_pdf_to_page_image"], transforms["original_pdf_to_page_pdf"])
        table_box = _box(grounding["table_bbox_page_image"])
        check = _project(to_page, native["pdf_table_bbox"])
        if max(abs(a - b) for a, b in zip((check.x1, check.y1, check.x2, check.y2), (table_box.x1, table_box.y1, table_box.x2, table_box.y2))) > TRANSFORM_TOLERANCE_PIXELS:
            raise ValueError("PDF-to-page transform does not reproduce the recorded table box")
        wrap = _project(to_page, native["pdf_table_wrap_bbox"])
        wrap = Box(x1=min(wrap.x1, table_box.x1), y1=min(wrap.y1, table_box.y1), x2=max(wrap.x2, table_box.x2), y2=max(wrap.y2, table_box.y2))
        crop_boxes = {c["cell_id"]: c for c in table["cells"]}
        row_index = {r["row_id"]: r["index"] for r in table["rows"]}
        column_index = {c["column_id"]: c["index"] for c in table["columns"]}
        placed = {c["cell_id"]: c for c in grounding["cells"]}
        if set(placed) != set(crop_boxes):
            raise ValueError("Page grounding does not place every logical cell")
        cells = []
        for cell in table["cells"]:
            at = placed[cell["cell_id"]]
            if at["row_ids"] != cell["row_ids"] or at["column_ids"] != cell["column_ids"]:
                raise ValueError("Page grounding changes a cell's row or column membership")
            linked = [native["cells"][i] for i in at["source_cell_indices"]]
            pdf_text = " ".join((c.get("pdf_text_content") or "").strip() for c in linked).strip() or None
            box = _box(at["bbox_page_image"])
            if box.x1 < -0.5 or box.y1 < -0.5 or box.x2 > width + 0.5 or box.y2 > height + 0.5:
                raise ValueError("A projected cell lies outside its page")
            cells.append(TableCellRecord(native_id=cell["cell_id"], row_ids=cell["row_ids"], column_ids=cell["column_ids"],
                row_indices=sorted(row_index[r] for r in cell["row_ids"]), column_indices=sorted(column_index[c] for c in cell["column_ids"]),
                bbox=box, crop_bbox=_box(cell["bbox"]), pdf_bboxes=[_project(to_page, b) for b in at.get("native_bboxes_pdf", [])],
                text=cell["text"]["raw"], text_method=cell["text"]["method"], text_status=cell["text"]["status"], pdf_text=pdf_text,
                roles=[TableRole(role=r["role"], status=r["status"], method=r["method"]) for r in cell["roles"]],
                native_flags=sorted({flag for c in linked for key, flag in NATIVE_FLAGS.items() if c.get(key)})))
        quality["logical_cells_without_text"] += sum(c.text is None for c in cells)
        quality["cells_with_pdf_text_only"] += sum(c.text is None and c.pdf_text is not None for c in cells)
        rows = [TableAxisRecord(native_id=r["row_id"], index=row_index[r["row_id"]], bbox=_box(r["bbox_page_image"]),
                                crop_bbox=_box(next(x["bbox"] for x in table["rows"] if x["row_id"] == r["row_id"]))) for r in grounding["rows"]]
        columns = [TableAxisRecord(native_id=c["column_id"], index=column_index[c["column_id"]], bbox=_box(c["bbox_page_image"]),
                                   crop_bbox=_box(next(x["bbox"] for x in table["columns"] if x["column_id"] == c["column_id"]))) for c in grounding["columns"]]
        crop = next(a for a in table["assets"] if a["kind"] == "table_crop")
        verification = grounding["verification"]
        tables.append(TableRecord(key=tid, table_id=tid, native_id=native_table, document_id=did, page_id=pid,
            ref=page_ref.model_copy(update={"table_id": native_table}), title=entry["title"], revision=entry["table_revision"],
            table_sha256=entry["table_sha256"], grounding_revision=grounding["grounding_revision"], grounding_sha256=binding["grounding_sha256"],
            bbox=table_box, wrap_bbox=wrap, crop_width=int(crop["width"]), crop_height=int(crop["height"]), crop_sha256=crop_hash,
            crop_to_page=binding["crop_to_page"]["matrix"], registration_correlation=float(verification["image_registration_normalized_correlation"]),
            geometry_verification=(f"Export checks passed; image registration correlation {verification['image_registration_normalized_correlation']:.3f}; "
                f"visual check {verification['visual_check']}; independent human review {'done' if verification['independent_semantic_human_review'] else 'pending'}."),
            row_count=len(rows), column_count=len(columns), rows=sorted(rows, key=lambda r: r.index), columns=sorted(columns, key=lambda c: c.index),
            cells=cells, quality_status=table["quality"]["status"], publishable=bool(table["quality"]["publishable"]),
            quality_issues=[TableQualityIssue(code=i["code"], severity=i["severity"], detail=i["detail"], cell_ids=i.get("affected_cell_ids", []))
                            for i in table["quality"]["issues"]],
            source={"dataset": table["source"]["dataset"], "dataset_revision": table["source"]["dataset_revision"],
                    "source_table_key": entry["source_table_key"], "table_path": entry["table_path"], "grounding_path": grounding_path,
                    "pdf_annotation": {k: grounding["source_annotation"][k] for k in ("path", "sha256", "archive_member", "document_json_sha256")},
                    "article_title": document_meta["title"], "doi": document_meta.get("doi"), "license": document_meta.get("license"),
                    "model_input_mode_declared": manifest.get("model_input_mode")}))
        quality["tables_needing_review"] += table["quality"]["status"] != "approved"

        # Questions: every record must pin this table revision and match its page link.
        records = sorted(questions_by_table.get(native_table, []), key=lambda q: q["question_id"])
        if len(records) != entry["question_count"]:
            raise ValueError("Question count differs from the manifest")
        cell_ids = {c.native_id: c for c in cells}
        flagged = 0
        for record in records:
            scope, answer, ground = record["scope"], record["answer"], record["grounding"]
            link = links[record["question_id"]]
            if (scope["table_id"], scope["table_revision"], scope["table_sha256"], scope["document_id"]) != (native_table, entry["table_revision"], entry["table_sha256"], native_document):
                raise ValueError("Question pins a different table revision")
            if (link["table_id"], link["page_id"], link["answer"]["cell_ids"]) != (native_table, native_page, answer["cell_ids"]):
                raise ValueError("Question page link differs from the question record")
            referenced = [*answer["cell_ids"], *ground["row_header_path"], *ground["column_header_path"], *ground["support_cell_ids"]]
            if any(i not in cell_ids for i in referenced):
                raise ValueError("Question grounding refers to a cell absent from its table")
            status = link["source_qa_binding_status"]
            flagged += status == "needs_source_qa_review"
            pdf_answer = " ".join(cell_ids[i].pdf_text or "" for i in answer["cell_ids"]).strip() or None
            qid = identity("question", version, domain, record["question_id"])
            qref = doc_ref.model_copy(update={"question_id": record["question_id"]})
            questions.append(QuestionRecord(key=qid, question_id=qid, document_id=did, ref=qref, native_id=record["question_id"],
                question=record["question"], document_scoped_question=record["question"], collection_scoped_question=record["question"],
                answer=answer["text"], question_type=record["question_type"], native_element_ids=[], support_element_ids=[],
                support_page_ids=[pid], support_element_refs=[], support_page_refs=[page_ref],
                table_grounding=TableGrounding(table_id=tid, native_table_id=native_table, table_revision=entry["table_revision"],
                    table_sha256=entry["table_sha256"], answer_cell_ids=answer["cell_ids"], answer_row_ids=ground["row_ids"],
                    answer_column_ids=ground["column_ids"], row_header_path=ground["row_header_path"], column_header_path=ground["column_header_path"],
                    support_cell_ids=ground["support_cell_ids"], accepted_answers=answer["accepted_texts"], value_type=answer["value_type"],
                    unit=answer.get("unit"), rationale=ground["rationale"], binding_status=status, pdf_answer_text=pdf_answer,
                    authored_input=record["input"]["mode"], review_status=record["review"]["status"])))
            if status == "needs_source_qa_review":
                issues.append(QualityIssue(code="qa_binding_needs_review", page_id=pid, document_id=did, ref=qref,
                    detail=f"Saved answer '{answer['text']}' differs from the bound PDF cell text '{pdf_answer}'. Excluded from grounding studies by default."))
            quality["unreviewed_questions"] += record["review"]["status"] == "unreviewed"
        quality["questions_needing_source_qa_review"] += flagged

        held = held_documents.setdefault(did, {"ref": doc_ref, "native": native_document, "filename": source_name, "meta": document_meta,
                                               "page_ids": [], "table_ids": [], "table_shas": [], "questions": 0})
        if pid not in held["page_ids"]:
            held["page_ids"].append(pid)
        held["table_ids"].append(tid)
        held["table_shas"].append(entry["table_sha256"])
        held["questions"] += len(records)

    documents = []
    for did, held in held_documents.items():
        shas = held["table_shas"]
        documents.append(DocumentRecord(key=did, document_id=did, ref=held["ref"], native_id=held["native"], filename=held["filename"],
            total_pages=len(held["page_ids"]), page_ids=held["page_ids"], question_count=held["questions"], element_count=0,
            # One table: its snapshot hash. Several: the hash of their hashes, in manifest order.
            annotation_version=shas[0] if len(shas) == 1 else sha256(canonical(shas)).hexdigest(),
            source_pdf_sha256=held["meta"]["source_pdf"]["sha256"], ocr_model="none", ocr_status="not_supplied",
            title=held["meta"]["title"], table_ids=held["table_ids"]))

    counts = {"documents": len(documents), "pages": len(pages), "questions": len(questions), "elements": 0,
              "tables": len(tables), "cells": sum(len(t.cells) for t in tables)}
    summary = DatasetSummary(dataset_id=dataset_id, name=name, dataset_version=version, adapter_version=ADAPTER_VERSION,
        counts=counts, domain_counts={domain: counts}, domain_aliases={}, element_types=[], element_categories=[],
        question_types=sorted({q.question_type for q in questions}), quality_counts=dict(quality), quality_issues=issues,
        source_manifest={**{k: v for k, v in manifest.items() if k != "resource_map"}, "page_sources": page_sources},
        source_file_count=len(hashes), json_snapshot_sha256=sha256(canonical({k: v for k, v in hashes.items() if k.endswith((".json", ".jsonl"))})).hexdigest(),
        coordinate_convention=manifest.get("coordinate_convention", COORDINATE_CONVENTION),
        geometry_validation=manifest.get("geometry_validation", GEOMETRY_VALIDATION),
        structure_kind="table_grounding")
    return Snapshot(schema_version="0.3.0", summary=summary, documents=documents, pages=pages, elements=[], questions=questions,
        file_sha256=hashes, image_files=image_files, source_records=source_records, tables=tables)
