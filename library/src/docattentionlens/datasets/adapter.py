"""Read-only adapter for the native doclayout export. No model calculations."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image

from .contracts import (
    Box, DatasetSummary, DocumentRecord, ElementRecord, PageRecord,
    QualityIssue, QuestionRecord, Snapshot, SourceRef,
)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def identity(kind: str, *parts: object) -> str:
    return kind + "_" + sha256(canonical(parts)).hexdigest()


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_dataset(root: Path, name: str = "doclayout_mm_rag_bench") -> Snapshot:
    """Read an export in the doclayout layout. `name` is its registered source and enters the dataset identity."""
    root = root.resolve(strict=True)
    source_records = {}
    hashes = {}

    def confined(base: Path, relative: str) -> Path:
        path = (base / relative).resolve(strict=True)
        if not path.is_relative_to(base.resolve()):
            raise ValueError("Source reference escapes its domain")
        if not path.is_file():
            raise ValueError("Source reference is not a file")
        return path

    def remember(path: Path) -> str:
        relative = path.relative_to(root).as_posix()
        if relative not in hashes:
            hashes[relative] = file_hash(path)
        return hashes[relative]

    def load(path: Path):
        raw = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        hashes[relative] = sha256(raw).hexdigest()
        result = json.loads(raw.decode("utf-8-sig"))
        source_records[relative] = result
        return result

    # Hash all native JSON plus every explicitly referenced image/PDF. Paths are
    # relative, making content versions independent of installation location.
    for path in sorted(root.rglob("*.json")):
        if not path.resolve().is_relative_to(root):
            raise ValueError("JSON source escapes dataset root")
        load(path)
    json_hash = sha256(canonical(hashes)).hexdigest()
    domains = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / "documents.json").is_file():
            continue
        docs = source_records[(directory / "documents.json").relative_to(root).as_posix()]["documents"]
        for doc in docs:
            remember(confined(directory, doc["source_pdf_file"]))
            for page in doc["pages"]:
                remember(confined(directory, page["image_file"]))
        domains.append((directory, docs))
    version = sha256(canonical(hashes)).hexdigest()
    dataset_id = identity("dataset", name, version)
    documents, pages, elements, questions, issues = [], [], [], [], []
    image_files = {}
    counts, quality, domain_counts = Counter(), Counter(), {}

    def payload(directory, relative):
        return source_records[confined(directory, relative).relative_to(root).as_posix()]

    for directory, docs in domains:
        domain = directory.name
        dc = Counter()
        seen_docs = set()
        for doc in docs:
            native = doc["id"]
            if (type(native), native) in seen_docs:
                raise ValueError("Duplicate document identity in domain")
            seen_docs.add((type(native), native))
            did = identity("document", version, domain, native)
            ref = SourceRef(dataset_id=dataset_id, dataset_version=version, domain=domain, document_id=native)
            layout = payload(directory, doc["layout_file"])
            qa = payload(directory, doc["annotations_file"])
            ocr = payload(directory, doc["ocr_file"])
            if any(record["document_id"] != native for record in (layout, qa, ocr)):
                raise ValueError("Native document join mismatch")
            page_map = {}
            for page in sorted(doc["pages"], key=lambda item: item["page_number"]):
                pn = page["page_number"]
                if pn in page_map:
                    raise ValueError("Duplicate page number")
                path = confined(directory, page["image_file"])
                with Image.open(path) as image:
                    width, height = image.size
                pid = identity("page", version, domain, native, pn)
                record = PageRecord(key=pid, page_id=pid, document_id=did,
                    ref=ref.model_copy(update={"page_number": pn}), page_number=pn,
                    width=width, height=height, image_sha256=remember(path),
                    image_url=f"/api/v1/pages/{pid}/image",
                    orientation="portrait" if height > width else "landscape" if width > height else "square")
                page_map[pn] = record
                pages.append(record)
                image_files[pid] = path.relative_to(root).as_posix()
            if len(page_map) != doc["total_pages"]:
                raise ValueError("Declared page count differs from source pages")
            ocr_map = {}
            for item in ocr["elements"]:
                if item["id"] in ocr_map:
                    raise ValueError("Duplicate OCR identity within document")
                ocr_map[item["id"]] = item["ocr_text"]
            element_map = {}
            for item in layout["structured_layout_elements"]:
                eid = item["id"]
                if eid in element_map:
                    raise ValueError("Duplicate element identity within document")
                page = page_map[item["page_number"]]
                box = Box.model_validate(item["bbox"])
                flags = []
                if box.x1 >= box.x2 or box.y1 >= box.y2:
                    flags.append("degenerate")
                if min(box.x1, box.y1) < 0 or box.x2 > page.width or box.y2 > page.height:
                    flags.append("out_of_bounds")
                display = Box(x1=max(0, min(page.width, box.x1)), y1=max(0, min(page.height, box.y1)),
                              x2=max(0, min(page.width, box.x2)), y2=max(0, min(page.height, box.y2)))
                available = display.x1 < display.x2 and display.y1 < display.y2 and "degenerate" not in flags
                text = ocr_map.get(eid)
                status = "missing" if eid not in ocr_map else "empty" if not text.strip() else "present"
                quality["empty_ocr_texts"] += status == "empty"
                quality["missing_ocr_joins"] += status == "missing"
                ek = identity("element", version, domain, native, eid)
                eref = ref.model_copy(update={"page_number": page.page_number, "element_id": eid})
                record = ElementRecord(key=ek, element_id=ek, page_id=page.page_id, document_id=did,
                    ref=eref, native_id=eid, page_number=page.page_number,
                    element_type=item["element_type"], element_category=item["element_category"],
                    bbox=box, level=item["level"], reading_order=item["reading_order"],
                    ocr={"status": status, "text": text}, geometry_flags=flags,
                    display_bbox=display if available else None,
                    display_geometry_policy="unavailable" if not available else "clipped_to_page" if flags else "raw")
                elements.append(record)
                element_map[eid] = record
                if flags:
                    quality["bbox_outside_or_degenerate"] += 1
                    issues.append(QualityIssue(code="bbox_outside_or_degenerate", element_id=ek,
                        page_id=page.page_id, document_id=did, ref=eref, raw_bbox=box,
                        detail=", ".join(flags) + "; original coordinates preserved"))
            if set(ocr_map) - set(element_map):
                raise ValueError("Dangling native OCR reference")
            seen_questions = set()
            for item in qa["questions_and_answers"]:
                qid = item["id"]
                if qid in seen_questions or item["document_id"] != native:
                    raise ValueError("Duplicate question identity or document join mismatch")
                seen_questions.add(qid)
                support = [element_map[eid] for eid in item["element_ids"]]
                support_pages = sorted({element.page_number for element in support})
                quality["multi_page_support_questions"] += len(support_pages) > 1
                qk = identity("question", version, domain, native, qid)
                questions.append(QuestionRecord(key=qk, question_id=qk, document_id=did,
                    ref=ref.model_copy(update={"question_id": qid}), native_id=qid,
                    question=item["question"], document_scoped_question=item["document_scoped_question"],
                    collection_scoped_question=item["collection_scoped_question"], answer=item["answer"],
                    question_type=item["question_type"], native_element_ids=item["element_ids"],
                    support_element_ids=[element.element_id for element in support],
                    support_page_ids=[page_map[pn].page_id for pn in support_pages],
                    support_element_refs=[element.ref for element in support],
                    support_page_refs=[page_map[pn].ref for pn in support_pages]))
            documents.append(DocumentRecord(key=did, document_id=did, ref=ref, native_id=native,
                filename=doc["filename"], total_pages=doc["total_pages"],
                page_ids=[page.page_id for page in page_map.values()], question_count=len(seen_questions),
                element_count=len(element_map), annotation_version=remember(confined(directory, doc["annotations_file"])),
                source_pdf_sha256=remember(confined(directory, doc["source_pdf_file"])),
                ocr_model=ocr["ocr_model"], ocr_status=ocr["ocr_status"]))
            dc.update(documents=1, pages=len(page_map), elements=len(element_map), questions=len(seen_questions))
        domain_counts[domain] = dict(dc)
        counts.update(dc)
    page_counts = Counter(element.page_id for element in elements)
    quality["maximum_elements_per_page"] = max(page_counts.values(), default=0)
    summary = DatasetSummary(dataset_id=dataset_id, name=name, dataset_version=version,
        counts=dict(counts), domain_counts=domain_counts,
        domain_aliases={"pwc_documents": "professional_services_reports"},
        element_types=sorted({element.element_type for element in elements}),
        element_categories=sorted({element.element_category for element in elements}),
        question_types=sorted({question.question_type for question in questions}),
        quality_counts=dict(quality), quality_issues=issues, source_manifest=source_records["manifest.json"],
        source_file_count=len(hashes), json_snapshot_sha256=json_hash)
    return Snapshot(summary=summary, documents=documents, pages=pages, elements=elements,
        questions=questions, file_sha256=hashes, image_files=image_files, source_records=source_records)
