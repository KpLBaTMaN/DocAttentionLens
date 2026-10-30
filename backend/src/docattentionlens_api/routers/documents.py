import base64
from hashlib import sha256
import json
import binascii

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response

from docattentionlens_api.errors import fail
from docattentionlens_api.repositories.jobs import ConflictError
from docattentionlens_api.schemas.contracts import JobAccepted, job_accepted
from docattentionlens_api.schemas.documents import (
    DatasetPage, DatasetSourcePage, DocumentPage, ElementPage, InvestigationPage, QuestionPage, RegistrationRequest, TablePage, TableSummaryPage,
)
from docattentionlens_api.services.dataset_sources import SOURCES
from docattentionlens.datasets.adapter import canonical
from docattentionlens.datasets.contracts import (
    CaseRequest, CaseRecord, DatasetSummary, DocumentRecord, ElementRecord, PageRecord, QuestionRecord,
    InvestigationCreate, InvestigationUpdate, InvestigationRecord,
)

router = APIRouter(prefix="/api/v1", tags=["documents"])


def repo(request):
    return request.app.state.documents


def page_slice(records, scope, limit, before):
    signature = sha256(canonical(scope)).hexdigest()
    offset = 0
    if before:
        try:
            cursor = json.loads(base64.urlsafe_b64decode(before))
            if cursor["scope"] != signature or type(cursor["offset"]) is not int or not 0 <= cursor["offset"] <= len(records):
                raise ValueError()
            offset = cursor["offset"]
        except (ValueError, KeyError, TypeError, UnicodeDecodeError, binascii.Error):
            fail("invalid_cursor", "Cursor does not match this list")
    selected = records[offset:offset + limit]
    next_offset = offset + len(selected)
    cursor = base64.urlsafe_b64encode(canonical({"scope": signature, "offset": next_offset})).decode() if next_offset < len(records) else None
    return {"items": selected, "next_cursor": cursor, "total": len(records)}


def native_order(record):
    return (record.ref.domain, 0 if isinstance(record.native_id, int) else 1, record.native_id)


@router.post("/datasets/register", response_model=JobAccepted, status_code=202)
def register(body: RegistrationRequest, request: Request):
    specification = {"kind": "dataset_import", "source": body.source, "operation_version": SOURCES[body.source].operation_version}
    try:
        job = request.app.state.repository.submit(body.idempotency_key, specification)
    except ConflictError:
        fail("idempotency_conflict", "Key already names a different operation", 409)
    return job_accepted(job)


@router.get("/datasets", response_model=DatasetPage)
def datasets(request: Request):
    return {"items": [row["summary"] for row in repo(request).datasets()]}


@router.get("/datasets/sources", response_model=DatasetSourcePage)
def dataset_sources(request: Request):
    """Registrable dataset folders, so the interface can offer each one that is present."""
    root = request.app.state.settings.data_root
    return {"items": [{"source": s.source, "label": s.label, "description": s.description, "structure_kind": s.structure_kind,
                       "present": (root / s.source / "manifest.json").is_file()} for s in SOURCES.values()]}


@router.get("/tables", response_model=TableSummaryPage)
def tables(request: Request, dataset_id: str):
    snapshot = repo(request).snapshot(dataset_id)
    pages = {p.page_id: p for p in snapshot.pages}
    sources = {s["page_id"]: s for s in snapshot.summary.source_manifest.get("page_sources", []) if isinstance(s, dict)}
    items = []
    for table in snapshot.tables:
        questions = [q for q in snapshot.questions if q.table_grounding and q.table_grounding.table_id == table.table_id]
        page = pages[table.page_id]
        items.append({"table_id": table.table_id, "native_id": table.native_id, "document_id": table.document_id, "page_id": table.page_id,
            "title": table.title, "source_page_number": page.page_number, "source_page_count": int(sources.get(table.page_id, {}).get("original_page_count", 1)),
            "row_count": table.row_count, "column_count": table.column_count, "cell_count": len(table.cells), "question_count": len(questions),
            "questions_needing_review": sum(q.table_grounding.binding_status == "needs_source_qa_review" for q in questions), "quality_status": table.quality_status})
    return {"items": items}


@router.get("/datasets/{dataset_id}/summary", response_model=DatasetSummary)
def summary(dataset_id: str, request: Request):
    return repo(request).snapshot(dataset_id).summary


@router.get("/documents", response_model=DocumentPage)
def documents(request: Request, dataset_id: str, limit: int = Query(25, ge=1, le=100), before: str | None = Query(None, max_length=1024), domain: str | None = None, search: str = Query("", max_length=200)):
    snapshot = repo(request).snapshot(dataset_id)
    records = sorted((doc for doc in snapshot.documents if (domain is None or doc.ref.domain == domain) and search.casefold() in doc.filename.casefold()), key=native_order)
    return page_slice(records, [dataset_id, domain, search], limit, before)


@router.get("/documents/{document_id}", response_model=DocumentRecord)
def document(document_id: str, request: Request):
    return repo(request).resource("documents", document_id)[1]


@router.get("/questions", response_model=QuestionPage)
def questions(request: Request, document_id: str, limit: int = Query(25, ge=1, le=100), before: str | None = Query(None, max_length=1024), question_type: str | None = None, search: str = Query("", max_length=200)):
    snapshot, _ = repo(request).resource("documents", document_id)
    records = sorted((q for q in snapshot.questions if q.document_id == document_id and (question_type is None or q.question_type == question_type) and search.casefold() in q.question.casefold()), key=native_order)
    return page_slice(records, [document_id, question_type, search], limit, before)


@router.get("/questions/{question_id}", response_model=QuestionRecord)
def question(question_id: str, request: Request):
    return repo(request).resource("questions", question_id)[1]


@router.get("/pages/{page_id}", response_model=PageRecord)
def page(page_id: str, request: Request):
    return repo(request).resource("pages", page_id)[1]


@router.get("/pages/{page_id}/elements", response_model=ElementPage)
def elements(page_id: str, request: Request):
    snapshot, _ = repo(request).resource("pages", page_id)
    items = [e for e in snapshot.elements if e.page_id == page_id]
    return {"items": items, "total": len(items)}


@router.get("/pages/{page_id}/tables", response_model=TablePage)
def page_tables(page_id: str, request: Request):
    snapshot, _ = repo(request).resource("pages", page_id)
    items = [t for t in snapshot.tables if t.page_id == page_id]
    return {"items": items, "total": len(items)}


@router.get("/pages/{page_id}/image", responses={200: {"content": {"image/jpeg": {}}, "description": "Verified original page image"}})
def image(page_id: str, request: Request):
    snapshot, page = repo(request).resource("pages", page_id)
    raw = repo(request).image_bytes(snapshot, page)
    media_type = "image/png" if raw.startswith(b"\x89PNG") else "image/jpeg"
    return Response(raw, media_type=media_type, headers={"ETag": f'"{page.image_sha256}"', "Cache-Control": "public, max-age=31536000, immutable"})


@router.get("/elements/{element_id}", response_model=ElementRecord)
def element(element_id: str, request: Request):
    return repo(request).resource("elements", element_id)[1]


@router.post("/cases", response_model=CaseRecord, status_code=201)
def create_case(body: CaseRequest, request: Request):
    return repo(request).create_case(body)


@router.get("/cases", response_model=list[CaseRecord], response_model_exclude_unset=True)
def cases(request: Request, limit: int = Query(25, ge=1, le=100), before: str | None = None):
    return [row["content"] for row in repo(request).cases(limit, before)]


@router.get("/cases/{case_id}", response_model=CaseRecord, response_model_exclude_unset=True)
def case(case_id: str, request: Request):
    return repo(request).case(case_id)


@router.post("/investigations", response_model=InvestigationRecord, status_code=201)
def create_investigation(body: InvestigationCreate, request: Request):
    return repo(request).save(body)


@router.patch("/investigations/{investigation_id}", response_model=InvestigationRecord)
def save_investigation(investigation_id: str, body: InvestigationUpdate, request: Request):
    return repo(request).save(body, investigation_id)


@router.get("/investigations", response_model=InvestigationPage, response_model_exclude_unset=True)
def investigations(request: Request, limit: int = Query(25, ge=1, le=100), before: str | None = None, case_id: str | None = None):
    items = [row["content"] for row in repo(request).investigations(limit + 1, before, case_id)]
    return {"items": items[:limit], "next_cursor": items[limit - 1]["investigation_id"] if len(items) > limit else None}


@router.get("/investigations/{investigation_id}", response_model=InvestigationRecord, response_model_exclude_unset=True)
def investigation(investigation_id: str, request: Request, revision: int | None = Query(None, ge=1)):
    return repo(request).investigation(investigation_id, revision)
