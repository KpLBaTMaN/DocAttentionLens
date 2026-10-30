"""Mark layout elements on uploaded pages: a revisioned draft, box fitting and frozen versions."""
from fastapi import APIRouter, Request

from docattentionlens_api.services import element_marking as marking

router = APIRouter(prefix="/api/v1/documents/{document_id}/marking", tags=["marking"])


@router.get("", response_model=marking.MarkingDraft)
def draft(document_id: str, request: Request):
    """The working elements; a marked version's id opens its upload's draft."""
    return marking.draft(request.app.state.documents, document_id)


@router.put("", response_model=marking.MarkingDraft)
def save(document_id: str, body: marking.MarkingSave, request: Request):
    return marking.save(request.app.state.documents, document_id, body)


@router.post("/fit", response_model=marking.FitResult)
def fit(document_id: str, body: marking.FitRequest, request: Request):
    return marking.fit(request.app.state.documents, document_id, body)


@router.post("/freeze", response_model=marking.FreezeResult, status_code=201)
def freeze(document_id: str, body: marking.FreezeRequest, request: Request):
    return marking.freeze(request.app.state.documents, document_id, body)
