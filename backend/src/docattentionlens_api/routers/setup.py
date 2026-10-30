"""Configuration inputs; raw file bodies avoid loading large weights into RAM."""
import json
from typing import Literal
from fastapi import APIRouter, Request, Query
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from docattentionlens.datasets.contracts import CaseRecord, DocumentRecord
from docattentionlens_api.errors import fail
from docattentionlens_api.services.imports import PDF_LOCK, ImportStore, ModelStore, import_documents, publish_json
from docattentionlens_api.services.model_catalogue import ModelRegistration, ModelRole

router = APIRouter(prefix="/api/v1/setup", tags=["setup"])
_completion_lock = PDF_LOCK

class UploadRequest(BaseModel):
    kind: Literal["documents", "model"]

class UploadSession(BaseModel):
    upload_id: str

class CompleteRequest(BaseModel):
    filenames: list[str] = Field(min_length=1, max_length=64)
    name: str = Field(default="My checkpoint", min_length=1, max_length=100)
    role: ModelRole = "generation"

class CustomCaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str
    query: str = Field(min_length=1, max_length=4000)

@router.get("/models", response_model=list[ModelRegistration])
def models(request: Request):
    return ModelStore(request.app.state.settings.artifact_root).list()

@router.post("/uploads", response_model=UploadSession, status_code=201)
def start(body: UploadRequest, request: Request):
    return {"upload_id": ImportStore(request.app.state.settings.artifact_root).start(body.kind)}

@router.put("/uploads/{upload_id}/files")
async def upload(upload_id: str, request: Request, filename: str = Query(min_length=1, max_length=180)):
    return await ImportStore(request.app.state.settings.artifact_root).upload(upload_id, filename, request)

def finish(request, upload_id, body, kind):
    # Native PDF rendering cannot run concurrently in this API process.
    # Also serialize finalization so retries cannot publish different inputs.
    with _completion_lock:
        return _finish(request, upload_id, body, kind)

def _finish(request, upload_id, body, kind):
    folder, session = ImportStore(request.app.state.settings.artifact_root).session(upload_id)
    if session["kind"] != kind:
        fail("invalid_input", "Upload type differs")
    if (folder / "result.json").exists():
        saved = json.loads((folder / "result.json").read_text())
        return ModelStore(request.app.state.settings.artifact_root).get(saved["registration_id"]) if kind == "model" else saved
    if not (folder / "files").exists() or set(body.filenames) != {p.name for p in (folder / "files").iterdir()}:
        fail("invalid_input", "Upload all selected files before importing")
    try:
        result = import_documents(request.app.state.documents, folder, body.filenames).model_dump(mode="json") if kind == "documents" else ModelStore(request.app.state.settings.artifact_root).register(folder, body.name, body.role)
    except (ValueError, OSError, RuntimeError, Image.DecompressionBombError, UnidentifiedImageError):
        fail("invalid_input", "Unable to import the document or checkpoint package; verify the file format")
    # Publication is atomic; retries return the same immutable input identity.
    publish_json(folder / "result.json", result)
    return result

@router.post("/uploads/{upload_id}/documents", response_model=DocumentRecord)
async def documents(upload_id: str, body: CompleteRequest, request: Request):
    return await run_in_threadpool(finish, request, upload_id, body, "documents")

@router.post("/uploads/{upload_id}/model", response_model=ModelRegistration)
async def model(upload_id: str, body: CompleteRequest, request: Request):
    return await run_in_threadpool(finish, request, upload_id, body, "model")

@router.post("/cases", response_model=CaseRecord, status_code=201)
def custom_case(body: CustomCaseRequest, request: Request):
    return request.app.state.documents.custom_case(body.document_id, body.query)
