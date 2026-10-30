from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from docattentionlens_api.dependencies import receipt_store, repository
from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens_api.repositories.receipts import FileReceiptStore
from docattentionlens_api.schemas.contracts import RuntimeStatus
from docattentionlens_api.services.plans import retrieval_plan
from docattentionlens.artifacts.records import OperationalReceipt

router = APIRouter(prefix="/api/v1", tags=["resources"])


@router.get("/runtime", response_model=RuntimeStatus)
def runtime(repo: Annotated[JobRepository, Depends(repository)]):
    plan = retrieval_plan(repo)
    return RuntimeStatus(workers=repo.workers(),model_execution="available" if plan.available else "unavailable",model_reason=plan.reason)


@router.get("/artifacts/operational_receipt/{record_id}", response_model=OperationalReceipt)
def receipt(record_id: str, repo: Annotated[JobRepository, Depends(repository)], store: Annotated[FileReceiptStore, Depends(receipt_store)]):
    metadata = repo.receipt(record_id)
    if metadata is None:
        raise HTTPException(404, {"code": "unknown_artifact", "message": "Receipt not found."})
    try:
        return store.read(record_id, metadata["content_hash"])
    except (ValueError, OSError):
        raise HTTPException(409, {"code": "artifact_incomplete", "message": "Receipt is missing or failed integrity verification."})
