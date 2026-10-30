from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from docattentionlens_api.dependencies import repository
from docattentionlens_api.repositories.jobs import ConflictError, JobRepository
from docattentionlens_api.schemas.contracts import CalculationRequest, JobAccepted, JobDetail, JobPage, JobRequest, JobSummary, job_accepted
from docattentionlens_api.services.jobs import submit_job
from docattentionlens.methods.registry import get_method

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])
Repo = Annotated[JobRepository, Depends(repository)]


@router.post("", response_model=JobAccepted, status_code=202)
def submit(request: JobRequest | CalculationRequest, repo: Repo):
    try:
        job = submit_job(repo, request)
    except NotImplementedError as exc:
        raise HTTPException(422, {"code": "capture_required" if get_method(request.method_id, request.variant_id).status != "planned" else "method_planned", "message": str(exc)})
    except KeyError:
        raise HTTPException(404, {"code": "unknown_method", "message": "Method variant not found."})
    except ConflictError as exc:
        raise HTTPException(409, {"code": "idempotency_conflict", "message": str(exc)})
    return job_accepted(job)


@router.get("", response_model=JobPage)
def history(repo: Repo, limit: int = Query(default=25, ge=1, le=100), before: str | None = None):
    if before is not None and repo.get(before) is None:
        raise HTTPException(404, {"code": "unknown_cursor", "message": "Job cursor not found."})
    rows = repo.list(limit + 1, before)
    return JobPage(items=[JobSummary.model_validate(row) for row in rows[:limit]], next_cursor=rows[limit - 1]["job_id"] if len(rows) > limit else None)


@router.get("/{job_id}", response_model=JobDetail)
def detail(job_id: str, repo: Repo):
    job = repo.get(job_id)
    if job is None:
        raise HTTPException(404, {"code": "unknown_job", "message": "Job not found."})
    return JobDetail.model_validate(job)


@router.post("/{job_id}/cancel", response_model=JobSummary)
def cancel(job_id: str, repo: Repo):
    job = repo.cancel(job_id)
    if job is None:
        raise HTTPException(404, {"code": "unknown_job", "message": "Job not found."})
    return JobSummary.model_validate(job)
