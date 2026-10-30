from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from docattentionlens.core.integrity import IntegrityInput
from docattentionlens.methods.contracts import DocumentStructure, MethodDescriptor, PreparedMethod, TableStructure

JobState = Literal["queued", "running", "cancellation_requested", "completed", "failed", "cancelled"]


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["integrity_check"] = "integrity_check"
    idempotency_key: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")
    resource_class: Literal["cpu"] = "cpu"
    inputs: IntegrityInput = Field(default_factory=IntegrityInput)


class CalculationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["calculation"]
    idempotency_key: str = Field(min_length=1, max_length=128)
    method_id: str = Field(max_length=128)
    variant_id: str = Field(max_length=128)
    inputs: dict[str, str] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)


class JobSummary(BaseModel):
    job_id: str
    kind: str
    resource_class: str
    state: JobState
    created_at: datetime
    updated_at: datetime
    attempt_id: str | None
    result_id: str | None
    output_resource_id: str | None = None
    output_resource_kind: str | None = None
    error_code: str | None
    error_message: str | None


class Attempt(BaseModel):
    attempt_id: str
    worker_id: str
    state: str
    started_at: datetime
    finished_at: datetime | None
    error_code: str | None


class JobEvent(BaseModel):
    event_id: int
    attempt_id: str | None
    state: str
    created_at: datetime


class JobDetail(JobSummary):
    specification: dict[str, Any]
    input_hash: str
    attempts: list[Attempt]
    events: list[JobEvent]


class JobPage(BaseModel):
    items: list[JobSummary]
    next_cursor: str | None


class JobAccepted(BaseModel):
    job_id: str
    state: JobState
    status_url: str


def job_accepted(job: dict) -> dict:
    """The 202 body for a newly queued or idempotently reused job row."""
    return {"job_id": job["job_id"], "state": job["state"], "status_url": f"/api/v1/jobs/{job['job_id']}"}


class MethodPage(BaseModel):
    items: list[MethodDescriptor]
    catalogue_version: str = "0.4.0"


class PreparedMethodPage(BaseModel):
    items: list[PreparedMethod]
    generation_capture_id: str | None = None
    retrieval_capture_id: str | None = None
    case_id: str | None = None
    # Present when a case is supplied: annotations on its input pages and their provenance.
    structure: DocumentStructure | None = None
    # Present for a table grounding case: the table table-aware methods aggregate into.
    table_structure: TableStructure | None = None


class WorkerStatus(BaseModel):
    worker_id: str
    resource_class: str
    heartbeat_at: datetime
    producer_version: str
    available: bool
    capabilities: dict[str, Any] = Field(default_factory=dict)


class RuntimeStatus(BaseModel):
    workers: list[WorkerStatus]
    model_execution: Literal["available", "unavailable"] = "unavailable"
    model_reason: str = "GPU execution unavailable; saved captures remain readable."


class HealthStatus(BaseModel):
    status: Literal["live", "ready", "unavailable"]
    dependencies: dict[str, Literal["ready", "unavailable"]] = Field(default_factory=dict)
