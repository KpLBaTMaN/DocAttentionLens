from docattentionlens_api.errors import fail
from docattentionlens_api.repositories.jobs import ConflictError, JobRepository
from docattentionlens_api.schemas.contracts import CalculationRequest, JobRequest, job_accepted
from docattentionlens.methods.registry import require_executable


def submit_job(repository: JobRepository, request: JobRequest | CalculationRequest) -> dict:
    if isinstance(request, CalculationRequest):
        require_executable(request.method_id, request.variant_id)
        raise NotImplementedError("Select the compatible saved retrieval or generator capture in Attention Tool to inspect this calculation.")
    specification = request.model_dump(mode="json", exclude={"idempotency_key"})
    specification["operation_version"] = "utf8_sha256_v1"
    return repository.submit(request.idempotency_key, specification)


def submit_once(jobs, key, specification, conflict, admit):
    """The 202 body for a model run, queued at most once per idempotency key.

    A replay of the same specification returns the existing job without calling `admit`, which checks
    (and fails) a new submission, e.g. for a live capable worker. A key that names a different
    specification is a 409 with the `conflict` message, also when it is taken between check and insert."""
    try:
        job = jobs.existing_submission(key, specification)
        if job is None:
            admit()
            job = jobs.submit(key, specification)
    except ConflictError:
        fail("idempotency_conflict", conflict, 409)
    return job_accepted(job)
