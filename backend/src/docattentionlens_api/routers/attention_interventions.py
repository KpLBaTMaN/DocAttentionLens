"""Prepare, review, run and reopen direct attention interventions."""
from fastapi import APIRouter,Request,Query
from pydantic import BaseModel
from docattentionlens_api.errors import fail,readable
from docattentionlens_api.repositories.generations import GenerationRepository
from docattentionlens_api.repositories.captures import CaptureRepository
from docattentionlens_api.repositories.attention_interventions import AttentionExperimentRepository
from docattentionlens_api.schemas.contracts import JobAccepted
from docattentionlens_api.services.jobs import submit_once
from docattentionlens_api.services.plans import generation_plan,retrieval_plan
from docattentionlens.experiments.attention_interventions import AttentionExperimentRequest,AttentionExperiment,prepare,validate_code,VERSION
from docattentionlens.experiments.contracts import ExperimentRunRequest
from docattentionlens.generation.contracts import GENERATORS

router=APIRouter(prefix="/api/v1/attention-interventions",tags=["attention interventions"])
class AttentionExperimentPage(BaseModel):
    items:list[AttentionExperiment]

@router.post("",response_model=AttentionExperiment,status_code=201)
def create(body:AttentionExperimentRequest,request:Request):
    jobs=request.app.state.repository
    capture=(GenerationRepository(jobs) if body.capture_id.startswith("generation_") else CaptureRepository(jobs)).get(body.capture_id)
    case=request.app.state.documents.case(capture.case_id)
    try: record=prepare(body,capture,case)
    except (KeyError,ValueError) as exc: fail("invalid_parameters",readable(exc))
    return AttentionExperimentRepository(jobs).admit(record)

@router.get("",response_model=AttentionExperimentPage)
def records(request:Request,case_id:str|None=None,limit:int=Query(25,ge=1,le=100)):
    return AttentionExperimentPage(items=AttentionExperimentRepository(request.app.state.repository).list(case_id,limit))

@router.get("/{experiment_id}",response_model=AttentionExperiment)
def record(experiment_id:str,request:Request):
    return AttentionExperimentRepository(request.app.state.repository).get(experiment_id)

@router.post("/{experiment_id}/runs",response_model=JobAccepted,status_code=202)
def run(experiment_id:str,body:ExperimentRunRequest,request:Request):
    jobs=request.app.state.repository; experiment=AttentionExperimentRepository(jobs).get(experiment_id)
    if experiment.kind!="prepared": fail("invalid_reference","Run a prepared attention experiment")
    try: validate_code(experiment)
    except ValueError as exc: fail("version_mismatch",str(exc),409)
    if experiment.role=="generation":
        available=generation_plan(jobs,GENERATORS[experiment.model_registration_id]).available
    else:
        available=retrieval_plan(jobs,experiment.model_registration_id).available
    spec={"kind":"attention_intervention","experiment_id":experiment_id,"experiment_sha256":experiment.content_sha256,"operation_version":VERSION}
    def admit():
        if not available or not any(w["available"] and VERSION in w["capabilities"].get("attention_intervention_operations",[]) for w in jobs.workers()):
            fail("model_unavailable","A compatible GPU worker with the attention-intervention adapter is required",503)
    return submit_once(jobs,body.idempotency_key,spec,"Key names different attention experiment inputs",admit)
