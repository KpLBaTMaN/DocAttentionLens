"""Prepare, inspect and explicitly run immutable fixed-evidence document experiments."""
from fastapi import APIRouter,Request,Query,Response
from pydantic import BaseModel
from docattentionlens_api.errors import fail
from docattentionlens_api.repositories.generations import GenerationRepository
from docattentionlens_api.repositories.document_studies import DocumentStudyRepository
from docattentionlens_api.services.document_context import annotations
from docattentionlens_api.services.jobs import submit_once
from docattentionlens_api.services.plans import generation_plan
from docattentionlens_api.schemas.contracts import JobAccepted
from docattentionlens.generation.contracts import GENERATORS
from docattentionlens.experiments.contracts import ExperimentRunRequest
from docattentionlens.experiments.document_studies import StudyRequest,DocumentStudy,prepare,VERSION,masked_png,validate_execution_code,execution_code_hashes

# A preview re-renders only the frozen input (masking and text serialisation), so only that code must
# match the plan. Analysis code (e.g. integrated gradients' notices) may change without breaking the
# previews of saved studies; running a plan still requires the complete execution code to match.
PREVIEW_CODE=("experiments/document_studies.py","generation/document_inputs.py")


def validate_preview_code(study):
    stored=study.provenance.get("execution_code_sha256") or {}
    current=execution_code_hashes()
    if any(stored.get(name)!=current[name] for name in PREVIEW_CODE):
        raise ValueError("Arm preview code differs from the code that froze this plan; the exact input cannot be re-rendered")

router=APIRouter(prefix="/api/v1/document-studies",tags=["document studies"])


class StudyPage(BaseModel):
    items:list[DocumentStudy]


@router.post("",response_model=DocumentStudy,status_code=201)
def create(body:StudyRequest,request:Request):
    jobs=request.app.state.repository; record=GenerationRepository(jobs).get(body.generation_capture_id)
    if record.provenance.get("model_registration_id") not in GENERATORS: fail("unsupported_model","Document experiments require a pinned built-in generator")
    if record.provenance.get("experiment_protocol_id") or record.provenance.get("document_study_id") or record.context.representation!="image":
        fail("unsupported_capture","Prepare document experiments from an original image-input case capture")
    try: study=prepare(body,record,annotations(jobs,record))
    except (KeyError,ValueError) as exc: fail("invalid_parameters",str(exc))
    return DocumentStudyRepository(jobs).admit(study)


@router.get("",response_model=StudyPage)
def records(request:Request,case_id:str|None=None,limit:int=Query(25,ge=1,le=100)):
    return StudyPage(items=DocumentStudyRepository(request.app.state.repository).list(case_id,limit))


@router.get("/{study_id}",response_model=DocumentStudy)
def record(study_id:str,request:Request): return DocumentStudyRepository(request.app.state.repository).get(study_id)


@router.get("/{study_id}/arms/{arm_name}/pages/{page_id}")
def preview(study_id:str,arm_name:str,page_id:str,request:Request):
    study=DocumentStudyRepository(request.app.state.repository).get(study_id)
    try: validate_preview_code(study)
    except ValueError as exc: fail("version_mismatch",str(exc),409)
    arm=next((a for a in study.arms if a.name==arm_name),None)
    if arm is None or page_id not in study.provenance["page_ids"]: fail("invalid_reference","Unknown document study arm or page")
    from docattentionlens_api.repositories.experiments import source_inputs
    page,raw,elements=source_inputs(request.app.state.documents,study.case,[page_id])[0]
    if arm.representation!="image":
        from docattentionlens.generation.document_inputs import serialize
        return Response(serialize(page,elements,arm.representation)[0],media_type="text/plain")
    if arm.mask_box and arm.page_id==page_id and arm.strength is None: raw=masked_png(raw,arm.mask_box,arm.replacement)
    return Response(raw,media_type="image/png" if raw.startswith(b"\x89PNG") else "image/jpeg")


@router.post("/{study_id}/runs",response_model=JobAccepted,status_code=202)
def run(study_id:str,body:ExperimentRunRequest,request:Request):
    jobs=request.app.state.repository; study=DocumentStudyRepository(jobs).get(study_id)
    if study.kind!="prepared": fail("invalid_reference","Run a prepared immutable document experiment")
    try: validate_execution_code(study)
    except ValueError as exc: fail("version_mismatch",str(exc),409)
    profile=GENERATORS[study.model_registration_id]
    spec={"kind":"document_experiment","study_id":study_id,"study_sha256":study.content_sha256,"operation_version":VERSION,
          "generator_adapter":profile.adapter_version,"generator_revision":profile.revision}
    def admit():
        if not any(w["available"] and w["resource_class"]=="gpu" and VERSION in w["capabilities"].get("document_experiment_operations",[]) for w in jobs.workers()):
            fail("model_unavailable","A GPU worker with the document experiment adapter is required",503)
        if not generation_plan(jobs,profile).available: fail("model_unavailable","A compatible GPU worker and the selected model's memory budget are required",503)
    return submit_once(jobs,body.idempotency_key,spec,"Key names different document experiment inputs",admit)
