from datetime import UTC, datetime
from uuid import uuid4
from fastapi import APIRouter, Request, Response, Query
from pydantic import BaseModel
from docattentionlens_api.errors import fail
from docattentionlens_api.repositories.experiments import ExperimentRepository, source_inputs
from docattentionlens_api.repositories.captures import CaptureStore
from docattentionlens_api.repositories.generations import GenerationStore
from docattentionlens_api.schemas.contracts import JobAccepted
from docattentionlens_api.services.jobs import submit_once
from docattentionlens_api.services.plans import generation_plan
from docattentionlens.experiments.contracts import ExperimentDraftRequest, Review, ExperimentRecord, ExperimentRunRequest, VERSION
from docattentionlens.experiments.relayout import build_draft, freeze
from docattentionlens.generation.contracts import AttentionRow, GenerationCapture
from docattentionlens.generation.inspection import attention_row, pixel_preview
from docattentionlens.retrieval.contracts import RetrievalCapture, ADAPTER_VERSION as RETRIEVAL_ADAPTER, MODEL_REVISION as RETRIEVAL_REVISION
from docattentionlens.generation.contracts import ADAPTER_VERSION as GENERATOR_ADAPTER, MODEL_REVISION as GENERATOR_REVISION

router=APIRouter(prefix="/api/v1/experiments",tags=["experiments"])

class ExperimentPage(BaseModel):
    items:list[ExperimentRecord]
    next_cursor:str|None=None

@router.post("/drafts",response_model=ExperimentRecord,status_code=201)
def draft(body:ExperimentDraftRequest,request:Request):
    case=request.app.state.documents.case(body.case_id)
    if body.page_id not in case.page_ids: fail("invalid_reference","Changed page must belong to the case")
    repo=ExperimentRepository(request.app.state.repository)
    try:
        page,raw,elements=source_inputs(request.app.state.documents,case,[body.page_id])[0]
        record,assets=build_draft(body,case,page,raw,elements)
        digest=repo.store.put(record,assets)
    except (OSError,ValueError) as exc: fail("invalid_manipulation",str(exc))
    repo.admit(record,digest)
    return repo.get(record.experiment_id)

@router.get("",response_model=ExperimentPage)
def records(request:Request,case_id:str|None=None,limit:int=Query(10,ge=1,le=25),before:str|None=None):
    items=ExperimentRepository(request.app.state.repository).list(case_id,limit+1,before)
    return ExperimentPage(items=items[:limit],next_cursor=items[limit-1].experiment_id if len(items)>limit else None)

@router.get("/{experiment_id}",response_model=ExperimentRecord)
def record(experiment_id:str,request:Request): return ExperimentRepository(request.app.state.repository).get(experiment_id)

@router.post("/{experiment_id}/review",response_model=ExperimentRecord,status_code=201)
def review(experiment_id:str,body:Review,request:Request):
    repo=ExperimentRepository(request.app.state.repository); draft=repo.get(experiment_id)
    if draft.record_kind!="experiment_draft": fail("invalid_reference","Review a draft; finalized records are immutable")
    for arm in draft.arms:
        try: repo.store.image(draft,arm.name)
        except (OSError,ValueError): fail("artifact_incomplete","Variant asset failed integrity verification",409)
    final=freeze(draft.model_copy(update={"experiment_id":"experiment_"+uuid4().hex,"record_kind":"experiment_protocol","created_at":datetime.now(UTC).isoformat(),"review":body,
        "dependencies":[*draft.dependencies,draft.experiment_id],"provenance":{**draft.provenance,"asset_owner":draft.experiment_id,"review_origin":"submitted reviewer declaration; not independently validated"}}))
    digest=repo.store.put(final); repo.admit(final,digest); return repo.get(final.experiment_id)

@router.post("/{experiment_id}/runs",response_model=JobAccepted,status_code=202)
def run(experiment_id:str,body:ExperimentRunRequest,request:Request):
    repo=ExperimentRepository(request.app.state.repository); protocol=repo.get(experiment_id)
    if protocol.record_kind!="experiment_protocol" or protocol.review is None: fail("review_required","Finalize semantic, relationship, readability and control review before running")
    spec={"kind":"layout_experiment","experiment_id":experiment_id,"protocol_sha256":protocol.content_sha256,"operation_version":VERSION,"retrieval_adapter":RETRIEVAL_ADAPTER,"retrieval_revision":RETRIEVAL_REVISION,"generator_adapter":GENERATOR_ADAPTER,"generator_revision":GENERATOR_REVISION}
    jobs=request.app.state.repository
    def admit():
        if not generation_plan(jobs).available: fail("model_unavailable","A GPU worker with 12 GiB available is required",503)
    return submit_once(jobs,body.idempotency_key,spec,"Key names different experiment inputs",admit)

@router.get("/{experiment_id}/arms/{arm}/image")
def image(experiment_id:str,arm:str,request:Request):
    repo=ExperimentRepository(request.app.state.repository); record=repo.get(experiment_id)
    if arm not in {a.name for a in record.arms}: fail("invalid_reference","Unknown experiment arm",422)
    try: data=repo.store.image(record,arm)
    except (OSError,ValueError): fail("artifact_incomplete","Saved page is missing or failed integrity verification",409)
    return Response(data,media_type="image/png",headers={"Cache-Control":"private, max-age=31536000, immutable"})

def captured(request,identifier,arm,kind):
    record=ExperimentRepository(request.app.state.repository).get(identifier)
    outcome=next((o for o in record.outcomes if o["arm"]==arm),None)
    if not outcome or kind not in outcome: fail("uncaptured_selection","This arm has no saved capture of the selected kind")
    ref=outcome[kind]; store=(GenerationStore if kind=="generation" else CaptureStore)(request.app.state.settings.artifact_root)
    try:
        value=store.read(ref["capture_id"],ref["sha256"])
        if value.case_id!=next(a.case.case_id for a in record.arms if a.name==arm): raise ValueError("Capture lineage differs")
        return store,value
    except (OSError,ValueError,KeyError): fail("artifact_incomplete","Saved model capture failed integrity verification",409)

@router.get("/{experiment_id}/arms/{arm}/retrieval",response_model=RetrievalCapture)
def retrieval(experiment_id:str,arm:str,request:Request): return captured(request,experiment_id,arm,"retrieval")[1]

@router.get("/{experiment_id}/arms/{arm}/generation",response_model=GenerationCapture)
def generation(experiment_id:str,arm:str,request:Request): return captured(request,experiment_id,arm,"generation")[1]

@router.get("/{experiment_id}/arms/{arm}/attention",response_model=AttentionRow)
def attention(experiment_id:str,arm:str,request:Request,step:int=Query(0,ge=0,le=7),layer:int=Query(0,ge=0,le=127),head:int=Query(0,ge=0,le=127)):
    store,cap=captured(request,experiment_id,arm,"generation")
    selected=next((s for s in cap.steps if s.step==step),None)
    if not selected: fail("uncaptured_selection","Prediction step was not captured")
    try: return attention_row(cap,store.array(cap,selected.array),step,layer,head)
    except (OSError,ValueError): fail("artifact_incomplete","Attention payload failed verification",409)

@router.get("/{experiment_id}/arms/{arm}/input-image/{page_id}")
def consumed_image(experiment_id:str,arm:str,page_id:str,request:Request):
    store,cap=captured(request,experiment_id,arm,"generation")
    page=next((p for p in cap.context.pages if p.page_id==page_id),None)
    if not page: fail("invalid_reference","Page was omitted from generation",404)
    try:
        ip=cap.provenance["image_processor"]
        data=pixel_preview(store.array(cap,page.pixel_array),page.grid_thw,ip["mean"],ip["std"],ip["rescale"])
    except (OSError,ValueError,KeyError): fail("artifact_incomplete","Consumed pixels failed verification",409)
    return Response(data,media_type="image/png")
