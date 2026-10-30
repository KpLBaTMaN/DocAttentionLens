from fastapi import APIRouter, Query, Request
from pydantic import BaseModel
from docattentionlens_api.repositories.captures import CaptureRepository
from docattentionlens_api.errors import fail, readable
from docattentionlens_api.schemas.contracts import JobAccepted
from docattentionlens_api.services.jobs import submit_once
from docattentionlens_api.services.plans import RetrievalPlan, retrieval_plan, retriever_profile
from docattentionlens.retrieval.contracts import (
    RetrievalRequest, RetrievalCapture, SimilarityRow, SimilarityColumn, RemovalRequest, RemovalResult,
    RetrievalReceivedRequest, RetrievalAttentionRequest,
)
from docattentionlens.retrieval.inspection import similarity_row, similarity_column, fixed_vector_removal
from docattentionlens.methods.contracts import Calculation, MethodDescriptor, ReceivedAttention

router = APIRouter(prefix="/api/v1/retrieval",tags=["retrieval"])


@router.get("/attention-methods", response_model=list[MethodDescriptor])
def attention_methods():
    from docattentionlens.retrieval.attention import methods
    return methods()


@router.post("/captures/{capture_id}/attention", response_model=Calculation)
def attention(capture_id: str, body: RetrievalAttentionRequest, request: Request):
    from docattentionlens.retrieval.attention import calculate
    repo = CaptureRepository(request.app.state.repository)
    record = repo.get(capture_id)
    read_verified = repo.store.verified_reader(record)
    try:
        from docattentionlens_api.services.document_context import annotations
        from docattentionlens.methods.registry import get_method
        method=get_method(body.method_id)
        table_aware=method.perspective=="table_aware"
        # Table cases have no layout elements; their verified structure comes from the frozen table revision.
        document=annotations(request.app.state.repository,record) if not table_aware and any(r.endswith("_annotations") for r in method.requires) else None
        table=None
        if table_aware:
            from docattentionlens_api.services.table_context import table_inputs
            table=table_inputs(request.app.state.repository,record)
        return calculate(record, body, read_verified, document, table)
    except OSError:
        fail("artifact_incomplete", "Attention payload is missing", 409)
    except (ValueError,KeyError) as exc:
        fail("invalid_parameters", readable(exc), 422)


class CapturePage(BaseModel):
    items: list[RetrievalCapture]
    next_cursor: str | None = None


@router.get("/plan",response_model=RetrievalPlan)
def plan(request:Request,model_registration_id:str=Query("builtin-retrieval",max_length=128),capture_joint_attention:bool=False,joint_answer:bool=False):
    return retrieval_plan(request.app.state.repository, model_registration_id, capture_joint_attention, joint_answer)


@router.post("/runs",response_model=JobAccepted,status_code=202)
def run(body:RetrievalRequest,request:Request):
    from docattentionlens_api.services.imports import ModelStore
    model = ModelStore(request.app.state.settings.artifact_root).get(body.model_registration_id)
    if model["role"] != "retrieval":
        fail("incompatible_model", "Choose a retrieval model with an installed retrieval adapter", 422)
    profile = retriever_profile(model["registration_id"])
    case = request.app.state.documents.case(body.case_id)
    if len(set(body.page_ids))!=len(body.page_ids) or not set(body.page_ids)<=set(case.page_ids):
        fail("invalid_reference","Candidate pages must be distinct frozen case inputs")
    # ColPali specifications are unchanged, so earlier idempotency keys still match.
    specification = {"kind":"retrieval_capture","case_id":case.case_id,"page_ids":body.page_ids,"adapter_version":profile["adapter_version"],"model_revision":profile["revision"],"case_sha256":case.content_sha256}
    if body.capture_attention:
        if body.attention_page_id not in body.page_ids:
            fail("invalid_reference", "Select one candidate page for document attention", 422)
        specification.update(capture_attention=True, attention_page_id=body.attention_page_id, attention_adapter_version=profile["attention_adapter_version"])
        if body.capture_joint_attention:
            specification["capture_joint_attention"] = True
            specification["joint_capture_policy"] = "retain_document_v1"
            if body.joint_answer_source:
                # The case's reference answer enters only this diagnostic pass; refuse a case without one.
                if not (case.reference_answer or "").strip():
                    fail("invalid_parameters", "This case has no reference answer to add to the diagnostic prompt", 422)
                specification["joint_answer_source"] = body.joint_answer_source
    elif body.attention_page_id:
        fail("invalid_parameters", "Document attention page requires attention capture", 422)
    if body.capture_joint_attention and not body.capture_attention:
        fail("invalid_parameters", "Joint attention requires attention capture", 422)
    if body.joint_answer_source and not body.capture_joint_attention:
        fail("invalid_parameters", "A joint answer requires the diagnostic joint capture", 422)
    def admit():
        plan = retrieval_plan(request.app.state.repository, model["registration_id"])
        if not plan.available:
            fail("model_unavailable",f"{profile['name']} GPU worker is unavailable or lacks required memory",503)
        if body.capture_attention:
            if not plan.attention_available:
                fail("model_unavailable", "The GPU worker lacks the retrieval attention adapter", 503)
            from shutil import disk_usage
            from docattentionlens.retrieval.contracts import attention_capture_budget
            required=attention_capture_budget(profile,body.capture_joint_attention,bool(body.joint_answer_source))[1]
            if disk_usage(request.app.state.settings.artifact_root).free < required:
                fail("capture_budget_exceeded", f"Retriever attention needs at least {required//1024**2} MiB free artifact storage", 422)
    return submit_once(request.app.state.repository, body.idempotency_key, specification, "Key already names different retrieval inputs", admit)


@router.get("/captures",response_model=CapturePage)
def captures(request:Request,case_id:str|None=None,limit:int=Query(10,ge=1,le=25),before:str|None=None):
    records = CaptureRepository(request.app.state.repository).list(case_id,limit+1,before)
    return CapturePage(items=records[:limit],next_cursor=records[limit-1].capture_id if len(records)>limit else None)


@router.get("/captures/{capture_id}",response_model=RetrievalCapture)
def capture(capture_id:str,request:Request):
    return CaptureRepository(request.app.state.repository).get(capture_id)


def arrays(request,capture_id,page_id):
    repo = CaptureRepository(request.app.state.repository)
    record = repo.get(capture_id)
    page = next((p for p in record.pages if p.page_id==page_id),None)
    if page is None:
        fail("invalid_reference","Page is outside captured candidate set",404)
    try:
        return record,page,repo.store.array(record,record.query_array),repo.store.array(record,page.vector_array)
    except (OSError,ValueError):
        fail("artifact_incomplete","Numeric payload is missing or failed integrity verification",409)


@router.get("/captures/{capture_id}/rows",response_model=SimilarityRow)
def row(capture_id:str,request:Request,page_id:str,query_index:int=Query(0,ge=0,le=255)):
    record,page,q,d = arrays(request,capture_id,page_id)
    try:
        return similarity_row(record,page,q,d,query_index)
    except ValueError as exc:
        fail("invalid_reference",str(exc))


@router.get("/captures/{capture_id}/columns",response_model=SimilarityColumn)
def column(capture_id:str,request:Request,page_id:str,position_index:int=Query(ge=0,le=1099)):
    """One page vector's signed similarity to every query vector."""
    record,page,q,d = arrays(request,capture_id,page_id)
    try:
        return similarity_column(record,page,q,d,position_index)
    except ValueError as exc:
        fail("invalid_reference",str(exc))


@router.post("/captures/{capture_id}/attention/received",response_model=ReceivedAttention)
def attention_received(capture_id: str, body: RetrievalReceivedRequest, request: Request):
    """Weight one encoding target receives from every active source row."""
    from docattentionlens.retrieval.attention import received_column
    repo = CaptureRepository(request.app.state.repository)
    record = repo.get(capture_id)
    try:
        return received_column(record, body, repo.store.verified_reader(record))
    except ValueError as exc:
        fail("invalid_parameters", str(exc), 422)


@router.post("/captures/{capture_id}/fixed-vector-removal",response_model=RemovalResult)
def removal(capture_id:str,body:RemovalRequest,request:Request):
    record,page,q,d = arrays(request,capture_id,body.page_id)
    try:
        return fixed_vector_removal(record,page,q,d,body.removed_position_indices)
    except ValueError as exc:
        fail("invalid_reference",str(exc))
