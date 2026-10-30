from fastapi import APIRouter, Request, Query, Response
from pydantic import BaseModel
from docattentionlens_api.repositories.captures import CaptureRepository
from docattentionlens_api.repositories.generations import GenerationRepository
from docattentionlens_api.errors import fail
from docattentionlens_api.schemas.contracts import JobAccepted
from docattentionlens.generation.contracts import GenerationCapture, GenerationRequest, AttentionRow, MAX_NEW_TOKENS
from docattentionlens_api.services.jobs import submit_once
from docattentionlens_api.services.model_catalogue import generator_profile
from docattentionlens_api.services.plans import GenerationPlan, generation_plan
from docattentionlens.generation.inspection import attention_row, pixel_preview, source_view, received_column
from docattentionlens.methods.contracts import ReceivedAttention

router=APIRouter(prefix="/api/v1/generation",tags=["generation"])


class GenerationPage(BaseModel):
    items: list[GenerationCapture]
    next_cursor: str | None = None


@router.get("/plan",response_model=GenerationPlan)
def plan(request:Request,model_registration_id:str=Query("builtin",min_length=1,max_length=128)):
    from docattentionlens_api.services.imports import ModelStore
    checkpoint = ModelStore(request.app.state.settings.artifact_root).get(model_registration_id)
    profile = generator_profile(checkpoint) if checkpoint["role"] == "generation" else None
    if profile is None:
        fail("incompatible_model", "Choose a generation checkpoint with a compatible attention adapter", 422)
    return generation_plan(request.app.state.repository, profile, model_registration_id)


def job_specification(body:GenerationRequest,case,retrieval,profile,checkpoint):
    """The hashed job specification: identical inputs and options submit identical specifications."""
    spec={"kind":"generation_capture","case_id":case.case_id,"case_sha256":case.content_sha256,
        "retrieval_capture_id":retrieval.capture_id if retrieval else None,"retrieval_sha256":retrieval.content_sha256 if retrieval else None,
        "page_ids":body.page_ids,"adapter_version":profile.adapter_version,"model_revision":checkpoint["revision"]}
    # Preserve replay hashes for the original pinned retrieval-based request: defaults are omitted.
    from docattentionlens.generation.contracts import SYSTEM_PROMPT
    if body.model_registration_id != "builtin":
        spec["model_registration_id"] = body.model_registration_id
    if body.system_prompt != SYSTEM_PROMPT:
        spec["system_prompt"] = body.system_prompt
    if body.capture_full_attention:
        spec["capture_full_attention"] = True
        # Full-attention captures keep every generated step (generation schema 0.8.0). The marker
        # gives these jobs a different input hash from the earlier eight-step full-attention jobs.
        spec["full_attention_steps"] = "all"
    if body.image_token_limit != 512:
        spec["image_token_limit"] = body.image_token_limit
    return spec


@router.post("/runs",response_model=JobAccepted,status_code=202)
def run(body:GenerationRequest,request:Request):
    from docattentionlens_api.services.imports import ModelStore
    checkpoint = ModelStore(request.app.state.settings.artifact_root).get(body.model_registration_id)
    profile = generator_profile(checkpoint) if checkpoint["role"] == "generation" and checkpoint["capabilities"]["can_run"] else None
    if profile is None:
        fail("incompatible_model", "Choose a generation checkpoint with a compatible attention adapter", 422)
    retrieval=CaptureRepository(request.app.state.repository).get(body.retrieval_capture_id) if body.retrieval_capture_id else None
    case=request.app.state.documents.case(retrieval.case_id if retrieval else body.case_id)
    if len(set(body.page_ids))!=len(body.page_ids) or not set(body.page_ids)<=(set(p.page_id for p in retrieval.pages) if retrieval else set(case.page_ids)):
        fail("invalid_reference","Generator pages must be distinct pages of the case or selected retrieval capture")
    spec=job_specification(body,case,retrieval,profile,checkpoint)
    def admit():
        if not generation_plan(request.app.state.repository, profile, body.model_registration_id).available:
            fail("model_unavailable","Generator worker unavailable or lacks required memory",503)
    return submit_once(request.app.state.repository, body.idempotency_key, spec, "Key already names different generation inputs", admit)


@router.get("/captures",response_model=GenerationPage)
def captures(request:Request,case_id:str|None=None,limit:int=Query(10,ge=1,le=25),before:str|None=None):
    records=GenerationRepository(request.app.state.repository).list(case_id,limit+1,before)
    return GenerationPage(items=records[:limit],next_cursor=records[limit-1].capture_id if len(records)>limit else None)


class AnswerCheck(BaseModel):
    verdict: str
    basis: str
    matched_answer: str | None
    accepted_answers: list[str]
    interpretation: str


@router.get("/captures/{capture_id}/answer-check",response_model=AnswerCheck)
def answer_check(capture_id:str,request:Request):
    """Evaluation only: the generated answer against a table question's accepted answers."""
    from docattentionlens.evaluation.analysis import assess_against_accepted
    record=GenerationRepository(request.app.state.repository).get(capture_id)
    case=request.app.state.documents.case(record.case_id)
    if case.table_grounding is None:
        fail("not_applicable","Accepted answers are recorded for table questions only",404)
    return assess_against_accepted(record.answer,case.table_grounding.accepted_answers)


class TableRelatedCapture(BaseModel):
    capture_id: str
    case_id: str
    question_id: str
    query: str
    answer: str
    created_at: str
    answer_row: int | None
    answer_column: int | None
    # How this capture's question relates to the selected one: the same answer row or column, or neither.
    relation: str
    binding_status: str
    comparable: bool
    reason: str | None = None


class TableRelatedPage(BaseModel):
    table_id: str
    items: list[TableRelatedCapture]
    # The newest comparable capture of each other unflagged question: the default question-swap baseline.
    baseline_capture_ids: list[str]
    # Of those, the questions sharing an answer row or column: candidates for a matched-question contrast.
    matched_capture_ids: list[str]


@router.get("/captures/{capture_id}/table-related",response_model=TableRelatedPage)
def table_related(capture_id:str,request:Request,limit:int=Query(200,ge=1,le=500)):
    """Saved captures of questions on the same table, and whether each can join a table-aware comparison
    (same model and revision, same layer/head axes, identical image positions on the page)."""
    from docattentionlens.methods._table import same_page_geometry
    jobs=request.app.state.repository; repo=GenerationRepository(jobs); documents=request.app.state.documents
    record=repo.get(capture_id); case=documents.case(record.case_id); ground=case.table_grounding
    if ground is None:
        fail("not_applicable","Related table captures exist for table questions only",404)
    with jobs.connect() as conn:
        rows=conn.execute("SELECT g.capture_id,g.case_id,g.created_at FROM generation_captures g JOIN cases c ON c.case_id=g.case_id "
                          "WHERE c.content->'table_grounding'->>'table_id'=%s AND g.capture_id<>%s ORDER BY g.created_at DESC,g.capture_id DESC LIMIT %s",
                          (ground.table_id,capture_id,limit)).fetchall()
    snapshot=documents.snapshot(case.dataset_id)
    table=next((t for t in snapshot.tables if t.table_id==ground.table_id),None)
    rows_index={a.native_id:a.index for a in table.rows} if table else {}
    columns_index={a.native_id:a.index for a in table.columns} if table else {}
    first=lambda ids,index:next((index[i] for i in ids if i in index),None)
    mine=(first(ground.answer_row_ids,rows_index),first(ground.answer_column_ids,columns_index))
    items=[]
    for row in rows:
        other_case=documents.case(row["case_id"]); other=other_case.table_grounding
        try:
            capture=repo.get(row["capture_id"])
        except Exception:
            continue
        position=(first(other.answer_row_ids,rows_index),first(other.answer_column_ids,columns_index))
        relation=("same_question" if other_case.question_id==case.question_id else "same_row" if position[0]==mine[0] and position[1]!=mine[1]
                  else "same_column" if position[1]==mine[1] and position[0]!=mine[0] else "other")
        reason=None
        if other.table_sha256!=ground.table_sha256: reason="Reads a different table revision"
        elif any(capture.provenance.get(k)!=record.provenance.get(k) for k in ("model_id","model_revision")): reason="Different model or revision"
        elif (capture.layer_count,capture.query_head_count)!=(record.layer_count,record.query_head_count): reason="Different layer or head axes"
        elif not same_page_geometry(record,capture): reason="Different page resolution"
        elif not any(s.step==0 for s in capture.steps): reason="No captured first prediction"
        items.append(TableRelatedCapture(capture_id=capture.capture_id,case_id=other_case.case_id,question_id=other_case.question_id,
            query=other_case.query,answer=capture.answer or "",created_at=row["created_at"].isoformat(),answer_row=position[0],
            answer_column=position[1],relation=relation,binding_status=other.binding_status,comparable=reason is None,reason=reason))
    baseline,seen=[],set()
    for item in items:
        if item.comparable and item.relation!="same_question" and item.binding_status!="needs_source_qa_review" and item.question_id not in seen:
            seen.add(item.question_id); baseline.append(item.capture_id)
    matched=[i.capture_id for i in items if i.capture_id in baseline and i.relation in ("same_row","same_column")]
    return TableRelatedPage(table_id=ground.table_id,items=items,baseline_capture_ids=baseline,matched_capture_ids=matched)


@router.get("/captures/{capture_id}",response_model=GenerationCapture)
def capture(capture_id:str,request:Request):
    return GenerationRepository(request.app.state.repository).get(capture_id)


@router.get("/captures/{capture_id}/rows",response_model=AttentionRow)
def row(capture_id:str,request:Request,step:int=Query(0,ge=0,le=MAX_NEW_TOKENS-1,description="Prediction step: the first eight on ordinary captures, every generated step on full-attention captures"),layer:int=Query(0,ge=0,le=127),head:int=Query(0,ge=0,le=127),
        source_index:int|None=Query(None,ge=0,le=1535,description="Retained question-token source row, or any prompt position on a full-attention capture; replaces the prediction step")):
    repo=GenerationRepository(request.app.state.repository)
    record=repo.get(capture_id)
    if source_index is not None:
        try:
            record,_=source_view(record,source_index); step=0
        except ValueError as exc:
            fail("uncaptured_selection",str(exc),422)
    selected=next((s for s in record.steps if s.step==step),None)
    if selected is None:
        fail("uncaptured_selection","This prediction step was not captured",422)
    try:
        values=repo.store.source_array(record,selected.array)
    except (OSError,ValueError):
        fail("artifact_incomplete","Attention payload is missing or failed integrity verification",409)
    try:
        return attention_row(record,values,step,layer,head)
    except ValueError as exc:
        fail("invalid_reference",str(exc))


@router.get("/captures/{capture_id}/received",response_model=ReceivedAttention)
def received(capture_id:str,request:Request,target_index:int=Query(ge=0,le=1599),layer:int=Query(0,ge=0,le=127),head:int=Query(0,ge=0,le=127)):
    """Weight one target receives from every retained prediction and question-token row."""
    repo=GenerationRepository(request.app.state.repository)
    record=repo.get(capture_id)
    try:
        return received_column(record,target_index,layer,head,repo.store.verified_reader(record))
    except ValueError as exc:
        fail("invalid_reference",str(exc))


@router.get("/captures/{capture_id}/pages/{page_id}/input-image")
def image(capture_id:str,page_id:str,request:Request):
    repo=GenerationRepository(request.app.state.repository); record=repo.get(capture_id)
    page=next((p for p in record.context.pages if p.page_id==page_id),None)
    if page is None:
        fail("invalid_reference","Page was not supplied to this generator",404)
    if page.pixel_array is None:
        fail("unsupported_representation","This capture consumed document text; it has no processed page-image payload",422)
    try:
        ip=record.provenance["image_processor"]
        data=pixel_preview(repo.store.array(record,page.pixel_array),page.grid_thw,ip["mean"],ip["std"],ip["rescale"])
    except (OSError,ValueError,KeyError):
        fail("artifact_incomplete","Consumed image payload is missing or incompatible",409)
    return Response(data,media_type="image/png",headers={"Cache-Control":"private, max-age=31536000, immutable"})
