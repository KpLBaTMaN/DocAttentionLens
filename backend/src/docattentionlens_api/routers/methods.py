from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from docattentionlens_api.errors import fail
from docattentionlens_api.schemas.contracts import MethodPage, PreparedMethodPage
from docattentionlens.methods.contracts import AnalysisRequest, AnalysisResult, MethodDescriptor
from docattentionlens.methods.registry import (get_method, list_methods,
    prepare_methods, prediction_capabilities, document_structure, table_structure)

router = APIRouter(prefix="/api/v1/methods", tags=["methods"])


class AnalysisPage(BaseModel):
    items: list[AnalysisResult]


@router.get("/saved-analyses",response_model=AnalysisPage)
def saved_analyses(request:Request,generation_capture_id:str|None=None,limit:int=Query(25,ge=1,le=100)):
    from docattentionlens_api.repositories.analyses import AnalysisRepository
    return AnalysisPage(items=AnalysisRepository(request.app.state.repository).list(generation_capture_id,limit))


@router.post("/analyses", response_model=AnalysisResult)
def analyse(body: AnalysisRequest, request: Request):
    from docattentionlens_api.services.analyses import calculate
    return calculate(request.app.state.repository, body)


@router.get("/analyses/{analysis_id}", response_model=AnalysisResult)
def analysis(analysis_id: str, request: Request):
    from docattentionlens_api.repositories.analyses import AnalysisRepository
    return AnalysisRepository(request.app.state.repository).get(analysis_id)


@router.get("", response_model=MethodPage)
def catalogue(family: str | None = None, evidence_kind: str | None = None, status: str | None = None):
    return MethodPage(items=[m for m in list_methods() if (family is None or m.family == family)
                            and (evidence_kind is None or m.evidence_kind == evidence_kind)
                            and (status is None or m.status == status)])


@router.get("/availability", response_model=PreparedMethodPage)
def availability(request: Request, generation_capture_id: str | None = None, retrieval_capture_id: str | None = None,
                 case_id: str | None = None):
    capabilities = set()
    generation = None
    structure = None
    table = None
    if case_id:
        # Document annotations come from the registered dataset, never from the client.
        documents = request.app.state.documents
        case = documents.case(case_id)
        snapshot = documents.snapshot(case.dataset_id)
        annotations, structure = document_structure(case, snapshot)
        capabilities.update(annotations)
        table_capabilities, table = table_structure(case, snapshot)
        capabilities.update(table_capabilities)
    if generation_capture_id:
        from docattentionlens_api.repositories.generations import GenerationRepository
        generation = GenerationRepository(request.app.state.repository).get(generation_capture_id)
        if case_id and generation.case_id != case_id:
            fail("invalid_reference", "Method context capture belongs to a different case")
        capabilities.update(prediction_capabilities(generation))
    if retrieval_capture_id:
        from docattentionlens_api.repositories.captures import CaptureRepository
        capture = CaptureRepository(request.app.state.repository).get(retrieval_capture_id)
        if generation_capture_id and generation.retrieval_capture_id != capture.capture_id:
            fail("invalid_reference", "Method context captures must share retrieval lineage")
        if case_id and capture.case_id != case_id:
            fail("invalid_reference", "Method context capture belongs to a different case")
        if capture.query_array and capture.pages and all(p.vector_array for p in capture.pages):
            capabilities.add("query_document_vectors")
        if capture.scorer:
            capabilities.add("scorer_configuration")
            if generation and generation.retrieval_capture_id==capture.capture_id:
                capabilities.add("retrieval_evidence")
        if capture.query_positions and all(p.positions for p in capture.pages):
            capabilities.update({"scored_masks", "position_identities"})
    return PreparedMethodPage(items=prepare_methods(capabilities, generation), generation_capture_id=generation_capture_id,
                              retrieval_capture_id=retrieval_capture_id, case_id=case_id, structure=structure,
                              table_structure=table if case_id else None)


@router.get("/{method_id}", response_model=MethodDescriptor)
def detail(method_id: str, variant_id: str | None = Query(default=None)):
    try:
        return get_method(method_id, variant_id)
    except KeyError:
        raise HTTPException(404, {"code": "unknown_method", "message": "Method variant not found."})
    except ValueError as exc:
        raise HTTPException(422, {"code": "variant_required", "message": str(exc)})
