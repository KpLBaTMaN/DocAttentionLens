from datetime import UTC, datetime
from uuid import uuid4
from fastapi import APIRouter, Request, Query
from pydantic import BaseModel
from docattentionlens import __version__
from docattentionlens.evaluation.contracts import (
    StudyRequest, StudyRecord, StudyRunRequest, ObservationRequest,
    AnswerReviewRequest, UtilityTrialRequest, EvaluationEvidence,
)
from docattentionlens.evaluation.analysis import (
    assess_answer, result_measurements, repeat_summary, utility_summary,
)
from docattentionlens.experiments.contracts import ExperimentRunRequest
from docattentionlens_api.errors import fail
from docattentionlens_api.repositories.experiments import ExperimentRepository
from docattentionlens_api.repositories.evaluation import EvaluationRepository
from docattentionlens_api.routers.experiments import run as run_experiment
from docattentionlens_api.schemas.contracts import JobAccepted

router = APIRouter(prefix='/api/v1/evaluation/studies', tags=['evaluation'])


class StudyPage(BaseModel):
    items: list[StudyRecord]
    next_cursor: str | None = None


def repository(request):
    return EvaluationRepository(request.app.state.repository)


def frozen_case(study, protocol_id, repetition=None):
    frozen = next((c for c in study.frozen_cases if c['protocol_id'] == protocol_id), None)
    if frozen is None or repetition is not None and repetition > study.specification.repetitions:
        fail('invalid_reference', 'Select a frozen study protocol and an in-range repetition')
    return frozen


def validated_result(request, study, identifier):
    result = ExperimentRepository(request.app.state.repository).get(identifier)
    if result.record_kind != 'experiment_result':
        fail('invalid_reference', 'Evaluation requires a completed measured result')
    frozen = frozen_case(study, result.provenance.get('protocol_id'))
    if result.provenance.get('protocol_sha256') != frozen['protocol_sha256']:
        fail('version_mismatch', 'Measured result differs from the frozen study protocol', 409)
    return result, frozen


@router.post('', response_model=StudyRecord, status_code=201)
def create(body: StudyRequest, request: Request):
    documents = request.app.state.documents
    experiments = ExperimentRepository(request.app.state.repository)
    frozen = []
    splits = {}
    for entry in body.cases:
        protocol = experiments.get(entry.protocol_id)
        if protocol.record_kind != 'experiment_protocol' or protocol.review is None:
            fail('review_required', 'Select finalized reviewed experiment protocols')
        case = documents.case(protocol.source_case_id)
        question = documents.resource('questions', case.question_id)[1]
        if entry.rubric == 'boolean_strict_v1' and question.question_type != 'true_or_false':
            fail('invalid_rubric', 'Use the boolean rubric only for true-or-false questions')
        if entry.rubric == 'normalized_exact_v1' and question.question_type != 'short':
            fail('invalid_rubric', 'Use exact matching only for short answers; long answers need human adjudication')
        try:
            assess_answer('', case.reference_answer, entry.rubric)
        except ValueError as exc:
            fail('invalid_rubric', str(exc))
        if case.document_id in splits and splits[case.document_id] != entry.split:
            fail('document_leakage', 'Discovery and validation must use disjoint documents')
        splits[case.document_id] = entry.split
        support = set(case.support_element_ids) & set(protocol.protocol.element_ids)
        frozen.append({**entry.model_dump(mode='json'), 'case_id': case.case_id,
                       'case_sha256': case.content_sha256, 'document_id': case.document_id,
                       'query': case.query, 'question_type': question.question_type,
                       'reference_answer': case.reference_answer, 'support_page_ids': case.support_page_ids,
                       'support_element_ids': case.support_element_ids,
                       'moved_designated_support_elements': sorted(support),
                       'changed_page_id': protocol.protocol.page_id,
                       'page_numbers': {pid: documents.resource('pages', pid)[1].page_number
                                        for pid in protocol.protocol.candidate_page_ids},
                       'protocol_sha256': protocol.content_sha256, 'stage': protocol.protocol.stage})
    if body.study_kind == 'researcher_evaluation' and not any(c['moved_designated_support_elements'] for c in frozen):
        fail('support_case_required', 'Include a reviewed protocol moving designated supporting elements')
    record = StudyRecord(study_id='study_' + uuid4().hex, created_at=datetime.now(UTC).isoformat(),
                         producer_version=__version__, dependencies=[c['protocol_id'] for c in frozen],
                         specification=body, frozen_cases=frozen, limitations=[
                             'Review roles are submitted declarations, not independently verified identities.',
                             'Repeated seed-zero acquisitions test reproducibility; they are not independent sampled questions.',
                             'Candidate-scoped retrieval and page-level context coverage are not causal grounding.',
                             'No inferential statistics, population claims or automatic semantic answer judgement.',
                             'Baseline access and task outcomes are observer reports; matched access requires researcher review.',
                         ])
    return repository(request).admit(record)


@router.get('', response_model=StudyPage)
def studies(request: Request, limit: int = Query(10, ge=1, le=25), before: str | None = None):
    items = repository(request).list(limit + 1, before)
    return StudyPage(items=items[:limit], next_cursor=items[limit - 1].study_id if len(items) > limit else None)


@router.get('/{study_id}', response_model=StudyRecord)
def study(study_id: str, request: Request):
    return repository(request).get(study_id)


@router.post('/{study_id}/runs', response_model=JobAccepted, status_code=202)
def run(study_id: str, body: StudyRunRequest, request: Request):
    repo = repository(request)
    plan = repo.get(study_id)
    frozen_case(plan, body.protocol_id, body.repetition)
    existing = next((r for r in repo.runs(study_id) if r['protocol_id'] == body.protocol_id
                     and r['repetition'] == body.repetition), None)
    if existing and existing['job_id'] is None:
        fail('repetition_conflict', 'This slot contains an explicitly attached result', 409)
    # Stable keys make retries safe even after a crash between job submission and membership admission.
    key = f'{study_id}:{body.protocol_id}:{body.repetition}'
    job = run_experiment(body.protocol_id, ExperimentRunRequest(idempotency_key=key), request)
    repo.assign(study_id, body.protocol_id, body.repetition, job_id=job['job_id'])
    return job


@router.post('/{study_id}/observations', status_code=201)
def observe(study_id: str, body: ObservationRequest, request: Request):
    repo = repository(request)
    plan = repo.get(study_id)
    frozen_case(plan, body.protocol_id, body.repetition)
    result, frozen = validated_result(request, plan, body.result_id)
    if frozen['protocol_id'] != body.protocol_id:
        fail('invalid_reference', 'Result belongs to another study protocol')
    retrospective = datetime.fromisoformat(result.created_at) < datetime.fromisoformat(plan.created_at)
    if retrospective and plan.specification.study_kind != 'engineering_pilot':
        fail('prospective_plan_required', 'Researcher evaluations cannot attach results acquired before the plan was frozen')
    return repo.assign(study_id, body.protocol_id, body.repetition, result_id=result.experiment_id,
                       retrospective=retrospective)


def admitted_results(request, study_id):
    repo = repository(request)
    plan = repo.get(study_id)
    rows = repo.runs(study_id)
    results = []
    for row in rows:
        identifier = row['result_id'] or (row['output_resource_id'] if row['state'] == 'completed' else None)
        if identifier:
            result, frozen = validated_result(request, plan, identifier)
            if frozen['protocol_id'] != row['protocol_id']:
                fail('version_mismatch', 'Study execution protocol differs', 409)
            results.append((row, result, frozen))
    return plan, rows, results


@router.post('/{study_id}/answer-reviews', response_model=EvaluationEvidence, status_code=201)
def answer_review(study_id: str, body: AnswerReviewRequest, request: Request):
    plan, _, results = admitted_results(request, study_id)
    selected = next((r for _, r, _ in results if r.experiment_id == body.result_id), None)
    if selected is None:
        fail('invalid_reference', 'Review a result admitted to this study')
    frozen = frozen_case(plan, selected.provenance['protocol_id'])
    if frozen['rubric'] != 'human_adjudication_v1' or frozen['stage'] == 'retrieval_only':
        fail('invalid_rubric', 'Human answer review requires generation and the human adjudication rubric')
    if plan.specification.study_kind == 'researcher_evaluation' and body.reviewer_role != 'independent_researcher':
        fail('review_required', 'This plan requires a declared independent assessor')
    evidence = EvaluationEvidence(record_kind='answer_review', evidence_id='evaluation_' + uuid4().hex,
                                  study_id=study_id, created_at=datetime.now(UTC).isoformat(),
                                  producer_version=__version__, dependencies=[study_id, body.result_id], payload=body)
    return repository(request).add_evidence(evidence)


@router.post('/{study_id}/utility-trials', response_model=EvaluationEvidence, status_code=201)
def utility_trial(study_id: str, body: UtilityTrialRequest, request: Request):
    repo = repository(request)
    plan = repo.get(study_id)
    frozen_case(plan, body.protocol_id)
    if plan.specification.study_kind == 'researcher_evaluation' and body.assessor_role != 'independent_researcher':
        fail('review_required', 'This plan requires a declared independent assessor')
    evidence = EvaluationEvidence(record_kind='utility_trial', evidence_id='evaluation_' + uuid4().hex,
                                  study_id=study_id, created_at=datetime.now(UTC).isoformat(),
                                  producer_version=__version__, dependencies=[study_id, body.protocol_id], payload=body)
    return repo.add_evidence(evidence)


@router.get('/{study_id}/report')
def report(study_id: str, request: Request):
    plan, slots, results = admitted_results(request, study_id)
    evidence = repository(request).evidence(study_id)
    reviews = [e.payload.model_dump(mode='json') for e in evidence if e.record_kind == 'answer_review']
    trials = [e.payload.model_dump(mode='json') for e in evidence if e.record_kind == 'utility_trial']
    cases = []
    for frozen in plan.frozen_cases:
        observed = []
        for slot, result, current in results:
            if current['protocol_id'] != frozen['protocol_id']:
                continue
            rows = result_measurements(result, frozen, [r for r in reviews if r['result_id'] == result.experiment_id])
            observed.append({'repetition': slot['repetition'], 'result_id': result.experiment_id,
                             'result_sha256': result.content_sha256, 'retrospective': slot['retrospective'],
                             'measurements': rows})
        cases.append({'protocol_id': frozen['protocol_id'], 'document_id': frozen['document_id'],
                      'split': frozen['split'], 'rubric': frozen['rubric'],
                      'support_bearing': bool(frozen['moved_designated_support_elements']),
                      'observations': observed, 'summary': repeat_summary([o['measurements'] for o in observed])})
    expected = len(plan.frozen_cases) * plan.specification.repetitions
    return {'schema_version': '0.6.0', 'study_id': study_id, 'study_sha256': plan.content_sha256,
            'model_evaluation': {'planned_acquisitions': expected, 'completed_acquisitions': len(results),
                                 'pending_or_missing_acquisitions': expected - len(results), 'cases': cases},
            'runs': slots, 'evidence': evidence, 'utility': utility_summary(trials),
            'scientific_validation': 'not established by software checks; inspect reviews, cases and trial evidence',
            'limitations': plan.limitations}
