"""Transport and integrity fixtures in disposable schemas; no real study claims."""
from datetime import UTC, datetime
from uuid import uuid4
from fastapi.testclient import TestClient
import pytest
from docattentionlens_api.main import create_app
from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens_api.repositories.experiments import ExperimentRepository
from docattentionlens_api.repositories.evaluation import EvaluationRepository
from docattentionlens.experiments.contracts import ExperimentRecord


@pytest.fixture
def evaluation_setup(registered, snapshot):
    doc = snapshot.documents[0]
    page = next(p for p in snapshot.pages if p.document_id == doc.document_id and p.page_number == 34)
    elements = [e for e in snapshot.elements if e.page_id == page.page_id]
    question = next(q for q in snapshot.questions if q.document_id == doc.document_id)
    with TestClient(create_app(registered)) as api:
        case = api.post('/api/v1/cases', json={'question_id': question.question_id}).json()
        body = {'name': 'evaluation fixture only', 'case_id': case['case_id'], 'page_id': page.page_id,
                'element_ids': [e.element_id for e in elements if e.native_id in (549, 550)],
                'non_support_element_ids': [e.element_id for e in elements if e.native_id in (551, 552)],
                'candidate_page_ids': [page.page_id, doc.page_ids[2]], 'fixed_context_page_ids': [page.page_id],
                'stage': 'end_to_end'}
        draft = api.post('/api/v1/experiments/drafts', json=body)
        assert draft.status_code == 201, draft.text
        review = {'reviewer': 'fixture reviewer', 'notes': 'Disposable transport fixture, not independent scientific review',
                  'relationship_confirmed': True, 'content_and_meaning_preserved': True, 'readable': True,
                  'answer_policy': 'unchanged_after_review', 'controls_and_resize_limitations_accepted': True}
        response = api.post('/api/v1/experiments/' + draft.json()['experiment_id'] + '/review', json=review)
        assert response.status_code == 201, response.text
        protocol = ExperimentRecord.model_validate(response.json())
        specification = {'name': 'Disposable evaluation', 'cases': [{'protocol_id': protocol.experiment_id,
                         'split': 'validation', 'rubric': 'human_adjudication_v1',
                         'answer_criteria': 'Compare the answer with the source evidence and retained reference; fixture only',
                         'reviewer': 'fixture reviewer',
                         'reviewer_role': 'engineering', 'review_notes': 'Transport fixtures with actual source pages only',
                         'support_labels_checked': True, 'reference_answer_checked': True}],
                         'repetitions': 2, 'selection_criteria': 'Declare the same actual document and explicit fixture selection',
                         'exclusions': 'Fixtures only', 'baseline_instructions': 'Declared external baseline using the same pages and compute access',
                         'task_instructions': 'Locate designated support and reopen saved references under fixed instructions',
                         'study_kind': 'engineering_pilot'}
        yield api, protocol, specification, registered


def admit_result(settings, protocol):
    """Fabricated transport values are never admitted to the real application DB."""
    result = protocol.model_copy(update={'experiment_id': 'experiment_' + uuid4().hex,
            'record_kind': 'experiment_result', 'created_at': datetime.now(UTC).isoformat(),
            'outcomes': [{'arm': a.name, 'answer': 'fixture only; not model output',
                          'scores': {a.page.page_id: -1.0}, 'supplied_page_ids': [a.page.page_id]}
                         for a in protocol.arms],
            'provenance': {**protocol.provenance, 'protocol_id': protocol.experiment_id,
                           'protocol_sha256': protocol.content_sha256, 'fixture': 'not_model_evidence'}})
    repo = ExperimentRepository(JobRepository(settings))
    digest = repo.store.put(result)
    repo.admit(result, digest)
    return result.experiment_id


def test_plan_membership_rubric_pending_disagreement_and_integrity(evaluation_setup):
    api, protocol, spec, settings = evaluation_setup
    response = api.post('/api/v1/evaluation/studies', json=spec)
    assert response.status_code == 201, response.text
    study = response.json(); base = '/api/v1/evaluation/studies/' + study['study_id']
    initial = api.get(base + '/report').json()
    assert initial['model_evaluation']['pending_or_missing_acquisitions'] == 2
    assert initial['utility']['complete_pairs'] == 0
    rid = admit_result(settings, protocol)
    observation = {'protocol_id': protocol.experiment_id, 'repetition': 1, 'result_id': rid}
    assert api.post(base + '/observations', json=observation).status_code == 201
    assert api.post(base + '/observations', json=observation).status_code == 201
    assert api.post(base + '/observations', json={**observation, 'repetition': 2}).status_code == 409
    assert api.post(base + '/observations', json={**observation, 'repetition': 3}).status_code == 422
    report = api.get(base + '/report').json()
    assert report['model_evaluation']['completed_acquisitions'] == 1
    assert report['model_evaluation']['cases'][0]['observations'][0]['measurements'][0]['answer_assessment']['verdict'] == 'pending'
    review = {'result_id': rid, 'reviewer': 'fixture judge A', 'reviewer_role': 'engineering',
              'judgements': [{'arm': a.name, 'verdict': 'correct', 'rationale': 'Fixture interpretation, not a model judgement'}
                             for a in protocol.arms]}
    assert api.post(base + '/answer-reviews', json=review).status_code == 201
    assert api.post(base + '/answer-reviews', json=review).status_code == 409
    review['reviewer'] = 'fixture judge B'
    review['judgements'][0]['verdict'] = 'incorrect'
    assert api.post(base + '/answer-reviews', json=review).status_code == 201
    measured = api.get(base + '/report').json()['model_evaluation']['cases'][0]['observations'][0]['measurements']
    assert measured[0]['answer_assessment']['disagreement']
    assert measured[0]['answer_assessment']['verdict'] == 'uncertain'
    assert api.get(base).json() == study
    path = EvaluationRepository(JobRepository(settings)).path(study['study_id'])
    path.write_bytes(b'corrupt fixture')
    assert api.get(base).status_code == 409


def test_review_scope_rubric_and_discovery_leakage_refusals(evaluation_setup):
    api, protocol, spec, settings = evaluation_setup
    independent = {**spec, 'study_kind': 'researcher_evaluation'}
    assert api.post('/api/v1/evaluation/studies', json=independent).status_code == 422
    independent['cases'] = [{**spec['cases'][0], 'reviewer_role': 'independent_researcher'}]
    # Both moved groups in the fixture are non-supporting for this real question.
    refusal = api.post('/api/v1/evaluation/studies', json=independent)
    assert refusal.status_code == 422 and refusal.json()['error']['code'] == 'support_case_required'
    second = protocol.model_copy(update={'experiment_id': 'experiment_' + uuid4().hex})
    repo = ExperimentRepository(JobRepository(settings)); digest = repo.store.put(second); repo.admit(second, digest)
    leaked = {**spec, 'cases': [spec['cases'][0], {**spec['cases'][0], 'protocol_id': second.experiment_id, 'split': 'discovery'}]}
    refusal = api.post('/api/v1/evaluation/studies', json=leaked)
    assert refusal.status_code == 422 and refusal.json()['error']['code'] == 'document_leakage'
    duplicate = {**spec, 'cases': [spec['cases'][0], spec['cases'][0]]}
    assert api.post('/api/v1/evaluation/studies', json=duplicate).status_code == 422


def test_utility_unique_pairs_and_repeat_job_idempotency(evaluation_setup, monkeypatch):
    api, protocol, spec, _ = evaluation_setup
    study = api.post('/api/v1/evaluation/studies', json=spec).json()
    base = '/api/v1/evaluation/studies/' + study['study_id']
    trial = {'protocol_id': protocol.experiment_id, 'participant': 'fixture p1', 'experience': 'Fixture research experience',
             'task': 'reproduce_investigation', 'condition': 'baseline', 'order': 1, 'elapsed_seconds': 100,
             'completed': True, 'diagnostic_correct': None, 'reproduction_success': True,
             'unsupported_causal_assertion': False, 'assessor': 'fixture assessor', 'assessor_role': 'engineering',
             'notes': 'Disposable test observations; no actual participant', 'matched_cases_and_compute_confirmed': True}
    assert api.post(base + '/utility-trials', json=trial).status_code == 201
    assert api.post(base + '/utility-trials', json=trial).status_code == 409
    linked = {**trial, 'condition': 'docattentionlens', 'elapsed_seconds': 90}
    assert api.post(base + '/utility-trials', json=linked).status_code == 422
    linked['order'] = 2
    assert api.post(base + '/utility-trials', json=linked).status_code == 201
    utility = api.get(base + '/report').json()['utility']
    assert utility['complete_pairs'] == 1 and utility['paired_time_difference_median_seconds'] == -10
    monkeypatch.setattr(JobRepository, 'workers', lambda self: [{'available': True, 'resource_class': 'gpu',
                        'capabilities': {'adapters': ['qwen25vl_prediction_rows_v1'], 'free_mib': 24000}}])
    body = {'protocol_id': protocol.experiment_id, 'repetition': 1}
    job = api.post(base + '/runs', json=body)
    assert job.status_code == 202, job.text
    monkeypatch.setattr(JobRepository, 'workers', lambda self: [])
    assert api.post(base + '/runs', json=body).json()['job_id'] == job.json()['job_id']
    assert api.get(base + '/report').json()['runs'][0]['state'] == 'queued'
