from dataclasses import replace
from datetime import UTC,datetime
from uuid import uuid4
from fastapi.testclient import TestClient
import numpy as np
import pytest
from docattentionlens_api.main import create_app
from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens_api.repositories.experiments import ExperimentRepository
from docattentionlens_worker.main import run_once
from docattentionlens.retrieval.contracts import RetrievalCapture,PageEvidence


@pytest.mark.parametrize('stage',['retrieval_only','generation_fixed_evidence','end_to_end'])
def test_review_paired_fencing_offline_recovery_and_true_stage_consumption(registered,snapshot,monkeypatch,stage):
    """Actual page fixtures, simulated model outputs confined to disposable schema."""
    calls=[]
    def retrieve(case,inputs,cache,interrupted):
        calls.append(('retrieval',[p.image_sha256 for p,_,_ in inputs]))
        pages=[PageEvidence(page_id=p.page_id,image_sha256=p.image_sha256,width=p.width,height=p.height,page_number=p.page_number,score=float(i),reference_score=float(i),reconstruction_error=0,positions=[],vector_array=f'page_{i}',ownership=[],contributions={},winner_indices=[],query_maxima=[]) for i,(p,_,_) in enumerate(inputs)]
        return RetrievalCapture(capture_id='capture_'+uuid4().hex,case_id=case.case_id,created_at=datetime.now(UTC).isoformat(),producer_version='test',dependencies=[],dataset_version=case.dataset_version,annotation_version=case.annotation_version,query=case.query,query_positions=[],query_array='query',pages=pages,provenance={'fixture':'not_model_evidence'},scorer={},geometry={},validation={}),{'query':np.zeros((0,1),dtype=np.float32),**{f'page_{i}':np.zeros((0,1),dtype=np.float32) for i in range(len(inputs))}}
    def generate(case,retrieval,inputs,cache,interrupted):
        calls.append(('generation',[p.image_sha256 for p,_,_ in inputs]))
        from docattentionlens.generation.contracts import GenerationCapture,ContextManifest
        return GenerationCapture(capture_id='generation_'+uuid4().hex,case_id=case.case_id,retrieval_capture_id=retrieval.capture_id,created_at=datetime.now(UTC).isoformat(),producer_version='test',dependencies=[],context=ContextManifest(retrieved_page_ids=[p.page_id for p in retrieval.pages],selected_page_ids=[p.page_id for p,_,_ in inputs],omitted_page_ids=[],query=case.query,prompt='fixture only',pages=[],input_token_ids=[],input_attention_mask=[]),positions=[],output_token_ids=[1],answer='fixture only; not model output',stop_reason='eos',steps=[],layer_count=36,query_head_count=16,kv_head_count=2,provenance={'fixture':'not_model_evidence'},coverage={},validation={}),{}
    monkeypatch.setattr('docattentionlens.retrieval.colpali.capture',retrieve)
    monkeypatch.setattr('docattentionlens.generation.qwen.capture',generate)
    cpu=JobRepository(registered); gpu_settings=replace(registered,worker_resource_class='gpu');gpu=JobRepository(gpu_settings)
    doc=snapshot.documents[0];page=next(p for p in snapshot.pages if p.document_id==doc.document_id and p.page_number==34)
    es=[e for e in snapshot.elements if e.page_id==page.page_id]
    q=next(q for q in snapshot.questions if q.document_id==doc.document_id)
    with TestClient(create_app(registered)) as api:
        case=api.post('/api/v1/cases',json={'question_id':q.question_id}).json()
        body={'name':'disposable fixture','case_id':case['case_id'],'page_id':page.page_id,'element_ids':[e.element_id for e in es if e.native_id in (549,550)],'non_support_element_ids':[e.element_id for e in es if e.native_id in (551,552)],'candidate_page_ids':[page.page_id,doc.page_ids[2]],'fixed_context_page_ids':[page.page_id],'stage':stage}
        response=api.post('/api/v1/experiments/drafts',json=body); assert response.status_code==201,response.text
        draft=response.json(); did=draft['experiment_id']
        assert api.post(f'/api/v1/experiments/{did}/runs',json={'idempotency_key':'unreviewed'}).status_code==422
        review={'reviewer':'disposable automated fixture','notes':'Transport fixture only, not a scientific review','relationship_confirmed':True,'content_and_meaning_preserved':True,'readable':True,'answer_policy':'unchanged_after_review','controls_and_resize_limitations_accepted':True}
        bad={**review,'readable':False};assert api.post(f'/api/v1/experiments/{did}/review',json=bad).status_code==422
        final=api.post(f'/api/v1/experiments/{did}/review',json=review);assert final.status_code==201,final.text
        pid=final.json()['experiment_id'];assert api.get(f'/api/v1/experiments/{did}').json()['review'] is None
        assert api.post(f'/api/v1/experiments/{pid}/review',json=review).status_code==422
        monkeypatch.setattr(JobRepository,'workers',lambda self:[{'available':True,'resource_class':'gpu','capabilities':{'adapters':['qwen25vl_prediction_rows_v1'],'free_mib':24000}}])
        job=api.post(f'/api/v1/experiments/{pid}/runs',json={'idempotency_key':'paired-fixture'});assert job.status_code==202,job.text
        jid=job.json()['job_id'];assert cpu.claim('cpu-fixture') is None
        gpu.heartbeat('gpu-fixture');run_once(gpu_settings,'gpu-fixture')
        detail=api.get('/api/v1/jobs/'+jid).json();assert detail['state']=='completed',detail
        rid=detail['output_resource_id'];result=api.get('/api/v1/experiments/'+rid).json();assert len(result['outcomes'])==4 and len(result['comparisons'])==3
        assert len([x for x in calls if x[0]=='retrieval'])==(1 if stage=='generation_fixed_evidence' else 4)
        assert len([x for x in calls if x[0]=='generation'])==(0 if stage=='retrieval_only' else 4)
        if stage=='generation_fixed_evidence':
            assert 'scores' not in result['outcomes'][1] and 'retrieval' not in result['outcomes'][1]
            assert result['outcomes'][1]['generation'] and result['outcomes'][1]['fixed_candidate_selection']['record_kind']=='fixed_context_selection'
        for outcome in result['outcomes']:
            for kind in ('retrieval','generation'):
                if kind in outcome:
                    assert api.get(f"/api/v1/{kind}/captures/{outcome[kind]['capture_id']}").status_code==200
        monkeypatch.setattr(JobRepository,'workers',lambda self:[])
        assert api.post(f'/api/v1/experiments/{pid}/runs',json={'idempotency_key':'paired-fixture'}).json()['job_id']==jid
        assert api.get('/api/v1/experiments/'+rid).json()==result
        assert api.get(f'/api/v1/experiments/{rid}/arms/variant/image').status_code==200
        assert api.get(f'/api/v1/experiments/{rid}/arms/invalid/image').status_code==422
        path=ExperimentRepository(cpu).store.path(did)/'variant.png';path.write_bytes(b'corrupt fixture')
        assert api.get(f'/api/v1/experiments/{rid}/arms/variant/image').status_code==409
