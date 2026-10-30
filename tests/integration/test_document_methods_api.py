"""Disposable-schema API fixtures, explicitly not model or research evidence."""
from datetime import datetime,UTC
from uuid import uuid4
import numpy as np
from fastapi.testclient import TestClient
from docattentionlens_api.main import create_app
from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens_api.repositories.generations import GenerationStore
from docattentionlens.generation.contracts import GenerationCapture,GenerationPosition,PredictionStep,ContextManifest,ContextPage,QWEN25_VL_3B


def capture_fixture(settings,api,snapshot):
    page=next(p for p in snapshot.pages if sum(e.page_id==p.page_id and not e.geometry_flags for e in snapshot.elements)>=2)
    question=next(q for q in snapshot.questions if q.document_id==page.document_id)
    case=api.post('/api/v1/cases',json={'question_id':question.question_id,'page_ids':[page.page_id]}).json()
    elements=[e for e in snapshot.elements if e.page_id==page.page_id and not e.geometry_flags][:2]
    identifier='generation_'+uuid4().hex
    positions=[GenerationPosition(index=i,token_id=i,piece='fixture',role='image' if i<2 else 'query' if i==2 else 'special',
        page_id=page.page_id if i<2 else None,bbox=[0,0,page.width,page.height] if i<2 else None,
        ownership={elements[i].element_id:1} if i<2 else {'non_image:query' if i==2 else 'non_image:special':1}) for i in range(4)]
    record=GenerationCapture(capture_id=identifier,case_id=case['case_id'],created_at=datetime.now(UTC).isoformat(),producer_version='fixture',dependencies=[],
        context=ContextManifest(retrieved_page_ids=[],selected_page_ids=[page.page_id],omitted_page_ids=[],query=case['query'],prompt='fixture not model evidence',
            input_token_ids=[0,1,2,3],input_attention_mask=[1]*4,pages=[ContextPage(page_id=page.page_id,page_number=page.page_number,image_sha256=page.image_sha256,
                source_width=page.width,source_height=page.height,resized_width=page.width,resized_height=page.height,grid_thw=[1,2,2],position_indices=[0,1],pixel_array=None,preview_url=page.image_url)]),
        positions=positions,output_token_ids=[4],answer='fixture output',stop_reason='eos',steps=[PredictionStep(step=0,pass_kind='prefill',source_index=3,
            consumed_token_id=3,consumed_piece='fixture',predicted_index=4,predicted_token_id=4,predicted_piece='fixture output',target_count=4,array='step_0')],
        layer_count=1,query_head_count=1,kv_head_count=1,provenance={'purpose':'unit_fixture_not_model_evidence','model_id':QWEN25_VL_3B.model_id,
            'model_revision':QWEN25_VL_3B.revision,'model_registration_id':'builtin'},coverage={'quantity':'post_softmax','all_causal_targets':True,'ownership':'fixture'},validation={})
    store=GenerationStore(settings.artifact_root);digest=store.put(record,{'step_0':np.array([[[.2,.3,.1,.4]]],dtype=np.float32)})
    jobs=JobRepository(settings);job=jobs.submit('fixture-'+uuid4().hex,{'kind':'generation_capture'})
    with jobs.connect() as conn:
        conn.execute("INSERT INTO generation_captures(capture_id,case_id,job_id,content_hash) VALUES (%s,%s,%s,%s)",(identifier,case['case_id'],job['job_id'],digest))
        conn.execute("UPDATE jobs SET state='completed' WHERE job_id=%s",(job['job_id'],))
    return identifier,elements


def test_annotation_execution_cache_multicapture_and_saved_result_integrity(registered,snapshot):
    with TestClient(create_app(registered)) as api:
        primary,_=capture_fixture(registered,api,snapshot)
        methods={m['method_id']:m for m in api.get('/api/v1/methods').json()['items']}
        def body(identifier):return {'generation_capture_id':primary,'method_id':identifier,'variant_id':methods[identifier]['variant_id'],'parameters':{}}
        for identifier in ('attention.element_type_profile','attention.reading_order','attention.layout_sinks','attention.section_rollup','attention.model_input_legibility','attention.evidence_funnel'):
            request=body(identifier);response=api.post('/api/v1/methods/analyses',json=request)
            assert response.status_code==200,response.text
            result=response.json();assert result['input_manifest']['inputs'][primary]['annotation_sha256']
            assert api.post('/api/v1/methods/analyses',json=request).json()['analysis_id']==result['analysis_id']
            assert api.get('/api/v1/methods/analyses/'+result['analysis_id']).json()['content_sha256']==result['content_sha256']
        assert api.get('/api/v1/methods/saved-analyses',params={'generation_capture_id':primary}).json()['items']
        assert api.post('/api/v1/methods/analyses',json={**body('attention.question_atlas'),'sample_capture_ids':[primary,primary]}).status_code==422
        assert api.post('/api/v1/methods/analyses',json={**body('attention.aligned_comparison'),'comparison_capture_id':primary,'source_index':0}).status_code==422


def test_document_experiment_prepare_is_immutable_and_run_requires_worker(registered,snapshot):
    with TestClient(create_app(registered)) as api:
        primary,elements=capture_fixture(registered,api,snapshot)
        methods={m['method_id']:m for m in api.get('/api/v1/methods').json()['items']}
        identifier='document.image_text_contrast'
        response=api.post('/api/v1/document-studies',json={'generation_capture_id':primary,'method_id':identifier,'variant_id':methods[identifier]['variant_id'],'parameters':{}})
        assert response.status_code==201,response.text
        study=response.json();assert study['kind']=='prepared' and len(study['arms'])==3
        assert api.get('/api/v1/document-studies/'+study['study_id']).json()['content_sha256']==study['content_sha256']
        page_id=study['provenance']['page_ids'][0]
        assert api.get(f"/api/v1/document-studies/{study['study_id']}/arms/ocr/pages/{page_id}").status_code==200
        result=api.post(f"/api/v1/document-studies/{study['study_id']}/runs",json={'idempotency_key':'study-fixture'})
        assert result.status_code==503
        assert api.post('/api/v1/document-studies',json={'generation_capture_id':primary,'method_id':'document.element_steering',
            'variant_id':methods['document.element_steering']['variant_id'],'parameters':{'element_ids':[elements[0].element_id],'strengths':['Infinity']}}).status_code==422
