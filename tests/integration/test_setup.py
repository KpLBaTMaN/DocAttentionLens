"""Real imports and publication in disposable schemas; no model inference."""
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
import json
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
from psycopg.types.json import Jsonb
import pymupdf
import pytest

from docattentionlens_api.main import create_app
from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens_worker.main import run_once
from docattentionlens_api.services.imports import publish_json
from docattentionlens.generation.contracts import GenerationCapture, GenerationPosition, PredictionStep, ContextManifest

def test_input_publication_retains_original_on_retry_and_conflict(tmp_path):
    path = tmp_path / "manifest.json"
    publish_json(path, {"files": {"page.png": "fixture"}})
    before = path.read_bytes()
    publish_json(path, {"files": {"page.png": "fixture"}})
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as conflict:
        publish_json(path, {"files": {"page.png": "changed"}})
    assert conflict.value.status_code == 409
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]

def import_pages(client):
    identifier = client.post('/api/v1/setup/uploads', json={'kind':'documents'}).json()['upload_id']
    image = BytesIO(); Image.new('RGB', (56,84), 'white').save(image,format='PNG')
    pdf = pymupdf.open(); pdf.new_page(width=200,height=300).insert_text((20,30),'Document import fixture')
    for name,raw in [('fixture.pdf',pdf.tobytes()),('fixture.png',image.getvalue())]:
        assert client.put(f'/api/v1/setup/uploads/{identifier}/files',params={'filename':name},content=raw).status_code==200
    result=client.post(f'/api/v1/setup/uploads/{identifier}/documents',json={'filenames':['fixture.pdf','fixture.png']})
    assert result.status_code==200,result.text
    return identifier,result.json()

def test_document_import_custom_question_and_direct_capture(database,monkeypatch):
    observed={}
    def fixture_capture(case,retrieval,inputs,cache,interrupted,**options):
        observed.update(retrieval=retrieval,query=case.query,inputs=inputs,options=options)
        record=GenerationCapture(capture_id='generation_'+uuid4().hex,case_id=case.case_id,retrieval_capture_id=None,created_at=datetime.now(UTC).isoformat(),producer_version='test',dependencies=[case.case_id],
            context=ContextManifest(input_source='supplied_pages',retrieved_page_ids=[],selected_page_ids=[inputs[0][0].page_id],omitted_page_ids=[],query=case.query,prompt='unit fixture only',system_prompt=options['system_prompt'],pages=[],input_token_ids=[1],input_attention_mask=[1]),
            positions=[GenerationPosition(index=0,token_id=1,piece='fixture',role='query',ownership={'non_image:query':1})],output_token_ids=[2],answer='unit fixture only',stop_reason='eos',steps=[PredictionStep(step=0,pass_kind='prefill',source_index=0,consumed_token_id=1,consumed_piece='fixture',predicted_index=1,predicted_token_id=2,predicted_piece='output',target_count=1,array='step_0')],layer_count=1,query_head_count=1,kv_head_count=1,provenance={'purpose':'transport_fixture_not_model_evidence'},coverage={},validation={})
        return record,{'step_0':np.ones((1,1,1),dtype=np.float32)}
    monkeypatch.setattr('docattentionlens.generation.qwen.capture',fixture_capture)
    monkeypatch.setattr(JobRepository,'heartbeat',lambda *args:None)
    with TestClient(create_app(database)) as client:
        upload,document=import_pages(client)
        assert document['total_pages']==2 and document['element_count']==0
        assert client.post(f'/api/v1/setup/uploads/{upload}/documents',json={'filenames':['fixture.pdf','fixture.png']}).json()==document
        assert client.put(f'/api/v1/setup/uploads/{upload}/files',params={'filename':'later.png'},content=b'x').status_code==409
        page_ids=document['page_ids']
        assert client.get('/api/v1/pages/'+page_ids[0]+'/image').headers['content-type']=='image/png'
        page=client.get('/api/v1/pages/'+page_ids[0]).json()
        assert (page['width'],page['height'])==(400,600)
        body={'document_id':document['document_id'],'query':'What does the page contain?'}
        case=client.post('/api/v1/setup/cases',json=body).json()
        assert case['reference_answer']=='' and case['support_element_ids']==[]
        document_view=client.post('/api/v1/investigations',json={'case_id':case['case_id'],'name':'Document-only annotation','view_state':{'page_id':page_ids[0],'evidence_signal':'document'}})
        assert document_view.status_code==201,document_view.text
        assert document_view.json()['view_state']['generation_capture_id'] is None
        assert client.post('/api/v1/setup/cases',json={**body,'query':'   '}).status_code==422
        assert client.post('/api/v1/setup/cases',json={**body,'reference_answer':'inject label'}).status_code==422
        with JobRepository(database).connect() as conn:
            conn.execute("INSERT INTO workers(worker_id,resource_class,heartbeat_at,producer_version,capabilities) VALUES ('direct-fixture','gpu',clock_timestamp(),'test',%s)",(Jsonb({'adapters':['qwen25vl_prediction_rows_v1'],'free_mib':15000}),))
        run={'case_id':case['case_id'],'page_ids':page_ids[:1],'idempotency_key':'direct-fixture','system_prompt':'Use the image only.'}
        assert client.post('/api/v1/generation/runs',json={**run,'retrieval_capture_id':'capture_wrong'}).status_code==422
        assert client.post('/api/v1/generation/runs',json={**run,'page_ids':['outside']}).status_code==422
        assert client.post('/api/v1/generation/runs',json={**run,'model_registration_id':'unknown'}).status_code==404
        accepted=client.post('/api/v1/generation/runs',json=run)
        assert accepted.status_code==202,accepted.text
        assert run_once(replace(database,worker_resource_class='gpu'),'direct-fixture')
        job=client.get('/api/v1/jobs/'+accepted.json()['job_id']).json()
        assert job['state']=='completed',job
        assert observed['retrieval'] is None and observed['query']==body['query']
        assert observed['inputs'][0][2]==[] and observed['options']['system_prompt']==run['system_prompt']
        capture=client.get('/api/v1/generation/captures/'+job['output_resource_id']).json()
        assert capture['retrieval_capture_id'] is None and capture['context']['input_source']=='supplied_pages'
        saved=client.post('/api/v1/investigations',json={'case_id':case['case_id'],'name':'Direct fixture','view_state':{'page_id':page_ids[0],'generation_capture_id':capture['capture_id'],'attention_lens':'context','investigation_focus':'generation','evidence_signal':'attention'}})
        assert saved.status_code==201,saved.text
        assert saved.json()['view_state']['attention_lens']=='context'
        assert client.post('/api/v1/generation/runs',json=run).json()['job_id']==job['job_id']
        assert client.post('/api/v1/generation/runs',json={**run,'system_prompt':'Changed instruction'}).status_code==409

def test_upload_validation_and_model_compatibility(database):
    with TestClient(create_app(database)) as client:
        identifier=client.post('/api/v1/setup/uploads',json={'kind':'model'}).json()['upload_id']
        for filename in ['../config.json','custom.py','pytorch_model.bin']:
            assert client.put(f'/api/v1/setup/uploads/{identifier}/files',params={'filename':filename},content=b'x').status_code==422
        assert client.put(f'/api/v1/setup/uploads/{identifier}/files',params={'filename':'config.json'},content=b'{}').status_code==200
        assert client.put(f'/api/v1/setup/uploads/{identifier}/files',params={'filename':'config.json'},content=b'{"changed":1}').status_code==409
        assert client.post(f'/api/v1/setup/uploads/{identifier}/model',json={'filenames':['config.json']}).json()['error']['code']=='incompatible_model'
        assert client.post(f'/api/v1/setup/uploads/{identifier}/documents',json={'filenames':['config.json']}).status_code==422
        assert client.get('/api/v1/setup/models').json()[0]['registration_id']=='builtin'
        doc=client.post('/api/v1/setup/uploads',json={'kind':'documents'}).json()['upload_id']
        assert client.put(f'/api/v1/setup/uploads/{doc}/files',params={'filename':'empty.pdf'},content=b'').status_code==422
        assert client.put(f'/api/v1/setup/uploads/{doc}/files',params={'filename':'bad.pdf'},content=b'not a PDF').status_code==200
        assert client.post(f'/api/v1/setup/uploads/{doc}/documents',json={'filenames':['bad.pdf']}).status_code==422


def test_checkpoint_registration_is_immutable_and_labels_runtime_validation_pending(database):
    """Tiny structural fixture, deliberately not a complete runnable model."""
    with TestClient(create_app(database)) as client:
        identifier=client.post('/api/v1/setup/uploads',json={'kind':'model'}).json()['upload_id']
        config={'model_type':'qwen2_5_vl','num_hidden_layers':36,'num_attention_heads':16,'num_key_value_heads':2,'hidden_size':2048,'intermediate_size':11008,'vocab_size':151936,'vision_config':{'depth':32,'hidden_size':1280,'num_heads':16,'out_hidden_size':2048,'patch_size':14,'spatial_merge_size':2,'temporal_patch_size':2}}
        header=json.dumps({'fixture_only':{'dtype':'BF16','shape':[1],'data_offsets':[0,2]}}).encode()
        header+=b' '*((-len(header))%8)
        files={'config.json':json.dumps(config).encode(),'tokenizer_config.json':b'{}','tokenizer.json':b'{}','preprocessor_config.json':json.dumps({'patch_size':14,'merge_size':2,'temporal_patch_size':2}).encode(),'model.safetensors':len(header).to_bytes(8,'little')+header+b'\0\0'}
        for name,raw in files.items():
            assert client.put(f'/api/v1/setup/uploads/{identifier}/files',params={'filename':name},content=raw).status_code==200
        body={'filenames':list(files),'name':'Structural fixture only'}
        registered=client.post(f'/api/v1/setup/uploads/{identifier}/model',json=body)
        assert registered.status_code==200,registered.text
        model=registered.json()
        assert model['registration_id'].startswith('model_') and set(model['files'])==set(files)
        assert 'full weight loading' in model['validation']
        assert client.post(f'/api/v1/setup/uploads/{identifier}/model',json=body).json()==model
        assert next(m for m in client.get('/api/v1/setup/models').json() if m['registration_id']==model['registration_id'])==model
        assert client.put(f'/api/v1/setup/uploads/{identifier}/files',params={'filename':'model.safetensors'},content=b'changed').status_code==409


def test_model_roles_describe_actual_architecture_and_gate_unknown_adapters(database):
    """A second architecture must not inherit Qwen's heads or capture claims."""
    with TestClient(create_app(database)) as client:
        models={m['registration_id']:m for m in client.get('/api/v1/setup/models').json()}
        generator,retriever,colqwen=models['builtin'],models['builtin-retrieval'],models['builtin-colqwen2']
        assert (generator['role'],retriever['role'],colqwen['role'])==('generation','retrieval','retrieval')
        # ColQwen2 declares its own causal Qwen2-VL language stack, not ColPali's dimensions.
        assert [(c['layers'],c['query_heads']) for c in colqwen['architecture']['components']]==[(28,12),(32,16)]
        assert colqwen['capabilities']['can_run'] and colqwen['capabilities']['adapter_id']=='colqwen2_transformers_v1'
        assert 'causal' in colqwen['capabilities']['summary'] and colqwen['capabilities']['limits']['image_positions_per_page']==768
        assert client.get('/api/v1/retrieval/plan',params={'model_registration_id':'builtin-colqwen2'}).json()['model_id']=='vidore/colqwen2-v1.0-hf'
        # ColQwen2.5 is the Qwen2.5-VL-3B stack, with its own adapter, its base named and a larger memory need.
        colqwen25=models['builtin-colqwen2.5']
        assert [(c['layers'],c['query_heads']) for c in colqwen25['architecture']['components']]==[(36,16),(32,16)]
        assert colqwen25['capabilities']['adapter_id']=='colqwen2_5_transformers_v1' and 'vidore/colqwen2.5-base' in colqwen25['capabilities']['summary']
        plan25=client.get('/api/v1/retrieval/plan',params={'model_registration_id':'builtin-colqwen2.5'}).json()
        assert (plan25['model_id'],plan25['required_free_gpu_mib'])==('vidore/colqwen2.5-v0.2',12288)
        assert client.get('/api/v1/retrieval/plan',params={'model_registration_id':'builtin'}).status_code==422
        assert generator['architecture']['components'][0]['layers']==36
        assert retriever['architecture']['components'][0]['query_heads']==8
        assert retriever['architecture']['components'][1]['patch_size']==14
        assert retriever['capabilities']['can_run'] and retriever['capabilities']['attention_available']
        assert client.post('/api/v1/retrieval/runs',json={'case_id':'unused','page_ids':['unused'],'model_registration_id':'builtin','idempotency_key':'wrong-role'}).status_code==422
        assert client.post('/api/v1/generation/runs',json={'case_id':'unused','page_ids':['unused'],'model_registration_id':'builtin-retrieval','idempotency_key':'wrong-role'}).status_code==422
        header=json.dumps({'fixture_only':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
        files={'config.json':json.dumps({'model_type':'bert','architectures':['BertModel'],'num_hidden_layers':12,'num_attention_heads':12,'hidden_size':768}).encode(),
            'tokenizer_config.json':b'{}','tokenizer.json':b'{}','model.safetensors':len(header).to_bytes(8,'little')+header+b'\0'*4}
        records=[]
        for role in ['retrieval','generation']:
            upload=client.post('/api/v1/setup/uploads',json={'kind':'model'}).json()['upload_id']
            for name,raw in files.items():
                assert client.put(f'/api/v1/setup/uploads/{upload}/files',params={'filename':name},content=raw).status_code==200
            response=client.post(f'/api/v1/setup/uploads/{upload}/model',json={'filenames':list(files),'name':'Metadata fixture','role':role})
            assert response.status_code==200,response.text
            record=response.json(); records.append(record)
            assert record['role']==role
            assert record['architecture']['components'][0]['layers']==12
            assert record['architecture']['components'][0]['query_heads']==12
            assert record['architecture']['components'][0]['kv_heads'] is None
            assert not record['capabilities']['can_run'] and not record['capabilities']['attention_available']
            endpoint=f'/api/v1/{role}/runs'
            assert client.post(endpoint,json={'case_id':'unused','page_ids':['unused'],'model_registration_id':record['registration_id'],'idempotency_key':'unknown-adapter'}).status_code==422
        assert records[0]['registration_id']!=records[1]['registration_id']
