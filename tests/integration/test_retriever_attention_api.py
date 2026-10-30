"""Verified shards, model-specific source admission and saved-view reconstruction."""
from datetime import UTC, datetime
from uuid import uuid4
import numpy as np
from fastapi.testclient import TestClient
from docattentionlens_api.main import create_app
from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens_api.repositories.captures import CaptureStore
from docattentionlens.retrieval.contracts import RetrievalCapture, Position, EncodingAttention


def test_retriever_rows_require_verified_shards_and_save_without_generation(registered,snapshot):
    jobs=JobRepository(registered)
    with TestClient(create_app(registered)) as client:
        case=client.post('/api/v1/cases',json={'question_id':snapshot.questions[0].question_id,
            'page_ids':[snapshot.documents[0].page_ids[0]]}).json()
        identifier='capture_'+uuid4().hex
        encoding=EncodingAttention(encoding_id='query',pass_kind='query',layer_count=1,query_head_count=1,
            position_count=2,attention_mask=[1,1],layer_arrays=['attn_query_l0'])
        record=RetrievalCapture(capture_id=identifier,case_id=case['case_id'],created_at=datetime.now(UTC).isoformat(),
            producer_version='test',dependencies=[],dataset_version=case['dataset_version'],annotation_version=case['annotation_version'],
            query=case['query'],query_array='query',query_positions=[Position(index=i,token_id=i,piece=str(i),role='query_text',eligible=True) for i in range(2)],
            pages=[],attention_encodings=[encoding],provenance={'purpose':'numeric fixture only'},scorer={},geometry={},validation={})
        store=CaptureStore(registered.artifact_root)
        digest=store.put(record,{'attn_query_l0':np.array([[[0,1],[.5,.5]]],dtype=np.float32),'query':np.ones((2,1),dtype=np.float32)})
        job=jobs.submit('retriever-attention-fixture',{'kind':'retrieval_capture'})
        with jobs.connect() as conn:
            conn.execute('INSERT INTO retrieval_captures(capture_id,case_id,job_id,content_hash) VALUES (%s,%s,%s,%s)',(identifier,case['case_id'],job['job_id'],digest))
        url='/api/v1/retrieval/captures/'+identifier+'/attention'
        body={'encoding_id':'query','source_index':0}
        response=client.post(url,json=body)
        assert response.status_code==200,response.text
        assert response.json()['row']['values']==[0,1] and response.json()['row']['predicted_index'] is None
        assert client.post(url,json={**body,'encoding_id':'document'}).status_code==422
        assert client.post(url,json={**body,'source_index':2}).status_code==422
        rollout=client.post(url,json={**body,'method_id':'attention.rollout'})
        assert rollout.status_code==200,rollout.text
        assert rollout.json()['row']['quantity']=='rollout_connectivity' and abs(sum(rollout.json()['row']['values'])-1)<1e-9
        assert client.post(url,json={**body,'method_id':'attribution.integrated_gradients'}).status_code==422
        view={'investigation_model_role':'retrieval','retrieval_capture_id':identifier,'retrieval_encoding_id':'query',
            'retrieval_source_index':1,'retrieval_attention_method':'attention.entropy'}
        saved=client.post('/api/v1/investigations',json={'name':'Numeric fixture only','case_id':case['case_id'],'view_state':view})
        assert saved.status_code==201,saved.text
        reopened=client.get('/api/v1/investigations/'+saved.json()['investigation_id']).json()
        assert reopened['view_state']['retrieval_source_index']==1 and reopened['view_state']['generation_capture_id'] is None
        bad=client.post('/api/v1/investigations',json={'name':'Invalid source','case_id':case['case_id'],'view_state':{**view,'retrieval_source_index':2}})
        assert bad.status_code==422
        path=store.path(identifier)/'attn_query_l0.npy'
        path.write_bytes(path.read_bytes()+b'corrupt')
        corrupt=client.post(url,json=body)
        assert corrupt.status_code==409
        assert corrupt.json()['error']=={'code':'artifact_incomplete','message':'Attention payload is missing or failed integrity verification'}
