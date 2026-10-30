from dataclasses import replace
from pathlib import Path
import shutil

from fastapi.testclient import TestClient

from docattentionlens_api.main import create_app
from docattentionlens_api.repositories.jobs import JobRepository
from docattentionlens_worker.main import run_once

ROOT = Path(__file__).resolve().parents[2]


def test_generation_publication_lineage_replay_and_saved_axis_validation(registered,snapshot,monkeypatch):
    """Transport fixtures in a disposable DB, never real model output."""
    from datetime import UTC,datetime
    import numpy as np
    from uuid import uuid4
    from psycopg.types.json import Jsonb
    from docattentionlens_api.repositories.captures import CaptureStore
    from docattentionlens_api.repositories.generations import GenerationStore
    from docattentionlens.retrieval.contracts import RetrievalCapture,Position
    from docattentionlens.generation.contracts import GenerationCapture,ContextManifest,GenerationPosition,PredictionStep
    from docattentionlens.artifacts.records import OperationalReceipt
    cpu=JobRepository(registered)
    gpu=JobRepository(replace(registered,worker_resource_class='gpu'))
    with TestClient(create_app(registered)) as api:
        q=snapshot.questions[0]
        case=api.post('/api/v1/cases',json={'question_id':q.question_id,'page_ids':[snapshot.documents[0].page_ids[0]]}).json()
        page_id=case['page_ids'][0]
        rid='capture_'+uuid4().hex
        retrieval=RetrievalCapture(capture_id=rid,case_id=case['case_id'],created_at=datetime.now(UTC).isoformat(),producer_version='test',dependencies=[],dataset_version=case['dataset_version'],annotation_version=case['annotation_version'],query=case['query'],query_positions=[Position(index=0,token_id=1,piece='fixture',role='query_text',eligible=True)],query_array='query',pages=[],provenance={'purpose':'unit_fixture_not_model_evidence'},scorer={},geometry={},validation={})
        # Route only needs candidate identity; full numerical scoring is separately
        # tested and the deliberately empty fixture never enters live artifacts.
        from docattentionlens.retrieval.contracts import PageEvidence
        retrieval.pages=[PageEvidence(page_id=page_id,page_number=1,image_sha256=case['page_image_sha256'][page_id],width=1,height=1,score=0,reference_score=0,reconstruction_error=0,positions=[],vector_array='page_0',ownership=[],contributions={},winner_indices=[],query_maxima=[])]
        digest=CaptureStore(registered.artifact_root).put(retrieval,{'query':np.zeros((1,1),dtype=np.float32),'page_0':np.zeros((0,1),dtype=np.float32)})
        rjob=cpu.submit('retrieval-fixture',{'kind':'retrieval_capture'})
        with cpu.connect() as conn:
            conn.execute('INSERT INTO retrieval_captures(capture_id,case_id,job_id,content_hash) VALUES (%s,%s,%s,%s)',(rid,case['case_id'],rjob['job_id'],digest))
            conn.execute("UPDATE jobs SET state='completed' WHERE job_id=%s",(rjob['job_id'],))
        request={'retrieval_capture_id':rid,'page_ids':[page_id],'idempotency_key':'generator-fixture'}
        assert api.post('/api/v1/generation/runs',json=request).status_code==503
        # Declared fixture hardware capability; no inference is executed.
        with cpu.connect() as conn:
            conn.execute("INSERT INTO workers(worker_id,resource_class,heartbeat_at,producer_version,capabilities) VALUES ('generator-fixture','gpu',clock_timestamp(),'test',%s)",(Jsonb({'adapters':['qwen25vl_prediction_rows_v1'],'free_mib':15000}),))
        accepted=api.post('/api/v1/generation/runs',json=request)
        assert accepted.status_code==202
        assert api.post('/api/v1/generation/runs',json={**request,'page_ids':[page_id,page_id]}).status_code==422
        assert api.post('/api/v1/generation/runs',json={**request,'page_ids':['outside']}).status_code==422
        claimed=gpu.claim('generator-fixture')
        assert claimed['job_id']==accepted.json()['job_id'] and claimed['resource_class']=='gpu'
        gid='generation_'+uuid4().hex
        position=GenerationPosition(index=0,token_id=1,piece='fixture',role='query',ownership={'non_image:query':1})
        record=GenerationCapture(capture_id=gid,case_id=case['case_id'],retrieval_capture_id=rid,created_at=datetime.now(UTC).isoformat(),producer_version='test',dependencies=[case['case_id'],rid],
            context=ContextManifest(retrieved_page_ids=[page_id],selected_page_ids=[page_id],omitted_page_ids=[],query=case['query'],prompt='test fixture',pages=[],input_token_ids=[1],input_attention_mask=[1]),
            positions=[position],output_token_ids=[2],answer='test fixture',stop_reason='eos',
            steps=[PredictionStep(step=0,pass_kind='prefill',source_index=0,consumed_token_id=1,consumed_piece='fixture',predicted_index=1,predicted_token_id=2,predicted_piece='fixture output',target_count=1,array='step_0')],layer_count=1,query_head_count=1,kv_head_count=1,provenance={'purpose':'unit_fixture_not_model_evidence'},coverage={'quantity':'post_softmax; dropout disabled','all_causal_targets':True,'ownership':'fixture partition'},validation={})
        store=GenerationStore(registered.artifact_root)
        gh=store.put(record,{'step_0':np.ones((1,1,1),dtype=np.float32)})
        receipt=OperationalReceipt(record_id='receipt_'+uuid4().hex,created_at=datetime.now(UTC),producer_version='test',dependencies=[],provenance={'purpose':'test_fixture'},output={})
        assert gpu.finish(claimed,receipt,'fixture_receipt_checksum',capture={'kind':'generation_capture','capture_id':gid,'case_id':case['case_id'],'retrieval_capture_id':rid,'content_hash':gh})
        assert not gpu.finish(claimed,receipt,'fixture_receipt_checksum',capture={'kind':'generation_capture','capture_id':gid,'case_id':case['case_id'],'retrieval_capture_id':rid,'content_hash':gh})
        assert api.get('/api/v1/jobs/'+claimed['job_id']).json()['output_resource_kind']=='generation_capture'
        with cpu.connect() as conn:conn.execute("UPDATE workers SET heartbeat_at=clock_timestamp()-interval '1 minute'")
        assert api.post('/api/v1/generation/runs',json=request).json()['job_id']==claimed['job_id']
        assert api.get('/api/v1/generation/captures/'+gid).json()['content_sha256']==gh
        row=api.get(f'/api/v1/generation/captures/{gid}/rows')
        assert row.status_code==200 and row.json()['image_mass']==0
        assert api.get(f'/api/v1/generation/captures/{gid}/rows?step=1').json()['error']['code']=='uncaptured_selection'
        body={'case_id':case['case_id'],'name':'Fixture recovery','view_state':{'page_id':page_id,'retrieval_capture_id':rid,'generation_capture_id':gid,'evidence_signal':'attention','attention_layer':1}}
        assert api.post('/api/v1/investigations',json=body).status_code==422
        body['view_state']['attention_layer']=0
        body['view_state'].update(attention_method_id='attention.element_allocation',attention_variant_id='captured_weights_v1')
        assert api.post('/api/v1/investigations',json=body).status_code==422  # exact method/variant pair required
        body['view_state']['attention_variant_id']='ownership_mass_v1'
        body['display_settings']={'attention_opacity':0.35}
        saved=api.post('/api/v1/investigations',json=body)
        assert saved.status_code==201
        reopened=api.get('/api/v1/investigations/'+saved.json()['investigation_id']).json()
        assert reopened['view_state']['generation_capture_id']==gid and gid in reopened['dependencies']
        assert reopened['view_state']['attention_method_id']=='attention.element_allocation'
        assert reopened['display_settings']['attention_opacity']==0.35
        prepared=api.get('/api/v1/methods/availability',params={'generation_capture_id':gid,'retrieval_capture_id':rid})
        assert prepared.status_code==200
        methods={p['method']['method_id']:p for p in prepared.json()['items']}
        assert methods['attention.row']['available']
        # The fixture maps no annotated element, so element mass is unavailable rather than zero.
        assert not methods['attention.element_allocation']['available']
        assert any('annotated element boxes mapped' in r for r in methods['attention.element_allocation']['reasons'])
        assert methods['attention.row']['method']['perspective']=='general'
        assert methods['attention.element_allocation']['method']['perspective']=='document_aware'
        assert prepared.json()['structure'] is None
        # Document annotations and their provenance come from the registered dataset.
        structured=api.get('/api/v1/methods/availability',params={'generation_capture_id':gid,'retrieval_capture_id':rid,'case_id':case['case_id']}).json()
        on_pages=[e for e in snapshot.elements if e.page_id==page_id]
        assert structured['structure']['provenance']=='dataset' and structured['case_id']==case['case_id']
        assert structured['structure']['element_count']==len(on_pages)
        assert structured['structure']['boxed_element_count']==sum(e.display_bbox is not None for e in on_pages)
        assert structured['structure']['support_element_count']==len(case['support_element_ids'])
        support={p['method']['method_id']:p for p in structured['items']}['attention.designated_support']
        assert not support['available'] and not any(r.startswith('Planned') for r in support['reasons'])
        assert any('designated supporting-element annotations' in r for r in support['reasons'])==(not case['support_element_ids'])
        other=api.post('/api/v1/cases',json={'question_id':snapshot.questions[1].question_id}).json()
        assert api.get('/api/v1/methods/availability',params={'generation_capture_id':gid,'case_id':other['case_id']}).status_code==422
        assert api.get('/api/v1/methods/availability',params={'retrieval_capture_id':rid,'case_id':other['case_id']}).status_code==422
        # Each perspective remembers a method of its own role and perspective.
        remembered={'generation:general':'attention.mean_heads/captured_prediction_v1','generation:document_aware':'attention.element_allocation/ownership_mass_v1',
                    'retrieval:document_aware':'attention.element_allocation/retriever_encoding_v1'}
        kept=api.post('/api/v1/investigations',json={**body,'view_state':{**body['view_state'],'perspective_methods':remembered}})
        assert kept.status_code==201, kept.text
        assert api.get('/api/v1/investigations/'+kept.json()['investigation_id']).json()['view_state']['perspective_methods']==remembered
        for wrong in ({'generation:general':'attention.element_allocation/ownership_mass_v1'},{'retrieval:general':'attention.row/captured_weights_v1'},
                      {'generation:document_aware':'attention.designated_missing/v1'},{'evaluation:general':'attention.row/captured_weights_v1'}):
            assert api.post('/api/v1/investigations',json={**body,'view_state':{**body['view_state'],'perspective_methods':wrong}}).status_code==422
        assert not methods['attention.rollout']['available']
        assert not methods['retrieval.maxsim']['available']  # empty fixture has no recorded scorer/masks
        assert not any(p['available'] for p in api.get('/api/v1/methods/availability').json()['items'])
        # Saved-data analysis does not submit GPU jobs and records exact defaults,
        # input hashes, method versions, support and immutable result identities.
        analysis_request={'generation_capture_id':gid,'method_id':'attention.mean_heads','variant_id':'captured_prediction_v1','parameters':{}}
        analysis=api.post('/api/v1/methods/analyses',json=analysis_request)
        assert analysis.status_code==200, analysis.text
        result=analysis.json()
        assert result['capture_sha256']==gh and result['parameters']['heads']==[0]
        assert result['valid_target_indices']==[0] and result['row']['total_mass']==1
        assert api.post('/api/v1/methods/analyses',json=analysis_request).json()==result
        assert api.get('/api/v1/methods/analyses/'+result['analysis_id']).json()==result
        # A changed calculation never reuses a cached result: identities hash the method source.
        import docattentionlens_api.services.analyses as analysis_service
        with monkeypatch.context() as changed_code:
            changed_code.setattr(analysis_service,'calculation_sha256',lambda:'0'*64)
            recalculated=api.post('/api/v1/methods/analyses',json=analysis_request).json()
        assert recalculated['analysis_id']!=result['analysis_id'] and recalculated['row']==result['row']
        assert api.post('/api/v1/methods/analyses',json=analysis_request).json()==result
        # Parameter errors name the parameter instead of returning Pydantic's report.
        assert api.post('/api/v1/methods/analyses',json={**analysis_request,'parameters':{'layer':128}}).json()['error']['message']=='Parameter layer: Input should be less than 128.'
        assert api.post('/api/v1/methods/analyses',json={**analysis_request,'parameters':{'layer':1}}).json()['error']['message']=='Layer is outside captured coverage'
        assert api.post('/api/v1/methods/analyses',json={**analysis_request,'parameters':{'bandwidth':2}}).json()['error']['message']=='Unknown parameter bandwidth.'
        for identifier in ('attention.max_heads','attention.min_heads','attention.weighted_heads','attention.concentration','attention.topk_mass','attention.sinks','attention.modality_ratio','attention.head_by_head','attention.across_layers','heads.distribution_similarity','attention.entropy'):
            variant='shannon_row_v1' if identifier=='attention.entropy' else 'captured_prediction_v1'
            response=api.post('/api/v1/methods/analyses',json={**analysis_request,'method_id':identifier,'variant_id':variant})
            assert response.status_code==200, response.text
            assert response.json()['capture_sha256']==gh
        assert api.post('/api/v1/methods/analyses',json={**analysis_request,'parameters':{'heads':[1]}}).status_code==422
        assert api.post('/api/v1/methods/analyses',json={**analysis_request,'method_id':'attention.rollout','variant_id':'residual_mean_square_v1'}).status_code==422
        assert api.post('/api/v1/methods/analyses',json={**analysis_request,'method_id':'attention.weighted_heads','parameters':{'weights':[0]}}).status_code==422
        body['view_state'].update(attention_method_id='attention.mean_heads',attention_variant_id='captured_prediction_v1',attention_parameters={'heads':[0]},attention_analysis_id=result['analysis_id'])
        stored=api.post('/api/v1/investigations',json=body)
        assert stored.status_code==201, stored.text
        assert result['analysis_id'] in stored.json()['dependencies']
        assert api.get('/api/v1/investigations/'+stored.json()['investigation_id']).json()['view_state']['attention_parameters']=={'heads':[0]}
        body['view_state']['attention_parameters']={'heads':[1]}
        assert api.post('/api/v1/investigations',json=body).status_code==422
        body['view_state']['attention_parameters']={'heads':[0]}
        # Retrieval's query selector must be valid too, even while attention shown.
        from docattentionlens.retrieval.contracts import Position
        retrieval.query_positions=[Position(index=0,token_id=1,piece='fixture',role='query_text',eligible=True)]
        # Publish a new immutable fixture, never rewrite the first manifest.
        rid2='capture_'+uuid4().hex; retrieval.capture_id=rid2
        rh=CaptureStore(registered.artifact_root).put(retrieval,{'query':np.zeros((1,1),dtype=np.float32),'page_0':np.zeros((0,1),dtype=np.float32)})
        with cpu.connect() as conn:conn.execute('INSERT INTO retrieval_captures(capture_id,case_id,job_id,content_hash) VALUES (%s,%s,%s,%s)',(rid2,case['case_id'],rjob['job_id'],rh))
        body['view_state']['retrieval_capture_id']=rid2
        assert api.get('/api/v1/methods/availability',params={'generation_capture_id':gid,'retrieval_capture_id':rid2}).status_code==422
        assert api.post('/api/v1/investigations',json=body).status_code==422  # lineage mismatch
        (store.path(gid)/'step_0.npy').write_bytes(b'corrupt')
        assert api.get(f'/api/v1/generation/captures/{gid}/rows').json()['error']['code']=='artifact_incomplete'
        assert api.get('/api/v1/methods/analyses/'+result['analysis_id']).status_code==200
        uncached={**analysis_request,'method_id':'attention.weighted_heads','parameters':{'weights':[3]}}
        assert api.post('/api/v1/methods/analyses',json=uncached).json()['error']['code']=='artifact_incomplete'
        (registered.artifact_root/'analyses'/(result['analysis_id']+'.json')).write_bytes(b'corrupt')
        assert api.get('/api/v1/methods/analyses/'+result['analysis_id']).json()['error']['code']=='artifact_incomplete'


def test_actual_counts_quality_pagination_and_native_references(registered, snapshot):
    with TestClient(create_app(registered)) as api:
        summary = api.get(f"/api/v1/datasets/{snapshot.summary.dataset_id}/summary").json()
        assert summary["counts"] == {"documents": 80, "pages": 1849, "elements": 25760, "questions": 1559}
        assert summary["quality_counts"]["bbox_outside_or_degenerate"] == 50
        assert summary["quality_counts"]["empty_ocr_texts"] == 163
        assert summary["quality_counts"]["multi_page_support_questions"] == 427
        params = {"dataset_id": snapshot.summary.dataset_id, "limit": 1}
        first = api.get("/api/v1/documents", params=params).json()
        second = api.get("/api/v1/documents", params={**params, "before": first["next_cursor"]}).json()
        assert first["items"][0]["document_id"] != second["items"][0]["document_id"]
        assert first["total"] == 80
        assert api.get("/api/v1/documents", params={**params,"before":first["next_cursor"],"search":"changed"}).status_code == 422
        for kind, records in (("elements", snapshot.elements), ("questions", snapshot.questions)):
            repeated = [r for r in records if r.native_id == records[0].native_id]
            assert len({r.key for r in repeated}) == len(repeated) > 1
        element = next(e for e in snapshot.elements if e.geometry_flags)
        result = api.get(f"/api/v1/pages/{element.page_id}/elements").json()
        actual = next(e for e in result["items"] if e["element_id"] == element.element_id)
        assert actual["bbox"] == element.bbox.model_dump()
        assert actual["geometry_flags"] == element.geometry_flags


def test_case_wording_and_revision_conflict_recovery(registered, snapshot):
    q = next(q for q in snapshot.questions if len(q.support_page_ids)>1)
    element = next(e for e in snapshot.elements if e.document_id==q.document_id)
    with TestClient(create_app(registered)) as api:
        bad = api.post("/api/v1/cases", json={"question_id":q.question_id,"query_variant":"document_scoped","task_scope":"collection"})
        assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_scope"
        case = api.post("/api/v1/cases", json={"question_id":q.question_id,"query_variant":"document_scoped","task_scope":"document"}).json()
        assert case["query"] == q.document_scoped_question
        assert case["content_sha256"] and case["created_at"]
        assert case["record_kind"]=="case" and q.question_id in case["dependencies"]
        body = {"case_id":case["case_id"],"name":"Recovery check","notes":"Original notes",
            "view_state":{"page_id":element.page_id,"selected_element_id":element.element_id,"source_element_id":element.element_id,"target_element_id":None},
            "display_settings":{"schema_version":"0.2.0","zoom":1.25,"center_x":20,"center_y":20,"show_support":True}}
        created = api.post("/api/v1/investigations",json=body)
        assert created.status_code == 201, created.text
        record = created.json()
        update = api.patch(f"/api/v1/investigations/{record['investigation_id']}", json={**body,"notes":"Revised notes","expected_revision":1})
        assert update.status_code == 200
        conflict = api.patch(f"/api/v1/investigations/{record['investigation_id']}", json={**body,"expected_revision":1})
        assert conflict.status_code == 409
        assert conflict.json()["error"]["current_revision_id"] == update.json()["revision_id"]
        iid=record["investigation_id"]
    # New app/repository instances simulate reopening, avoiding process-local state.
    with TestClient(create_app(registered)) as api:
        latest = api.get(f"/api/v1/investigations/{iid}").json()
        exact = api.get(f"/api/v1/investigations/{iid}?revision=1").json()
        assert latest["notes"] == "Revised notes" and exact["notes"] == "Original notes"
        assert latest["view_state"] == record["view_state"]
        assert latest["display_settings"] == record["display_settings"]
        assert api.get(f"/api/v1/cases/{case['case_id']}").json()["query"] == q.document_scoped_question
        foreign = next(e for e in snapshot.elements if e.document_id!=q.document_id)
        invalid = api.post("/api/v1/investigations", json={**body,"view_state":{"page_id":element.page_id,"selected_element_id":foreign.element_id}})
        assert invalid.status_code == 422


def test_case_list_is_newest_first_with_limit_and_case_cursor(registered, snapshot):
    question = snapshot.questions[0]
    with TestClient(create_app(registered)) as api:
        ids = [api.post("/api/v1/cases", json={"question_id": question.question_id}).json()["case_id"] for _ in range(3)]
        listed = api.get("/api/v1/cases").json()
        assert [c["case_id"] for c in listed] == ids[::-1]
        assert listed == [api.get(f"/api/v1/cases/{i}").json() for i in ids[::-1]]
        assert [c["case_id"] for c in api.get("/api/v1/cases", params={"limit": 2}).json()] == [ids[2], ids[1]]
        assert [c["case_id"] for c in api.get("/api/v1/cases", params={"before": ids[1]}).json()] == [ids[0]]
        assert api.get("/api/v1/cases", params={"before": "case_unknown"}).json() == []
        assert api.get("/api/v1/cases", params={"limit": 101}).status_code == 422


def test_source_image_hash_and_missing_asset(registered, snapshot, tmp_path):
    page=snapshot.pages[0]
    root=tmp_path / "sources"
    image=root / "doclayout_mm_rag_bench" / snapshot.image_files[page.page_id]
    image.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "data/doclayout_mm_rag_bench" / snapshot.image_files[page.page_id],image)
    settings=replace(registered,data_root=root)
    with TestClient(create_app(settings)) as api:
        ok=api.get(f"/api/v1/pages/{page.page_id}/image")
        assert ok.status_code==200 and ok.headers["etag"]==f'"{page.image_sha256}"'
        image.write_bytes(b"modified test copy")
        assert api.get(f"/api/v1/pages/{page.page_id}/image").json()["error"]["code"]=="version_mismatch"
        image.unlink()
        assert api.get(f"/api/v1/pages/{page.page_id}/image").json()["error"]["code"]=="source_missing"


def test_worker_registration_is_durable_and_idempotent(database):
    settings=replace(database,data_root=ROOT / "data",lease_seconds=60)
    with TestClient(create_app(settings)) as api:
        request={"source":"doclayout_mm_rag_bench","idempotency_key":"register-real"}
        first=api.post("/api/v1/datasets/register",json=request).json()
        assert api.post("/api/v1/datasets/register",json=request).json()["job_id"]==first["job_id"]
    assert run_once(settings,"dataset-worker")
    with TestClient(create_app(settings)) as api:
        job=api.get(f"/api/v1/jobs/{first['job_id']}").json()
        assert job["state"]=="completed", job
        assert job["output_resource_kind"]=="dataset"
        summary=api.get(f"/api/v1/datasets/{job['output_resource_id']}/summary").json()
        assert summary["counts"]["documents"]==80
        assert api.get(f"/api/v1/artifacts/operational_receipt/{job['result_id']}").status_code==200
