"""Attention-intervention API contracts in isolated disposable PostgreSQL schemas."""
from fastapi.testclient import TestClient
from docattentionlens_api.main import create_app
from docattentionlens_api.repositories.jobs import JobRepository
from test_document_methods_api import capture_fixture

def test_prepare_reopen_worker_gate_and_gpu_job_kind(registered,snapshot):
    with TestClient(create_app(registered)) as api:
        source,_=capture_fixture(registered,api,snapshot)
        request={"capture_id":source,"method_id":"attention.head_ablation","variant_id":"captured_attention_v1","parameters":{}}
        response=api.post("/api/v1/attention-interventions",json=request)
        assert response.status_code==201,response.text
        prepared=response.json()
        assert prepared["kind"]=="prepared" and prepared["outcomes"]==[] and prepared["result"] is None
        assert prepared["provenance"]["inputs_unchanged"] is True
        assert api.get("/api/v1/attention-interventions/"+prepared["experiment_id"]).json()["content_sha256"]==prepared["content_sha256"]
        assert api.get("/api/v1/attention-interventions",params={"case_id":prepared["source_case_id"]}).json()["items"]
        run=api.post("/api/v1/attention-interventions/"+prepared["experiment_id"]+"/runs",json={"idempotency_key":"attention-fixture-run"})
        assert run.status_code==503,run.text
        assert api.post("/api/v1/attention-interventions",json={**request,"method_id":"attention.entropy","variant_id":"shannon_row_v1"}).status_code==422
        assert api.post("/api/v1/attention-interventions",json={**request,"method_id":"attention.edge_ablation","parameters":{"source_index":0,"target_index":3}}).status_code==422
        jobs=JobRepository(registered)
        queued=jobs.submit("attention-kind-fixture",{"kind":"attention_intervention","experiment_id":prepared["experiment_id"]})
        assert queued["resource_class"]=="gpu"
        assert api.post("/api/v1/jobs/"+queued["job_id"]+"/cancel").json()["state"]=="cancelled"

def test_prepared_parameters_cannot_be_tampered_with(registered,snapshot):
    import json
    with TestClient(create_app(registered)) as api:
        source,_=capture_fixture(registered,api,snapshot)
        response=api.post("/api/v1/attention-interventions",json={"capture_id":source,"method_id":"attention.head_ablation","parameters":{}})
        assert response.status_code==201,response.text
        record=response.json()
        path=registered.artifact_root/"attention_experiments"/(record["experiment_id"]+".json")
        tampered=json.loads(path.read_bytes());tampered["parameters"]["head"]=1
        path.write_text(json.dumps(tampered),encoding="utf-8")
        assert api.get("/api/v1/attention-interventions/"+record["experiment_id"]).status_code==409
