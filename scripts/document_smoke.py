"""Verify real registration and recovery against the running Docker application."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument("--restart", action="store_true")
args=parser.parse_args()
client=httpx.Client(base_url="http://127.0.0.1:8080",timeout=30)


def get(path):
    response=client.get(path)
    response.raise_for_status()
    return response.json()


job=client.post("/api/v1/datasets/register",json={"source":"doclayout_mm_rag_bench","idempotency_key":"dataset-smoke-"+uuid4().hex})
job.raise_for_status()
job_id=job.json()["job_id"]
print("Registration job submitted: "+job_id,flush=True)
deadline=time.monotonic()+300
while time.monotonic()<deadline:
    saved_job=get("/api/v1/jobs/"+job_id)
    if saved_job["state"]=="completed":
        break
    if saved_job["state"] in ("failed","cancelled"):
        raise RuntimeError(saved_job)
    time.sleep(1)
else:
    raise TimeoutError("Dataset registration did not complete")
summary=get("/api/v1/datasets/"+saved_job["output_resource_id"]+"/summary")
assert summary["counts"]=={"documents":80,"pages":1849,"elements":25760,"questions":1559}
doc=get("/api/v1/documents?dataset_id="+summary["dataset_id"])["items"][0]
question=get("/api/v1/questions?document_id="+doc["document_id"])["items"][0]
case_response=client.post("/api/v1/cases",json={"question_id":question["question_id"],"query_variant":"collection_scoped","task_scope":"collection"})
case_response.raise_for_status()
case=case_response.json()
page=get("/api/v1/pages/"+case["page_ids"][0])
element=get("/api/v1/pages/"+page["page_id"]+"/elements")["items"][0]
response=client.post("/api/v1/investigations",json={"name":"First real document investigation","case_id":case["case_id"],"notes":"Source geometry and native annotations inspected. No model capture.","view_state":{"page_id":page["page_id"],"selected_element_id":element["element_id"],"source_element_id":element["element_id"],"target_element_id":None},"display_settings":{"schema_version":"0.2.0","zoom":1.25,"center_x":page["width"]/2,"center_y":page["height"]/2,"show_support":True}})
response.raise_for_status()
investigation=response.json()
image=client.get(page["image_url"])
image.raise_for_status()
if args.restart:
    subprocess.run(["docker","compose","-f","infra/compose.yaml","restart","metadata","api","worker-cpu","frontend"],cwd=ROOT,check=True)
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:
        try:
            ready=client.get("/health/ready")
            if ready.status_code==200:
                break
        except httpx.TransportError:
            pass
        time.sleep(1)
reopened=get("/api/v1/investigations/"+investigation["investigation_id"])
assert reopened==investigation
assert get("/api/v1/cases/"+case["case_id"])["query"]==question["collection_scoped_question"]
assert client.get(page["image_url"]).content==image.content
evidence={"dataset":summary["dataset_id"],"dataset_version":summary["dataset_version"],"counts":summary["counts"],"quality":summary["quality_counts"],"registration_job":job_id,"case_id":case["case_id"],"investigation":investigation,"restart_verified":args.restart,"url":"http://localhost:8080/attention-tool/"+investigation["investigation_id"]}
destination=ROOT / "var/verification/document-smoke.json"
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(evidence,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"counts":summary["counts"],"quality":summary["quality_counts"],"url":evidence["url"],"recovery":"passed"},indent=2),flush=True)
