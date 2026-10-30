"""Live API smoke checks for saved attention and optional real intervention/joint runs."""
import argparse,json,time,urllib.request,urllib.error
from pathlib import Path
from uuid import uuid4
API="http://127.0.0.1:8000"
ROOT=Path(__file__).resolve().parents[1]

def call(path,body=None):
    request=urllib.request.Request(API+path,data=None if body is None else json.dumps(body).encode(),headers={"Content-Type":"application/json"},method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)
    except urllib.error.HTTPError as e:raise RuntimeError(f"{path}: {e.code}: {e.read().decode()[:600]}") from None

def wait(job_id):
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        job=call("/api/v1/jobs/"+job_id)
        if job["state"]=="completed":return job
        if job["state"] in ("failed","cancelled"):raise RuntimeError(json.dumps(job))
        time.sleep(2)
    raise RuntimeError("GPU smoke job did not finish within 10 minutes")

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--gpu",action="store_true");parser.add_argument("--joint",action="store_true")
    args=parser.parse_args()
    report={"scope":"Engineering smoke checks on existing real captures; not scientific validation","saved_methods":[],"experiments":[],"joint":[]}
    retrievers=call("/api/v1/retrieval/captures?limit=25")["items"]
    record=next(c for c in retrievers if any(e["pass_kind"]=="document" for e in c.get("attention_encodings",[])))
    enc=next(e for e in record["attention_encodings"] if e["pass_kind"]=="document")
    page=next(p for p in record["pages"] if p["page_id"]==enc["page_id"])
    image=next(p["index"] for p in page["positions"] if p["role"]=="visual_patch" and p["eligible"])
    text=next(p["index"] for p in page["positions"] if p["role"]!="visual_patch" and p["eligible"])
    ids=["attention.whole_query","attention.text_to_text","attention.text_to_vision","attention.vision_to_text","attention.vision_to_vision",
         "attention.spatial_direction","attention.local_global","attention.row_structure","attention.column_structure","attention.sink_detection",
         "heads.clustering","heads.pattern_taxonomy","attention.partial_rollout","attention.rollout_evolution","attention.graph","attention.flow"]
    for identifier in ids:
        parameters={"layers":[0]} if identifier in ("heads.clustering","heads.pattern_taxonomy","attention.graph","attention.rollout_evolution") else {}
        source=text if identifier in ("attention.text_to_text","attention.text_to_vision") else image
        started=time.monotonic()
        result=call(f"/api/v1/retrieval/captures/{record['capture_id']}/attention",
            {"encoding_id":enc["encoding_id"],"source_index":source,"method_id":identifier,"parameters":parameters})
        assert result.get("row") or result.get("table") or result.get("statistics") or result.get("graph") or result.get("comparisons")
        report["saved_methods"].append({"method_id":identifier,"capture_id":record["capture_id"],"passed":True,"seconds":round(time.monotonic()-started,3)})
        print(identifier,"passed",flush=True)
    generators=call("/api/v1/generation/captures?limit=25")["items"]
    generation=next(c for c in generators if c.get("source_rows") and c["provenance"].get("model_registration_id")=="builtin-qwen2-vl-2b" and not c["provenance"].get("document_study_id"))
    for identifier in ("attention.whole_query","attention.sink_detection","heads.clustering","heads.pattern_taxonomy","attention.text_to_vision"):
        variant=next(m["variant_id"] for m in call("/api/v1/methods")["items"] if m["method_id"]==identifier)
        result=call("/api/v1/methods/analyses",{"generation_capture_id":generation["capture_id"],"method_id":identifier,"variant_id":variant,"parameters":{}})
        assert result["analysis_id"] and result["capture_sha256"]==generation["content_sha256"]
        report["saved_methods"].append({"method_id":identifier,"capture_id":generation["capture_id"],"passed":True})
        print("generator",identifier,"passed",flush=True)
    if args.gpu:
        for source in (record,generation):
            prepared=call("/api/v1/attention-interventions",{"capture_id":source["capture_id"],"method_id":"attention.head_ablation","parameters":{"layer":0,"head":0}})
            accepted=call("/api/v1/attention-interventions/"+prepared["experiment_id"]+"/runs",{"idempotency_key":"attention-smoke-"+uuid4().hex})
            job=wait(accepted["job_id"]); result=call("/api/v1/attention-interventions/"+job["output_resource_id"])
            assert result["kind"]=="result" and len(result["outcomes"])==2 and result["result"]["table"]
            for outcome in result["outcomes"]:
                path="/api/v1/generation/captures/" if result["role"]=="generation" else "/api/v1/retrieval/captures/"
                assert call(path+outcome["capture_id"])["content_sha256"]==outcome["capture_sha256"]
            report["experiments"].append({"role":result["role"],"method_id":result["method_id"],"experiment_id":result["experiment_id"],"job_id":job["job_id"],"passed":True})
            print("real",result["role"],"intervention passed",flush=True)
    if args.joint:
        case=call("/api/v1/cases/"+record["case_id"])
        for registration in ("builtin-retrieval","builtin-colqwen2"):
            accepted=call("/api/v1/retrieval/runs",{"case_id":case["case_id"],"page_ids":[page["page_id"]],"model_registration_id":registration,
                "capture_attention":True,"capture_joint_attention":True,"attention_page_id":page["page_id"],"idempotency_key":"joint-smoke-"+uuid4().hex})
            job=wait(accepted["job_id"]); joint=call("/api/v1/retrieval/captures/"+job["output_resource_id"])
            e=next(e for e in joint["attention_encodings"] if e["pass_kind"]=="joint")
            query=next(p["index"] for p in e["positions"] if p["role"]=="query_text")
            block=call("/api/v1/retrieval/captures/"+joint["capture_id"]+"/attention",{"encoding_id":e["encoding_id"],"source_index":query,"method_id":"attention.text_to_vision","parameters":{}})
            assert block["row"]["image_mass"]>0
            assert joint["provenance"]["diagnostic_joint_pass"] is True
            report["joint"].append({"model_registration_id":registration,"capture_id":joint["capture_id"],"job_id":job["job_id"],"query_source":query,"image_mass":block["row"]["image_mass"],"passed":True})
            print(registration,"diagnostic joint passed",flush=True)
    path=ROOT/"var/verification/attention-only-smoke.json";path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"saved_checks":len(report["saved_methods"]),"experiments":len(report["experiments"]),"joint":len(report["joint"]),"report":str(path)}))

if __name__=="__main__":main()
