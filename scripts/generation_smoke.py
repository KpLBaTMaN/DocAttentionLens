"""Real local GPU acceptance: retrieval lineage -> consumed context -> answer."""
import argparse
import json
import time
import urllib.request
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from PIL import Image

BASE="http://localhost:8000"
ROOT=Path(__file__).resolve().parents[1]


def request(path,body=None):
    req=urllib.request.Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=30) as response:
        return json.load(response)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--retrieval",default="capture_d48684012e9e484e804eab6d9631ccce")
    parser.add_argument("--capture",help="Verify existing capture instead of acquiring")
    args=parser.parse_args()
    retrieval=request("/api/v1/retrieval/captures/"+args.retrieval)
    case=request("/api/v1/cases/"+retrieval["case_id"])
    ordered=sorted(retrieval["pages"],key=lambda p:-p["score"])
    # Two actual page shapes exercise page boundaries and omitted candidates.
    selected=[ordered[0]["page_id"],retrieval["pages"][1]["page_id"]]
    if args.capture:
        identifier=args.capture
    else:
        accepted=request("/api/v1/generation/runs",{"retrieval_capture_id":retrieval["capture_id"],"page_ids":selected,"idempotency_key":"generation-smoke-"+uuid4().hex})
        print(json.dumps(accepted),flush=True)
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            job=request("/api/v1/jobs/"+accepted["job_id"])
            if job["state"]=="completed":break
            if job["state"] in ("failed","cancelled"):raise RuntimeError(json.dumps(job))
            time.sleep(2)
        else:raise TimeoutError("Generator acceptance exceeded time budget")
        identifier=job["output_resource_id"]
    capture=request("/api/v1/generation/captures/"+identifier)
    assert capture["retrieval_capture_id"]==retrieval["capture_id"]
    assert capture["context"]["query"]==case["query"]
    assert capture["context"]["evaluation_labels_consumed"] is False and capture["context"]["ocr_consumed"] is False
    assert capture["context"]["selected_page_ids"]==selected
    assert set(capture["context"]["selected_page_ids"])|set(capture["context"]["omitted_page_ids"])==set(capture["context"]["retrieved_page_ids"])
    assert capture["validation"]["capture_on_off_token_equality"] and capture["validation"]["capture_on_off_logits_max_error"]<=1e-5
    assert (capture["layer_count"],capture["query_head_count"],capture["kv_head_count"])==(36,16,2)
    n=len(capture["context"]["input_token_ids"])
    positional=capture["provenance"].get("positional_configuration")
    if positional:
        assert positional["prefill_axes"]==["text_causal","temporal","height","width"]
        assert len(positional["prefill_position_ids"])==4
        assert positional["prefill_position_ids"][0]==list(range(n))
        assert positional["query_head_to_kv_head"]==[0]*8+[1]*8
        assert [p["cache_position"] for p in positional["prediction_source_position_ids"]]==[n+s["step"]-1 for s in capture["steps"]]
    for step in capture["steps"]:
        k=step["step"]
        assert (step["source_index"],step["predicted_index"],step["target_count"])==(n+k-1,n+k,n+k)
        assert capture["positions"][step["source_index"]]["token_id"]==step["consumed_token_id"]
        assert capture["output_token_ids"][k]==step["predicted_token_id"]
        row=request(f"/api/v1/generation/captures/{identifier}/rows?step={k}&layer=35&head=15")
        assert len(row["values"])==n+k and abs(sum(row["values"])-1)<.02
        assert abs(sum(a["mass"] for a in row["allocations"])-row["total_mass"])<1e-8
        assert row["allocation_error"]<1e-8
    image_hashes={}
    for page in capture["context"]["pages"]:
        assert page["image_sha256"]==case["page_image_sha256"][page["page_id"]]
        with urllib.request.urlopen(BASE+page["preview_url"]) as response:data=response.read()
        image=Image.open(BytesIO(data))
        assert image.size==(page["resized_width"],page["resized_height"])
        image_hashes[page["page_id"]]=sha256(data).hexdigest()
        positions=[capture["positions"][i] for i in page["position_indices"]]
        assert all(p["page_id"]==page["page_id"] and p["role"]=="image" for p in positions)
        assert all(abs(sum(p["ownership"].values())-1)<1e-8 for p in positions)
    elements=request("/api/v1/pages/"+selected[0]+"/elements")["items"]
    reviewed=[e for e in elements if not e["geometry_flags"]]
    inspected=next((e for e in reviewed if e["element_type"]=="figure"),reviewed[0] if reviewed else None)
    view={"selected_element_id":inspected["element_id"] if inspected else None,"page_id":selected[0],"retrieval_capture_id":retrieval["capture_id"],"generation_capture_id":identifier,
        "evidence_signal":"attention","attention_step":capture["steps"][0]["step"],"attention_layer":35,"attention_head":15}
    investigation=request("/api/v1/investigations",{"name":"Milestone 4 · real context, answer and attention","case_id":case["case_id"],"view_state":view,"display_settings":{"show_boxes":False,"show_reading_order":False},"notes":"Real Qwen2.5-VL output from two selected ColPali candidates. Attention is original-domain allocation; no attribution or answer-quality claim."})
    reopened=request("/api/v1/investigations/"+investigation["investigation_id"])
    assert reopened["view_state"]["attention_head"]==15 and identifier in reopened["dependencies"]
    row=request(f"/api/v1/generation/captures/{identifier}/rows?step=0&layer=35&head=15")
    result={"capture_id":identifier,"capture_sha256":capture["content_sha256"],"retrieval_capture_id":retrieval["capture_id"],"investigation_id":investigation["investigation_id"],"view_state":reopened["view_state"],"row":row,"image_sha256":image_hashes,"answer":capture["answer"],"input_positions":n,"captured_steps":len(capture["steps"]),"generated_steps":len(capture["output_token_ids"]),"validation":capture["validation"],"retained_bytes":sum(r["bytes"] for r in capture["arrays"]),"peak_gpu_allocated_bytes":capture["provenance"]["peak_gpu_allocated_bytes"]}
    path=ROOT/"var/verification/generation-smoke.json";path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("row","view_state","image_sha256")},indent=2))


if __name__=="__main__":main()
