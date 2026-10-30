"""Real saved-capture checks for the four formerly planned methods.

Drives the running API (default http://127.0.0.1:8000) and writes
var/verification/completed-methods.json. Rollout and distance use a saved ColPali
document encoding; profile similarity uses saved Qwen2.5-VL generation captures;
integrated gradients prepares and runs one Document studies GPU job. Each check
recomputes or cross-checks a value independently of the method code where the API
exposes the inputs. Usage: python scripts/completed_methods_check.py [--skip-gpu]
"""
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[1]


def call(path, body=None):
    request = urllib.request.Request(API+path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{path}: {error.code} {error.read().decode()[:400]}") from None


def stats(result):
    return {s["name"]: s["value"] for s in result["statistics"]}


def retriever_checks():
    capture = next(c for c in call("/api/v1/retrieval/captures?limit=25")["items"]
                   if any(e["pass_kind"] == "document" for e in c.get("attention_encodings") or []))
    encoding = next(e for e in capture["attention_encodings"] if e["pass_kind"] == "document")
    page = next(p for p in capture["pages"] if p["page_id"] == encoding["page_id"])
    patches = [p for p in page["positions"] if p["bbox"]]
    source = patches[len(patches)//2]["index"]
    url = f"/api/v1/retrieval/captures/{capture['capture_id']}/attention"
    body = lambda method, **parameters: {"encoding_id": "document", "source_index": source, "method_id": method, "parameters": parameters}
    out = {"capture_id": capture["capture_id"], "model": capture["provenance"].get("model_id"), "source_index": source,
           "layers": encoding["layer_count"], "heads": encoding["query_head_count"], "positions": encoding["position_count"]}
    started = time.time()
    full = call(url, body("attention.rollout", layer=encoding["layer_count"]-1, residual_coefficient=.5))
    out["rollout_seconds"] = round(time.time()-started, 2)
    values = full["row"]["values"]
    out["rollout_sum_error"] = abs(sum(values)-1)
    out["rollout_min"] = min(values)
    out["rollout_image_share"] = stats(full)["Image share"]
    identity = call(url, body("attention.rollout", layer=encoding["layer_count"]-1, residual_coefficient=1.0))["row"]["values"]
    out["identity_rollout_is_one_hot"] = identity[source] == 1.0 and sum(identity) == 1.0
    # With no residual and one layer, rollout equals the renormalized head mean of layer 0.
    single = call(url, body("attention.rollout", layer=0, residual_coefficient=0.0))["row"]["values"]
    mean = call(url, body("attention.mean_heads", layer=0, heads=[]))["row"]["values"]
    total = sum(mean)
    out["single_layer_matches_head_mean_max_error"] = max(abs(a-b/total) for a, b in zip(single, mean))
    # Distance recomputed from the raw row and the capture's own patch boxes.
    row = call(url, body("attention.row", layer=5, head=2))["row"]["values"]
    origin = next(p for p in page["positions"] if p["index"] == source)["bbox"]
    centre = lambda b: ((b[0]+b[2])/2, (b[1]+b[3])/2)
    sx, sy = centre(origin)
    numerator = mass = 0.0
    for p in patches:
        cx, cy = centre(p["bbox"])
        numerator += row[p["index"]]*math.hypot(cx-sx, cy-sy)
        mass += row[p["index"]]
    distance = call(url, body("attention.spatial_distance", layer=5, head=2, units="source_pixels"))
    measured = stats(distance)
    out["distance_source_pixels"] = measured["Mean attention distance"]
    out["distance_independent"] = numerator/mass
    out["distance_relative_error"] = abs(measured["Mean attention distance"]-numerator/mass)/(numerator/mass)
    out["distance_excluded_mass"] = measured["Excluded non-image or unresolved mass"]
    out["distance_layer_series_points"] = len(distance["plots"][0]["points"]) if distance["plots"] else 0
    prompt = next(p["index"] for p in page["positions"] if not p["bbox"])
    try:
        call(url, {**body("attention.spatial_distance", layer=5, head=2), "source_index": prompt})
        out["distance_prompt_source_refused"] = False
    except RuntimeError as error:
        out["distance_prompt_source_refused"] = "image-patch source" in str(error)
    return out


def generator_captures():
    captures = [c for c in call("/api/v1/generation/captures?limit=25")["items"]
                if c["provenance"].get("model_id") == "Qwen/Qwen2.5-VL-3B-Instruct" and c["context"].get("representation") == "image"
                and not c["provenance"].get("document_study_id") and not c["provenance"].get("experiment_protocol_id")]
    by_case = {}
    for capture in captures:
        by_case.setdefault(capture["case_id"], capture)
    return list(by_case.values())


def profile_checks(captures):
    # The catalogue refuses a sample row without mapped annotated elements; the similarity method reports
    # such heads as unavailable instead. Cross-check on captures the catalogue accepts individually.
    sample = []
    for capture in captures:
        try:
            call("/api/v1/methods/analyses", {"generation_capture_id": capture["capture_id"], "sample_capture_ids": [capture["capture_id"]],
                "method_id": "attention.head_specialisation", "variant_id": "document_saved_rows_v1",
                "parameters": {"step": 0, "layer": 0, "head": 0, "layers": [0, 20], "heads": list(range(16))}})
            sample.append(capture["capture_id"])
        except RuntimeError:
            continue
        if len(sample) == 5:
            break
    request = {"generation_capture_id": sample[0], "sample_capture_ids": sample,
               "parameters": {"step": 0, "layer": 0, "head": 0, "layers": [0, 20], "heads": list(range(16))}}
    result = call("/api/v1/methods/analyses", {**request, "method_id": "heads.profile_similarity", "variant_id": "reviewed_profile_cosine_v1"})
    catalogue = call("/api/v1/methods/analyses", {**request, "method_id": "attention.head_specialisation", "variant_id": "document_saved_rows_v1"})
    profiles = {f"L{int(v[0])} · H{int(v[1])}": v[2:-1] for v in catalogue["table"]["values"]}
    cosine = lambda p, q: sum(a*b for a, b in zip(p, q))/math.sqrt(sum(a*a for a in p)*sum(b*b for b in q))
    labels = result["table"]["labels"]
    error = 0.0
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            value = result["table"]["values"][i][j]
            if value is not None:
                error = max(error, abs(value-cosine(profiles[a], profiles[b])))
    available = [v for row in result["table"]["values"] for v in row if v is not None]
    return {"sample_capture_ids": sample, "compared_heads": len(labels), "available_pairs": len(available),
            "max_error_vs_catalogue_profiles": error, "minimum_cosine": min(available), "notices": result["notices"]}


def attribution_check(captures, steps=128):
    capture = next(c for c in captures if len(c["context"]["pages"]) == 1)
    study = call("/api/v1/document-studies", {"generation_capture_id": capture["capture_id"], "method_id": "attribution.integrated_gradients",
                                              "variant_id": "input_path_v1", "parameters": {"step": 0, "steps": steps, "baseline": "black"}})
    job = call(f"/api/v1/document-studies/{study['study_id']}/runs", {"idempotency_key": "completed-methods-"+study["study_id"]})
    started = time.time()
    while True:
        detail = call(f"/api/v1/jobs/{job['job_id']}")
        if detail["state"] in ("completed", "failed", "cancelled"):
            break
        time.sleep(3)
    if detail["state"] != "completed":
        return {"capture_id": capture["capture_id"], "job_state": detail["state"], "error": detail.get("error_message")}
    result_study = call(f"/api/v1/document-studies/{detail['output_resource_id']}")
    analysis = call("/api/v1/methods/analyses", {"generation_capture_id": capture["capture_id"], "method_id": "attribution.integrated_gradients",
                                                 "variant_id": "input_path_v1", "parameters": result_study["parameters"],
                                                 "experiment_result_id": result_study["study_id"]})
    outcome = result_study["outcomes"][0]
    measured = stats(analysis)
    return {"capture_id": capture["capture_id"], "study_id": result_study["study_id"], "job_seconds": round(time.time()-started, 1),
            "steps": steps, "baseline": "black", "target_token_id": outcome["target_token_id"], "target_is_greedy": outcome["target_is_greedy"],
            "target_log_probability": outcome["target_log_probability"], "baseline_log_probability": outcome["baseline_log_probability"],
            "sum_of_attributions": measured["Sum of patch attributions"], "completeness_gap": measured["Completeness gap"],
            "relative_completeness_gap": measured["Relative completeness gap"],
            "half_rule_totals": outcome["half_rule_totals"], "half_rule_disagreement": measured["Interleaved half-rule disagreement"],
            "native_log_probability": result_study["provenance"]["attribution_runtime"]["native_log_probability"],
            "attributed_positions": sum(v is not None for v in outcome["attribution"]),
            "peak_gpu_allocated_bytes": result_study["provenance"]["attribution_runtime"]["peak_gpu_allocated_bytes"],
            "top_elements": list(zip(analysis["table"]["labels"][:5], [row[0] for row in analysis["table"]["values"][:5]]))}


def main():
    evidence = {"api": API, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "retriever": retriever_checks()}
    captures = generator_captures()
    evidence["profile_similarity"] = profile_checks(captures)
    if "--skip-gpu" not in sys.argv:
        evidence["integrated_gradients"] = attribution_check(captures)
    destination = ROOT/"var"/"verification"/"completed-methods.json"
    destination.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
