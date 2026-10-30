"""Model matrix: every registered model, checked the same way.

Models come from GET /api/v1/setup/models, so a newly added generator or retriever (built in
or uploaded) is tested without editing this script. Expected layer and head counts come from
each registration, not from this file. Every model gets these checks:

  registration  an installed adapter can run it, with declared language layers and heads
  weights       its pinned snapshot is complete in the model cache (read in the API container)
  kernels       a tiny random model of its language family runs the attention interventions on
                the installed kernels (scripts/attention_kernel_conformance.py, GPU container)
  plan          a GPU worker can admit it now
  capture       one small real capture through the API: identity, axes, the adapter's own
                runtime checks, sampled attention rows, causal support and page mapping
  oracle        the independent conformance oracle on that capture (scripts/attention_conformance.py)
  baseline      the same inputs give the same outputs as the last accepted run

Each check reports pass, fail, warn or skip with a reason. The exit code is 1 when any check
fails, and with --require-all also when any is skipped. The report is written to
var/verification/model-matrix.json and accepted outputs to var/verification/model-matrix-baseline.json.

    .venv/Scripts/python.exe scripts/model_matrix.py                    # all models, real captures
    .venv/Scripts/python.exe scripts/model_matrix.py --skip-runs        # no GPU captures
    .venv/Scripts/python.exe scripts/model_matrix.py --models builtin-colqwen2
    .venv/Scripts/python.exe scripts/model_matrix.py --update-baseline  # accept changed outputs

Captures are saved like any other run. The first passing run of each model records its
baseline; later runs on the same case, revision and adapter are compared with it.
"""
import argparse
import inspect
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ("registration", "weights", "kernels", "plan", "capture", "oracle", "baseline")
ROW_TOLERANCE = .02  # BF16 softmax rounding, declared by the capture contracts.
EXACT = 1e-8


def log(message):
    print(message, file=sys.stderr, flush=True)


def outcome(status, detail="", **extra):
    return {"status": status, "detail": detail, **extra}


class Api:
    def __init__(self, base):
        self.base = base.rstrip("/") + "/api/v1"

    def call(self, path, body=None, raw=False, timeout=300):
        url = path if path.startswith("http") else self.base + path.removeprefix("/api/v1")
        request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
                return response.status, data if raw else json.loads(data)
        except urllib.error.HTTPError as error:
            data = error.read()
            try:
                return error.code, json.loads(data)
            except ValueError:
                return error.code, data.decode(errors="replace")

    def get(self, path):
        status, body = self.call(path)
        if status != 200:
            raise RuntimeError(f"GET {path} returned {status}: {message(body)}")
        return body


def message(body):
    if isinstance(body, dict):
        error = body.get("error")
        return error.get("message", str(error)) if isinstance(error, dict) else str(body.get("detail", body))[:300]
    return str(body)[:300]


def component(model, name):
    return next((c for c in model["architecture"]["components"] if c["name"] == name), {})


class Checks:
    """Named sub-checks folded into one pass/fail outcome."""
    def __init__(self):
        self.items = []

    def __call__(self, name, passed, **detail):
        self.items.append({"check": name, "passed": bool(passed), **detail})
        return passed

    def outcome(self, **extra):
        failed = [c["check"] for c in self.items if not c["passed"]]
        return outcome("fail" if failed else "pass", f"failed: {', '.join(failed)}" if failed else f"{len(self.items)} checks passed",
                       checks=self.items, **extra)


# Steps that run inside the compose containers ---------------------------------------------

def in_container(container, source, env=None, timeout=1800):
    """Python source run inside a compose container: (exit code, stdout, stderr)."""
    command = ["docker", "exec", "-i", *[a for k, v in (env or {}).items() for a in ("-e", f"{k}={v}")], container, "python", "-"]
    done = subprocess.run(command, input=source.encode(), capture_output=True, timeout=timeout)
    return done.returncode, done.stdout.decode(errors="replace"), done.stderr.decode(errors="replace")


def container_json(container, source, env=None, timeout=1800):
    """The JSON report a script printed in a container: (report, None), or (None, outcome) when it gave none."""
    try:
        code, out, err = in_container(container, source, env, timeout)
    except FileNotFoundError:
        return None, outcome("skip", "docker is not installed or not on PATH")
    except subprocess.TimeoutExpired:
        return None, outcome("fail", f"no result from {container} within {timeout} s")
    if "No such container" in err or "is not running" in err:
        return None, outcome("skip", f"container {container} is not running")
    try:
        return json.loads(out[out.index("{"):out.rindex("}") + 1]), None
    except ValueError:
        # The script started but crashed, for example on a model family it cannot build.
        return None, outcome("fail", f"script in {container} printed no report (exit {code}): {err.strip()[-500:]}")


def weights_probe(models, cache):
    """Runs where the model cache is mounted: is each pinned snapshot complete?"""
    import json
    from pathlib import Path
    cache, result = Path(cache), {}
    for m in models:
        if m["uploaded"]:
            result[m["registration_id"]] = {"status": "skip", "detail": "uploaded checkpoint; its files are hash-verified when loaded"}
            continue
        name = "models--" + m["model_id"].replace("/", "--")
        # Generators use the Hugging Face hub layout; the retriever adapters pass the cache root directly.
        folder = next((f for f in (cache / "hub" / name / "snapshots" / m["revision"], cache / name / "snapshots" / m["revision"]) if f.is_dir()), None)
        if folder is None:
            result[m["registration_id"]] = {"status": "fail", "detail": f"revision {m['revision'][:12]} of {m['model_id']} is not in the model cache"}
            continue
        index = folder / "model.safetensors.index.json"
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() else ["model.safetensors"]
        # A link whose blob never finished downloading is not a file.
        missing = [n for n in ("config.json", "tokenizer_config.json", "preprocessor_config.json", *shards) if not (folder / n).is_file()]
        size = sum((folder / n).stat().st_size for n in shards if (folder / n).is_file())
        result[m["registration_id"]] = {"status": "fail" if missing else "pass", "path": str(folder), "weights_bytes": size,
                                        "detail": f"missing {missing}" if missing else f"{len(shards)} weight file(s), {size / 1e9:.2f} GB"}
    return result


def check_weights(models, container):
    if not container:
        return {m["registration_id"]: outcome("skip", "no API container given (--api-container)") for m in models}
    listed = [{"registration_id": m["registration_id"], "model_id": m["model_id"], "revision": m["revision"], "uploaded": bool(m.get("files"))} for m in models]
    source = inspect.getsource(weights_probe) + "\nimport json, os\nprint(json.dumps(weights_probe(json.loads(os.environ['DOCATTENTIONLENS_MATRIX_MODELS']), os.environ.get('DOCATTENTIONLENS_MODEL_CACHE', '/model-cache'))))\n"
    found, problem = container_json(container, source, {"DOCATTENTIONLENS_MATRIX_MODELS": json.dumps(listed)}, timeout=120)
    if problem:
        return {m["registration_id"]: problem for m in models}
    return {m["registration_id"]: outcome(**found[m["registration_id"]]) for m in models}


def check_kernels(models, container):
    if not container:
        return {m["registration_id"]: outcome("skip", "no GPU worker container given (--gpu-container)") for m in models}
    report, problem = container_json(container, (ROOT / "scripts" / "attention_kernel_conformance.py").read_text(encoding="utf-8"), timeout=600)
    if problem:
        return {m["registration_id"]: problem for m in models}
    by_registration = {r: e for e in report["families"] for r in e["registrations"]}
    results = {}
    for m in models:
        # Uploaded checkpoints run on a built-in adapter, so they share that adapter's family result.
        twin = next((r["registration_id"] for r in models if not r.get("files") and r["capabilities"].get("adapter_id") == m["capabilities"].get("adapter_id")), m["registration_id"])
        entry = by_registration.get(twin)
        if entry is None:
            results[m["registration_id"]] = outcome("fail", "no kernel conformance family covers this registration",
                                                    families=sorted({e["family"] for e in report["families"]}))
        elif entry["passed"]:
            results[m["registration_id"]] = outcome("pass", f"{entry['family']} ({'causal' if entry['causal'] else 'bidirectional'}, {entry['rope']} rotary): "
                                                    f"{len(entry['checks'])} interventions", transformers=report["transformers"])
        else:
            results[m["registration_id"]] = outcome("fail", f"{entry['family']}: {entry['error']}", transformers=report["transformers"])
    return results


def check_oracle(captures, container):
    """Independent oracle on the new captures, grouped by capture id."""
    if not captures:
        return {}
    if not container:
        return {c: outcome("skip", "no API container given (--api-container)") for c in captures}
    log(f"Conformance oracle on {len(captures)} new capture(s)...")
    report, problem = container_json(container, (ROOT / "scripts" / "attention_conformance.py").read_text(encoding="utf-8"),
                                     {"DOCATTENTIONLENS_CONFORMANCE_CAPTURES": ",".join(captures)}, timeout=3600)
    if problem:
        return {c: problem for c in captures}
    results = {}
    for capture in captures:
        mine = [r for r in report["results"] if r["subject"].split("/")[0] == capture]
        failed = [r for r in mine if r["status"] == "fail"]
        if not mine:
            results[capture] = outcome("fail", "the oracle recorded no checks for this capture")
        else:
            results[capture] = outcome("fail" if failed else "pass", f"{len(mine) - len(failed)} of {len(mine)} checks passed",
                                       failed=[{k: v for k, v in r.items() if k != "status"} for r in failed[:10]])
    return results


# Real captures ---------------------------------------------------------------------------

def check_plan(api, model, wait=0):
    """Whether a GPU worker admits this model, polled for up to `wait` seconds: (outcome, plan)."""
    deadline = time.monotonic() + wait
    while True:
        status, plan = api.call(f"/{model['role']}/plan?model_registration_id={quote(model['registration_id'])}")
        if status != 200:
            return outcome("fail", f"plan refused ({status}): {message(plan)}"), {}
        if plan["available"]:
            attention = "" if model["role"] == "generation" else (" with attention" if plan.get("attention_available") else " without attention")
            return outcome("pass", f"GPU worker admits it{attention}"), plan
        if time.monotonic() >= deadline:
            return outcome("skip", plan["reason"]), plan
        time.sleep(10)


def choose_case(api, args, baseline):
    """The case every model reads. Reused from the baseline so outputs stay comparable."""
    remembered = args.case or baseline.get("inputs", {}).get("case_id")
    if remembered:
        status, case = api.call(f"/cases/{remembered}")
        if status == 200:
            return case
        if args.case:
            raise RuntimeError(f"Case {args.case} is unavailable: {message(case)}")
    dataset = api.get("/datasets")["items"][0]
    for document in api.get(f"/documents?dataset_id={dataset['dataset_id']}")["items"]:
        question = next((q for q in api.get(f"/questions?document_id={document['document_id']}")["items"] if q["support_page_ids"]), None)
        if question and len(document["page_ids"]) >= 2:
            # The supporting page first, so one-page generation still sees the evidence.
            support = question["support_page_ids"][0]
            pages = [support, next(p for p in document["page_ids"] if p != support)]
            status, case = api.call("/cases", {"question_id": question["question_id"], "page_ids": pages})
            if status != 201:
                raise RuntimeError(f"Could not create the matrix case: {message(case)}")
            return case
    raise RuntimeError("No document with a supported question and two pages")


def worker_error(container, job):
    """The exception the GPU worker logged for this job, since the API only reports a generic failure."""
    if not container:
        return None
    try:
        logs = subprocess.run(["docker", "logs", "--since", "2h", container], capture_output=True, timeout=60)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    lines = (logs.stdout + logs.stderr).decode(errors="replace").splitlines()
    start = next((i for i in range(len(lines) - 1, -1, -1) if job["job_id"] in lines[i] and job["attempt_id"] in lines[i]), None)
    if start is None:
        return None
    # The traceback ends at its first unindented line naming the exception.
    return next((line for line in lines[start + 1:start + 200] if line and not line[0].isspace() and ("Error" in line or "Exception" in line)
                 and not line.startswith("Traceback")), None)


def run_job(api, path, body, timeout, label, container=None):
    status, accepted = api.call(path, body)
    if status != 202:
        return None, f"launch refused ({status}): {message(accepted)}"
    started, reported = time.monotonic(), 0.0
    while time.monotonic() - started < timeout:
        job = api.get(f"/jobs/{accepted['job_id']}")
        if job["state"] == "completed":
            log(f"[{label}] capture {job['output_resource_id']} in {time.monotonic() - started:.0f} s")
            return job["output_resource_id"], None
        if job["state"] in ("failed", "cancelled"):
            cause = worker_error(container, job) if job["state"] == "failed" else None
            return None, f"job {accepted['job_id']} {job['state']}: {cause or (str(job.get('error_code')) + ' ' + str(job.get('error_message')))}"
        if time.monotonic() - reported > 30:
            log(f"[{label}] {job['state']} for {time.monotonic() - started:.0f} s ({accepted['job_id']})")
            reported = time.monotonic()
        time.sleep(3)
    api.call(f"/jobs/{accepted['job_id']}/cancel", {})
    return None, f"job {accepted['job_id']} did not finish within {timeout} s and was cancelled"


def tiling(boxes, width, height):
    """Patch boxes must stay on the page and their areas sum to the page area (the oracle checks the exact grid)."""
    problems = []
    if not all(0 <= b[0] < b[2] <= width + 1e-6 and 0 <= b[1] < b[3] <= height + 1e-6 for b in boxes):
        problems.append("patch box outside the page")
    area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
    if abs(area - width * height) > 1e-6 * width * height:
        problems.append(f"patches cover {area / (width * height):.6f} of the page")
    return problems


def row_problems(row, targets):
    values, problems = row["values"], []
    if len(values) != targets:
        problems.append(f"{len(values)} values for {targets} targets")
    if values and min(values) < 0:
        problems.append("negative weight")
    if abs(row["total_mass"] - 1) > ROW_TOLERANCE:
        problems.append(f"row mass {row['total_mass']:.6f}")
    if abs(sum(a["mass"] for a in row["allocations"]) - row["total_mass"]) > EXACT or row["allocation_error"] > EXACT:
        problems.append("allocation does not conserve the row")
    if row["image_mass"] > row["total_mass"] + EXACT:
        problems.append("image mass exceeds row mass")
    return problems


def png_size(data):
    return (int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")) if data[:8] == b"\x89PNG\r\n\x1a\n" else None


def check_generation(api, model, capture, pages):
    check = Checks()
    language, vision = component(model, "Language"), component(model, "Vision")
    provenance, validation, context = capture["provenance"], capture["validation"], capture["context"]
    check("identity", (provenance.get("model_id"), provenance.get("model_revision"), provenance.get("adapter_version"))
          == (model["model_id"], model["revision"], model["capabilities"]["adapter_id"]), recorded=[provenance.get("model_id"), provenance.get("model_revision"), provenance.get("adapter_version")])
    declared = [language.get("layers"), language.get("query_heads"), language.get("kv_heads")]
    captured = [capture["layer_count"], capture["query_head_count"], capture["kv_head_count"]]
    check("axes", captured == declared, captured=captured, declared=declared)
    check("runtime_checks", validation.get("status") == "passed_for_this_capture" and validation.get("capture_on_off_token_equality") is True
          and validation.get("capture_on_off_logits_max_error", 1) <= validation.get("prediction_equivalence_tolerance", 0)
          and validation.get("max_attention_mass_error", 1) <= validation.get("attention_mass_tolerance", 0)
          and validation.get("question_rows_future_mass", 0) == 0,
          validation={k: validation.get(k) for k in ("status", "capture_on_off_token_equality", "capture_on_off_logits_max_error",
                                                    "max_attention_mass_error", "question_rows_future_mass")})
    check("inputs", context["selected_page_ids"] == pages and context["evaluation_labels_consumed"] is False)
    n, steps = len(context["input_token_ids"]), capture["steps"]
    check("prediction_alignment", bool(steps) and all(
        (s["source_index"], s["predicted_index"], s["target_count"]) == (n + s["step"] - 1, n + s["step"], n + s["step"])
        and capture["output_token_ids"][s["step"]] == s["predicted_token_id"] for s in steps), steps=len(steps))
    # First and last captured step, layer and head: the corners of every axis the adapter declares.
    sampled, problems = 0, []
    for step in sorted({steps[0]["step"], steps[-1]["step"]} if steps else set()):
        for layer in sorted({0, capture["layer_count"] - 1}):
            for head in sorted({0, capture["query_head_count"] - 1}):
                status, row = api.call(f"/generation/captures/{capture['capture_id']}/rows?step={step}&layer={layer}&head={head}")
                sampled += 1
                found = row_problems(row, n + step) if status == 200 else [f"HTTP {status}: {message(row)}"]
                problems += [f"step {step} layer {layer} head {head}: {p}" for p in found]
    check("attention_rows", sampled and not problems, sampled=sampled, problems=problems[:10])
    merge = vision.get("spatial_merge_size")
    for page in context["pages"]:
        positions = [capture["positions"][i] for i in page["position_indices"]]
        found = tiling([p["bbox"] for p in positions], page["source_width"], page["source_height"])
        t, h, w = page["grid_thw"]
        if merge and len(positions) != t * h * w // merge ** 2:
            found.append(f"{len(positions)} image positions for a {t}x{h}x{w} grid merged {merge}x{merge}")
        if not all(p["role"] == "image" and p["page_id"] == page["page_id"] and abs(sum(p["ownership"].values()) - 1) < EXACT for p in positions):
            found.append("image positions with the wrong page or unconserved ownership")
        status, data = api.call(page["preview_url"], raw=True)
        if status != 200 or png_size(data) != (page["resized_width"], page["resized_height"]):
            found.append("consumed input image does not match its declared size")
        check(f"page_mapping:{page['page_number']}", not found, positions=len(positions), problems=found)
    return check.outcome(capture_id=capture["capture_id"], answer=capture["answer"])


def check_retrieval(api, model, capture, pages, attention):
    check = Checks()
    language, limits = component(model, "Language"), model["capabilities"].get("limits", {})
    provenance, validation = capture["provenance"], capture["validation"]
    check("identity", (provenance.get("model_id"), provenance.get("model_revision"), provenance.get("adapter_version"))
          == (model["model_id"], model["revision"], model["capabilities"]["adapter_id"])
          and provenance.get("model_registration_id") in (None, model["registration_id"]),
          recorded=[provenance.get("model_id"), provenance.get("model_revision"), provenance.get("adapter_version")])
    runtime = (validation.get("max_reconstruction_error", 1) <= validation.get("score_tolerance_absolute", 0)
               and validation.get("ownership_conserved") is True and validation.get("processor_fixture") is True)
    if attention:
        a = validation.get("attention") or {}
        # Each adapter declares its own embedding gate: ColQwen2 a mean per-position L2 distance, ColPali a maximum coordinate.
        embeddings = (a.get("mean_embedding_kernel_l2_error", 1) <= a["embedding_kernel_l2_tolerance"] if "embedding_kernel_l2_tolerance" in a
                      else a.get("max_embedding_kernel_error", 1) <= a.get("embedding_kernel_tolerance", 0))
        runtime = runtime and (a.get("capture_on_off_embeddings_equal") is True and a.get("candidate_ranking_equal") is True
                               and a.get("runtime_mask_verified") is True and a.get("max_row_sum_error", 1) <= ROW_TOLERANCE
                               and a.get("max_score_kernel_error", 1) <= a.get("score_kernel_tolerance", 0)
                               and embeddings and a.get("runtime_positions_verified", True) is True
                               and a.get("causal_future_mass", 0) == 0)
    check("runtime_checks", runtime, validation=validation)
    check("inputs", sorted(p["page_id"] for p in capture["pages"]) == sorted(pages))
    query = next(q["index"] for q in capture["query_positions"] if q["eligible"])
    for page in capture["pages"]:
        visual = [q for q in page["positions"] if q["bbox"]]
        found = tiling([q["bbox"] for q in visual], page["width"], page["height"])
        if page["reconstruction_error"] > 1e-4 or abs(sum(page["contributions"].values()) - page["score"]) > 1e-4:
            found.append("MaxSim score is not reconstructed by its contributions")
        if not 0 < len(visual) <= limits.get("image_positions_per_page", len(visual)):
            found.append(f"{len(visual)} image positions (limit {limits.get('image_positions_per_page')})")
        if not all(abs(sum(o.values()) - 1) < EXACT for o in page["ownership"]):
            found.append("ownership not conserved")
        status, row = api.call(f"/retrieval/captures/{capture['capture_id']}/rows?page_id={page['page_id']}&query_index={query}")
        present = [s for s in row["similarities"] if s is not None] if status == 200 else []
        if not present or abs(max(present) - row["maximum"]) > 1e-6:
            found.append("similarity row maximum disagrees with its values")
        check(f"page:{page['page_number']}", not found, positions=len(visual), score=page["score"], problems=found)
    if attention:
        encodings = capture.get("attention_encodings") or []
        check("encodings_present", any(e["pass_kind"] == "query" for e in encodings) and any(e["pass_kind"] != "query" for e in encodings),
              encodings=[e["encoding_id"] for e in encodings])
        for encoding in encodings:
            found = []
            declared = [language.get("layers"), language.get("query_heads")]
            if [encoding["layer_count"], encoding["query_head_count"]] != declared:
                found.append(f"axes {[encoding['layer_count'], encoding['query_head_count']]}, declared {declared}")
            active = [i for i, m in enumerate(encoding["attention_mask"]) if m]
            # The middle source checks causal support (later keys must be zero); the last sees every key.
            for source in sorted({active[len(active) // 2], active[-1]}):
                status, result = api.call(f"/retrieval/captures/{capture['capture_id']}/attention", {
                    "encoding_id": encoding["encoding_id"], "source_index": source, "method_id": "attention.row",
                    "parameters": {"layer": encoding["layer_count"] - 1, "head": encoding["query_head_count"] - 1}})
                if status != 200 or not result.get("row"):
                    found.append(f"source {source}: HTTP {status}: {message(result)}")
                    continue
                found += [f"source {source}: {p}" for p in row_problems(result["row"], encoding["position_count"])]
                if encoding.get("causal") and any(v != 0 for v in result["row"]["values"][source + 1:]):
                    found.append(f"source {source}: attention to later keys in a causal encoder")
            check(f"encoding:{encoding['encoding_id']}", not found, causal=bool(encoding.get("causal")), problems=found)
    return check.outcome(capture_id=capture["capture_id"])


def capture_model(api, model, inputs, args):
    """Launch one small capture of this model and check it. Returns (outcome, capture or None)."""
    label, identifier = model["registration_id"], model["registration_id"]
    key = f"model-matrix-{uuid4().hex}"
    if model["role"] == "generation":
        pages = inputs["generation_page_ids"]
        capture_id, error = run_job(api, "/generation/runs", {"case_id": inputs["case_id"], "page_ids": pages, "model_registration_id": identifier,
                                                              "idempotency_key": key}, args.timeout, label, args.gpu_container)
        if error:
            return outcome("fail", error), None
        capture = api.get(f"/generation/captures/{capture_id}")
        return check_generation(api, model, capture, pages), capture
    pages, attention = inputs["retrieval_page_ids"], inputs["attention"][identifier]
    body = {"case_id": inputs["case_id"], "page_ids": pages, "model_registration_id": identifier, "idempotency_key": key}
    if attention:
        body.update(capture_attention=True, attention_page_id=pages[0])
    capture_id, error = run_job(api, "/retrieval/runs", body, args.timeout, label, args.gpu_container)
    if error:
        return outcome("fail", error), None
    capture = api.get(f"/retrieval/captures/{capture_id}")
    return check_retrieval(api, model, capture, pages, attention), capture


# Baselines -------------------------------------------------------------------------------

def fingerprint(role, capture):
    arrays = {a["name"]: a["sha256"] for a in capture["arrays"]}
    if role == "generation":
        return {"answer": capture["answer"], "output_token_ids": capture["output_token_ids"], "arrays": arrays}
    return {"scores": {p["page_id"]: p["score"] for p in capture["pages"]}, "arrays": arrays}


def versions(capture):
    p = capture["provenance"]
    return {k: p[k] for k in ("frameworks", "torch", "transformers", "code_sha256") if k in p}


def compare(role, accepted, capture):
    old, new = accepted["fingerprint"], fingerprint(role, capture)
    context = {"accepted_capture": accepted["capture_id"], "accepted_versions": accepted.get("versions"), "versions": versions(capture)}
    if role == "generation" and old["output_token_ids"] != new["output_token_ids"]:
        return outcome("fail", "generated tokens changed", accepted_answer=old["answer"], answer=new["answer"], **context)
    if role == "retrieval":
        moved = sorted(k for k in old["scores"].keys() | new["scores"].keys() if abs(old["scores"].get(k, float("inf")) - new["scores"].get(k, float("-inf"))) > 1e-4)
        if moved:
            return outcome("fail", f"MaxSim scores changed on {len(moved)} page(s)", accepted_scores=old["scores"], scores=new["scores"], **context)
    differing = sorted(k for k in old["arrays"].keys() | new["arrays"].keys() if old["arrays"].get(k) != new["arrays"].get(k))
    if differing:
        return outcome("warn", f"outputs unchanged but {len(differing)} of {len(new['arrays'])} saved arrays differ", arrays=differing[:10], **context)
    return outcome("pass", f"identical to accepted capture {accepted['capture_id']}")


# Report ----------------------------------------------------------------------------------

def table(results):
    width = max(len(r["registration_id"]) for r in results) + 2
    lines = ["model".ljust(width) + "role".ljust(12) + "".join(c.ljust(14) for c in CHECKS)]
    for r in results:
        lines.append(r["registration_id"].ljust(width) + r["role"].ljust(12) + "".join(r["checks"][c]["status"].ljust(14) for c in CHECKS))
    # Skips that only follow from an earlier choice or check are left out of the notes.
    consequences = ("--skip-runs", "no new capture", "registration cannot run")
    notes = [f"  {r['registration_id']} - {c}: {r['checks'][c]['status']}: {r['checks'][c]['detail']}"
             for r in results for c in CHECKS if r["checks"][c]["status"] in ("fail", "warn", "skip") and r["checks"][c]["detail"] not in consequences]
    return "\n".join(lines + (["", "Not passing:"] + notes if notes else []))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--models", nargs="+", help="Registration ids to check (default: every registration)")
    parser.add_argument("--case", help="Case every model reads (default: the baseline's case, or a new one)")
    parser.add_argument("--generation-pages", type=int, choices=(1, 2), default=1, help="Pages supplied to generators")
    parser.add_argument("--skip-runs", action="store_true", help="Registration, weights, kernels and plan only; no GPU captures")
    parser.add_argument("--allow-download", action="store_true", help="Run models whose weights are incomplete (the GPU job downloads them)")
    parser.add_argument("--api-container", default="docattentionlens-api-1", help="Reads the model cache and runs the oracle; '' to skip")
    parser.add_argument("--gpu-container", default="docattentionlens-worker-gpu-1", help="Runs kernel conformance; '' to skip")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds allowed for each capture")
    parser.add_argument("--plan-wait", type=int, default=120, help="Seconds to wait for a GPU worker to admit a model before skipping it")
    parser.add_argument("--baseline", default="var/verification/model-matrix-baseline.json")
    parser.add_argument("--update-baseline", action="store_true", help="Accept this run's outputs for models whose other checks pass")
    parser.add_argument("--require-all", action="store_true", help="Treat skipped checks as failures")
    parser.add_argument("--output", default="var/verification/model-matrix.json")
    args = parser.parse_args()
    api, started = Api(args.base_url), datetime.now(UTC).isoformat()
    models = api.get("/setup/models")
    if args.models:
        unknown = set(args.models) - {m["registration_id"] for m in models}
        if unknown:
            parser.error(f"unknown registration(s): {', '.join(sorted(unknown))}")
        models = [m for m in models if m["registration_id"] in args.models]
    log(f"{len(models)} model(s): {', '.join(m['registration_id'] for m in models)}")
    baseline_path = ROOT / args.baseline
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else {}
    results = {m["registration_id"]: {"registration_id": m["registration_id"], "name": m["name"], "role": m["role"], "model_id": m["model_id"],
                                      "revision": m["revision"], "checks": {}} for m in models}

    for m in models:
        language, capabilities = component(m, "Language"), m["capabilities"]
        if not capabilities.get("can_run"):
            results[m["registration_id"]]["checks"]["registration"] = outcome("skip", f"no installed adapter: {capabilities.get('reason')}")
        elif not (language.get("layers") and language.get("query_heads")):
            results[m["registration_id"]]["checks"]["registration"] = outcome("fail", "registration declares no language layers or heads")
        else:
            results[m["registration_id"]]["checks"]["registration"] = outcome("pass", f"{capabilities['adapter_id']}: {language['layers']} layers x "
                                                                              f"{language['query_heads']} query heads")
    runnable = [m for m in models if results[m["registration_id"]]["checks"]["registration"]["status"] == "pass"]
    log("Checking model cache...")
    for identifier, result in check_weights(runnable, args.api_container).items():
        results[identifier]["checks"]["weights"] = result
    log("Running kernel conformance...")
    for identifier, result in check_kernels(runnable, args.gpu_container).items():
        results[identifier]["checks"]["kernels"] = result
    for m in runnable:
        results[m["registration_id"]]["checks"]["plan"] = check_plan(api, m)[0]

    captures, inputs = {}, None
    for m in runnable:
        identifier, checks = m["registration_id"], results[m["registration_id"]]["checks"]
        if args.skip_runs:
            checks["capture"] = outcome("skip", "--skip-runs")
            continue
        if checks["weights"]["status"] == "fail" and not args.allow_download:
            checks["capture"] = outcome("skip", "weights incomplete; --allow-download lets the GPU job download them")
            continue
        # Re-checked at launch: free GPU memory changes as the worker swaps models.
        checks["plan"], plan = check_plan(api, m, args.plan_wait)
        if checks["plan"]["status"] != "pass":
            checks["capture"] = outcome("skip", "the GPU worker cannot admit it")
            continue
        if inputs is None:
            case = choose_case(api, args, baseline)
            inputs = {"case_id": case["case_id"], "generation_page_ids": case["page_ids"][:args.generation_pages],
                      "retrieval_page_ids": case["page_ids"][:2], "attention": {}}
            log(f"Case {case['case_id']}: {case['query'][:90]}")
        if m["role"] == "retrieval":
            inputs["attention"][identifier] = bool(plan.get("attention_available"))
        log(f"[{identifier}] launching {m['role']} capture...")
        try:
            checks["capture"], capture = capture_model(api, m, inputs, args)
        except Exception as error:
            checks["capture"], capture = outcome("fail", f"{type(error).__name__}: {error}"), None
        if capture:
            captures[identifier] = capture
            results[identifier]["capture_id"] = capture["capture_id"]

    oracle = check_oracle([c["capture_id"] for c in captures.values()], args.api_container)
    accepted = baseline.get("models", {}) if baseline.get("inputs", {}).get("case_id") == (inputs or {}).get("case_id") else {}
    recorded = {}
    for m in models:
        identifier, checks = m["registration_id"], results[m["registration_id"]]["checks"]
        capture = captures.get(identifier)
        checks["oracle"] = oracle.get(capture["capture_id"]) if capture else outcome("skip", "no new capture")
        if not capture:
            checks["baseline"] = outcome("skip", "no new capture")
            continue
        options = {"pages": inputs["generation_page_ids"] if m["role"] == "generation" else inputs["retrieval_page_ids"],
                   "attention": inputs["attention"].get(identifier)}
        entry = accepted.get(identifier)
        comparable = bool(entry) and (entry["model_revision"], entry["adapter_version"], entry["options"]) == (m["revision"], m["capabilities"]["adapter_id"], options)
        healthy = checks["capture"]["status"] == "pass" and checks["oracle"]["status"] != "fail"
        if comparable:
            checks["baseline"] = compare(m["role"], entry, capture)
        elif entry:
            checks["baseline"] = outcome("skip", "the accepted run used another revision, adapter or input pages; --update-baseline accepts this one")
        else:
            checks["baseline"] = outcome("skip", "no accepted run yet")
        # The first healthy run becomes the baseline; replacing one needs --update-baseline.
        if healthy and (entry is None or args.update_baseline):
            recorded[identifier] = {"model_revision": m["revision"], "adapter_version": m["capabilities"]["adapter_id"], "options": options,
                                    "capture_id": capture["capture_id"], "accepted_at": datetime.now(UTC).isoformat(),
                                    "versions": versions(capture), "fingerprint": fingerprint(m["role"], capture)}
            if entry is None:
                checks["baseline"] = outcome("pass", f"baseline recorded from {capture['capture_id']}")
            elif checks["baseline"]["status"] != "pass":
                checks["baseline"] = outcome("pass", f"--update-baseline accepted {capture['capture_id']} ({checks['baseline']['detail']})")
    for result in results.values():
        for c in CHECKS:
            result["checks"].setdefault(c, outcome("skip", "registration cannot run"))
    if recorded:
        models_entry = (accepted if accepted else {}) | recorded
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps({"inputs": {k: v for k, v in inputs.items() if k != "attention"}, "models": models_entry}, indent=1), encoding="utf-8")

    ordered = [results[m["registration_id"]] for m in models]
    statuses = [r["checks"][c]["status"] for r in ordered for c in CHECKS]
    failed = statuses.count("fail") + (statuses.count("skip") if args.require_all else 0)
    report = {"started_at": started, "finished_at": datetime.now(UTC).isoformat(), "api": args.base_url, "inputs": inputs,
              "summary": {s: statuses.count(s) for s in ("pass", "warn", "skip", "fail")}, "models": ordered}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(table(ordered))
    print(f"\n{report['summary']} - report {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
