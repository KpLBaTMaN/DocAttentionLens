"""Independent conformance oracle for saved attention, similarity and geometry.

Runs where the artifact store is readable (normally inside the API container):

    docker exec -i docattentionlens-api-1 python - < scripts/attention_conformance.py > var/verification/attention-conformance.json

From PowerShell, wrap that command in `& $env:ComSpec /c "..."`: PowerShell has no `<`
redirection and Windows PowerShell 5.1 writes `>` output as UTF-16.

Set DOCATTENTIONLENS_CONFORMANCE_CAPTURES to a comma-separated list of capture ids to check only
those captures, all deeply (`scripts/model_matrix.py` does this for its new captures).

It never imports `docattentionlens`. Reference values come from the
raw saved `.npy` arrays and manifests, using formulas written separately from the
method modules (for example Gini from the mean absolute difference rather than
sorted ranks, Jensen-Shannon from the entropy identity, ownership from point
sampling rather than the exact cell partition). The API results are then
compared with those references. Method analyses are content-addressed caches, so
POST /methods/analyses may add cache records; nothing else is written.
"""
import json
import math
import os
import sys
import urllib.error
import urllib.request
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image

API = os.environ.get("DOCATTENTIONLENS_CONFORMANCE_API", "http://127.0.0.1:8000/api/v1")
ROOT = Path(os.environ.get("DOCATTENTIONLENS_ARTIFACT_ROOT", "/artifacts"))
SAMPLES = int(os.environ.get("DOCATTENTIONLENS_CONFORMANCE_SAMPLES", "3"))
RNG = np.random.default_rng(20261012)
ROW_TOLERANCE = .02  # BF16 softmax rounding, declared by the capture contract.
EXACT = 1e-6
results, observations = [], {}


def call(path, body=None, raw=False):
    request = urllib.request.Request(API + path if path.startswith("/") else path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"content-type": "application/json"}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
            return response.status, data if raw else json.loads(data)
    except urllib.error.HTTPError as error:
        data = error.read()
        try:
            return error.code, json.loads(data)
        except ValueError:
            return error.code, data.decode(errors="replace")


def record(check, subject, passed, **detail):
    results.append({"check": check, "subject": subject, "status": "pass" if passed else "fail", **detail})
    if not passed:
        print(f"FAIL {check} {subject} {json.dumps(detail, default=str)[:400]}", file=sys.stderr)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def manifest(kind, identifier):
    folder = ROOT / kind / identifier
    content = json.loads((folder / "manifest.json").read_bytes())
    digest = sha256(canonical({k: v for k, v in content.items() if k != "content_sha256"})).hexdigest()
    record("manifest_hash", identifier, digest == content["content_sha256"])
    return folder, content


def array(folder, content, name):
    ref = next(r for r in content["arrays"] if r["name"] == name)
    raw = (folder / f"{name}.npy").read_bytes()
    values = np.load(BytesIO(raw), allow_pickle=False)
    if sha256(raw).hexdigest() != ref["sha256"] or list(values.shape) != ref["shape"] or values.dtype != np.float32:
        record("array_integrity", f"{content['capture_id']}/{name}", False)
    return values


# Independent reference formulas -------------------------------------------------

def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * (np.log(p) / math.log(2))).sum())


def gini_mean_difference(p):
    # G = sum_i sum_j |x_i - x_j| / (2 n^2 mean), the textbook definition.
    n = len(p)
    return float(np.abs(p[:, None] - p[None, :]).sum() / (2 * n * n * p.mean()))


def jensen_shannon_bits(p, q):
    # JSD = H(M) - (H(P) + H(Q)) / 2, independent of the KL form used in the method module.
    return entropy_bits((p + q) / 2) - (entropy_bits(p) + entropy_bits(q)) / 2


def close(a, b, tolerance=EXACT):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= tolerance)), float(np.max(np.abs(a - b))) if a.shape == b.shape and a.size else None


def allocation_reference(weights, positions, valid=None):
    """Ownership ledger over the key support; positions outside it (masked, or later keys of a causal encoder) are left out."""
    ledger = {}
    for i, (position, weight) in enumerate(zip(positions, weights)):
        if valid is not None and not valid[i]:
            continue
        for bucket, fraction in position["ownership"].items():
            key = f"{bucket}@{position.get('page_id')}"
            ledger[key] = ledger.get(key, 0.0) + float(weight) * fraction
    return ledger


def compare_row(check, subject, row, weights, positions, quantity="post_softmax_attention_weight", valid=None):
    ok, error = close(row["values"], weights)
    image = float(sum(w for w, p in zip(weights, positions) if p["role"] == "image"))
    ledger = allocation_reference(weights, positions, valid)
    returned = {f"{a['bucket']}@{a['page_id']}": a["mass"] for a in row["allocations"]}
    ledger_ok = set(ledger) == set(returned) and all(abs(ledger[k] - returned[k]) < 1e-6 for k in ledger)
    passed = (ok and abs(row["total_mass"] - float(np.sum(weights))) < 1e-6 and abs(row["image_mass"] - image) < 1e-6
              and ledger_ok and abs(row["display_max"] - float(np.max(weights))) < EXACT and row["quantity"] == quantity
              and abs(sum(returned.values()) - row["total_mass"]) < 1e-6)
    record(check, subject, passed, max_value_error=error, ledger_matches=ledger_ok)


# Geometry ---------------------------------------------------------------------

def page_assets(page_id):
    status, page = call(f"/pages/{page_id}")
    status_e, elements = call(f"/pages/{page_id}/elements")
    _, raw = call(page["image_url"].removeprefix("/api/v1"), raw=True)
    return page, elements["items"], np.asarray(Image.open(BytesIO(raw)).convert("RGB"), dtype=np.float64)


def sampled_ownership(box, elements, samples=40):
    x1, y1, x2, y2 = box
    xs = x1 + (np.arange(samples) + .5) * (x2 - x1) / samples
    ys = y1 + (np.arange(samples) + .5) * (y2 - y1) / samples
    shares = {}
    for y in ys:
        for x in xs:
            labels = ["unreviewed_geometry" if e["geometry_flags"] else e["element_id"] for e in elements
                      if e["bbox"]["x2"] > e["bbox"]["x1"] and e["bbox"]["y2"] > e["bbox"]["y1"]
                      and e["bbox"]["x1"] <= x < e["bbox"]["x2"] and e["bbox"]["y1"] <= y < e["bbox"]["y2"]] or ["background"]
            for label in labels:
                shares[label] = shares.get(label, 0.0) + 1 / len(labels) / samples ** 2
    return shares


def ownership_error(expected, sampled):
    return max(abs(expected.get(k, 0) - sampled.get(k, 0)) for k in set(expected) | set(sampled))


def tiling(boxes, width, height, columns, rows):
    """Boxes in row-major order must tile the page exactly, with no gaps or overlaps."""
    boxes = np.asarray(boxes, dtype=np.float64)
    expected = np.array([[c * width / columns, r * height / rows, (c + 1) * width / columns, (r + 1) * height / rows]
                         for r in range(rows) for c in range(columns)])
    area = float(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).sum())
    return len(boxes) == rows * columns and np.allclose(boxes, expected, atol=1e-6) and abs(area - width * height) < 1e-3 * width * height


def mean_colours(image, boxes):
    integral = np.pad(image.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    colours = []
    for x1, y1, x2, y2 in boxes:
        a, b, c, d = int(round(y1)), int(round(y2)), int(round(x1)), int(round(x2))
        total = integral[b, d] - integral[a, d] - integral[b, c] + integral[a, c]
        colours.append(total / max(1, (b - a) * (d - c)))
    return np.asarray(colours)


# Generator captures -----------------------------------------------------------

def generator(identifier, deep):
    folder, capture = manifest("generation", identifier)
    positions, steps, n = capture["positions"], capture["steps"], len(capture["context"]["input_token_ids"])
    tokens = capture["context"]["input_token_ids"] + capture["output_token_ids"]
    L, H = capture["layer_count"], capture["query_head_count"]
    rows = {}
    for step in steps:
        values = array(folder, capture, step["array"])
        k = step["step"]
        sums = values.astype(np.float64).sum(-1)
        aligned = (step["source_index"] == n + k - 1 and step["target_count"] == n + k and step["predicted_index"] == n + k
                   and step["consumed_token_id"] == tokens[n + k - 1] and step["predicted_token_id"] == tokens[n + k]
                   and values.shape == (L, H, n + k))
        record("prediction_alignment", f"{identifier}/step_{k}", aligned)
        record("row_normalization", f"{identifier}/step_{k}", float(np.abs(sums - 1).max()) <= ROW_TOLERANCE and float(values.min()) >= 0,
               max_row_sum_error=float(np.abs(sums - 1).max()))
        rows[k] = values
    for source in capture.get("source_rows", []):
        values = array(folder, capture, source["array"])
        i = source["source_index"]
        sums = values.astype(np.float64).sum(-1)
        record("question_row_alignment", f"{identifier}/question_{i}",
               positions[i]["role"] == "query" and source["target_count"] == i + 1 and values.shape == (L, H, i + 1)
               and source["predicted_token_id"] == tokens[i + 1] and float(np.abs(sums - 1).max()) <= ROW_TOLERANCE,
               max_row_sum_error=float(np.abs(sums - 1).max()))
    # Geometry: patch footprints tile each consumed page in row-major order and show the consumed pixels.
    image_count = 0
    for page in capture["context"]["pages"]:
        t, h, w = page["grid_thw"]
        indices = page["position_indices"]
        image_count += len(indices)
        boxes = [positions[i]["bbox"] for i in indices]
        own = all(positions[i]["page_id"] == page["page_id"] and positions[i]["role"] == "image" for i in indices)
        record("patch_tiling", f"{identifier}/{page['page_id'][:20]}", own and len(indices) == t * h * w // 4
               and indices == list(range(indices[0], indices[0] + len(indices))) and tiling(boxes, page["source_width"], page["source_height"], w // 2, h // 2))
        if not deep:
            continue
        info, elements, original = page_assets(page["page_id"])
        # Patch side from the saved row width (3 channels x 2 frames x side²): 14 px Qwen2/2.5-VL, 16 px Qwen3-VL.
        saved = array(folder, capture, page["pixel_array"])
        side = int(round((saved.shape[1] / 6) ** .5))
        pixels = saved.reshape(-1, 4, 3, 2, side, side)
        ip = capture["provenance"]["image_processor"]
        consumed = (pixels[:, :, :, 0].mean(axis=(1, 3, 4)) * np.asarray(ip["std"]) + np.asarray(ip["mean"])) / ip["rescale"]
        original_colours = mean_colours(original, boxes)
        flipped = mean_colours(original, [boxes[(h // 2 - 1 - r) * (w // 2) + c] for r in range(h // 2) for c in range(w // 2)])
        mapped = float(np.corrcoef(consumed.ravel(), original_colours.ravel())[0, 1])
        control = float(np.corrcoef(consumed.ravel(), flipped.ravel())[0, 1])
        difference = float(np.abs(consumed - original_colours).mean())
        control_difference = float(np.abs(consumed - flipped).mean())
        # Near-uniform pages make any mapping look similar; the control reports whether this page discriminates.
        record("patch_pixel_correspondence", f"{identifier}/{page['page_id'][:20]}", mapped > .99 and difference < 3,
               correlation=mapped, mean_abs_rgb_difference=difference, flipped_correlation=control,
               flipped_mean_abs_rgb_difference=control_difference, control_discriminative=control_difference > 5 * difference)
        sample = RNG.choice(indices, size=min(40, len(indices)), replace=False)
        errors = [ownership_error(positions[i]["ownership"], sampled_ownership(positions[i]["bbox"], elements)) for i in sample]
        record("ownership_sampling", f"{identifier}/{page['page_id'][:20]}", max(errors) <= .06,
               sampled_patches=len(sample), max_error=max(errors), mean_error=float(np.mean(errors)))
    record("image_position_count", identifier, image_count == sum(p["role"] == "image" for p in positions))
    record("ownership_conservation", identifier, all(abs(sum(p["ownership"].values()) - 1) < 1e-8 and min(p["ownership"].values()) >= 0 for p in positions))

    # Raw rows through the API; GET requests store nothing.
    for k, values in rows.items():
        for layer, head in {(0, 0), (L - 1, H - 1), (int(RNG.integers(L)), int(RNG.integers(H)))}:
            status, row = call(f"/generation/captures/{identifier}/rows?step={k}&layer={layer}&head={head}")
            weights = values[layer, head].astype(np.float64)
            if status != 200:
                record("api_raw_row", f"{identifier}/s{k}/l{layer}/h{head}", False, http_status=status, body=row)
                continue
            compare_row("api_raw_row", f"{identifier}/s{k}/l{layer}/h{head}", row, weights, positions[:len(weights)])
    if not deep:
        return
    methods(identifier, capture, rows)
    columns(identifier, capture, folder, rows)
    question_sources(identifier, capture, folder)
    describe(identifier, capture, rows)


def analysis(identifier, method, variant, parameters, source_index=None):
    body = {"generation_capture_id": identifier, "method_id": method, "variant_id": variant, "parameters": parameters}
    if source_index is not None:
        body["source_index"] = source_index
    return call("/methods/analyses", body)


def statistics(result):
    return {s["name"]: s["value"] for s in result["statistics"]}


def check_methods(identifier, label, positions, row_values, run, valid=None, mapped=True):
    """`row_values[layer, head]` is the saved row; `run(method, variant, parameters)` returns the API result.

    `valid` is the declared key support (all targets for causal generator rows). Without
    `mapped` document elements in that support, element allocation must be refused.
    """
    L, H = row_values.shape[:2]
    valid = np.ones(row_values.shape[-1], dtype=bool) if valid is None else valid
    for _ in range(SAMPLES):
        layer, head = int(RNG.integers(L)), int(RNG.integers(H))
        heads = sorted(RNG.choice(H, size=int(RNG.integers(2, H + 1)), replace=False).tolist())
        rows = row_values[layer].astype(np.float64)
        a = rows[head]
        p = a[valid] / a[valid].sum()
        subject = f"{label}/l{layer}/h{head}"
        expected = {
            ("attention.row", "captured_weights_v1", "row"): a,
            ("attention.element_allocation", "ownership_mass_v1", "row"): a,
            ("attention.mean_heads", "captured_prediction_v1", "mean"): rows[heads].mean(0),
            ("attention.max_heads", "captured_prediction_v1", "max"): rows[heads].max(0),
            ("attention.min_heads", "captured_prediction_v1", "min"): rows[heads].min(0),
        }
        for (method, variant, kind), weights in expected.items():
            parameters = {"layer": layer, **({"head": head} if kind == "row" else {"heads": heads})}
            status, result = run(method, variant, parameters)
            quantity = {"max": "head_maximum_envelope", "min": "head_minimum_envelope"}.get(kind, "post_softmax_attention_weight")
            if method == "attention.element_allocation" and not mapped:
                record(f"refusal:{method}", subject, 400 <= status < 500, http_status=status)
                continue
            if status != 200:
                record(f"method:{method}", subject, False, http_status=status, body=result)
                continue
            compare_row(f"method:{method}", subject, result["row"], weights, positions[:len(weights)], quantity, valid)
        raw_weights = RNG.uniform(0, 3, size=len(heads)).round(3).tolist()
        status, result = run("attention.weighted_heads", "captured_prediction_v1", {"layer": layer, "heads": heads, "weights": raw_weights})
        if status == 200:
            compare_row("method:attention.weighted_heads", subject, result["row"], np.average(rows[heads], axis=0, weights=raw_weights), positions[:len(a)],
                        valid=valid)
        else:
            record("method:attention.weighted_heads", subject, False, http_status=status, body=result)
        status, result = run("attention.entropy", "shannon_row_v1", {"layer": layer, "head": head})
        s = statistics(result) if status == 200 else {}
        h = entropy_bits(p)
        # One valid target (a causal encoder's first source) has no spread to normalize by.
        normalized = h / math.log2(len(p)) if len(p) > 1 else 0.0
        record("method:attention.entropy", subject, status == 200 and abs(s["Shannon entropy"] - h) < 1e-6
               and abs(s["Normalized entropy"] - normalized) < 1e-6 and s["Valid targets"] == len(p), expected_bits=h, returned=s)
        status, result = run("attention.concentration", "captured_prediction_v1", {"layer": layer, "head": head})
        s = statistics(result) if status == 200 else {}
        reference = {"Maximum probability": p.max(), "Gini concentration": gini_mean_difference(p),
                     "Inverse Simpson effective support": 2 ** -math.log2(float((p ** 2).sum()))}
        record("method:attention.concentration", subject, status == 200 and all(abs(s[k] - v) < 1e-6 * max(1, abs(v)) for k, v in reference.items()),
               expected=reference, returned=s)
        k = int(RNG.integers(1, min(64, len(p)) + 1))
        status, result = run("attention.topk_mass", "captured_prediction_v1", {"layer": layer, "head": head, "k": k})
        s = statistics(result) if status == 200 else {}
        reference = float(np.partition(a[valid], len(p) - k)[len(p) - k:].sum())
        record("method:attention.topk_mass", f"{subject}/k{k}", status == 200 and abs(s[f"Top-{k} attention mass"] - reference) < 1e-6,
               expected=reference, returned=s)
        roles = sorted(RNG.choice(["image", "query", "prompt", "special", "generated"], size=int(RNG.integers(1, 4)), replace=False).tolist())
        status, result = run("attention.sinks", "captured_prediction_v1", {"layer": layer, "head": head, "roles": roles})
        s = statistics(result) if status == 200 else {}
        reference = float(sum(w for w, q, v in zip(a, positions, valid) if v and q["role"] in roles))
        record("method:attention.sinks", f"{subject}/{'+'.join(roles)}", status == 200 and abs(s["Declared-group attention mass"] - reference) < 1e-6,
               expected=reference, returned=s)
        status, result = run("attention.modality_ratio", "captured_prediction_v1", {"layer": layer, "head": head})
        ok = status == 200
        if ok:
            table = dict(zip(result["table"]["labels"], result["table"]["values"]))
            for role in ("image", "query", "prompt", "special", "generated"):
                members = [i for i, q in enumerate(positions[:len(a)]) if valid[i] and q["role"] == role]
                mass, share = float(a[members].sum()), len(members) / int(valid.sum())
                enrichment = mass / a.sum() / share if members else None
                got = table[role]
                ok &= abs(got[0] - mass) < 1e-6 and got[1] == len(members) and abs(got[2] - share) < 1e-9 and (
                    got[3] is None if enrichment is None else abs(got[3] - enrichment) < 1e-6)
        record("method:attention.modality_ratio", subject, ok)
        status, result = run("attention.head_by_head", "captured_prediction_v1", {"layer": layer, "heads": heads})
        ok = status == 200 and [c["row"]["head"] for c in result["comparisons"]] == heads and all(
            close(c["row"]["values"], rows[h])[0] for c, h in zip(result["comparisons"], heads))
        record("method:attention.head_by_head", subject, ok)
        layers = sorted(RNG.choice(L, size=min(L, int(RNG.integers(2, 6))), replace=False).tolist())
        status, result = run("attention.across_layers", "captured_prediction_v1", {"layer": layer, "head": head, "layers": layers})
        ok = status == 200 and [c["row"]["layer"] for c in result["comparisons"]] == layers and all(
            close(c["row"]["values"], row_values[l, head])[0] for c, l in zip(result["comparisons"], layers))
        record("method:attention.across_layers", subject, ok)
        status, result = run("heads.distribution_similarity", "captured_prediction_v1", {"layer": layer, "heads": heads})
        if status == 200:
            distributions = rows[heads][:, valid] / rows[heads][:, valid].sum(1, keepdims=True)
            reference = np.array([[1 - jensen_shannon_bits(x, y) for y in distributions] for x in distributions])
            ok, error = close(result["table"]["values"], np.clip(reference, 0, 1), 1e-6)
            record("method:heads.distribution_similarity", subject, ok, max_error=error)
        else:
            record("method:heads.distribution_similarity", subject, False, http_status=status, body=result)


def methods(identifier, capture, rows):
    for k in sorted(rows)[:2]:
        check_methods(identifier, f"{identifier}/s{k}", capture["positions"], rows[k],
                      lambda m, v, p, k=k: analysis(identifier, m, v, {"step": k, **p}))


def question_sources(identifier, capture, folder):
    sources = capture.get("source_rows", [])
    if not sources:
        status, body = analysis(identifier, "attention.row", "captured_weights_v1", {"step": 0, "layer": 0, "head": 0},
                                source_index=next(i for i, p in enumerate(capture["positions"]) if p["role"] == "query"))
        record("question_source_unavailable_on_old_capture", identifier, status == 422, http_status=status)
        return
    for source in [sources[0], sources[-1]]:
        values = array(folder, capture, source["array"])
        check_methods(identifier, f"{identifier}/q{source['source_index']}", capture["positions"], values,
                      lambda m, v, p, i=source["source_index"]: analysis(identifier, m, v, {"step": 0, **p}, source_index=i))


def columns(identifier, capture, folder, rows):
    sources = capture.get("source_rows", [])
    image = [p["index"] for p in capture["positions"] if p["role"] == "image"]
    query = [p["index"] for p in capture["positions"] if p["role"] == "query"]
    L, H = capture["layer_count"], capture["query_head_count"]
    for target in [image[0], image[len(image) // 2], query[0], query[-1]]:
        layer, head = int(RNG.integers(L)), int(RNG.integers(H))
        status, column = call(f"/generation/captures/{identifier}/received?target_index={target}&layer={layer}&head={head}")
        expected = [(s["source_index"], float(rows[s["step"]][layer, head, target])) for s in capture["steps"] if target < s["target_count"]]
        for source in sources:
            if target < source["target_count"]:
                expected.append((source["source_index"], float(array(folder, capture, source["array"])[layer, head, target])))
        got = [(s["source_index"], s["value"]) for s in column["sources"]] if status == 200 else []
        ok = status == 200 and [i for i, _ in got] == [i for i, _ in expected] and all(abs(a - b) < EXACT for (_, a), (_, b) in zip(got, expected))
        record("api_received_column", f"{identifier}/t{target}/l{layer}/h{head}", ok, sources=len(expected))


def describe(identifier, capture, rows):
    """Descriptive patterns compared with published observations; never a pass/fail claim about the model."""
    positions = capture["positions"]
    values = rows[0].astype(np.float64)
    image = np.array([p["role"] == "image" for p in positions[:values.shape[-1]]])
    source = capture["steps"][0]["source_index"]
    argmax = values.argmax(-1)
    image_max = values[..., image].max(-1)
    observations[identifier] = {
        "image_mass_by_layer_mean_over_heads": values[..., image].sum(-1).mean(1).round(4).tolist(),
        "position0_mass_by_layer_mean_over_heads": values[..., 0].mean(1).round(4).tolist(),
        "fraction_heads_argmax_position0": float((argmax == 0).mean()),
        "fraction_heads_argmax_self_or_previous": float(((argmax == source) | (argmax == source - 1)).mean()),
        "fraction_heads_argmax_image": float(image[argmax].mean()),
        "image_contrast_gain_median": float(np.median(values.max(-1) / np.maximum(image_max, 1e-12))),
        "fraction_heads_image_mass_below_5_percent": float((values[..., image].sum(-1) < .05).mean()),
    }


# Retrieval captures -----------------------------------------------------------

def retrieval(identifier, deep):
    folder, capture = manifest("retrieval", identifier)
    q = array(folder, capture, capture["query_array"]).astype(np.float64)
    query_mask = np.array([p["eligible"] for p in capture["query_positions"]])
    norms = np.linalg.norm(q[query_mask], axis=1)
    for page in capture["pages"]:
        d = array(folder, capture, page["vector_array"]).astype(np.float64)
        mask = np.array([p["eligible"] for p in page["positions"]])
        similarity = q[query_mask] @ d[mask].T
        eligible = np.flatnonzero(mask)
        maxima = similarity.max(1)
        score = float(maxima.sum())
        winners_ok = all(similarity[i, list(eligible).index(w)] >= maxima[i] - 1e-6 for i, w in enumerate(page["winner_indices"]))
        record("maxsim_score", f"{identifier}/{page['page_id'][:20]}", abs(score - page["score"]) < 1e-4 and abs(score - page["reference_score"]) < 1e-4
               and winners_ok and close(page["query_maxima"], maxima, 1e-5)[0], score=score, saved=page["score"], upstream=page["reference_score"])
        ledger = {}
        for winner, value in zip(page["winner_indices"], page["query_maxima"]):
            for bucket, weight in page["ownership"][winner].items():
                ledger[bucket] = ledger.get(bucket, 0.0) + value * weight
        saved = {k: v for k, v in page["contributions"].items() if k != "scorer_rounding"}
        record("maxsim_contributions", f"{identifier}/{page['page_id'][:20]}", set(saved) == set(ledger)
               and all(abs(saved[k] - ledger[k]) < 1e-6 for k in ledger) and abs(sum(page["contributions"].values()) - page["score"]) < 1e-5)
        visual = [p for p in page["positions"] if p["bbox"]]
        # Grid size differs by retriever (ColPali 32 x 32; aspect-preserving pages otherwise), so it is read
        # from the distinct cell edges; tiling then requires a complete uniform row-major grid of that size.
        columns = len({round(p["bbox"][0], 6) for p in visual})
        rows = len({round(p["bbox"][1], 6) for p in visual})
        record("patch_tiling", f"{identifier}/{page['page_id'][:20]}", tiling([p["bbox"] for p in visual], page["width"], page["height"], columns, rows),
               grid=[columns, rows])
        observations.setdefault(identifier, {})["vector_norm_range"] = [float(norms.min()), float(norms.max()),
                                                                          float(np.linalg.norm(d[mask], axis=1).min()), float(np.linalg.norm(d[mask], axis=1).max())]
        for index in [int(i) for i in RNG.choice(np.flatnonzero(query_mask), size=min(3, int(query_mask.sum())), replace=False)]:
            status, row = call(f"/retrieval/captures/{identifier}/rows?page_id={page['page_id']}&query_index={index}")
            expected = q[index] @ d.T
            got = np.array([np.nan if v is None else v for v in row["similarities"]]) if status == 200 else None
            ok = status == 200 and np.array_equal(np.isnan(got), ~mask) and np.allclose(got[mask], expected[mask], atol=1e-5) \
                and abs(row["maximum"] - expected[mask].max()) < 1e-5 and expected[row["winner_index"]] >= expected[mask].max() - 1e-6
            record("api_similarity_row", f"{identifier}/q{index}", ok)
        for position in [visual[0]["index"], visual[-1]["index"], page["winner_indices"][0]]:
            status, column = call(f"/retrieval/captures/{identifier}/columns?page_id={page['page_id']}&position_index={position}")
            expected = q @ d[position]
            order = [p["index"] for p in capture["query_positions"] if p["eligible"]]
            winners = [i for i, w in zip(order, page["winner_indices"]) if w == position]
            got = np.array([np.nan if v is None else v for v in column["similarities"]]) if status == 200 else None
            ok = status == 200 and np.allclose(got[query_mask], expected[query_mask], atol=1e-5) and column["winning_query_indices"] == winners
            record("api_similarity_column", f"{identifier}/p{position}", ok)
        if deep:
            _, elements, _ = page_assets(page["page_id"])
            sample = RNG.choice(len(visual), size=30, replace=False)
            errors = [ownership_error(page["ownership"][visual[i]["index"]], sampled_ownership(visual[i]["bbox"], elements)) for i in sample]
            record("ownership_sampling", f"{identifier}/{page['page_id'][:20]}", max(errors) <= .06, max_error=max(errors), mean_error=float(np.mean(errors)))
    for encoding in capture.get("attention_encodings", []):
        encoder(identifier, capture, folder, encoding, deep)


def encoder(identifier, capture, folder, encoding, deep):
    n, mask = encoding["position_count"], np.array(encoding["attention_mask"], dtype=bool)
    # Causal encoders (for example ColQwen2) give each source only itself and earlier keys.
    causal = bool(encoding.get("causal"))
    page = next((p for p in capture["pages"] if p["page_id"] == encoding["page_id"]), None)
    # A joint page-plus-question encoding records its own positions and ownership; the others reuse the pass's positions.
    original = encoding.get("positions") or (capture["query_positions"] if encoding["pass_kind"] == "query" else page["positions"])
    ownership = encoding.get("ownership") or (page["ownership"] if page and encoding["pass_kind"] != "query" else [{"non_page": 1.0}] * n)
    record("encoding_mask_identity", f"{identifier}/{encoding['encoding_id']}", len(original) == n and all(bool(m) == p["eligible"] for m, p in zip(mask, original)))
    sources = np.flatnonzero(mask)
    chosen = [int(sources[0]), int(sources[len(sources) // 2]), int(sources[-1])][:3 if deep else 1]
    targets = [(int(sources[0]), int(RNG.integers(encoding["layer_count"])), int(RNG.integers(encoding["query_head_count"]))),
               (int(sources[-1]), int(RNG.integers(encoding["layer_count"])), int(RNG.integers(encoding["query_head_count"])))]
    # Keep only the slices the checks need; a document encoding is about 34 MB per layer.
    stacks, received, future, future_max = {s: [] for s in chosen}, {}, [], 0.0
    upper = np.triu(np.ones((n, n), dtype=bool), 1)[mask]
    # Sources that can see each sampled target: all active rows, or for a causal encoder the target and later rows.
    seeing = {target: sources[sources >= target] if causal else sources for target, _, _ in targets}
    for layer, name in enumerate(encoding["layer_arrays"]):
        values = array(folder, capture, name)
        active = values[:, mask].astype(np.float64)
        error = float(np.abs(active.sum(-1) - 1).max())
        masked_zero = not (~mask).any() or float(values[:, :, ~mask].max()) == 0
        future_mass = (active * upper[None]).sum(-1)
        future.append(float(future_mass.mean()))
        future_max = max(future_max, float(future_mass.max()))
        record("encoding_row_normalization", f"{identifier}/{encoding['encoding_id']}/l{layer}", error <= ROW_TOLERANCE and masked_zero, max_row_sum_error=error)
        for s in chosen:
            stacks[s].append(values[:, s].copy())
        for target, target_layer, head in targets:
            if target_layer == layer:
                received[target, target_layer, head] = values[head, seeing[target], target].copy()
        del values, active
    observations.setdefault(identifier, {})[f"{encoding['encoding_id']}_future_key_mass_by_layer"] = [round(v, 4) for v in future]
    if causal:
        record("encoding_causal_support", f"{identifier}/{encoding['encoding_id']}", future_max == 0, max_future_mass=future_max)
    roles = {"visual_patch": "image", "query_text": "query", "visual_prompt": "prompt", "augmentation": "special",
             "prefix_or_special": "special", "bos": "special", "newline": "special"}
    positions = [{"role": roles[p["role"]], "page_id": encoding["page_id"], "ownership": ownership[i]} for i, p in enumerate(original)]
    element = np.array([any(b.startswith("element_") and f > 0 for b, f in o.items()) for o in ownership[:n]], dtype=bool)
    for source in chosen:
        def run(method, variant, parameters, source=source):
            return call(f"/retrieval/captures/{identifier}/attention", {"encoding_id": encoding["encoding_id"], "source_index": source,
                                                                       "method_id": method, "parameters": parameters})
        valid = mask & (np.arange(n) <= source) if causal else mask
        mapped = encoding["pass_kind"] != "query" and bool((element & valid).any())
        check_methods(identifier, f"{identifier}/{encoding['encoding_id']}/src{source}", positions, np.stack(stacks[source]), run, valid, mapped)
    for (target, layer, head), expected in received.items():
        status, column = call(f"/retrieval/captures/{identifier}/attention/received",
                              {"encoding_id": encoding["encoding_id"], "target_index": target, "layer": layer, "head": head})
        ok = status == 200 and [s["source_index"] for s in column["sources"]] == seeing[target].tolist() and np.allclose([s["value"] for s in column["sources"]], expected, atol=EXACT)
        record("api_encoding_received_column", f"{identifier}/{encoding['encoding_id']}/t{target}", ok)


# Error handling ---------------------------------------------------------------

def probes(generation_id, steps, retrieval_id, layers, heads):
    """Invalid requests must be refused with 4xx and a message, never a 500 or a silent result."""
    cases = {
        "uncaptured_step": analysis(generation_id, "attention.row", "captured_weights_v1", {"step": steps}),
        "layer_out_of_range": analysis(generation_id, "attention.row", "captured_weights_v1", {"layer": layers}),
        "head_out_of_range": analysis(generation_id, "attention.row", "captured_weights_v1", {"head": heads}),
        "duplicate_heads": analysis(generation_id, "attention.mean_heads", "captured_prediction_v1", {"heads": [1, 1]}),
        "negative_head_weight": analysis(generation_id, "attention.weighted_heads", "captured_prediction_v1", {"heads": [0, 1], "weights": [-1, 2]}),
        "zero_head_weights": analysis(generation_id, "attention.weighted_heads", "captured_prediction_v1", {"heads": [0, 1], "weights": [0, 0]}),
        "topk_zero": analysis(generation_id, "attention.topk_mass", "captured_prediction_v1", {"k": 0}),
        "topk_beyond_support": analysis(generation_id, "attention.topk_mass", "captured_prediction_v1", {"k": 1600}),
        "unknown_parameter": analysis(generation_id, "attention.row", "captured_weights_v1", {"bandwidth": 2}),
        "string_layer": analysis(generation_id, "attention.row", "captured_weights_v1", {"layer": "3"}),
        "unknown_method": analysis(generation_id, "attention.unknown", "v1", {}),
        "variant_mismatch": analysis(generation_id, "attention.row", "shannon_row_v1", {}),
        "planned_method": analysis(generation_id, "attention.rollout", "residual_mean_square_v1", {}),
        "retrieval_method_on_generator": analysis(generation_id, "retrieval.maxsim", "actual_scorer_v1", {}),
        "non_query_source_index": analysis(generation_id, "attention.row", "captured_weights_v1", {}, source_index=0),
        "duplicate_sink_roles": analysis(generation_id, "attention.sinks", "captured_prediction_v1", {"roles": ["image", "image"]}),
        "raw_row_uncaptured_step": call(f"/generation/captures/{generation_id}/rows?step={steps}"),
        "received_target_beyond_domain": call(f"/generation/captures/{generation_id}/received?target_index=1599"),
        "retrieval_step_parameter": call(f"/retrieval/captures/{retrieval_id}/attention", {"encoding_id": "document", "method_id": "attention.row", "parameters": {"step": 0}}),
        "retrieval_unsupported_method": call(f"/retrieval/captures/{retrieval_id}/attention", {"encoding_id": "document", "method_id": "attention.rollout"}),
        "retrieval_masked_or_absent_source": call(f"/retrieval/captures/{retrieval_id}/attention", {"encoding_id": "document", "source_index": 5000}),
        "retrieval_unknown_encoding": call(f"/retrieval/captures/{retrieval_id}/attention", {"encoding_id": "vision"}),
    }
    for name, (status, body) in cases.items():
        message = body.get("error", {}).get("message") or body.get("detail") if isinstance(body, dict) else body
        record("refusal", name, 400 <= status < 500, http_status=status, message=str(message)[:200])


def main():
    only = [c for c in os.environ.get("DOCATTENTIONLENS_CONFORMANCE_CAPTURES", "").split(",") if c]
    if only:
        generation, retrieval_items = [], []
        for identifier in only:
            kind = "generation" if identifier.startswith("generation_") else "retrieval"
            status, body = call(f"/{kind}/captures/{identifier}")
            record("capture_lookup", identifier, status == 200, http_status=status)
            if status == 200:
                (generation if kind == "generation" else retrieval_items).append(body)
        deep_generation = {c["capture_id"] for c in generation}
        deep_retrieval = {c["capture_id"] for c in retrieval_items}
    else:
        generation = call("/generation/captures?limit=25")[1]["items"]
        retrieval_items, cursor = [], None
        while True:
            page = call("/retrieval/captures?limit=25" + (f"&before={cursor}" if cursor else ""))[1]
            retrieval_items += page["items"]
            cursor = page["next_cursor"]
            if not cursor:
                break
        # Deep checks (methods, pixels, ownership) on question-row, two-page and longest-answer captures; light checks on all.
        deep_generation = {c["capture_id"] for c in generation if c.get("source_rows")}
        deep_generation |= {c["capture_id"] for c in generation if len(c["context"]["pages"]) == 2}
        deep_generation |= {max(generation, key=lambda c: len(c["steps"]))["capture_id"]}
        with_attention = [c for c in retrieval_items if c.get("attention_encodings")]
        deep_retrieval = {with_attention[0]["capture_id"], max(retrieval_items, key=lambda c: len(c["pages"]))["capture_id"]}
    for capture in generation:
        generator(capture["capture_id"], capture["capture_id"] in deep_generation)
    for capture in retrieval_items:
        retrieval(capture["capture_id"], capture["capture_id"] in deep_retrieval)
    # Refusal probes test the API rather than a capture, so a filtered run leaves them out.
    if not only:
        short = min(generation, key=lambda c: len(c["steps"]))
        probes(short["capture_id"], len(short["steps"]), with_attention[0]["capture_id"], short["layer_count"], short["query_head_count"])
    failures = [r for r in results if r["status"] == "fail"]
    summary = {}
    for r in results:
        summary.setdefault(r["check"], {"pass": 0, "fail": 0})[r["status"]] += 1
    print(json.dumps({"api": API, "generation_captures": len(generation), "retrieval_captures": len(retrieval_items),
                      "deep_generation": sorted(deep_generation), "deep_retrieval": sorted(deep_retrieval),
                      "checks": len(results), "failures": len(failures), "summary": summary,
                      "failed": failures, "observations": observations, "results": results}, indent=1, default=str))
    print(f"{len(results)} checks, {len(failures)} failures", file=sys.stderr)


if __name__ == "__main__":
    main()
