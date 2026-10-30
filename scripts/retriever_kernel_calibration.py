"""Evidence for the retrievers' eager-extraction versus SDPA-scoring tolerances.

The attention capture extracts weights from an eager forward pass and scores with SDPA, and
checks that both passes compute the same encoder. This script measures, on real questions and
pages of the dataset:

  noise       eager extraction against SDPA scoring: per-encoding mean per-position L2
              distance and maximum coordinate difference, and MaxSim score differences
              per eligible query position (ColQwen2 and ColQwen2.5 with their FP32-arithmetic
              eager kernel, and with stock BF16 eager for comparison; ColPali with stock eager)
  mechanism   the largest attention logit in each ColQwen layer, and which BF16 rounding
              step of stock eager produces the drift
  reference   the same weights in FP32: eager against SDPA, which isolates masks and positions
              from kernel arithmetic
  positions   runtime rotary position ids against colqwen.rope_positions
  errors      deliberate masking and position errors in the eager pass, which the
              tolerance must reject

Runs in the GPU worker image against the metadata database and the model cache:

    docker exec -i docattentionlens-worker-gpu-1 python - < scripts/retriever_kernel_calibration.py > var/verification/colqwen-kernel-calibration.json

DOCATTENTIONLENS_CALIBRATION_CASE, DOCATTENTIONLENS_CALIBRATION_QUERIES and DOCATTENTIONLENS_CALIBRATION_PAGES change the case and sample sizes;
DOCATTENTIONLENS_CALIBRATION_RETRIEVERS (comma-separated colqwen2, colqwen2.5, colpali) selects the retrievers. ColQwen2.5
is recorded in var/verification/colqwen25-kernel-calibration.json:

    docker exec -i -e DOCATTENTIONLENS_CALIBRATION_RETRIEVERS=colqwen2.5 -e DOCATTENTIONLENS_CALIBRATION_CASE=case_95ee01dfbee043e485c9ac1f38f456e0 \\
        docattentionlens-worker-gpu-1 python - < scripts/retriever_kernel_calibration.py > var/verification/colqwen25-kernel-calibration.json

The FP32 reference holds a second copy of the model in FP32 (about 15 GiB for ColQwen2.5), so no other
process should keep a model resident on the GPU.
"""
import json
import os
import sys
from contextlib import contextmanager
from io import BytesIO
import numpy as np
from PIL import Image
from docattentionlens_api.config import Settings
from docattentionlens_api.repositories.documents import DocumentRepository
from docattentionlens.retrieval import colpali, colqwen
from docattentionlens.retrieval.encoding_capture import FLOAT32_EAGER, register_float32_eager
from docattentionlens.retrieval.scoring import maxsim

CASE = os.environ.get("DOCATTENTIONLENS_CALIBRATION_CASE", "case_3317451817bf4dbe8746c0cd2b91bd0a")
QUERIES = int(os.environ.get("DOCATTENTIONLENS_CALIBRATION_QUERIES", 200))
PAGES = int(os.environ.get("DOCATTENTIONLENS_CALIBRATION_PAGES", 30))
RETRIEVERS = os.environ.get("DOCATTENTIONLENS_CALIBRATION_RETRIEVERS", "colqwen2,colqwen2.5,colpali").split(",")
COLQWEN = {"colqwen2": colqwen.REGISTRATION_ID, "colqwen2.5": colqwen.COLQWEN25_REGISTRATION_ID}
if set(RETRIEVERS) - {*COLQWEN, "colpali"}:
    sys.exit(f"Unknown retrievers: {sorted(set(RETRIEVERS) - {*COLQWEN, 'colpali'})}")


def log(message):
    print(message, file=sys.stderr, flush=True)


def distribution(values):
    v = np.asarray(values, dtype=float)
    return {"count": len(v), "min": v.min(), "median": float(np.median(v)), "p99": float(np.quantile(v, .99)), "max": v.max()}


def difference(a, b):
    l2 = np.linalg.norm(a - b, axis=-1)
    return {"mean_l2": float(l2.mean()), "max_abs": float(np.abs(a - b).max()), "max_l2": float(l2.max())}


def score(q, d):
    return float(maxsim(q, d, [True] * len(q), [True] * len(d))[0])


settings = Settings.from_env()
documents = DocumentRepository(settings)
case = documents.case(CASE)
snapshot = documents.snapshot(case.dataset_id)
rng = np.random.default_rng(2026)
questions = sorted({q.question for q in snapshot.questions} - {case.query})
queries = [case.query] + [questions[i] for i in rng.choice(len(questions), min(QUERIES, len(questions)), replace=False)]
case_pages = [next(p for p in snapshot.pages if p.page_id == pid) for pid in case.page_ids]
others = [p for p in snapshot.pages if p.page_id not in case.page_ids]
pages = case_pages + [others[i] for i in rng.choice(len(others), min(PAGES, len(others)), replace=False)]


def images(processor):
    for page in pages:
        with Image.open(BytesIO(documents.image_bytes(snapshot, page))) as opened:
            yield page, processor(images=[opened.convert("RGB")], return_tensors="pt").to("cuda")


def measure(torch, processor, model, kernels, record_positions=None):
    """Embeddings of every query and page under SDPA and each named eager kernel."""
    language = model.vlm.model.language_model

    @contextmanager
    def kernel(name):
        original = language.config._attn_implementation
        language.set_attn_implementation(name)
        try:
            yield
        finally:
            language.set_attn_implementation(original)

    def embed(batch, name):
        with kernel(name), torch.inference_mode():
            return model(**batch, use_cache=False).embeddings[0].float().cpu().numpy()

    out = {"queries": [], "pages": []}
    batches = [("queries", q, processor(text=[q], return_tensors="pt", padding="longest").to("cuda")) for q in queries]
    for kind, label, batch in batches + [("pages", page.page_id, batch) for page, batch in images(processor)]:
        if record_positions:
            record_positions(batch)
        out[kind].append({"id": label, "positions": int(batch["input_ids"].shape[1]),
                          **{name: embed(batch, name) for name in ("sdpa", *kernels)}})
    return out


def noise(measured, kernel):
    """Kernel noise of one eager kernel against SDPA: encodings and all query/page scores."""
    result = {}
    for kind in ("queries", "pages"):
        rows = [difference(r[kernel], r["sdpa"]) for r in measured[kind]]
        result[kind] = {m: distribution([r[m] for r in rows]) for m in ("mean_l2", "max_abs", "max_l2")}
    per_position = [abs(score(q[kernel], p[kernel]) - score(q["sdpa"], p["sdpa"])) / q["positions"]
                    for q in measured["queries"] for p in measured["pages"]]
    absolute = [x * q["positions"] for x, q in zip(per_position, [q for q in measured["queries"] for _ in measured["pages"]])]
    result["scores"] = {"per_query_position": distribution(per_position), "absolute": distribution(absolute)}
    return result


report = {"case_id": CASE, "query": case.query, "queries": len(queries), "pages": len(pages)}

# ColQwen2 and ColQwen2.5 ---------------------------------------------------------------
# Helpers read the resident model through the globals the loop below rebinds per checkpoint.
seen, mismatched = {}, []


def check_positions(batch):
    with torch.inference_mode():
        model(**batch, use_cache=False)
    tokens = batch["input_ids"][0].tolist()
    grid = batch["image_grid_thw"][0].tolist() if "image_grid_thw" in batch else None
    if not np.array_equal(seen["positions"], colqwen.rope_positions(tokens, image_token_id, grid)):
        mismatched.append(len(tokens))


# Mechanism: largest logit per layer, and the stock-eager rounding step responsible.
logits, rounding = [], {"logits": False, "probabilities": False}


def probe(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    k = key.repeat_interleave(module.num_key_value_groups, dim=1).float()
    raw = query.float() @ k.transpose(2, 3) * scaling
    logits.append(float(raw.abs().max()))
    if rounding["logits"]:
        raw = raw.bfloat16().float()
    if attention_mask is not None:
        raw = raw + attention_mask[:, :, :, :k.shape[-2]].float()
    weights = raw.softmax(-1)
    if rounding["probabilities"]:
        weights = weights.bfloat16().float()
    v = value.repeat_interleave(module.num_key_value_groups, dim=1).float()
    return (weights @ v).to(query.dtype).transpose(1, 2).contiguous(), weights


from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, eager_mask
AttentionInterface.register("calibration_probe", probe)
AttentionMaskInterface.register("calibration_probe", eager_mask)


def run(batch, name, **kwargs):
    original = language.config._attn_implementation
    language.set_attn_implementation(name)
    try:
        with torch.inference_mode():
            return model(**batch, use_cache=False, **kwargs).embeddings[0].float().cpu().numpy()
    finally:
        language.set_attn_implementation(original)


# Deliberate masking and position errors in the extraction pass, against SDPA scoring.
@contextmanager
def tamper(allowed=None, positions=None):
    def hook(module, args, kwargs):
        if allowed is not None:
            n = kwargs["inputs_embeds"].shape[1]
            kwargs["attention_mask"] = torch.where(allowed(n), 0.0, torch.finfo(torch.bfloat16).min).to(torch.bfloat16)[None, None]
        if positions is not None:
            kwargs["position_ids"] = positions(kwargs["position_ids"].clone())
        return args, kwargs
    handle = language.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def causal(n, extra=0):
    i = torch.arange(n, device="cuda")
    return i[None, :] <= i[:, None] + extra


def hidden(*keys):
    def allowed(n):
        mask = causal(n)
        for k in keys:
            mask[:, k] = False
            mask[k, k] = True
        return mask
    return allowed


def shifted(start):
    def positions(p):
        p[..., start:] += 1
        return p
    return positions


def text_positions(p):
    return torch.arange(p.shape[-1], device=p.device).view(1, 1, -1).expand_as(p).clone()


def swapped(p):
    p[[1, 2]] = p[[2, 1]]
    return p


everything = lambda n: torch.ones(n, n, dtype=torch.bool, device="cuda")

for name in [r for r in RETRIEVERS if r in COLQWEN]:
    registration = COLQWEN[name]
    label = colqwen.BACKBONES[registration]["label"]
    embedding_tolerance = colqwen.EMBEDDING_KERNEL_TOLERANCES[registration]
    torch, processor, model = colqwen.runtime(settings.model_cache, registration)
    register_float32_eager()
    language = model.vlm.model.language_model
    image_token_id = model.config.vlm_config.image_token_id
    report["versions"] = {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "gpu": torch.cuda.get_device_name()}
    mismatched.clear()
    hook = language.register_forward_pre_hook(lambda m, a, k: seen.__setitem__("positions", k["position_ids"][:, 0].cpu().numpy()), with_kwargs=True)
    log(f"{label}: {len(queries)} queries, {len(pages)} pages")
    measured = measure(torch, processor, model, (FLOAT32_EAGER, "eager"), check_positions)
    hook.remove()
    result = report[name] = {"registration": registration, "float32_eager": noise(measured, FLOAT32_EAGER), "stock_bf16_eager": noise(measured, "eager"),
                             "rope_positions": {"checked": len(queries) + len(pages), "mismatched": len(mismatched)},
                             "tolerances": {"embedding_mean_l2": embedding_tolerance, "score_per_query_position": colqwen.SCORE_KERNEL_TOLERANCES[registration]}}

    case_query = measured["queries"][0]
    query_batch = processor(text=[case.query], return_tensors="pt", padding="longest").to("cuda")
    logits.clear()
    run(query_batch, "calibration_probe")
    mechanism = {"case_query_max_abs_logit_by_layer": [round(x, 1) for x in logits]}
    for setting_label, setting in (("only_logits_rounded_to_bf16", {"logits": True, "probabilities": False}),
                                   ("only_probabilities_rounded_to_bf16", {"logits": False, "probabilities": True})):
        rounding.update(setting)
        rows = [difference(run(processor(text=[q["id"]], return_tensors="pt", padding="longest").to("cuda"), "calibration_probe"), q["sdpa"])
                for q in measured["queries"][:20]]
        mechanism[setting_label] = {m: distribution([r[m] for r in rows]) for m in ("mean_l2", "max_abs")}
    rounding.update(logits=False, probabilities=False)
    result["mechanism"] = mechanism

    page_batch = next(batch for page, batch in images(processor) if page.page_id == case_pages[0].page_id)
    query_tokens, page_tokens = query_batch["input_ids"][0].tolist(), page_batch["input_ids"][0].tolist()
    augmentation = query_tokens.index(processor.tokenizer.convert_tokens_to_ids(processor.query_augmentation_token))
    first_image = page_tokens.index(image_token_id)
    errors = [("query", "bidirectional (no causal mask)", dict(allowed=everything)),
              ("query", "one future key visible", dict(allowed=lambda n: causal(n, 1))),
              ("query", "first key hidden from later rows", dict(allowed=hidden(0))),
              ("query", "one question key hidden from later rows", dict(allowed=hidden(augmentation // 2))),
              ("query", "augmentation keys treated as padding", dict(allowed=lambda n: hidden(*range(augmentation, n))(n))),
              ("query", "first augmentation key hidden", dict(allowed=hidden(augmentation))),
              ("query", "positions shifted by one from the augmentation", dict(positions=shifted(augmentation))),
              ("query", "position of the last token shifted by one", dict(positions=shifted(len(query_tokens) - 1))),
              ("document", "bidirectional (no causal mask)", dict(allowed=everything)),
              ("document", "one future key visible", dict(allowed=lambda n: causal(n, 1))),
              ("document", "first key hidden from later rows", dict(allowed=hidden(0))),
              ("document", "one image key hidden from later rows", dict(allowed=hidden(first_image + 300))),
              ("document", "image positions as 1D text positions", dict(positions=text_positions)),
              ("document", "image height and width positions swapped", dict(positions=swapped)),
              ("document", "positions shifted by one after the first image row", dict(positions=shifted(first_image + 1))),
              ("document", "position of the last token shifted by one", dict(positions=shifted(len(page_tokens) - 1)))]
    case_page = measured["pages"][0]
    result["deliberate_errors"] = []
    for kind, error_label, options in errors:
        batch, reference = (query_batch, case_query) if kind == "query" else (page_batch, case_page)
        with tamper(**options):
            wrong = run(batch, FLOAT32_EAGER)
        found = difference(wrong, reference["sdpa"])
        result["deliberate_errors"].append({"encoding": kind, "error": error_label, **found,
            "rejected_by_embedding_tolerance": found["mean_l2"] > embedding_tolerance,
            "rejected_by_position_check": "positions" in options})
        log(f"{kind:8} {error_label:52} mean L2 {found['mean_l2']:.4f} max abs {found['max_abs']:.4f}")

    # FP32 reference: masks and positions with kernel arithmetic out of the way.
    model.float()
    reference = {"queries": [], "pages": []}
    for q in measured["queries"][:40]:
        batch = processor(text=[q["id"]], return_tensors="pt", padding="longest").to("cuda")
        reference["queries"].append(difference(run(batch, "eager"), run(batch, "sdpa")))
    for page, batch in images(processor):
        if page.page_id in case.page_ids:
            reference["pages"].append(difference(run(batch, "eager"), run(batch, "sdpa")))
    result["fp32_eager_vs_fp32_sdpa"] = {kind: {m: distribution([r[m] for r in rows]) for m in ("mean_l2", "max_abs")}
                                         for kind, rows in reference.items()}
    del measured, model, language, query_batch, page_batch, batch
    colqwen.release_runtime()


# ColPali, stock BF16 eager --------------------------------------------------------------
if "colpali" in RETRIEVERS:
    torch, processor, model = colpali.runtime(settings.model_cache)
    language = model.vlm.model.language_model
    log(f"ColPali: {len(queries)} queries, {len(pages)} pages")

    def colpali_measure():
        out = {"queries": [], "pages": []}
        for q in queries:
            batch = processor(text=[q], return_tensors="pt", padding="longest").to("cuda")
            out["queries"].append({"id": q, "positions": int(batch["input_ids"].shape[1]), "sdpa": run(batch, "sdpa"), "eager": run(batch, "eager")})
        for page, _ in zip(pages, range(len(pages))):
            with Image.open(BytesIO(documents.image_bytes(snapshot, page))) as opened:
                batch = processor(images=[opened.convert("RGB")], return_tensors="pt").to("cuda")
            out["pages"].append({"id": page.page_id, "positions": int(batch["input_ids"].shape[1]), "sdpa": run(batch, "sdpa"), "eager": run(batch, "eager")})
        return out

    report["colpali"] = {"stock_bf16_eager": noise(colpali_measure(), "eager")}
    logits.clear()
    run(processor(text=[case.query], return_tensors="pt", padding="longest").to("cuda"), "calibration_probe")
    report["colpali"]["case_query_max_abs_logit_by_layer"] = [round(x, 1) for x in logits]
    colpali.release_runtime()
print(json.dumps(report, indent=1, default=float))
