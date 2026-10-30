"""Pinned Transformers ColQwen adapter (ColQwen2 on Qwen2-VL-2B, ColQwen2.5 on Qwen2.5-VL-3B; both causal).
Torch is an optional GPU dependency.

Pages keep their aspect ratio: the Qwen2-VL processor resizes each page to multiples
of 28 px with at most 768 merged image positions, then groups 14 px patches in 2x2
blocks. Merged positions follow row-major order over the merged grid. Qwen2-VL has
no vision window permutation; Qwen2.5-VL attends in windows over a permuted patch
order and inverts it before the language model, which every page forward verifies.
The language model is causal, so page patches precede the prompt and never attend to it.
"""
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from uuid import uuid4
import json
import platform
import numpy as np
from PIL import Image

from .contracts import (COLQWEN_MODEL_ID as MODEL_ID, COLQWEN_MODEL_REVISION as MODEL_REVISION, RETRIEVER_PROFILES,
    Position, PageEvidence, RetrievalCapture)
from .scoring import maxsim, ownership, contributions
from docattentionlens import __version__

REGISTRATION_ID = "builtin-colqwen2"
COLQWEN25_REGISTRATION_ID = "builtin-colqwen2.5"
# What differs between the checkpoints this adapter loads; names, pins and budgets are in RETRIEVER_PROFILES.
BACKBONES = {
    REGISTRATION_ID: {"label": "ColQwen2", "vision": "Qwen2-VL", "window_permutation": False},
    # colpali-engine's ColQwen2.5 processor and the checkpoint's own prompts declare no query prefix
    # (ColQwen2 keeps Transformers' "Query: ").
    COLQWEN25_REGISTRATION_ID: {"label": "ColQwen2.5", "vision": "Qwen2.5-VL", "window_permutation": True, "query_prefix": ""},
}
MAX_VISUAL_TOKENS = 768  # checkpoint max_pixels 602112 = 768 merged 28 px cells
QUERY_AUGMENTATION_TOKENS = 10
# FP32 eager extraction against BF16 SDPA scoring, calibrated per checkpoint by
# scripts/retriever_kernel_calibration.py (results in docs/python-notes.md). Embeddings compare by
# mean per-position L2 distance, scores per query position; the position check catches rotary errors.
EMBEDDING_KERNEL_TOLERANCES = {REGISTRATION_ID: .04, COLQWEN25_REGISTRATION_ID: .09}
SCORE_KERNEL_TOLERANCES = {REGISTRATION_ID: .01, COLQWEN25_REGISTRATION_ID: .03}

_runtime = None
_loaded = None  # registration of the resident model; one ColQwen checkpoint at a time
_checkpoint_files = {}
_weight_loading = {}


def release_runtime():
    global _runtime, _loaded
    _runtime = _loaded = None
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def merged_boxes(grid, width, height):
    """Source-page rectangles of merged image positions, in the processor's row-major order."""
    t, h, w = (int(x) for x in grid)
    if t != 1 or h % 2 or w % 2:
        raise ValueError("Only still images with 2x2 spatial merging are supported")
    rows, columns = h // 2, w // 2
    return [[c * width / columns, r * height / rows, (c + 1) * width / columns, (r + 1) * height / rows]
            for r in range(rows) for c in range(columns)]


def rope_positions(tokens, image_token_id, grid=None):
    """Qwen2-VL rotary indices (temporal, height, width) for one unpadded sequence, rebuilt independently.

    Text advances all three axes together. A still image's merged cells share one temporal
    index and offset height and width by their row and column; text after the image resumes
    one past the largest image index.
    """
    n = len(tokens)
    visual = [i for i, token in enumerate(tokens) if token == image_token_id]
    if not visual:
        return np.tile(np.arange(n), (3, 1))
    t, h, w = (int(x) for x in grid)
    rows, columns = h // 2, w // 2
    start = visual[0]
    if t != 1 or visual != list(range(start, start + rows * columns)):
        raise ValueError("Rotary reconstruction expects one still image of contiguous merged cells")
    row, column = np.divmod(np.arange(rows * columns), columns)
    image = start + np.stack([np.zeros_like(row), row, column])
    after = start + max(rows, columns) + np.arange(n - start - rows * columns)
    return np.concatenate([np.tile(np.arange(start), (3, 1)), image, np.tile(after, (3, 1))], axis=1)


def kernel_difference(eager, baseline, eligible):
    """Eager-extraction against SDPA-scoring embeddings on eligible positions: (mean L2 distance, max absolute coordinate)."""
    difference = (eager - baseline)[np.asarray(eligible, dtype=bool)]
    return float(np.linalg.norm(difference, axis=-1).mean()), float(np.abs(difference).max())


def token_role(token_id, piece, image_token_id, special_ids):
    if token_id == image_token_id:
        return "visual_patch"
    if token_id in special_ids:
        return "prefix_or_special"
    return "newline" if piece in ("Ċ", "\n") else "visual_prompt"


def special_token_ids(tokenizer):
    """Special and chat-control tokens such as <|im_start|> and <|vision_end|>."""
    ids = set(tokenizer.all_special_ids)
    ids.update(i for token, i in tokenizer.get_added_vocab().items() if token.startswith("<|") and token.endswith("|>"))
    return ids


def file_sha256(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_colqwen2(torch, cache):
    """ColQwen2 v1.0: a complete Transformers checkpoint."""
    from huggingface_hub import snapshot_download
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor, Qwen2TokenizerFast, Qwen2VLImageProcessor
    options = dict(revision=MODEL_REVISION, cache_dir=str(cache))
    # Slow image processing reproduces the PIL resize; the fast tokenizer gives exact offsets.
    processor = ColQwen2Processor(image_processor=Qwen2VLImageProcessor.from_pretrained(MODEL_ID, **options),
                                  tokenizer=Qwen2TokenizerFast.from_pretrained(MODEL_ID, **options))
    model, loading = ColQwen2ForRetrieval.from_pretrained(MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                          output_loading_info=True, **options)
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("Checkpoint weight loading is incomplete or incompatible")
    folder = Path(snapshot_download(MODEL_ID, local_files_only=True, **options))
    files = {path.name: file_sha256(path) for path in sorted(folder.iterdir()) if path.is_file() and path.suffix in (".json", ".safetensors")}
    return processor, model, files, {}


# Checkpoint pieces ColQwen2.5 needs: the adapter repository's LoRA, processor and the projection its
# sentence-transformers export stores merged; the base repository's configuration and BF16 weights.
COLQWEN25_ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors", "1_Dense/model.safetensors", "1_Dense/config.json",
    "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json"]
COLQWEN25_BASE_FILES = ["config.json", "model.safetensors.index.json", "model-*.safetensors"]
# Base checkpoint module names (colpali-engine's ColQwen2_5) to Transformers' ColQwen2ForRetrieval.
COLQWEN25_RENAMES = (("model.layers.", "vlm.model.language_model.layers."), ("model.embed_tokens.", "vlm.model.language_model.embed_tokens."),
    ("model.norm.", "vlm.model.language_model.norm."), ("visual.", "vlm.model.visual."), ("custom_text_proj.", "embedding_proj_layer."))
# Qwen2.5-VL-3B's vision settings, several of them unstated in the base configuration and filled by defaults.
COLQWEN25_VISION = {"depth": 32, "hidden_size": 1280, "intermediate_size": 3420, "num_heads": 16, "out_hidden_size": 2048,
    "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2, "window_size": 112, "fullatt_block_indexes": [7, 15, 23, 31]}
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def colqwen25_name(key):
    for old, new in COLQWEN25_RENAMES:
        if key.startswith(old):
            return new + key[len(old):]
    raise RuntimeError("Checkpoint weight loading is incomplete or incompatible")


def merge_lora(base, adapter, scale, layers):
    """Fold every LoRA pair into its base matrix in place: W + scale * B @ A in FP32, cast back to W's dtype.

    The adapter must cover exactly the language layers' attention and MLP projections and the embedding
    projection. Returns the number of merged matrices and the FP32 merged projection, before the cast.
    """
    prefix = "base_model.model."
    expected = {f"model.layers.{i}.{part}.{name}" for i in range(layers)
                for part, names in (("self_attn", LORA_TARGETS[:4]), ("mlp", LORA_TARGETS[4:])) for name in names} | {"custom_text_proj"}
    modules = {key[len(prefix):-len(".lora_A.weight")] for key in adapter if key.startswith(prefix) and key.endswith(".lora_A.weight")}
    if modules != expected or len(adapter) != 2*len(modules):
        raise RuntimeError("Checkpoint weight loading is incomplete or incompatible")
    projection = None
    for module in sorted(modules):
        a, b = adapter[f"{prefix}{module}.lora_A.weight"].float(), adapter[f"{prefix}{module}.lora_B.weight"].float()
        weight = base[module+".weight"]
        if b.shape[1] != a.shape[0] or (b.shape[0], a.shape[1]) != tuple(weight.shape):
            raise RuntimeError("Checkpoint weight loading is incomplete or incompatible")
        merged = weight.float() + scale*(b @ a)
        if module == "custom_text_proj":
            projection = merged
        base[module+".weight"] = merged.to(weight.dtype)
    return len(modules), projection


def load_colqwen25(torch, cache):
    """ColQwen2.5 v0.2: a rank-32 LoRA over the language layers and projection of vidore/colqwen2.5-base.

    Transformers 4.57 cannot load this adapter itself (its key mapping needs Transformers 5.15, and older
    versions silently randomise the adapter), so the pinned files are read directly. Base weights are renamed
    to ColQwen2ForRetrieval's modules and each adapted matrix is merged in FP32, then cast to BF16. The
    checkpoint's own FP32 merged projection (its sentence-transformers export) must match the merge.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import ColQwen2Config, ColQwen2ForRetrieval, ColQwen2Processor, Qwen2TokenizerFast, Qwen2VLImageProcessor
    profile = RETRIEVER_PROFILES[COLQWEN25_REGISTRATION_ID]
    adapter = Path(snapshot_download(profile["model_id"], revision=profile["revision"], cache_dir=str(cache), allow_patterns=COLQWEN25_ADAPTER_FILES))
    base = Path(snapshot_download(profile["base_model_id"], revision=profile["base_revision"], cache_dir=str(cache), allow_patterns=COLQWEN25_BASE_FILES))
    settings = json.loads((adapter/"adapter_config.json").read_text())
    if ((settings["peft_type"], settings["r"], settings["lora_alpha"], settings["bias"], settings["use_dora"], settings["use_rslora"],
            settings["base_model_name_or_path"], settings["modules_to_save"]) != ("LORA", 32, 32, "none", False, False, profile["base_model_id"], None)):
        raise RuntimeError("Checkpoint weight loading is incomplete or incompatible")
    config = ColQwen2Config(vlm_config=json.loads((base/"config.json").read_text()), embedding_dim=128)
    vision, text = config.vlm_config.vision_config, config.vlm_config.text_config
    if (type(config.vlm_config).__name__ != "Qwen2_5_VLConfig" or {k: getattr(vision, k) for k in COLQWEN25_VISION} != COLQWEN25_VISION
            or (text.num_hidden_layers, text.num_attention_heads, text.num_key_value_heads, text.hidden_size) != (profile["layers"], profile["query_heads"], 2, 2048)):
        raise RuntimeError("Processor or vision ordering differs from the validated Qwen2.5-VL mapping")
    shards = sorted(set(json.loads((base/"model.safetensors.index.json").read_text())["weight_map"].values()))
    weights = {key: value for shard in shards for key, value in load_file(base/shard).items()}
    count, projection = merge_lora(weights, load_file(adapter/"adapter_model.safetensors"), settings["lora_alpha"]/settings["r"], text.num_hidden_layers)
    dense = load_file(adapter/"1_Dense"/"model.safetensors")
    projection_error = max(float((projection-dense["linear.weight"]).abs().max()),
                           float((weights["custom_text_proj.bias"].float()-dense["linear.bias"]).abs().max()))
    if projection_error > 1e-5:
        raise RuntimeError("Checkpoint weight loading is incomplete or incompatible")
    model, loading = ColQwen2ForRetrieval.from_pretrained(None, config=config, state_dict={colqwen25_name(k): v for k, v in weights.items()},
                                                          dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("Checkpoint weight loading is incomplete or incompatible")
    # Transformers' ColQwen2 forward asks its vision tower for a dtype; Qwen2.5-VL's tower only exposes `dtype`.
    visual = model.vlm.model.visual
    visual.get_dtype = lambda: visual.dtype
    processor = ColQwen2Processor(image_processor=Qwen2VLImageProcessor.from_pretrained(adapter), tokenizer=Qwen2TokenizerFast.from_pretrained(adapter),
                                  query_prefix=BACKBONES[COLQWEN25_REGISTRATION_ID]["query_prefix"])
    files = {f"{repo}/{path.relative_to(folder).as_posix()}": file_sha256(path)
             for repo, folder in ((profile["model_id"], adapter), (profile["base_model_id"], base))
             for path in sorted(folder.rglob("*")) if path.is_file() and path.suffix in (".json", ".safetensors")}
    loading = {"base_model_id": profile["base_model_id"], "base_model_revision": profile["base_revision"],
        "lora": f"rank {settings['r']}, alpha {settings['lora_alpha']}: {count} matrices merged as W + (alpha/r)·B·A in FP32, then cast to BF16",
        "lora_targets": "language self-attention and MLP projections of every layer, and the embedding projection; vision tower unchanged",
        "merged_projection_max_abs_error_vs_checkpoint_dense": projection_error}
    return processor, model, files, loading


def runtime(cache, registration=REGISTRATION_ID):
    global _runtime, _loaded, _checkpoint_files, _weight_loading
    if _runtime is not None and _loaded != registration:
        release_runtime()
    if _runtime is None:
        from docattentionlens.generation.qwen import release_runtime as release_generator, pack_pixels
        from docattentionlens.retrieval.colpali import release_runtime as release_colpali
        release_generator()
        release_colpali()
        import torch
        backbone, profile = BACKBONES[registration], RETRIEVER_PROFILES[registration]
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError(f"{backbone['label']} requires a BF16-capable CUDA device")
        free, _ = torch.cuda.mem_get_info()
        if free < profile["required_free_mib"]*1024**2:
            raise RuntimeError(f"At least {profile['required_free_mib']//1024} GiB free GPU memory is required for this bounded plan")
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        processor, model, files, loading = (load_colqwen25 if registration == COLQWEN25_REGISTRATION_ID else load_colqwen2)(torch, cache)
        model = model.to("cuda").eval()
        ip = processor.image_processor
        text = model.config.vlm_config.text_config
        if (type(ip).__name__ != "Qwen2VLImageProcessor" or (ip.patch_size, ip.merge_size, ip.temporal_patch_size) != (14, 2, 2)
                or ip.max_pixels != MAX_VISUAL_TOKENS*28*28 or not ip.do_resize or not ip.do_rescale or not ip.do_normalize
                or text.use_sliding_window or hasattr(model.vlm.model.visual, "get_window_index") != backbone["window_permutation"]):
            raise RuntimeError(f"Processor or vision ordering differs from the validated {backbone['vision']} mapping")
        # Arbitrary colours across patch boundaries catch transposed or regrouped patches.
        fixture = Image.fromarray(np.random.default_rng(17).integers(0, 256, (112, 168, 3), dtype=np.uint8))
        actual = ip(images=fixture, return_tensors="np")
        if not np.allclose(actual["pixel_values"], pack_pixels(fixture, actual["image_grid_thw"][0], ip), atol=1e-6, rtol=0):
            raise RuntimeError("Processor resize/temporal/grouping fixture failed")
        _checkpoint_files, _weight_loading = files, loading
        _runtime, _loaded = (torch, processor, model), registration
    return _runtime


@contextmanager
def vision_order_check(torch, model):
    """Qwen2.5-VL's vision tower attends in windows over a permuted patch order and inverts it at the end:
    the merged tokens it hands the language model must be its merger output restored to row-major order."""
    visual = model.vlm.model.visual
    merger_output, checked = [None], []

    def merger_hook(_module, _args, output):
        merger_output[0] = output

    def vision_hook(module, _args, kwargs, output):
        merged = output[0] if isinstance(output, tuple) else output
        order, _ = module.get_window_index(kwargs["grid_thw"])
        if (not torch.equal(torch.sort(order).values, torch.arange(len(order), device=order.device))
                or not torch.equal(merged, merger_output[0][torch.argsort(order)])):
            raise RuntimeError("Runtime vision merge/window inversion differs from row-major mapping")
        checked.append(True)

    hooks = [visual.merger.register_forward_hook(merger_hook), visual.register_forward_hook(vision_hook, with_kwargs=True)]
    try:
        yield checked
    finally:
        for hook in hooks:
            hook.remove()


def capture(case, inputs, cache: Path, interrupted=lambda: False, *, capture_attention=False, attention_page_id=None, intervention=None, force_eager=False, capture_joint_attention=False, joint_answer_source=None, registration=REGISTRATION_ID):
    # Joint captures score with SDPA like every other capture; FP32 eager extraction removed the drift
    # that once made them score with BF16 eager.
    if not intervention and not force_eager:
        return _capture(case,inputs,cache,interrupted,capture_attention=capture_attention,attention_page_id=attention_page_id,capture_joint_attention=capture_joint_attention,joint_answer_source=joint_answer_source,registration=registration)
    # Controlled baselines and interventions score every forward with the FP32-arithmetic eager kernel;
    # stock BF16 eager scores drift by up to 2.1 from the SDPA source capture.
    from .encoding_capture import FLOAT32_EAGER, register_float32_eager
    _torch,_processor,model=runtime(cache,registration)
    language=model.vlm.model.language_model
    original=language.config._attn_implementation
    try:
        register_float32_eager()
        language.set_attn_implementation(FLOAT32_EAGER)
        record,arrays=_capture(case,inputs,cache,interrupted,capture_attention=capture_attention,attention_page_id=attention_page_id,intervention=intervention,capture_joint_attention=capture_joint_attention,joint_answer_source=joint_answer_source,registration=registration)
        eligible=sum(p.eligible for p in record.query_positions)
        score_tolerance=SCORE_KERNEL_TOLERANCES[registration]
        return record.model_copy(update={"provenance":{**record.provenance,"attention_implementation":"FP32-arithmetic eager controlled baseline/intervention"},
            "validation":{**record.validation,"eager_score_kernel_tolerance":score_tolerance*eligible,
                          "eager_score_kernel_tolerance_per_query_position":score_tolerance}}),arrays
    finally:
        language.set_attn_implementation(original)


def _capture(case, inputs, cache: Path, interrupted=lambda: False, *, capture_attention=False, attention_page_id=None, intervention=None, capture_joint_attention=False, joint_answer_source=None, registration=REGISTRATION_ID):
    """Inputs contain verified image bytes, PageRecord and raw ElementRecords.

    Reference answers and support labels never enter processing. Each page has its
    own forward pass; the query never conditions that pass.
    """
    from docattentionlens.generation.qwen import pack_pixels
    torch, processor, model = runtime(cache, registration)
    backbone, profile = BACKBONES[registration], RETRIEVER_PROFILES[registration]
    tokenizer = processor.tokenizer
    query_input = processor(text=[case.query], return_tensors="pt", padding="longest")
    if query_input["input_ids"].shape[1] > 256:
        raise ValueError("Processed query exceeds the 256-position capture limit; no truncation was applied")
    augmentation = processor.query_augmentation_token
    formatted = processor.query_prefix + case.query + augmentation*QUERY_AUGMENTATION_TOKENS
    offset_input = tokenizer(formatted, return_offsets_mapping=True, add_special_tokens=False)
    if offset_input["input_ids"] != query_input["input_ids"][0].tolist():
        raise ValueError("Query offset alignment failed")
    start = len(processor.query_prefix)
    end = start+len(case.query)
    augmentation_id = tokenizer.convert_tokens_to_ids(augmentation)
    qp = []
    for i, (token, mask, offsets) in enumerate(zip(query_input["input_ids"][0].tolist(), query_input["attention_mask"][0].tolist(), offset_input["offset_mapping"], strict=True)):
        role = "query_text" if offsets[1] > start and offsets[0] < end else "augmentation" if token == augmentation_id and mask else "prefix_or_special"
        qp.append(Position(index=i, token_id=token, piece=tokenizer.convert_ids_to_tokens(token), role=role, eligible=bool(mask), offsets=list(offsets)))
    with torch.inference_mode():
        q = model(**query_input.to("cuda")).embeddings[0].float().cpu()
    arrays = {"query": q.numpy()}
    encodings, attention_validation, attention_scores = [], {}, []
    eager_q = q.numpy()
    image_token_id = model.config.vlm_config.image_token_id
    query_eligible = [p.eligible for p in qp]
    embedding_tolerance, score_tolerance = EMBEDDING_KERNEL_TOLERANCES[registration], SCORE_KERNEL_TOLERANCES[registration]

    def check_embeddings(eager, baseline, eligible, encoding_id):
        mean_l2, max_abs = kernel_difference(eager, baseline, eligible)
        attention_validation["embedding_kernel_errors"][encoding_id] = {"mean_position_l2": mean_l2, "max_abs": max_abs}
        attention_validation["mean_embedding_kernel_l2_error"] = max(attention_validation["mean_embedding_kernel_l2_error"], mean_l2)
        attention_validation["max_embedding_kernel_error"] = max(attention_validation["max_embedding_kernel_error"], max_abs)
        if mean_l2 > embedding_tolerance:
            raise ValueError(f"Retriever eager/SDPA {encoding_id} embeddings exceed tolerance: mean per-position L2 distance "
                             f"{mean_l2:.4f} > {embedding_tolerance} (max absolute coordinate difference {max_abs:.4f})")

    if capture_attention:
        from .encoding_capture import capture_encoding
        if not attention_page_id or attention_page_id not in {p.page_id for p, _, _ in inputs}:
            raise ValueError("Select one candidate page for document encoding attention")
        eager_q, encoding, error = capture_encoding(torch, model, query_input.to("cuda"), "query", "query", arrays, causal=True, float32_attention=True,
                                                    rope_positions=rope_positions(query_input["input_ids"][0].tolist(), image_token_id))
        encodings.append(encoding)
        attention_validation = {"max_row_sum_error": error, "capture_on_off_embeddings_equal": True, "runtime_mask_verified": True,
            "runtime_positions_verified": True, "causal_future_mass": 0.0, "attention_arithmetic": "float32",
            "embedding_kernel_metric": "mean per-position L2 distance between unit embeddings, eligible positions",
            "embedding_kernel_l2_tolerance": embedding_tolerance, "mean_embedding_kernel_l2_error": 0.0,
            "max_embedding_kernel_error": 0.0, "embedding_kernel_errors": {},
            "score_kernel_tolerance_per_query_position": score_tolerance,
            "score_kernel_tolerance": score_tolerance * sum(query_eligible)}
        check_embeddings(eager_q, q.numpy(), query_eligible, "query")

    special = special_token_ids(tokenizer)
    ip = processor.image_processor
    evidence, grids = [], {}
    repeat_error = processor_error = 0.0
    for page, raw, elements in inputs:
        if interrupted():
            raise RuntimeError("Capture interrupted before page processing")
        if sha256(raw).hexdigest() != case.page_image_sha256[page.page_id]:
            raise ValueError("Page image differs from frozen case")
        with Image.open(BytesIO(raw)) as opened:
            if opened.size != (page.width, page.height):
                raise ValueError("Source dimensions differ from the frozen page")
            if opened.getexif().get(274, 1) != 1:
                raise ValueError("Unvalidated EXIF orientation")
            image = opened.convert("RGB")
        batch = processor(images=[image], return_tensors="pt")
        grid = batch["image_grid_thw"][0].tolist()
        patches = int(np.prod(grid))
        # The exact consumed pixels must equal an independent row-major regrouping of the resized page.
        processor_error = max(processor_error, float(np.abs(batch["pixel_values"][0][:patches].numpy()-pack_pixels(image, grid, ip)).max()))
        if processor_error > 1e-5:
            raise ValueError("Actual page pixel mapping differs from independent reconstruction")
        boxes = merged_boxes(grid, page.width, page.height)
        tokens = batch["input_ids"][0].tolist()
        visual = [i for i, token in enumerate(tokens) if token == image_token_id]
        if len(visual) != len(boxes) or len(boxes) > MAX_VISUAL_TOKENS or visual != list(range(visual[0], visual[0]+len(visual))):
            raise ValueError("Page processing changed the validated merged grid")
        positions, mappings = [], []
        for i, (token, mask) in enumerate(zip(tokens, batch["attention_mask"][0].tolist(), strict=True)):
            piece = tokenizer.convert_ids_to_tokens(token)
            bbox = boxes[i-visual[0]] if token == image_token_id else None
            positions.append(Position(index=i, token_id=token, piece=piece, role=token_role(token, piece, image_token_id, special), eligible=bool(mask), bbox=bbox))
            mappings.append(ownership(bbox, elements) if bbox else {"non_page": 1.0})
        grids[page.page_id] = grid
        from ..attention_interventions import apply as apply_attention_intervention
        from .encoding_capture import FLOAT32_EAGER
        active_intervention=intervention if intervention and intervention["page_id"]==page.page_id else None
        order_check = vision_order_check(torch, model) if backbone["window_permutation"] else nullcontext([True])
        with torch.inference_mode(), apply_attention_intervention(model,torch,active_intervention,positions,kernel=FLOAT32_EAGER), order_check as vision_checked:
            d = model(**batch.to("cuda")).embeddings[0].float().cpu()
            if not evidence:
                plain = model(**batch.to("cuda")).embeddings[0].float().cpu()
                repeat_error = float((plain-d).abs().max())
                if repeat_error != 0:
                    raise ValueError("Repeated plain forward differed from captured embeddings")
        if not vision_checked:
            raise RuntimeError("Runtime vision merge/window inversion differs from row-major mapping")
        score, winners, maxima, _ = maxsim(q.numpy(), d.numpy(), query_eligible, [p.eligible for p in positions])
        if capture_attention:
            eager_d = d.numpy()
            if page.page_id == attention_page_id:
                from .encoding_capture import capture_encoding
                eager_d, encoding, error = capture_encoding(torch, model, batch.to("cuda"), "document", "document", arrays, page.page_id, causal=True,
                                                            float32_attention=True, rope_positions=rope_positions(tokens, image_token_id, grid))
                check_embeddings(eager_d, d.numpy(), [p.eligible for p in positions], "document")
                encodings.append(encoding)
                attention_validation["max_row_sum_error"] = max(attention_validation["max_row_sum_error"], error)
                if capture_joint_attention:
                    from .joint_attention import capture_joint
                    if interrupted():
                        raise RuntimeError("Capture interrupted before diagnostic joint processing")
                    joint, joint_error = capture_joint(torch,model,processor,image,case.query,positions,mappings,arrays,page.page_id,True,float32_attention=True)
                    encodings.append(joint)
                    attention_validation["max_row_sum_error"] = max(attention_validation["max_row_sum_error"], joint_error)
                    if joint_answer_source == "reference":
                        if interrupted():
                            raise RuntimeError("Capture interrupted before diagnostic joint answer processing")
                        # The only place a case annotation enters the model; excluded from scoring.
                        answered, answer_error = capture_joint(torch,model,processor,image,case.query,positions,mappings,arrays,page.page_id,True,
                                                               float32_attention=True,answer=case.reference_answer,encoding_id="joint_answer")
                        encodings.append(answered)
                        attention_validation["max_row_sum_error"] = max(attention_validation["max_row_sum_error"], answer_error)
            attention_score = maxsim(eager_q, eager_d, query_eligible, [p.eligible for p in positions])[0]
            attention_scores.append(attention_score)
            if abs(attention_score-score) > attention_validation["score_kernel_tolerance"]:
                raise ValueError(f"Retriever eager/SDPA scores exceed tolerance on {page.page_id}: |{attention_score:.4f} - {score:.4f}| > "
                                 f"{attention_validation['score_kernel_tolerance']:.4f} ({score_tolerance} per eligible query position)")
        # The upstream scorer on precisely the eligible, unpadded FP32 domain.
        reference = float(processor.score_retrieval([q[[p.eligible for p in qp]]], [d[[p.eligible for p in positions]]], batch_size=1)[0, 0])
        error = abs(score-reference)
        if error > 1e-4:
            raise ValueError(f"Scorer reconstruction failed: {error}")
        name = "page_"+str(len(evidence)); arrays[name] = d.numpy()
        ledger = contributions(winners, maxima, mappings)
        ledger["scorer_rounding"] = score-sum(ledger.values())
        evidence.append(PageEvidence(page_id=page.page_id, image_sha256=page.image_sha256, width=page.width, height=page.height, page_number=page.page_number,
            score=score, reference_score=reference, reconstruction_error=error, positions=positions, vector_array=name, ownership=mappings,
            contributions=ledger, winner_indices=winners, query_maxima=maxima))
    if capture_attention:
        baseline = [p.score for p in evidence]
        if sorted(range(len(evidence)), key=lambda i: -baseline[i]) != sorted(range(len(evidence)), key=lambda i: -attention_scores[i]):
            raise ValueError(f"Retriever eager/SDPA candidate rankings differ: SDPA scores {baseline}, eager scores {attention_scores}")
        attention_validation.update({"candidate_ranking_equal": True, "max_score_kernel_error": max(abs(a-b) for a, b in zip(baseline, attention_scores, strict=True)),
            "baseline_scores": baseline, "eager_attention_scores": attention_scores})
    import transformers.models.colqwen2.modeling_colqwen2 as modeling
    import transformers.models.colqwen2.processing_colqwen2 as processing
    import transformers.models.qwen2_vl.image_processing_qwen2_vl as image_processing
    modules = [modeling, processing, image_processing]
    if backbone["window_permutation"]:
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as backbone_modeling
        modules.append(backbone_modeling)
    code = {Path(m.__file__).name: sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules}
    answered = capture_attention and capture_joint_attention and joint_answer_source == "reference"
    return RetrievalCapture(schema_version="0.12.0" if answered else "0.11.0" if capture_joint_attention else "0.10.0", capture_id="capture_"+uuid4().hex, case_id=case.case_id, created_at=datetime.now(UTC).isoformat(),
        producer_version=__version__, dependencies=[case.case_id, *[p.page_id for p in evidence]], dataset_version=case.dataset_version,
        annotation_version=case.annotation_version, query=case.query, query_positions=qp, query_array="query", pages=evidence,
        attention_encodings=encodings,
        provenance={"adapter_version": profile["adapter_version"], "model_registration_id": registration, "model_name": profile["name"],
            "model_id": profile["model_id"], "model_revision": profile["revision"], "processor_revision": profile["revision"], **_weight_loading,
            "query_prefix": processor.query_prefix,
            "query_augmentation_token": augmentation, "query_augmentation_tokens": QUERY_AUGMENTATION_TOKENS, "formatted_query": formatted,
            "visual_prompt": processor.visual_prompt_prefix, "model_input": "RGB page image only; separate query pass", "evaluation_labels_consumed": answered,
            **({"evaluation_labels_consumed_by": "diagnostic joint_answer encoding only (reference answer in its prompt); query, page and scores use no labels"} if answered else {}),
            "torch": version("torch"), "transformers": version("transformers"), "numpy": version("numpy"), "pillow": version("Pillow"),
            "python": platform.python_version(), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "eval_mode": True,
            "encoder_dtype": "bfloat16", "stored_dtype": "float32 lossless cast of BF16 outputs", "attention_implementation": "sdpa", "seed": 0, "tf32": False,
            "checkpoint_file_sha256": _checkpoint_files, "adapter_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "scoring_source_sha256": sha256(Path(__file__).with_name("scoring.py").read_bytes()).hexdigest(), "transformers_code_sha256": code,
            "image_processor": type(ip).__name__, "tokenizer": type(tokenizer).__name__,
            **({"attention_adapter_version": profile["attention_adapter_version"],
                "attention_implementation": "eager language extraction with FP32 logits, softmax and value mixing; SDPA baseline scoring",
                "stored_dtype": "embeddings: FP32 lossless cast of BF16 outputs; attention: FP32 post-softmax weights from FP32 logits",
                "attention_adapter_source_sha256": sha256(Path(__file__).with_name("encoding_capture.py").read_bytes()).hexdigest(),
                "attention_coverage": "separate query and document plus diagnostic joint page-plus-question; full causal language encoding matrices; joint pass excluded from scoring; vision tower excluded" if capture_joint_attention else "query and one selected document; full causal language encoding matrices; vision tower excluded",
                **({"diagnostic_joint_answer_source":"reference"} if answered else {}),
                **({"diagnostic_joint_pass":True,"joint_capture_policy":"retain_document_v1",
                    "joint_attention_source_sha256":sha256(Path(__file__).with_name("joint_attention.py").read_bytes()).hexdigest()} if capture_joint_attention else {})} if capture_attention else {})},
        scorer={"quantity": "late_interaction_similarity", "formula": "sum_i max_j dot(q_i,d_j)", "precision": "float32 CPU",
            "normalization": "encoder L2 normalization in BF16; no subsequent normalization",
            "eligible_domain": "all active positions, including query prefix/augmentation and non-page visual prompt", "padding": "inactive positions excluded; no zero padding in max domain",
            "tie_policy": "first eligible position", "candidate_scope": "explicit selected pages; not corpus retrieval",
            "ledger_precision": "float64 allocation; scorer_rounding reconciles FP32 score reduction"},
        geometry={"policy": "qwen2vl_smart_resize_row_major_v1", "patch_size": 14, "merge_size": 2, "max_visual_tokens": MAX_VISUAL_TOKENS,
            "grids": grids, "order": ("row-major merged 28 px cells; vision windows permute patches and are inverted before the language model"
                                      if backbone["window_permutation"] else "row-major merged 28 px cells; no vision window permutation"),
            "operations": ["RGB conversion", "aspect-preserving smart resize to multiples of 28 px, at most 768 merged cells; no crop/pad/tile",
                           "rescale and channel normalization", "14 px patches, duplicated temporal frame, 2x2 spatial merge", "one merged cell per image-token position"],
            "resample": int(ip.resample), "ownership_policy": "exact_cell_equal_overlap_v1",
            "flagged_geometry": "positive raw intersection allocated to unreviewed_geometry; no silent repair", "residuals": ["background", "non_page", "unreviewed_geometry"],
            "coverage": "full source page; rectangles describe nominal merged-cell footprints, not receptive-field attribution"},
        validation={"processor_fixture": True, "actual_page_processor_max_error": processor_error,
            **({"runtime_vision_window_inversion": True} if backbone["window_permutation"] else {"no_vision_window_permutation": True}),
            "source_dimensions_and_hashes": True, "score_tolerance_absolute": 1e-4, "max_reconstruction_error": max(p.reconstruction_error for p in evidence),
            "ownership_conserved": True, "signed_values_preserved": True, "repeat_plain_forward_max_abs_error": repeat_error,
            **({"attention": attention_validation} if capture_attention else {})}), arrays
