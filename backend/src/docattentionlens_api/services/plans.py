"""Model execution plans: whether a live GPU worker can run a retriever or generator job now, and its limits.

Run endpoints check these before queueing a new job; the plan endpoints and runtime status report them."""
from pydantic import BaseModel
from docattentionlens_api.errors import fail
from docattentionlens.generation.contracts import (QWEN25_VL_3B, MAX_INPUT_POSITIONS, MAX_NEW_TOKENS,
    CAPTURE_STEPS, SINGLE_PAGE_IMAGE_POSITIONS)
from docattentionlens.retrieval.contracts import MODEL_ID, MODEL_REVISION


class RetrievalPlan(BaseModel):
    available: bool
    reason: str
    model_registration_id: str = "builtin-retrieval"
    model_name: str = "ColPali v1.3"
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    candidate_limit: int = 8
    required_free_gpu_mib: int = 8192
    estimated_retained_bytes_per_page: int = 700000
    maximum_retained_attention_bytes: int = 735000000
    attention_available: bool = False
    execution: str = "One BF16 page forward at a time; query forward separate; FP32 scoring. Cold model download ~6 GB."
    scope: str = "Explicit case pages only; no corpus-wide ranking. No evaluation labels consumed."


def retriever_profile(registration_id):
    from docattentionlens.retrieval.contracts import RETRIEVER_PROFILES
    profile = RETRIEVER_PROFILES.get(registration_id)
    if profile is None:
        fail("incompatible_model", "Choose a retrieval model with an installed retrieval adapter", 422)
    return profile


def retrieval_plan(jobs, registration_id="builtin-retrieval", capture_joint_attention=False, joint_answer=False):
    from docattentionlens.retrieval.contracts import attention_capture_budget
    profile = retriever_profile(registration_id)
    required = profile["required_free_mib"]
    def ready(w, key, legacy, adapter):
        capabilities = w["capabilities"]
        return (w["available"] and w["resource_class"]=="gpu" and adapter in capabilities.get(key, [capabilities.get(legacy)])
                and (capabilities.get("free_mib",0)+capabilities.get("reclaimable_mib",0))>=required)
    workers = jobs.workers()
    eligible = any(ready(w, "adapters", "adapter", profile["adapter_version"]) for w in workers)
    attention_eligible = any(ready(w, "retrieval_attention_adapters", "retrieval_attention_adapter", profile["attention_adapter_version"]) for w in workers)
    # ColQwen pages keep their aspect ratio, so they retain fewer vectors than ColPali's fixed grid.
    download = {"builtin-colqwen2": "Cold model download ~4.4 GB.", "builtin-colqwen2.5": "Cold download ~7.3 GB (base weights and LoRA adapter, merged at load)."}.get(registration_id)
    return RetrievalPlan(available=eligible,attention_available=attention_eligible,model_registration_id=registration_id,model_name=profile["name"],
        model_id=profile["model_id"],model_revision=profile["revision"],maximum_retained_attention_bytes=attention_capture_budget(profile,capture_joint_attention,joint_answer)[0],
        required_free_gpu_mib=required,estimated_retained_bytes_per_page=450000 if profile["causal"] else 700000,
        execution=(f"One BF16 page forward at a time; query forward separate; FP32 scoring. Pages keep their aspect ratio (at most {profile['image_positions_per_page']} image positions). {download}"
                   if download else RetrievalPlan.model_fields["execution"].default),
        reason=f"Local {profile['name']} GPU worker available" if eligible else f"Start the GPU worker with at least {required//1024} GiB free memory. Saved captures remain readable.")


class GenerationPlan(BaseModel):
    available: bool
    reason: str
    model_registration_id: str = QWEN25_VL_3B.registration_id
    model_id: str = QWEN25_VL_3B.model_id
    model_revision: str = QWEN25_VL_3B.revision
    page_limit: int = 2
    max_pixels_per_page: int = QWEN25_VL_3B.max_pixels
    # A single supplied page may use the finer plan (more merged image positions).
    single_page_image_positions: int = SINGLE_PAGE_IMAGE_POSITIONS
    input_position_limit: int = MAX_INPUT_POSITIONS
    max_new_tokens: int = MAX_NEW_TOKENS
    captured_steps: int = CAPTURE_STEPS
    # Full-attention captures keep the row of every generated answer token, up to max_new_tokens.
    full_attention_captured_steps: int = MAX_NEW_TOKENS
    layers: int = QWEN25_VL_3B.layers
    query_heads: int = QWEN25_VL_3B.query_heads
    kv_heads: int = QWEN25_VL_3B.kv_heads
    required_free_gpu_mib: int = QWEN25_VL_3B.required_free_gib*1024
    retained_row_budget_bytes: int = (8+64)*QWEN25_VL_3B.layers*QWEN25_VL_3B.query_heads*(MAX_INPUT_POSITIONS+MAX_NEW_TOKENS)*4
    execution: str = "BF16 generation, one model resident at a time. Eager language attention materializes dense prefill matrices; hooks retain prediction rows and question-token rows. Two runs check capture-on/off equivalence. Cold checkpoint ~7.5 GB."
    context_policy: str = "Explicit ordered case pages or retrieval candidates as images plus frozen query; aspect-preserving resize, no text truncation, no OCR or evaluation labels."


def generation_plan(jobs, profile=QWEN25_VL_3B, registration_id=None):
    required=profile.required_free_gib*1024
    eligible=any(w["available"] and w["resource_class"]=="gpu" and profile.adapter_version in w["capabilities"].get("adapters",[]) and (w["capabilities"].get("free_mib",0)+w["capabilities"].get("reclaimable_mib",0))>=required for w in jobs.workers())
    details={} if profile is QWEN25_VL_3B and registration_id in (None, "builtin") else {
        "model_registration_id":registration_id or profile.registration_id,"model_id":profile.model_id,"model_revision":profile.revision,
        "max_pixels_per_page":profile.max_pixels,"layers":profile.layers,"query_heads":profile.query_heads,"kv_heads":profile.kv_heads,
        "required_free_gpu_mib":required,"retained_row_budget_bytes":(8+64)*profile.layers*profile.query_heads*(MAX_INPUT_POSITIONS+MAX_NEW_TOKENS)*4,
        "execution":f"BF16 generation, one model resident at a time. Eager language attention materializes dense prefill matrices; hooks retain prediction rows and question-token rows. Two runs check capture-on/off equivalence. {profile.name} needs {profile.required_free_gib} GiB free GPU memory."}
    return GenerationPlan(available=eligible,reason="Local generator GPU worker available" if eligible else f"Start the GPU worker with at least {profile.required_free_gib} GiB free memory. Saved answers and attention remain readable.",**details)
