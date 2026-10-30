"""Model descriptions derived from configuration, with explicit adapter coverage.

Architecture is checkpoint metadata, not a claim that its attention is captured.
New profiles/imported configurations need no model-specific frontend prose.
"""
from typing import Literal
from pydantic import BaseModel, Field
from docattentionlens.generation import contracts as generation
from docattentionlens.retrieval import contracts as retrieval

ModelRole = Literal["generation", "retrieval"]


class ModelComponent(BaseModel):
    name: str
    layers: int | None = None
    query_heads: int | None = None
    kv_heads: int | None = None
    hidden_size: int | None = None
    patch_size: int | None = None
    spatial_merge_size: int | None = None
    image_size: int | None = None


class ModelArchitecture(BaseModel):
    model_type: str
    architectures: list[str] = Field(default_factory=list)
    components: list[ModelComponent] = Field(default_factory=list)
    embedding_dim: int | None = None
    dtype: str | None = None


class ModelCapabilities(BaseModel):
    adapter_id: str | None = None
    can_run: bool = False
    attention_available: bool = False
    signal_description: str
    input_description: str
    summary: str
    limits: dict[str, int] = Field(default_factory=dict)
    reason: str = ""


class ModelRegistration(BaseModel):
    registration_id: str
    name: str
    model_id: str
    revision: str
    role: ModelRole
    architecture: ModelArchitecture
    capabilities: ModelCapabilities
    validation: str
    files: dict[str, str] = Field(default_factory=dict)


def positive(config, *keys):
    return next((config[k] for k in keys if type(config.get(k)) is int and config[k] > 0), None)


def describe_architecture(config, processor=None):
    nested = config.get("vlm_config", {})
    if not isinstance(nested, dict):
        nested = {}
    text = config.get("text_config", nested.get("text_config", config))
    vision = config.get("vision_config", nested.get("vision_config", {}))
    components = []
    for name, block in (("Language", text), ("Vision", vision)):
        if not isinstance(block, dict) or not block:
            continue
        layers = positive(block, "num_hidden_layers", "depth", "n_layer", "num_layers")
        heads = positive(block, "num_attention_heads", "num_heads", "n_head")
        if not layers and not heads:
            continue
        components.append(ModelComponent(name=name, layers=layers, query_heads=heads,
            kv_heads=positive(block, "num_key_value_heads"), hidden_size=positive(block, "hidden_size", "d_model"),
            patch_size=positive(block, "patch_size"), spatial_merge_size=positive(block, "spatial_merge_size"),
            image_size=positive(block, "image_size")))
    declared = config.get("architectures", [])
    return ModelArchitecture(model_type=str(config.get("model_type", "unknown")),
        architectures=[a for a in declared if isinstance(a, str)] if isinstance(declared, list) else [], components=components,
        embedding_dim=positive(config, "embedding_dim", "projection_dim"),
        dtype=config.get("dtype", config.get("torch_dtype")) if isinstance(config.get("dtype", config.get("torch_dtype")), str) else None).model_dump(mode="json")


def generator_profile(registration):
    """The adapter profile of a built-in or uploaded generation registration, if any."""
    if registration["registration_id"] in generation.GENERATORS:
        return generation.GENERATORS[registration["registration_id"]]
    return next((p for p in generation.GENERATORS.values() if p.adapter_version == registration["capabilities"].get("adapter_id")), None)


def describe_capabilities(config, processor, role, builtin=None):
    """`builtin` names a built-in retriever's registration; uploaded checkpoints are described from their configuration."""
    profile = generation.compatible_profile(config, processor) if role == "generation" else None
    if profile:
        deepstack = (" Image positions also receive DeepStack features from vision layers "
                     + ", ".join(map(str, profile.deepstack_visual_indexes)) + " after the first decoder layers.") if profile.deepstack_visual_indexes else ""
        return ModelCapabilities(adapter_id=profile.adapter_version, can_run=True, attention_available=True,
            signal_description="Language self-attention · mapped document images",
            input_description="Question and ordered page images; generated answer predictions",
            summary=f"Prediction attention over all causal targets: {profile.layers} layers × {profile.query_heads} query heads "
                    f"({profile.kv_heads} KV heads), {profile.cell} px merged image cells. Capture-on/off runs check output equivalence.{deepstack}",
            limits={"page_limit":2,"captured_steps":generation.CAPTURE_STEPS,"full_attention_captured_steps":generation.MAX_NEW_TOKENS,
                "max_new_tokens":generation.MAX_NEW_TOKENS,
                "input_positions":generation.MAX_INPUT_POSITIONS,"image_positions_per_page":generation.STANDARD_IMAGE_POSITIONS,
                "single_page_image_positions":generation.SINGLE_PAGE_IMAGE_POSITIONS,"required_free_gib":profile.required_free_gib},
            reason="Compatible generation adapter; full loading and mapping are checked at launch.").model_dump(mode="json")
    if role == "retrieval" and builtin in retrieval.RETRIEVER_PROFILES:
        profile = retrieval.RETRIEVER_PROFILES[builtin]
        heads = f"{profile['layers']} language layers and {profile['query_heads']} query heads"
        limits = {"candidate_limit":8,"query_positions":256,"image_positions_per_page":profile["image_positions_per_page"],"attention_page_limit":1}
        if profile["causal"]:
            # A LoRA checkpoint names the base it is merged into, which pins its own revision.
            merged = f" The LoRA adapter is merged into {profile['base_model_id']} at load." if profile.get("base_model_id") else ""
            return ModelCapabilities(adapter_id=profile["adapter_version"], can_run=True, attention_available=True,
                signal_description="Retriever causal language attention · embeddings and MaxSim similarity",
                input_description=f"Separate query and page encodings; pages keep their aspect ratio with at most {profile['image_positions_per_page']} image positions",
                summary=f"Capture all {heads} for query encoding and one selected document encoding. The encoder is causal: page patches precede the prompt and cannot attend to it. Vision-tower attention is excluded.{merged}",
                limits=limits,
                reason="Pinned retrieval attention adapter; pixel mapping, causal masks, normalization and capture-on/off equivalence checked at launch.").model_dump(mode="json")
        return ModelCapabilities(adapter_id=profile["adapter_version"], can_run=True, attention_available=True,
            signal_description="Retriever language attention · embeddings and MaxSim similarity",
            input_description="Separate query and page encodings; the query does not condition the page encoder",
            summary=f"Capture all {heads} for query encoding and one selected document encoding. Query and page passes are separate; vision-tower attention is excluded.",
            limits=limits,
            reason="Pinned retrieval attention adapter; masks, normalization and capture-on/off equivalence checked at launch.").model_dump(mode="json")
    return ModelCapabilities(signal_description="Checkpoint metadata · extraction adapter required",
        input_description="Inputs and capture coverage depend on a compatible model adapter",
        summary="Architecture metadata is available. This checkpoint cannot be launched with the installed adapters.",
        reason=f"No compatible {role} extraction adapter is installed for this checkpoint.").model_dump(mode="json")


def builtin_models():
    # Architecture values from the exact pinned checkpoint configurations.
    qwen = {"model_type":"qwen2_5_vl","architectures":["Qwen2_5_VLForConditionalGeneration"],"torch_dtype":"bfloat16",
        "num_hidden_layers":36,"num_attention_heads":16,"num_key_value_heads":2,"hidden_size":2048,"intermediate_size":11008,"vocab_size":151936,
        "vision_config":{"depth":32,"hidden_size":1280,"num_heads":16,"out_hidden_size":2048,"patch_size":14,"spatial_merge_size":2,"temporal_patch_size":2}}
    colpali = {"model_type":"colpali","architectures":["ColPaliForRetrieval"],"torch_dtype":"bfloat16","embedding_dim":128,
        "text_config":{"num_hidden_layers":18,"num_attention_heads":8,"num_key_value_heads":1,"hidden_size":2048},
        "vision_config":{"num_hidden_layers":27,"num_attention_heads":16,"hidden_size":1152,"patch_size":14,"image_size":448}}
    # From the pinned vidore/colqwen2-v1.0-hf configuration.
    colqwen = {"model_type":"colqwen2","architectures":["ColQwen2ForRetrieval"],"torch_dtype":"bfloat16","embedding_dim":128,
        "text_config":{"num_hidden_layers":28,"num_attention_heads":12,"num_key_value_heads":2,"hidden_size":1536},
        "vision_config":{"depth":32,"num_heads":16,"patch_size":14,"spatial_merge_size":2}}
    # From the pinned vidore/colqwen2.5-base configuration; its unstated Qwen2.5-VL vision settings are the adapter's validated defaults.
    colqwen25 = {"model_type":"qwen2_5_vl","architectures":["ColQwen2_5"],"torch_dtype":"bfloat16","embedding_dim":128,
        "text_config":{"num_hidden_layers":36,"num_attention_heads":16,"num_key_value_heads":2,"hidden_size":2048},
        "vision_config":{"depth":32,"num_heads":16,"hidden_size":1280,"patch_size":14,"spatial_merge_size":2}}
    generators = []
    for profile in generation.GENERATORS.values():
        # The first generator stays the default "builtin" registration used by quick acquisition.
        config = qwen if profile is generation.QWEN25_VL_3B else {"model_type":profile.model_type,"architectures":[profile.model_class],
            "torch_dtype":"bfloat16",**profile.text,"vision_config":profile.vision}
        settings = {"patch_size":profile.patch_size,"merge_size":profile.merge_size,"temporal_patch_size":profile.temporal_patch_size}
        generators.append({"registration_id":profile.registration_id,"name":profile.name,"model_id":profile.model_id,"revision":profile.revision,"role":"generation",
            "architecture":describe_architecture(config),"capabilities":describe_capabilities(config, settings, "generation"),
            "validation":"Pinned adapter; capture-on/off and page-mapping checks run for every capture"})
    return [*generators,
        {"registration_id":"builtin-retrieval","name":"ColPali v1.3","model_id":retrieval.MODEL_ID,"revision":retrieval.MODEL_REVISION,"role":"retrieval",
        "architecture":describe_architecture(colpali),"capabilities":describe_capabilities(colpali, {}, "retrieval", builtin="builtin-retrieval"),
        "validation":"Pinned retrieval adapter; image mapping and MaxSim reconstruction checked for every capture"},
        {"registration_id":"builtin-colqwen2","name":"ColQwen2 v1.0","model_id":retrieval.COLQWEN_MODEL_ID,"revision":retrieval.COLQWEN_MODEL_REVISION,"role":"retrieval",
        "architecture":describe_architecture(colqwen),"capabilities":describe_capabilities(colqwen, {}, "retrieval", builtin="builtin-colqwen2"),
        "validation":"Pinned retrieval adapter; pixel mapping, causal masks and MaxSim reconstruction checked for every capture"},
        {"registration_id":"builtin-colqwen2.5","name":"ColQwen2.5 v0.2","model_id":retrieval.COLQWEN25_MODEL_ID,"revision":retrieval.COLQWEN25_MODEL_REVISION,"role":"retrieval",
        "architecture":describe_architecture(colqwen25),"capabilities":describe_capabilities(colqwen25, {}, "retrieval", builtin="builtin-colqwen2.5"),
        "validation":"Pinned retrieval adapter and base checkpoint; LoRA merge, pixel mapping, vision window inversion, causal masks and MaxSim reconstruction checked for every capture"}]
