"""Conformance against installed Transformers attention kernels on synthetic tiny models.

No checkpoint weights or scientific cases are used. Checks actual head-output axes,
edge masking, causal support, normalization and restoration for the language-model
family of every built-in model registration, on both stock eager and the registered
FP32-arithmetic eager kernel. Families come from each registration's
`model_type`, so a newly registered generator or retriever is covered without editing
this script; a family that cannot be built here is reported as a failure.

Runs where torch and transformers are installed (the GPU worker image):

    docker exec -i docattentionlens-worker-gpu-1 python - < scripts/attention_kernel_conformance.py
"""
import json
import sys
from importlib import import_module
from types import SimpleNamespace as NS
import torch
import transformers
from transformers import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES
from docattentionlens_api.services.model_catalogue import builtin_models
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from docattentionlens.attention_interventions import apply,plan
from docattentionlens.retrieval.encoding_capture import FLOAT32_EAGER,eager_float32_attention,register_float32_eager
from docattentionlens.methods.registry import get_definition,get_method
from docattentionlens.retrieval.contracts import RETRIEVER_PROFILES

torch.manual_seed(0)
positions=[NS(index=i,role=role,page_id="p" if role=="image" else None,
    bbox=[0,0,2,2] if i==1 else [2,0,4,2] if i==2 else None)
    for i,role in enumerate(["special","image","image","query","query"])]
small={"vocab_size":32,"hidden_size":16,"intermediate_size":32,"num_hidden_layers":2,
       "num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"max_position_embeddings":64}
# Multimodal rotary families take (temporal, height, width) position ids; the section sums to head_dim / 2.
ROPE={"standard":({},lambda n:torch.arange(n)[None,:]),
      "multimodal":({"rope_scaling":{"type":"default","mrope_section":[1,1,2]}},lambda n:torch.arange(n)[None,None,:].expand(3,1,n))}


def language_config(model_type):
    """The language-model configuration inside a (possibly nested) multimodal configuration."""
    config=AutoConfig.for_model(model_type)
    while True:
        nested=getattr(config,"text_config",None) or getattr(config,"vlm_config",None)
        if nested is None:return config
        config=nested


def families():
    """(language model_type, causal) -> registrations, from every built-in registration."""
    groups={}
    for registration in builtin_models():
        identifier=registration["registration_id"]
        if registration["role"]=="generation":
            causal=True
        elif identifier in RETRIEVER_PROFILES:
            causal=RETRIEVER_PROFILES[identifier]["causal"]
        else:
            raise ValueError(f"{identifier}: retrieval registration without a retriever profile")
        family=language_config(registration["architecture"]["model_type"]).model_type
        groups.setdefault((family,causal),[]).append(identifier)
    return groups


def tiny(family,rope):
    """A two-layer random language model of this family on the installed kernels."""
    if family not in MODEL_MAPPING_NAMES:
        raise ValueError(f"No Transformers model class for language family {family}")
    model_type=getattr(transformers,MODEL_MAPPING_NAMES[family])
    config_type=type(AutoConfig.for_model(family))
    return model_type(config_type(**small,**ROPE[rope][0])).float().eval()


def conform(family,causal,kernel="eager"):
    """Every intervention method on one tiny model and attention kernel; returns the rotary layout used and per-method results."""
    if kernel==FLOAT32_EAGER:
        register_float32_eager()
    errors={}
    for rope in ROPE:
        try:
            language=tiny(family,rope)
            language.set_attn_implementation(kernel)
            kwargs={"input_ids":torch.tensor([[0,1,2,3,4]]),"output_attentions":True,"use_cache":False,"position_ids":ROPE[rope][1](5)}
            mask=torch.zeros((1,1,5,5))
            if causal:mask.masked_fill_(torch.triu(torch.ones(5,5,dtype=torch.bool),1),float("-inf"))
            kwargs["attention_mask"]=mask
            with torch.inference_mode(): language(**kwargs)
            break
        except Exception as error:  # Try the next rotary layout; report all if none runs.
            errors[rope]=f"{type(error).__name__}: {error}"
    else:
        raise RuntimeError(f"No rotary layout runs {family}: {errors}")
    carrier=NS(model=NS(language_model=language))
    projection_inputs=[]
    handle=language.layers[0].self_attn.o_proj.register_forward_pre_hook(lambda module,args:projection_inputs.append(args[0].detach().clone()))
    with torch.inference_mode(): baseline=language(**kwargs)
    original=import_module(type(language.layers[0].self_attn).__module__).eager_attention_forward

    def restored_kernels():
        return (import_module(type(language.layers[0].self_attn).__module__).eager_attention_forward is original
                and (kernel=="eager" or ALL_ATTENTION_FUNCTIONS[FLOAT32_EAGER] is eager_float32_attention)
                and language.config._attn_implementation==kernel)
    checks=[]
    for identifier in ("attention.head_ablation","attention.layer_ablation","attention.edge_ablation","attention.modality_ablation","attention.region_masking"):
        descriptor=get_method(identifier); definition=get_definition(identifier,descriptor.variant_id)
        p={"source_index":4,"target_index":1} if identifier=="attention.edge_ablation" else {"region":[0,0,.4,1]} if identifier=="attention.region_masking" else {}
        parameters=definition.parameters.model_validate(p).model_dump(mode="json")
        spec=plan(identifier,parameters,positions,[1]*5,causal,2,2,"p")
        with torch.inference_mode(),apply(carrier,torch,spec,positions,kernel=kernel): output=language(**kwargs)
        weights=output.attentions[0][0]
        assert torch.allclose(weights.sum(-1),torch.ones((2,5)),atol=1e-6)
        if causal:assert torch.count_nonzero(weights[:,torch.triu(torch.ones(5,5,dtype=torch.bool),1)])==0
        if identifier=="attention.head_ablation":
            assert torch.count_nonzero(projection_inputs[-1][...,:8])==0
            assert torch.equal(projection_inputs[-1][...,8:],projection_inputs[0][...,8:])
            assert torch.equal(output.attentions[0],baseline.attentions[0])
        elif identifier=="attention.layer_ablation":
            assert torch.count_nonzero(projection_inputs[-1])==0
        elif identifier=="attention.edge_ablation":
            assert weights[0,4,1]==0 and weights[1,4,1]>0
        elif identifier=="attention.modality_ablation":
            for i in (0,3,4):assert torch.count_nonzero(weights[:,i,1:3])==0
        else:
            assert weights[0,4,1]==0 and weights[0,1,1]>0
        assert restored_kernels()
        with torch.inference_mode(): restored=language(**kwargs)
        assert torch.equal(restored.last_hidden_state,baseline.last_hidden_state)
        checks.append({"method_id":identifier,"kernel":kernel,"passed":True})
    handle.remove()
    try:
        with apply(carrier,torch,spec,positions,kernel=kernel):raise RuntimeError("intentional restoration probe")
    except RuntimeError:pass
    assert restored_kernels()
    return rope,checks


def main():
    report=[]
    for (family,causal),registrations in families().items():
        entry={"family":family,"causal":causal,"registrations":registrations}
        try:
            entry["rope"],entry["checks"]=conform(family,causal)
            # The FP32-arithmetic kernel ColQwen2 scores interventions with, on every family.
            entry["checks"]+=conform(family,causal,FLOAT32_EAGER)[1]
            entry["passed"]=True
        except Exception as error:
            entry.update(passed=False,error=f"{type(error).__name__}: {error}"[:600])
        report.append(entry)
    failed=[e for e in report if not e["passed"]]
    print(json.dumps({"scope":"Synthetic tiny models on installed attention kernels; not pretrained model validation",
                      "transformers":transformers.__version__,"torch":torch.__version__,"families":report,
                      "registrations":{r:{"family":e["family"],"causal":e["causal"],"passed":e["passed"]} for e in report for r in e["registrations"]}},indent=2))
    print(f"{len(report)} language families, {len(failed)} failed",file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__=="__main__":
    main()
