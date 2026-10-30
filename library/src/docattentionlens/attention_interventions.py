"""Validated plans and scoped eager-attention interventions for pinned language adapters.

The model side of an attention intervention: `plan` checks a method's parameters against the frozen
input positions and `apply` patches the attention kernel while `generation/qwen.py`,
`retrieval/colpali.py` or `retrieval/colqwen.py` re-runs. The immutable experiment record that freezes
a plan and reports outcomes is `experiments/attention_interventions.py`.
"""
from contextlib import contextmanager
from importlib import import_module
import numpy as np

METHODS={"attention.head_ablation","attention.layer_ablation","attention.edge_ablation",
         "attention.modality_ablation","attention.region_masking"}

def removed_edges(spec,positions,query_indices,key_count):
    """Boolean source-target removals, extending generated positions as non-image sources."""
    image=np.zeros(key_count,dtype=bool)
    for p in positions:
        if p.index<key_count: image[p.index]=p.role in ("image","visual_patch")
    queries=np.asarray(query_indices,dtype=int)
    mode=spec["method_id"]
    drop=np.zeros((len(queries),key_count),dtype=bool)
    if mode=="attention.edge_ablation":
        drop[queries==spec["parameters"]["source_index"],spec["parameters"]["target_index"]]=True
    elif mode=="attention.modality_ablation":
        source_image,target_image={"text_to_vision":(False,True),"vision_to_text":(True,False),
            "text_to_text":(False,False),"vision_to_vision":(True,True)}[spec["parameters"]["block"]]
        sources=(image[queries] if source_image else ~image[queries])
        targets=image if target_image else ~image
        drop=sources[:,None]&targets[None,:]
    elif mode=="attention.region_masking":
        region=np.zeros(key_count,dtype=bool)
        region[spec["region_targets"]]=True
        drop=(~region[queries])[:,None]&region[None,:]
    return drop

def validate_plan(spec,positions,mask,causal):
    n=len(positions); active=np.asarray(mask,bool)
    if len(active)!=n or not active.any(): raise ValueError("Intervention input mask differs")
    if spec["method_id"] in ("attention.head_ablation","attention.layer_ablation"): return
    removed=removed_edges(spec,positions,range(n),n)
    allowed=np.broadcast_to(active,(n,n)).copy()
    if causal: allowed &= np.tri(n,dtype=bool)
    if np.any(active & ~(allowed&~removed).any(-1)):
        raise ValueError("This intervention would remove every valid key from a source row")
    if not np.any(removed&allowed&active[:,None]):
        raise ValueError("This intervention removes no valid attention edges on the frozen input")

def plan(method_id,parameters,positions,mask,causal,layer_count,head_count,page_id=None):
    if method_id not in METHODS: raise ValueError("Unknown attention intervention")
    p=dict(parameters)
    layers=list(range(layer_count)) if p.get("all_layers") else [p["layer"]]
    heads=list(range(head_count)) if method_id=="attention.layer_ablation" or p.get("all_heads") else [p.get("head",0)]
    if any(not 0<=i<layer_count for i in layers) or any(not 0<=i<head_count for i in heads):
        raise ValueError("Layer or head is outside the frozen capture")
    spec={"method_id":method_id,"parameters":p,"layers":layers,"heads":heads,"page_id":page_id}
    if method_id=="attention.edge_ablation":
        if not 0<=p["source_index"]<len(positions) or not 0<=p["target_index"]<len(positions):
            raise ValueError("Edge positions must belong to the frozen input prefix")
    if method_id=="attention.region_masking":
        a,b,c,d=p["region"]
        if not 0<=a<c<=1 or not 0<=b<d<=1: raise ValueError("Region must be a positive unit-page rectangle")
        page=[x for x in positions if x.role in ("image","visual_patch") and x.bbox and (not hasattr(x,"page_id") or x.page_id==page_id)]
        if not page: raise ValueError("Selected page has no mapped image patches")
        width=max(x.bbox[2] for x in page); height=max(x.bbox[3] for x in page)
        spec["region_targets"]=[x.index for x in page if a<=(x.bbox[0]+x.bbox[2])/2/width<=c and b<=(x.bbox[1]+x.bbox[3])/2/height<=d]
        if not spec["region_targets"]: raise ValueError("Region contains no patch centres")
    validate_plan(spec,positions,mask,causal)
    return spec

@contextmanager
def apply(model,torch,spec,positions,kernel="eager"):
    """kernel: "eager" wraps each modeling module's stock eager function; encoding_capture.FLOAT32_EAGER
    wraps the registered FP32-arithmetic kernel that ColQwen2 scores and extracts with."""
    if not spec:
        yield
        return
    language=model.vlm.model.language_model if hasattr(model,"vlm") else model.model.language_model
    targets={id(language.layers[layer].self_attn):layer for layer in spec["layers"]}
    if kernel=="eager":
        modules={import_module(type(layer.self_attn).__module__) for layer in language.layers}
        originals={module:module.eager_attention_forward for module in modules}
    else:
        from transformers import AttentionInterface
        from .retrieval.encoding_capture import FLOAT32_EAGER,eager_float32_attention,register_float32_eager
        if kernel!=FLOAT32_EAGER: raise ValueError("Unknown intervention attention kernel")
        register_float32_eager()
    implementation=language.config._attn_implementation
    def wrapper(original):
        def eager(attention,query,key,value,attention_mask,*args,**kwargs):
            if id(attention) not in targets:
                return original(attention,query,key,value,attention_mask,*args,**kwargs)
            if spec["method_id"] in ("attention.head_ablation","attention.layer_ablation"):
                output,probabilities=original(attention,query,key,value,attention_mask,*args,**kwargs)
                if output.ndim!=4 or output.shape[2]!=query.shape[1]:
                    raise ValueError("Attention head-output axes differ from the pinned eager adapter")
                output=output.clone(); output[:,:,spec["heads"],:]=0
                return output,probabilities
            q,k=query.shape[-2],key.shape[-2]
            removed=removed_edges(spec,positions,range(k-q,k),k)
            mask=(torch.zeros((query.shape[0],1,q,k),device=query.device,dtype=query.dtype) if attention_mask is None
                  else attention_mask[:,:,:q,:k])
            mask=mask.expand(query.shape[0],query.shape[1],q,k).clone()
            drop=torch.zeros_like(mask,dtype=torch.bool)
            drop[:,spec["heads"],:,:]=torch.as_tensor(removed,device=query.device)[None,None,:,:]
            valid=torch.isfinite(mask)&(mask>torch.finfo(mask.dtype).min/2)
            if not torch.all((valid&~drop).any(-1)):
                raise ValueError("Intervention removes every valid native key from a source row")
            mask.masked_fill_(drop,float("-inf"))
            return original(attention,query,key,value,mask,*args,**kwargs)
        return eager
    if kernel!="eager":
        try:
            AttentionInterface.register(kernel,wrapper(eager_float32_attention))
            language.set_attn_implementation(kernel)
            yield
        finally:
            AttentionInterface.register(kernel,eager_float32_attention)
            language.set_attn_implementation(implementation)
        return
    try:
        language.set_attn_implementation("eager")
        for module,original in originals.items(): module.eager_attention_forward=wrapper(original)
        yield
    finally:
        for module,original in originals.items(): module.eager_attention_forward=original
        language.set_attn_implementation(implementation)
