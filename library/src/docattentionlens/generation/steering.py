"""Scoped eager-attention override for pinned decoder adapters; always restored on failure."""
from contextlib import contextmanager
from importlib import import_module
import numpy as np


def biased_probabilities(probabilities,ownership,strength):
    """Independent numerical oracle for p'_j proportional to p_j exp(alpha*w_j)."""
    p=np.asarray(probabilities,dtype=float); w=np.asarray(ownership,dtype=float)
    if p.shape!=w.shape or not np.isfinite(p).all() or not np.isfinite(w).all() or np.any(p<0) or np.any(w<0) or np.any(w>1) or not np.isfinite(strength):
        raise ValueError("Invalid steering probability or ownership support")
    exponent=strength*w; exponent-=exponent.max()
    values=p*np.exp(exponent)
    if not values.sum(): raise ValueError("Steering requires positive valid probability mass")
    return values/values.sum()


@contextmanager
def apply_steering(model,torch,spec,positions):
    if not spec or spec["strength"]==0:
        yield; return
    layer,head,strength=spec["layer"],spec["head"],spec["strength"]
    target=model.model.language_model.layers[layer].self_attn
    module=import_module(type(target).__module__)
    original=module.eager_attention_forward
    weights=[]
    for p in positions:
        fraction=0.0
        if p.role=="image":
            if spec.get("control_box") and p.page_id==spec["page_id"] and p.bbox:
                x1,y1,x2,y2=p.bbox; a,b,c,d=spec["control_box"]
                fraction=max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))/((x2-x1)*(y2-y1))
            else: fraction=sum(p.ownership.get(i,0) for i in spec["element_ids"])
        if not 0<=fraction<=1+1e-8: raise ValueError("Intervention ownership is outside the probability domain")
        weights.append(min(1,fraction))
    if not any(weights): raise ValueError("Intervention has no mapped image targets")
    vector=torch.tensor(weights,device=next(target.parameters()).device,dtype=torch.float32)
    def eager(attention,query,key,value,attention_mask,scaling,dropout=0.0,**kwargs):
        if attention is not target: return original(attention,query,key,value,attention_mask,scaling,dropout,**kwargs)
        keys=module.repeat_kv(key,attention.num_key_value_groups)
        values=module.repeat_kv(value,attention.num_key_value_groups)
        logits=torch.matmul(query,keys.transpose(2,3))*scaling
        if attention_mask is not None: logits=logits+attention_mask[:,:,:,:keys.shape[-2]]
        # Native QK logits may be BF16. Fractional ownership biases must not be
        # rounded away before the existing FP32 softmax. Zero strength bypasses
        # this override entirely and keeps exact baseline kernel behaviour.
        logits=logits.float()
        bias=torch.zeros(keys.shape[-2],device=query.device,dtype=logits.dtype)
        bias[:len(vector)]=vector.to(logits.dtype)*strength
        logits[:,head,:,:]=logits[:,head,:,:]+bias
        probabilities=torch.nn.functional.softmax(logits,dim=-1,dtype=torch.float32).to(query.dtype)
        probabilities=torch.nn.functional.dropout(probabilities,p=dropout,training=attention.training)
        output=torch.matmul(probabilities,values).transpose(1,2).contiguous()
        return output,probabilities
    module.eager_attention_forward=eager
    try: yield
    finally: module.eager_attention_forward=original
