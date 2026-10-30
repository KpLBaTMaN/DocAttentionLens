"""Small random-weight adapter fixtures; no downloaded model or scientific result."""
from importlib import import_module
from types import SimpleNamespace as NS
import torch
from docattentionlens.generation.steering import apply_steering,biased_probabilities

profiles=[('qwen2_5_vl','Qwen2_5_VLAttention','Qwen2_5_VLTextConfig'),('qwen2_vl','Qwen2VLAttention','Qwen2VLTextConfig'),('qwen3_vl','Qwen3VLTextAttention','Qwen3VLTextConfig')]
for name,attention_name,config_name in profiles:
    module=import_module('transformers.models.'+name+'.modeling_'+name)
    config_module=import_module('transformers.models.'+name+'.configuration_'+name)
    config=getattr(config_module,config_name)(hidden_size=16,num_attention_heads=4,num_key_value_heads=2,head_dim=4)
    attention=getattr(module,attention_name)(config,layer_idx=0).eval()
    model=NS(model=NS(language_model=NS(layers=[NS(self_attn=attention)])))
    torch.manual_seed(13)
    q=torch.randn(1,4,5,4);k=torch.randn(1,2,5,4);v=torch.randn(1,2,5,4)
    mask=torch.zeros(1,1,5,5).masked_fill(~torch.ones(5,5,dtype=torch.bool).tril(),float('-inf'))
    original=module.eager_attention_forward
    baseline=original(attention,q,k,v,mask,.5)
    positions=[NS(role='image',page_id='page',bbox=[i,0,i+1,1],ownership={'element_a':1} if i==0 else {'background':1}) for i in range(5)]
    spec={'layer':0,'head':1,'strength':0,'element_ids':['element_a']}
    with apply_steering(model,torch,spec,positions):
        identity=module.eager_attention_forward(attention,q,k,v,mask,.5)
    assert all(torch.equal(a,b) for a,b in zip(baseline,identity))
    with apply_steering(model,torch,{**spec,'strength':1},positions):
        changed=module.eager_attention_forward(attention,q,k,v,mask,.5)
    expected=biased_probabilities(baseline[1][0,1,-1].numpy(),[1,0,0,0,0],1)
    assert torch.allclose(changed[1][0,1,-1],torch.tensor(expected,dtype=torch.float32),atol=1e-7,rtol=0)
    assert torch.equal(changed[1][:,[0,2,3]],baseline[1][:,[0,2,3]])
    assert torch.all(changed[1].masked_select(mask==float('-inf'))==0)
    assert module.eager_attention_forward is original
    # A fractional ownership bias is smaller than the BF16 spacing at these
    # equal QK logits. It must survive in FP32 before probabilities are cast.
    low_q=torch.full_like(q,4,dtype=torch.bfloat16)
    low_k=torch.full_like(k,4,dtype=torch.bfloat16)
    low_v=v.to(torch.bfloat16); low_mask=mask.to(torch.bfloat16)
    low_baseline=original(attention,low_q,low_k,low_v,low_mask,.5)
    fractional=[NS(role='image',page_id='page',bbox=[i,0,i+1,1],ownership={'element_a':.01} if i==0 else {'background':1}) for i in range(5)]
    with apply_steering(model,torch,{**spec,'strength':1},fractional):
        low_changed=module.eager_attention_forward(attention,low_q,low_k,low_v,low_mask,.5)
    low_expected=torch.tensor(biased_probabilities([.2]*5,[.01,0,0,0,0],1),dtype=torch.bfloat16)
    assert torch.equal(low_changed[1][0,1,-1],low_expected)
    assert not torch.equal(low_changed[1][0,1,-1],low_baseline[1][0,1,-1])
    assert torch.equal(low_changed[1][:,[0,2,3]],low_baseline[1][:,[0,2,3]])
    try:
        with apply_steering(model,torch,{**spec,'strength':1},positions):raise RuntimeError('fixture interruption')
    except RuntimeError:pass
    assert module.eager_attention_forward is original
    print(name,'zero identity, log-odds bias, fractional BF16 precision, unchanged heads, causal mask and restoration passed',flush=True)
