#!/usr/bin/env python3
"""Export the exact DROID checkpoint as three unquantized FP16-weight IRs."""
import argparse,gc,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from contract import ROOT,settings
from ov_common import load_model,load_observation,transforms,prepared

class Vision(nn.Module):
    def __init__(self,m):
        super().__init__();p=m.paligemma_with_expert.paligemma.model
        self.vision=p.vision_tower;self.projector=p.multi_modal_projector
        self.vision.config._attn_implementation='sdpa'
        self.vision.vision_model.config._attn_implementation='sdpa'
    def forward(self,images):
        return self.projector(self.vision(pixel_values=images).last_hidden_state)

class Prefix(nn.Module):
    def __init__(self,m):
        super().__init__();self.lm=m.paligemma_with_expert.paligemma.language_model
        self.lm.config._attn_implementation='sdpa'
    def forward(self,vision,tokens,pad):
        emb=torch.cat([vision.reshape(1,768,2048),self.lm.embed_tokens(tokens)*math.sqrt(2048)],dim=1)
        mask=torch.where(pad[:,None,:,None]&pad[:,None,None,:],0.,-2.3819763e38)
        out=self.lm(inputs_embeds=emb,attention_mask=mask,position_ids=pad.long().cumsum(1)-1,use_cache=True)
        cache=out.past_key_values.to_legacy_cache()
        return torch.cat([x[0] for x in cache],dim=0),torch.cat([x[1] for x in cache],dim=0)

class Denoise(nn.Module):
    def __init__(self,m):
        super().__init__();self.lm=m.paligemma_with_expert.gemma_expert.model
        self.lm.config._attn_implementation='sdpa'
        self.action_in=m.action_in_proj;self.action_out=m.action_out_proj
        self.time_in=m.time_mlp_in;self.time_out=m.time_mlp_out
        self.register_buffer('freq',1./(.004*(4./.004)**torch.linspace(0.,1.,512))*2*math.pi)
    def forward(self,noise,timestep,pad,keys,values):
        time=timestep[:,None]*self.freq[None]
        cond=nn.functional.silu(self.time_out(nn.functional.silu(self.time_in(torch.cat([torch.sin(time),torch.cos(time)],dim=-1)))))
        suffix=self.action_in(noise)
        valid=torch.cat([pad[:,None,:].expand(1,15,968),torch.ones(1,15,15,dtype=torch.bool,device=noise.device)],dim=2)
        mask=torch.where(valid[:,None],0.,-2.3819763e38)
        pos=pad.long().sum(-1)[:,None]+torch.arange(15,device=noise.device)[None]
        cache=DynamicCache.from_legacy_cache(tuple((keys[i:i+1],values[i:i+1]) for i in range(18)))
        out=self.lm(inputs_embeds=suffix,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=False,adarms_cond=cond)
        return self.action_out(out.last_hidden_state.float())

def reference():
    from openpi.policies.policy import Policy
    from openpi.models_pytorch import pi0_pytorch
    old=pi0_pytorch.get_safe_dtype
    pi0_pytorch.get_safe_dtype=lambda dtype,device:torch.float32 if device=='xpu' and dtype==torch.float64 else old(dtype,device)
    inputs,outputs=transforms();m=load_model().to('xpu')
    policy=Policy(m,transforms=[inputs],output_transforms=[outputs],sample_kwargs={'num_steps':10},is_pytorch=True,pytorch_device='xpu')
    paths=sorted((ROOT/'evidence').glob('fci-policy-*-observation-*.npz'))
    selected=[paths[0],paths[len(paths)//2],paths[-1]]
    records=[]
    for i,path in enumerate(selected):
        obs=load_observation(path);noise=np.random.default_rng(i).standard_normal((1,15,32)).astype(np.float32)
        with torch.inference_mode():result=policy.infer(obs,noise=noise)
        out=ROOT/'evidence'/f'ov-reference-{i}.npz'
        np.savez_compressed(out,noise=noise,actions=result['actions'])
        records.append({'observation':str(path),'reference':str(out)})
    (ROOT/'evidence/ov-reference.json').write_text(json.dumps(records,indent=2))
    print('REFERENCE_COMPLETE',flush=True)

def export(stage):
    import openvino as ov
    torch.set_num_threads(6)
    dest=ROOT/'openvino_fp16';dest.mkdir(exist_ok=True)
    m=load_model(stage)
    pad=torch.ones((1,968),dtype=torch.bool);pad[:,512:768]=False;pad[:,810:]=False
    if stage=='vision':module=Vision(m);example=(torch.zeros(3,3,224,224),);names=['images'];outputs=['vision']
    elif stage=='prefix':module=Prefix(m);example=(torch.zeros(3,256,2048),torch.ones(1,200,dtype=torch.long),pad);names=['vision','tokens','pad'];outputs=['keys','values']
    else:
        module=Denoise(m);example=(torch.zeros(1,15,32),torch.ones(1),pad,torch.zeros(18,1,968,256),torch.zeros(18,1,968,256));names=['noise','timestep','pad','keys','values'];outputs=['velocity']
    module.eval();del m;gc.collect()
    started=time.monotonic();print('TRACE',stage,flush=True)
    with torch.inference_mode():traced=torch.jit.trace(module,example,check_trace=False,strict=False)
    print('CONVERT',stage,flush=True)
    model=ov.convert_model(traced,example_input=example,input=[(n,list(t.shape),np.bool_ if t.dtype==torch.bool else np.int64 if t.dtype==torch.int64 else np.float32) for n,t in zip(names,example)])
    for port,name in zip(model.inputs,names):port.get_tensor().set_names({name})
    for output,name in zip(model.outputs,outputs):output.get_tensor().set_names({name})
    ov.save_model(model,dest/f'{stage}.xml',compress_to_fp16=True)
    report={'stage':stage,'elapsed_s':time.monotonic()-started,'openvino':ov.__version__,'source_checkpoint':settings()['policy']['checkpoint'],
            'weights':'FP16; no integer quantization','inputs':[{ 'name':x.any_name,'shape':str(x.partial_shape),'dtype':str(x.element_type)} for x in model.inputs]}
    (dest/f'{stage}-export.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reference','vision','prefix','denoise']);a=p.parse_args()
    if a.stage=='reference':reference()
    else:export(a.stage)
