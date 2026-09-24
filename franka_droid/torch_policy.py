"""FP16 Torch eager / Inductor execution of the same three DROID neural stages."""
import collections,math,time
import numpy as np
import torch
from contract import settings,validate_observation
from ov_common import transforms,prepared,load_model
from export_openvino import Vision,Prefix,Denoise

class CompactPrefix(Prefix):
    def forward(self,vision,tokens,pad):
        emb=torch.cat([vision.reshape(1,-1,2048),self.lm.embed_tokens(tokens)*math.sqrt(2048)],dim=1)
        mask=torch.where(pad[:,None,:,None]&pad[:,None,None,:],0.,-10000.)
        out=self.lm(inputs_embeds=emb,attention_mask=mask,position_ids=pad.long().cumsum(1)-1,use_cache=True)
        cache=out.past_key_values.to_legacy_cache()
        return torch.cat([x[0] for x in cache],dim=0),torch.cat([x[1] for x in cache],dim=0)

class CompactDenoise(Denoise):
    def forward(self,noise,timestep,pad,keys,values):
        from transformers.cache_utils import DynamicCache
        nn=torch.nn
        time=timestep[:,None]*self.freq[None]
        cond=nn.functional.silu(self.time_out(nn.functional.silu(self.time_in(torch.cat([torch.sin(time),torch.cos(time)],dim=-1)))))
        suffix=self.action_in(noise)
        valid=torch.cat([pad[:,None,:].expand(1,15,pad.shape[-1]),torch.ones(1,15,15,dtype=torch.bool,device=noise.device)],dim=2)
        mask=torch.where(valid[:,None],0.,-10000.)
        pos=pad.long().sum(-1)[:,None]+torch.arange(15,device=noise.device)[None]
        cache=DynamicCache.from_legacy_cache(tuple((keys[i:i+1],values[i:i+1]) for i in range(18)))
        out=self.lm(inputs_embeds=suffix,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=False,adarms_cond=cond)
        return self.action_out(out.last_hidden_state.float())

class TorchPolicy:
    def __init__(self,compile_model=False):
        if not torch.xpu.is_available():raise RuntimeError('Intel XPU is unavailable')
        torch.set_num_threads(6 if compile_model else 1)
        self.device_name=torch.xpu.get_device_name(0)
        if 'B580' not in self.device_name:raise RuntimeError('Expected Intel Arc B580')
        self.steps=int(settings()['policy']['denoise_steps'])
        self.input_transform,self.output_transform=transforms()
        self.compiled=compile_model
        # Load the identical checkpoint; half weights and FP16 autocast for neural
        # stages, FP32 Euler integration and output normalization, as in OV.
        model=load_model()
        self.modules=[Vision(model),CompactPrefix(model),CompactDenoise(model)]
        self.live_equivalence_checked=False
        for module in self.modules:
            for parameter in module.parameters():parameter.data=parameter.data.to(dtype=torch.float16)
            module.eval().to(device='xpu')
        self.weight_dtypes=dict(collections.Counter(str(p.dtype) for m in self.modules for p in m.parameters()))
        del model
        self.stages=[torch.compile(m,backend='inductor',dynamic=False,fullgraph=False) for m in self.modules] if compile_model else self.modules
        self.runtime={'torch':torch.__version__,'device':self.device_name,'weight_dtypes':self.weight_dtypes,
                      'cpu_threads':torch.get_num_threads(),'autocast':'torch.float16','integration':'torch.float32','compile_backend':'inductor' if compile_model else None,
                      'compiled_stages':['vision','prefix','denoise'] if compile_model else [],
                      'masked_unused_camera_skipped':True,'prefix_tokens':712}
        print('TORCH_RUNTIME',self.runtime,flush=True)
    def forward_stages(self,stages,images,tokens,pad,noise):
        with torch.inference_mode(),torch.autocast(device_type='xpu',dtype=torch.float16):
            if not self.live_equivalence_checked:print("FIRST_EXECUTION vision",tuple(images.shape),flush=True)
            vision=stages[0](images)
            if not self.live_equivalence_checked:print("FIRST_EXECUTION prefix",tuple(pad.shape),flush=True)
            keys,values=stages[1](vision,tokens,pad)
            if not self.live_equivalence_checked:print("FIRST_EXECUTION denoise",flush=True)
            for i in range(self.steps):
                velocity=stages[2](noise,torch.tensor([1.-i/self.steps],device='xpu'),pad,keys,values)
                noise=noise-velocity.float()/self.steps
        return noise
    def infer(self,obs):
        validate_observation(obs);begin=time.perf_counter()
        data,images,tokens,pad=prepared(obs,self.input_transform)
        valid_tokens=np.flatnonzero(pad[0,768:])
        token_count=tokens.shape[-1] if self.compiled else max(1,int(valid_tokens[-1])+1 if len(valid_tokens) else 1)
        self.runtime['prefix_tokens']=512+token_count
        images=torch.from_numpy(images).to('xpu');tokens=torch.from_numpy(tokens).to('xpu');pad=torch.from_numpy(pad).to('xpu')
        noise=torch.randn(1,15,32,device='xpu',dtype=torch.float32)
        torch.xpu.synchronize();t0=time.perf_counter()
        if torch.any(pad[:,512:768]):raise RuntimeError('Expected the unused third camera to be masked')
        reference=None
        if not self.live_equivalence_checked:
            reference=self.forward_stages(self.modules,images,tokens,pad,noise.clone())
        compact_pad=torch.cat([pad[:,:512],pad[:,768:768+token_count]],dim=1)
        noise=self.forward_stages(self.stages,images[:2],tokens[:,:token_count],compact_pad,noise)
        if reference is not None:
            difference=float((noise-reference).abs().max().item())
            self.runtime['live_masked_slot_equivalence_max_abs']=difference
            if difference>.01:raise RuntimeError(f'Masked camera compaction changed normalized output by {difference}')
            self.live_equivalence_checked=True
        actions=noise[0].float().cpu().numpy();t1=time.perf_counter()
        result=self.output_transform({'state':data['state'],'actions':actions})
        if np.asarray(result['actions']).shape!=(15,8) or not np.isfinite(result['actions']).all():
            raise RuntimeError('Torch produced invalid actions')
        if self.compiled:
            from torch._dynamo.utils import counters
            self.runtime['inductor_unique_graphs']=int(counters['stats']['unique_graphs'])
            if not self.runtime['inductor_unique_graphs']:raise RuntimeError('No Torch compiled graphs executed')
        result['policy_timing']={'infer_ms':(t1-t0)*1000,'total_ms':(time.perf_counter()-begin)*1000}
        return result
