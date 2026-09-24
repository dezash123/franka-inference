"""Shared official OpenPI transforms and memory-efficient checkpoint loading."""
import dataclasses
import numpy as np
from contract import ROOT,settings

def train_config():
    from openpi.training.config import get_config
    c=get_config(settings()['policy']['config_name'])
    return dataclasses.replace(c,model=dataclasses.replace(c.model,pytorch_compile_mode=None))

def transforms():
    from openpi import transforms as t
    from openpi.training import checkpoints
    c=train_config();d=c.data.create(c.assets_dirs,c.model)
    n=checkpoints.load_norm_stats(ROOT.parent/'checkpoints/pi05_droid_pytorch/assets',d.asset_id)
    return (t.compose([t.InjectDefaultPrompt(None),*d.data_transforms.inputs,
                       t.Normalize(n,use_quantiles=d.use_quantile_norm),*d.model_transforms.inputs]),
            t.compose([*d.model_transforms.outputs,t.Unnormalize(n,use_quantiles=d.use_quantile_norm),
                       *d.data_transforms.outputs]))

def load_observation(path):
    with np.load(path,allow_pickle=False) as f:obs={k:f[k] for k in f.files}
    obs['prompt']=str(obs['prompt'].item())
    return obs

def load_model(stage='all'):
    import torch
    from safetensors import safe_open
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from transformers.models.gemma.modeling_gemma import GemmaRotaryEmbedding
    with torch.device('meta'):m=PI0Pytorch(train_config().model)
    prefixes={
        'vision':('paligemma_with_expert.paligemma.model.vision_tower.',
                  'paligemma_with_expert.paligemma.model.multi_modal_projector.'),
        'prefix':('paligemma_with_expert.paligemma.model.language_model.',
                  'paligemma_with_expert.paligemma.lm_head.weight'),
        'denoise':('paligemma_with_expert.gemma_expert.model.','action_','time_')}
    state={}
    with safe_open(settings()['policy']['checkpoint']+'/model.safetensors',framework='pt',device='cpu') as f:
        for k in f.keys():
            if stage!='all' and not k.startswith(prefixes[stage]):continue
            if 'gemma_expert.lm_head' in k:continue
            value=f.get_tensor(k)
            state[k]=value if stage=='all' else value.float()
    embed='paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'
    head='paligemma_with_expert.paligemma.lm_head.weight'
    if head in state:state[embed]=state[head]
    m.load_state_dict(state,assign=True,strict=False)
    m.paligemma_with_expert.paligemma.lm_head=None
    m.paligemma_with_expert.gemma_expert.lm_head=None
    e=m.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings
    e.position_ids=torch.arange(e.num_positions).expand((1,-1))
    for lm in (m.paligemma_with_expert.paligemma.language_model,m.paligemma_with_expert.gemma_expert.model):
        lm.rotary_emb=GemmaRotaryEmbedding(lm.config,device='cpu')
    if stage=='all':
        m.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
        for n in ('action_in_proj','action_out_proj','time_mlp_in','time_mlp_out'):getattr(m,n).float()
    m.eval()
    return m

def prepared(obs,input_transform):
    import jax
    data=input_transform(jax.tree.map(lambda x:x.copy() if isinstance(x,np.ndarray) else x,obs))
    keys=('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')
    images=np.stack([data['image'][k] for k in keys]).astype(np.float32)/255.*2.-1.
    images=images.transpose(0,3,1,2).copy()
    mask=np.concatenate([np.repeat(bool(data['image_mask'][k]),256) for k in keys]+[data['tokenized_prompt_mask']])[None]
    return data,images,np.asarray(data['tokenized_prompt'],np.int64)[None],mask
