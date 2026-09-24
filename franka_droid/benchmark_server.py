#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--runs',type=int,default=30);a=p.parse_args()
root=Path(__file__).resolve().parent
files=sorted((root/'evidence').glob('fci-policy-*-observation-*.npz'))[-10:]
observations=[]
for path in files:
    with np.load(path,allow_pickle=False) as f:obs={k:f[k] for k in f.files}
    obs['prompt']=str(obs['prompt'].item());observations.append(obs)
client=WebsocketClientPolicy(host='127.0.0.1',port=8000)
metadata=client.get_server_metadata()
times=[];model_times=[]
for i in range(a.runs+3):
    start=time.perf_counter();result=client.infer(observations[i%len(observations)]);dt=time.perf_counter()-start
    action=np.asarray(result['actions'])
    assert action.shape==(15,8) and np.isfinite(action).all()
    if i>=3:times.append(dt*1000);model_times.append(result.get('policy_timing',{}).get('infer_ms'))
report={'runs':a.runs,'warmup':3,'metadata':metadata,'observations':[str(p) for p in files],
        'boundary':f'localhost websocket request through complete 15x8 action response, includes preprocessing, {metadata.get("denoise_steps","configured")} denoise steps and postprocessing; excludes image capture and robot control',
        'latency_ms':{'median':float(np.median(times)),'p95':float(np.percentile(times,95)),'min':min(times),'max':max(times),'samples':times},
        'reported_model_ms':model_times}
Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
