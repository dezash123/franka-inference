#!/usr/bin/env python3
"""Manual inference client; saves actions without any robot command transport."""
import argparse
import json
import numpy as np
from contract import settings,validate_observation

def main():
    p=argparse.ArgumentParser();p.add_argument('--observation',required=True);p.add_argument('--output',required=True);p.add_argument('--infer',action='store_true');a=p.parse_args()
    if not a.infer:p.error('Explicit --infer is required')
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    with np.load(a.observation,allow_pickle=False) as f:obs={k:f[k] for k in f.files}
    obs['prompt']=str(obs['prompt'].item());validate_observation(obs)
    cfg=settings()['policy'];client=WebsocketClientPolicy(host=cfg['host'],port=cfg['port'])
    result=client.infer(obs);actions=np.asarray(result['actions'])
    if actions.shape!=(cfg['action_horizon'],8) or not np.isfinite(actions).all():raise ValueError('Invalid action output')
    np.save(a.output,actions)
    print(json.dumps({'actions_saved':a.output,'shape':list(actions.shape),'robot_commands_sent':False}))

if __name__=='__main__':main()
