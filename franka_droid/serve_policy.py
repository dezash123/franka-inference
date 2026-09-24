#!/usr/bin/env python3
"""Manually started local inference service; no robot command interfaces."""
import argparse,json,logging,os,signal
import numpy as np
from contract import ROOT,settings,validate_observation
from runtime_contract import ABSOLUTE,RUNTIMES

def main():
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--backend',choices=list(RUNTIMES));a=p.parse_args()
    if not a.start:p.error('Explicit --start required')
    def shutdown(*_):raise SystemExit(0)
    signal.signal(signal.SIGTERM,shutdown)
    cfg=settings()['policy'];backend=a.backend or cfg['backend']
    if backend=='openvino':
        from ov_policy import OpenVINOPolicy
        policy=OpenVINOPolicy()
    elif backend=='ssog_mc2':
        from ssog_policy import SSOGPolicy
        policy=SSOGPolicy()
    elif backend=='ov_a_w8a8':
        from ov_a_policy import OvAPolicy
        policy=OvAPolicy()
    else:
        from torch_policy import TorchPolicy
        policy=TorchPolicy(compile_model=backend=='torch_compile')
    metadata={'config':'pi05_droid_jointpos' if RUNTIMES[backend]['action_space']==ABSOLUTE else 'pi05_droid','backend':backend,
              'precision':RUNTIMES[backend]['precision'],'action_space':RUNTIMES[backend]['action_space'],
              'device':policy.device_name,'checkpoint':getattr(policy,'checkpoint',cfg['checkpoint']),
              'action_horizon':15,'action_dimensions':8,'denoise_steps':policy.steps,'robot_commands_enabled':False,'pid':os.getpid()}
    class ValidatedPolicy:
        def infer(self,obs):
            validate_observation(obs);result=policy.infer(obs)
            if np.asarray(result['actions']).shape!=(15,8) or not np.isfinite(result['actions']).all():raise ValueError('Invalid model output')
            record={**metadata,'runtime':policy.runtime}
            dest=ROOT/'evidence/policy-runtime.json';tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));os.replace(tmp,dest)
            return result
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    logging.basicConfig(level=logging.INFO)
    try:WebsocketPolicyServer(ValidatedPolicy(),host=cfg['host'],port=cfg['port'],metadata=metadata).serve_forever()
    finally:
        if hasattr(policy,'close'):policy.close()
if __name__=='__main__':main()
