#!/usr/bin/env python3
"""Fresh camera and read-only FCI/gripper observation; no robot commands."""
import json,subprocess,time
import cv2
import numpy as np
from cameras import ZedSource,RealSenseSource,LatestCamera
from gripper import GripperReader
from contract import ROOT,settings,make_observation
from openpi_client.websocket_client_policy import WebsocketClientPolicy
cfg=settings();prefix=ROOT/'evidence'/f'raw-inference-{time.time_ns()}'
streams=[];gripper=None
try:
    streams=[LatestCamera(ZedSource(cfg['external_camera'])),LatestCamera(RealSenseSource(cfg['wrist_camera']))]
    gripper=GripperReader(cfg['robot']['gripper'])
    deadline=time.monotonic()+10
    while True:
        try:
            frames=[c.get(.25) for c in streams]
            if abs(frames[0][1]-frames[1][1])>.1:raise RuntimeError('Camera pair skew')
            gs=gripper.read()
            state=json.loads(subprocess.check_output([str(ROOT/'read_fci_snapshot')],text=True))
            if state['has_errors'] or not state['idle']:raise RuntimeError('Robot not idle and error-free')
            now=time.monotonic()
            if max(now-f[1] for f in frames)>.25:raise RuntimeError('Observation stale')
            break
        except RuntimeError:
            if time.monotonic()>deadline:raise
            time.sleep(.02)
    obs=make_observation(frames[0][0],frames[1][0],state['q'],gs['closure'],'pick up the marker and place it in the mug')
    np.savez_compressed(prefix.with_suffix('.observation.npz'),**obs)
    for name,frame in zip(('external','wrist'),frames):
        cv2.imwrite(str(prefix.with_suffix('.'+name+'.jpg')),cv2.cvtColor(frame[0],cv2.COLOR_RGB2BGR))
    client=WebsocketClientPolicy(host=cfg['policy']['host'],port=cfg['policy']['port'])
    metadata=client.get_server_metadata()
    if metadata.get('backend')!='openvino' or metadata.get('precision')!='FP16' or metadata.get('denoise_steps')!=5:
        raise RuntimeError('Wrong inference backend')
    latencies=[];outputs=[]
    for i in range(33):
        t=time.perf_counter();result=client.infer(obs);ms=(time.perf_counter()-t)*1000
        actions=np.asarray(result['actions'])
        if actions.shape!=(15,8) or not np.isfinite(actions).all():raise RuntimeError('Invalid model output')
        if i>=3:latencies.append(ms);outputs.append(actions.copy())
    raw=np.stack(outputs);np.save(prefix.with_suffix('.raw-actions.npy'),raw)
    report={'metadata':metadata,'robot_commands_sent':False,'motion_limits_changed':False,'motion_layer_scaling_applied':False,
            'input':'one fresh camera/FCI/gripper observation, repeated for timing','prompt':obs['prompt'],
            'observation':str(prefix.with_suffix('.observation.npz')),'state':state,'gripper':gs,
            'warmups':3,'runs':30,'latency_ms':{'median':float(np.median(latencies)),'p95':float(np.percentile(latencies,95)),
            'min':min(latencies),'max':max(latencies),'samples':latencies},
            'action_shape':list(raw.shape),'maximum_raw_joint_action_absolute':float(np.max(np.abs(raw[:,:,:7]))),
            'maximum_droid_joint_increment_rad_before_adapter_scaling':float(np.max(np.abs(.2*raw[:,:,:7]))),
            'first_raw_action_chunk':raw[0].tolist(),
            'boundary':'localhost request through raw 15x8 action response; includes preprocessing, five denoise steps and postprocessing; excludes capture and robot motion'}
    prefix.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'report':str(prefix.with_suffix('.json')),'latency_ms':report['latency_ms'],
                      'max_raw_delta_rad':report['maximum_droid_joint_increment_rad_before_adapter_scaling'],
                      'robot_commands_sent':False}),flush=True)
finally:
    if gripper:gripper.close()
    for c in reversed(streams):c.close()
