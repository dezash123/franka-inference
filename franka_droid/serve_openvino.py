#!/usr/bin/env python3
import argparse,logging
from contract import settings
from ov_policy import OpenVINOPolicy
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');a=p.parse_args()
if not a.start:p.error('Pass --start to start inference')
logging.basicConfig(level=logging.INFO)
policy=OpenVINOPolicy();cfg=settings()['policy']
metadata={'config':'pi05_droid','backend':'openvino','precision':'FP16','device':policy.device_name,
          'action_horizon':15,'action_dimensions':8,'denoise_steps':policy.steps,'robot_commands_enabled':False}
WebsocketPolicyServer(policy,host=cfg['host'],port=cfg['port'],metadata=metadata).serve_forever()
