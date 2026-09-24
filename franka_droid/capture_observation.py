#!/usr/bin/env python3
"""Read-only one-shot DROID observation from real cameras and ROS state."""
import argparse
import json
import time
import numpy as np
from cameras import ZedSource, RealSenseSource, LatestCamera
from contract import settings, make_observation, gripper_closure_from_width
from gripper import GripperReader

def main():
    p=argparse.ArgumentParser();p.add_argument('--prompt',required=True);p.add_argument('--output',required=True)
    p.add_argument('--inference-only-allow-unready-gripper',action='store_true',help='Capture measured gripper position for inference only; no activation or commands')
    args=p.parse_args();cfg=settings();grip=cfg['robot']['gripper']
    if not grip['port'] or not grip['max_width_m']:
        raise RuntimeError('Configure the real gripper status source first; no placeholder state is allowed')
    if not cfg['wrist_camera']['mount_and_orientation_verified']:
        raise RuntimeError('Verify wrist mounting and orientation first')
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    rclpy.init();node=rclpy.create_node('droid_read_only_observation');state={};streams=[];reader=None
    def receive(key,msg): state[key]=(msg,time.monotonic())
    node.create_subscription(JointState,cfg['robot']['joint_state_topic'],lambda msg:receive('arm',msg),qos_profile_sensor_data)
    try:
        reader=GripperReader(grip)
        streams=[LatestCamera(ZedSource(cfg['external_camera']))]
        streams.append(LatestCamera(RealSenseSource(cfg['wrist_camera'])))
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.02)
            if 'arm' not in state: continue
            try: frames=[c.get(cfg['freshness']['maximum_age_s']) for c in streams]
            except RuntimeError: continue
            now=time.monotonic()
            if any(now-t>cfg['freshness']['maximum_age_s'] for _,t in state.values()):continue
            if abs(frames[0][1]-frames[1][1])>cfg['freshness']['maximum_pair_skew_s']:continue
            arm=state['arm'][0]
            stamp=arm.header.stamp.sec+arm.header.stamp.nanosec/1e9
            if not 0<=time.time()-stamp<=cfg['freshness']['maximum_age_s']:
                raise RuntimeError('ROS source timestamp is stale or invalid')
            q=dict(zip(arm.name,arm.position))
            positions=[q[name] for name in cfg['robot']['joints']]
            gs=reader.read()
            gripper_ready=not gs['fault'] and gs['activation']==1 and gs['activation_status']==3
            if not gripper_ready and not args.inference_only_allow_unready_gripper:
                raise RuntimeError('Gripper is not ready; this reader will not activate it')
            closure=gs['closure']
            now=time.monotonic()
            if max(now-frames[0][1],now-frames[1][1],now-state['arm'][1])>cfg['freshness']['maximum_age_s']:
                continue
            obs=make_observation(frames[0][0],frames[1][0],positions,closure,args.prompt)
            np.savez_compressed(args.output,**obs)
            print(json.dumps({'saved':args.output,'inference_run':False,'robot_commands_sent':False,'camera_pair_skew_s':abs(frames[0][1]-frames[1][1]),'gripper_ready':gripper_ready,'gripper_status':gs}))
            return
        raise RuntimeError('Missing fresh, synchronized real observation')
    finally:
        for cam in reversed(streams):cam.close()
        if reader is not None:reader.close()
        node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
