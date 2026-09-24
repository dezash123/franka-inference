#!/usr/bin/env python3
"""Ten-second camera-only check. Never loads weights or accesses robot commands."""
import argparse
import json
import time
import cv2
from cameras import ZedSource, RealSenseSource, LatestCamera
from contract import ROOT, settings
from openpi_client.image_tools import resize_with_pad

def main():
    p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=10);args=p.parse_args()
    cfg=settings();streams=[];report={'model_loaded':False,'robot_commands_sent':False}
    output=ROOT/'evidence';output.mkdir(exist_ok=True)
    try:
        for role,cls,key in [('external',ZedSource,'external_camera'),('wrist',RealSenseSource,'wrist_camera')]:
            camera=LatestCamera(cls(cfg[key]));streams.append((role,camera))
        # Exposure settling and startup need not satisfy inference freshness.
        time.sleep(2)
        beginning=time.monotonic();counts={k:c.count for k,c in streams}
        max_skew=0;valid_pairs=0;last=None
        while time.monotonic()-beginning<args.seconds:
            try: pair=[c.get(cfg['freshness']['maximum_age_s']) for _,c in streams]
            except RuntimeError:
                time.sleep(.05);continue
            skew=abs(pair[0][1]-pair[1][1]);max_skew=max(max_skew,skew)
            if skew<=cfg['freshness']['maximum_pair_skew_s']:
                valid_pairs+=1;last=pair
            time.sleep(1/15)
        elapsed=time.monotonic()-beginning
        report.update(elapsed_s=elapsed,valid_pairs=valid_pairs,max_observed_pair_skew_s=max_skew)
        for role,c in streams:
            fps=(c.count-counts[role])/elapsed
            report[role]={**c.source.info,'delivered_fps':fps}
            if fps<13: raise RuntimeError(f'{role} only delivered {fps:.2f} FPS')
        if last is None or valid_pairs/elapsed<12:
            raise RuntimeError('Insufficient fresh simultaneous camera pairs')
        for (role,c),(rgb,stamp,seq) in zip(streams,last):
            cv2.imwrite(str(output/(role+'_rectified.png')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            small=resize_with_pad(rgb,224,224)
            cv2.imwrite(str(output/(role+'_policy_224.png')),cv2.cvtColor(small,cv2.COLOR_RGB2BGR))
        report['passed']=True
    except BaseException as e:
        report.update(passed=False,error=str(e));raise
    finally:
        for _,camera in reversed(streams): camera.close()
        report['timestamp_unix']=time.time()
        (output/'camera_pair_check.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))

if __name__=='__main__': main()
