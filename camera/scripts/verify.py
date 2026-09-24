#!/usr/bin/env python3
"""Validate real HTTP frames and sustained frame delivery, without robot motion."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import cv2
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--url',default='http://127.0.0.1:8765')
p.add_argument('--seconds',type=int,default=30)
p.add_argument('--minimum-cameras',type=int,default=1)
p.add_argument('--require-wrist',action='store_true')
args=p.parse_args()
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out=Path(__file__).resolve().parents[1]/'evidence'/('verify-'+stamp)
out.mkdir(parents=True)
def get(path):
    with urllib.request.urlopen(args.url+path,timeout=5) as r:
        return r.read()
samples=[]
for n in range(args.seconds+1):
    samples.append(json.loads(get('/api/status')))
    if n<args.seconds:
        time.sleep(1)
first={c['id']:c for c in samples[0]['cameras']}
last=samples[-1]['cameras']
failures=[]
if len(last)<args.minimum_cameras:
    failures.append('Fewer cameras than requested')
if args.require_wrist and not any(c.get('role')=='wrist' for c in last):
    failures.append('Wrist camera video device is missing')
results=[]
for c in last:
    baseline=first.get(c['id'])
    elapsed=samples[-1]['timestamp']-samples[0]['timestamp']
    delivered=(c['frame_count']-baseline['frame_count'])/elapsed if baseline else 0
    observations=[x for s in samples for x in s['cameras'] if x['id']==c['id']]
    continuous=len(observations)==len(samples) and all(x['healthy'] for x in observations)
    if not continuous or delivered<10:
        failures.append(f"{c['model']} {c['id']}: unstable or below 10 delivered fps")
    images={}
    for eye in c.get('views',('left','right','stereo')):
        data=get(f"/snapshot.jpg?camera={c['id']}&eye={eye}")
        img=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
        expected_width=c['width'] if eye in ('stereo','color') else c['width']//2
        if img is None or img.shape != (c['height'],expected_width,3):
            failures.append(f"Invalid {eye} image for {c['id']}")
            continue
        (out/f"{c['id']}-{eye}.jpg").write_bytes(data)
        images[eye]={'shape_bgr':list(img.shape),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'mean_pixel':float(img.mean()),'std_pixel':float(img.std())}
    results.append(dict(c,measured_delivery_fps=round(delivered,3),continuous=continuous,images=images))
report={'timestamp_utc':stamp,'duration_seconds':args.seconds,'passed':not failures,'failures':failures,'cameras':results,'limitations':['ZED unrectified RGB and D405 UVC color; no SDK calibration equivalence or policy accuracy claim.','Frame delivery and image decoding do not prove absence of all USB visual corruption.']}
(out/'samples.json').write_text(json.dumps(samples,indent=2)+'\n')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
print('Evidence:',out)
raise SystemExit(bool(failures))
