#!/usr/bin/env python3
"""Read-only live-camera warm-up. Does not activate or command robot/gripper."""
import json,os,subprocess,time
import numpy as np
from cameras import ZedSource,wrist_source,LatestCamera
from camera_preview import CameraPreview
from contract import ROOT,settings,make_observation
from gripper import GripperReader
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from runtime_contract import validate_runtime

class WarmupClient(WebsocketClientPolicy):
    def _wait_for_server(self):
        import websockets.sync.client
        from openpi_client import msgpack_numpy
        while True:
            try:
                conn=websockets.sync.client.connect(self._uri,compression=None,max_size=None,ping_interval=None)
                return conn,msgpack_numpy.unpackb(conn.recv())
            except ConnectionRefusedError:time.sleep(2)

def main():
    cfg=settings();streams=[];reader=None;preview=None
    report={'backend':cfg['policy']['backend'],'live_cameras':True,'robot_commands_sent':False,'started_unix':time.time()}
    dest=ROOT/'evidence/policy-readiness.json'
    try:
        state=json.loads(subprocess.check_output([str(ROOT/'read_fci_snapshot')],text=True,timeout=10))
        reader=GripperReader(cfg['robot']['gripper']);gripper=reader.read()
        client=WarmupClient(host=cfg['policy']['host'],port=cfg['policy']['port'])
        metadata=client.get_server_metadata();report['metadata']=metadata
        validate_runtime(metadata,report['backend'])
        for c in cfg['external_cameras']:streams.append(LatestCamera(ZedSource({**c,'eye':'left'})))
        streams.append(LatestCamera(wrist_source(cfg['wrist_camera'])))
        preview=CameraPreview(streams,cfg)
        timings=[]
        for i in range(8):
            deadline=time.monotonic()+10
            while True:
                try:
                    frames=[s.get(.25) for s in streams]
                    if any(abs(f[1]-frames[-1][1])>.1 for f in frames[:-1]):raise RuntimeError('Camera skew over 100 ms')
                    break
                except RuntimeError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.02)
            external=i%(len(streams)-1)
            obs=make_observation(frames[external][0],frames[-1][0],state['q'],gripper['closure'],cfg['task']['prompt'])
            obs['_session_id']='warmup-'+str(report['started_unix']);obs['_external_view']=cfg['external_cameras'][external]['serial']+':left'
            t=time.monotonic();out=client.infer(obs);dt=time.monotonic()-t
            if np.asarray(out['actions']).shape!=(15,8) or not np.isfinite(out['actions']).all():raise RuntimeError('Invalid model output')
            timings.append(dt*1000);print(json.dumps({'warmup':i+1,'inference_ms':dt*1000}),flush=True)
        report.update(inference_ms=timings,ready=max(timings[-5:])<=250,finished_unix=time.time())
        if not report['ready']:report['error']='Runtime loaded, but live inference exceeds the existing 250 ms motion deadline.'
    except Exception as e:report.update(ready=False,error=str(e),finished_unix=time.time());raise
    finally:
        if preview:preview.close()
        if reader:reader.close()
        for stream in reversed(streams):stream.close()
        tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));os.replace(tmp,dest)
        (ROOT/'evidence'/f"policy-readiness-{report['backend']}-{time.time_ns()}.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
if __name__=='__main__':main()
