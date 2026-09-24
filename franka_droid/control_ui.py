#!/usr/bin/env python3
"""Tailnet-only DROID control panel, using the existing arm launcher."""
import argparse,json,math,os,secrets,signal,socket,statistics,subprocess,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs,urlencode
from urllib.request import urlopen
from runtime_contract import RUNTIMES
from live_telemetry import LiveTelemetry
from camera_crop import center_crop_bounds
from io import BytesIO
from PIL import Image
ROOT=Path(__file__).resolve().parent
CONFIG=ROOT/'config/setup.json'
BACKENDS={key:value['label'] for key,value in RUNTIMES.items()}
LOCK=threading.RLock();TOKEN=secrets.token_urlsafe(32)
STATE={'busy':False,'phase':'idle','message':'Load the selected runtime to prepare a session.','ready':False,'run':None,'robot':None}
PROCESS=None;RUN_SINCE=0.;RECEIPT=None;PAUSE_REQUESTED=False;STOP_REQUESTED=False
TELEMETRY=LiveTelemetry()
CAMERA_LOCK=threading.Lock();CAMERA_CACHE=None;CAMERA_CACHE_TIME=0.

def load(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def config():return load(CONFIG,{})
def maintenance():return load(ROOT/'evidence/policy-maintenance.json')
def write_config(value):
    tmp=CONFIG.with_suffix('.ui.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.chown(tmp,1000,1000);os.replace(tmp,CONFIG)
def userctl(*args):return subprocess.run(['runuser','-u','spring','--','env','XDG_RUNTIME_DIR=/run/user/1000','systemctl','--user',*args],capture_output=True,text=True,timeout=25)
def clients():
    found=[]
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            args=(entry/'cmdline').read_bytes().decode().split('\0')
            if Path(args[0]).name.startswith('python') and any(Path(a).name=='run_policy_stream.py' for a in args[1:] if a):found.append(int(entry.name))
        except (OSError,UnicodeError):pass
    return found

def controller_running():
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            args=(entry/'cmdline').read_bytes().split(b'\0')
            if args and Path(os.fsdecode(args[0])).name=='stream_fci':return True
        except OSError:pass
    return False

def policy_pid():
    try:return int(userctl('show','franka-droid-policy.service','-p','MainPID','--value').stdout.strip())
    except (ValueError,subprocess.SubprocessError):return 0

def usable_readiness():
    r=load(ROOT/'evidence/policy-readiness.json',{})
    return bool(r.get('ready') and r.get('backend')==config()['policy']['backend'] and r.get('metadata',{}).get('pid')==policy_pid())

def public_state():
    with LOCK:
        c=config();s=dict(STATE);s['settings']={'prompt':c['task']['prompt'],'actions':c['policy']['execution_horizon'],'backend':c['policy']['backend'],'action_hz':c['policy'].get('control_hz',15)}
        s['csrf']=TOKEN;s['server_unix']=time.time();s['deadline_s']=None;s['continuous']=True;s['denoise_steps']=RUNTIMES[c['policy']['backend']]['denoise_steps']
        m=maintenance()
        if m:s.update(busy=True,ready=False,phase='maintenance',message=m.get('message','Inference runtime maintenance in progress.'))
        return s

def runtime_worker():
    global PROCESS
    logpath=ROOT/'evidence'/f'ui-runtime-{time.time_ns()}.log'
    try:
        userctl('stop','zed-camera-viewer.service')
        with LOCK:STATE.update(phase='loading',message='Loading '+BACKENDS[config()['policy']['backend']]+'…',runtime_log=logpath.name)
        result=userctl('restart','franka-droid-policy.service')
        if result.returncode:raise RuntimeError(result.stderr.strip() or 'Could not start policy service')
        with logpath.open('w') as log:
            proc=subprocess.Popen(['runuser','-u','spring','--','/bin/bash','-c','source ./env.sh && exec venv/bin/python -u warmup_policy_live.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with LOCK:PROCESS=proc;STATE.update(phase='warming',message='Warming the runtime with live cameras. First load may take several minutes.')
            deadline=time.monotonic()+1200
            while proc.poll() is None:
                if time.monotonic()>deadline or userctl('is-failed','--quiet','franka-droid-policy.service').returncode==0:
                    os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
                    raise RuntimeError('Policy service failed or warm-up timed out; inspect the runtime log.')
                time.sleep(2)
            code=proc.returncode
        r=load(ROOT/'evidence/policy-readiness.json',{})
        with LOCK:STATE['warmup']=r
        if code or not r.get('ready'):
            detail=r.get('error') if r.get('started_unix',0)>logpath.stat().st_mtime-1200 else None
            raise RuntimeError(detail or 'Runtime did not become ready; inspect runtime log.')
        with LOCK:STATE.update(ready=True,phase='idle',message=BACKENDS[r['backend']]+' ready.',warmup=r)
    except Exception as e:
        with LOCK:STATE.update(ready=False,phase='error',message=str(e))
    finally:
        PROCESS=None;userctl('start','zed-camera-viewer.service')
        with LOCK:STATE['busy']=False

def configure(data):
    if maintenance():raise ValueError('Inference maintenance is in progress; please wait')
    prompt=data.get('prompt');steps=data.get('actions');backend=data.get('backend');action_hz=data.get('action_hz',config()['policy'].get('control_hz',15))
    if not isinstance(prompt,str) or not 1<=len(prompt.strip())<=1000:raise ValueError('Prompt must contain 1–1000 characters')
    if type(steps) is not int or not 1<=steps<=15:raise ValueError('Actions must be an integer from 1 to 15')
    if backend not in BACKENDS:raise ValueError('Unknown runtime')
    if type(action_hz) not in (int,float) or not math.isfinite(action_hz) or action_hz<=0:raise ValueError('Action playback frequency must be a finite number greater than 0 Hz')
    with LOCK:
        if STATE['busy'] or clients() or controller_running():raise ValueError('Stop the current session before editing settings')
        c=config();changed=c['policy']['backend']!=backend or (backend=='torch_eager' and c['task']['prompt']!=prompt.strip());c['task']['prompt']=prompt.strip();c['policy']['execution_horizon']=steps;c['policy']['backend']=backend
        c['policy']['control_hz']=float(action_hz)
        c['policy']['denoise_steps']=RUNTIMES[backend]['denoise_steps'];c['policy']['precision']=RUNTIMES[backend]['precision']
        write_config(c)
        if changed or not STATE['ready']:
            STATE.update(busy=True,ready=False,phase='loading',message='Loading selected runtime…');threading.Thread(target=runtime_worker,daemon=True).start()
        else:STATE.update(message='Settings saved. Ready to run until paused or stopped.',phase='idle')

def failure_detail(receipt):
    failure=receipt.get('controller_error') or receipt.get('failure')
    if failure not in ('Streaming FCI stopped; see controller stderr','FCI process exited','Controller did not become ready','FCI state timestamp stale','FCI feedback stale','Controller did not stop cleanly') or not RECEIPT:
        return failure
    try:
        with RECEIPT.with_suffix('.stderr.log').open('rb') as log:
            log.seek(0,2);size=log.tell();log.seek(max(0,size-8192))
            lines=log.read().decode(errors='replace').splitlines()
        details=[line.split('STREAM_FCI_STOPPED: ',1)[1] for line in lines if line.startswith('STREAM_FCI_STOPPED: ')]
        if not details:return failure
        detail=details[-1]
        prefix='DROID joint position limit at joint '
        if detail.startswith(prefix):
            joint=int(detail[len(prefix):])-1
            limits=receipt['constraint_profile']['limits']
            detail+=f". Allowed range: {limits['joint_pos_lower'][joint]:.3f} to {limits['joint_pos_upper'][joint]:.3f} rad. Reposition with manual guiding before starting."
        return detail
    except (OSError,ValueError,KeyError,IndexError):return failure

def run_worker():
    global PROCESS,RECEIPT,RUN_SINCE,STOP_REQUESTED
    logpath=ROOT/'evidence'/f'ui-arm-{time.time_ns()}.log'
    try:
        RUN_SINCE=time.time();RECEIPT=None
        with logpath.open('w') as log:
            proc=subprocess.Popen(['/bin/bash',str(ROOT/'run-policy-stream.sh'),'--execute-confirmed','--duration','0'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with LOCK:
                PROCESS=proc;STATE['run_log']=logpath.name
                if STOP_REQUESTED:os.killpg(proc.pid,signal.SIGINT)
            proc.wait()
        time.sleep(.1);refresh_receipt()
        with LOCK:
            r=load(RECEIPT,{}) if RECEIPT else {}
            if r.get('complete'):STATE.update(phase='completed',message='Session completed.')
            elif r.get('stopped_by_user') or STOP_REQUESTED:STATE.update(phase='stopped',message='Session stopped.')
            else:STATE.update(phase='error',message=failure_detail(r) or 'Session stopped before motion began. Check the robot state and run log.')
    except Exception as e:
        with LOCK:STATE.update(phase='error',message=str(e))
    finally:
        PROCESS=None
        with LOCK:STATE['busy']=False

def start():
    global PAUSE_REQUESTED,STOP_REQUESTED
    if maintenance():raise ValueError('Inference maintenance is in progress; please wait')
    with LOCK:
        if STATE['busy'] or clients() or controller_running():raise ValueError('An operation or arm session is already active')
        if not STATE['ready'] or not usable_readiness():STATE['ready']=False;raise ValueError('Load and warm the selected runtime first')
        robot=STATE.get('robot') or {}
        if robot.get('operating_mode')!='Execution' or robot.get('brakes')!=['Unlocked']*7 or robot.get('robot_errors') or not robot.get('fci_active'):
            raise ValueError('Robot is not ready: switch to Execution, unlock brakes and clear reported errors')
        PAUSE_REQUESTED=False;STOP_REQUESTED=False
        STATE.update(busy=True,phase='starting',run=None,message='Checking robot, cameras and controller before motion…')
        threading.Thread(target=run_worker,daemon=True).start()

def pause(play=False):
    global PAUSE_REQUESTED
    with LOCK:
        if STATE['phase'] not in ('playing','pausing','paused','resuming'):raise ValueError('No running session to pause or play')
        pids=clients()
        if len(pids)!=1:raise ValueError('Arm client is not available')
        os.kill(pids[0],signal.SIGUSR2 if play else signal.SIGUSR1);PAUSE_REQUESTED=not play
        STATE.update(phase='resuming' if play else 'pausing',message='Taking a fresh observation…' if play else 'Holding the arm and gripper…')

def stop():
    global STOP_REQUESTED
    with LOCK:
        if STATE['phase'] in ('loading','warming'):raise ValueError('Runtime is loading; no arm motion is active')
        pids=clients();STOP_REQUESTED=True
        for pid in pids:os.kill(pid,signal.SIGINT)
        if not pids and PROCESS and PROCESS.poll() is None:os.killpg(PROCESS.pid,signal.SIGINT)
        if STATE['busy']:STATE.update(phase='stopping',message='Stopping the session…')

def latency_summary(receipt):
    # Read only the bounded recent receipt history, never the growing event journal.
    chunks=receipt.get('chunks',[])[-32:]
    if not chunks:return None
    def valid(value):return type(value) in (int,float) and math.isfinite(value) and value>=0
    values=[c.get('policy_timing',{}).get('infer_ms') for c in chunks if isinstance(c.get('policy_timing'),dict)]
    values=sorted(v for v in values if valid(v))
    latest=chunks[-1];timing=latest.get('policy_timing') or {}
    latest_ms=timing.get('infer_ms');round_trip=latest.get('inference_ms')
    started=receipt.get('control_started_unix');offset=latest.get('t')
    return {'backend':receipt.get('policy_metadata',{}).get('backend'),
            'sample_id':latest.get('id'),'samples':len(values),
            'latest_ms':latest_ms if valid(latest_ms) else None,
            'median_ms':statistics.median(values) if values else None,
            'p95_ms':values[math.ceil(.95*len(values))-1] if values else None,
            'round_trip_ms':round_trip if valid(round_trip) else None,
            'updated_unix':started+offset if valid(started) and valid(offset) else None}

def refresh_receipt():
    global RECEIPT
    if not RUN_SINCE:return
    if RECEIPT is None:
        paths=[p for p in (ROOT/'evidence').glob('stream-policy-*.json') if p.stat().st_mtime>=RUN_SINCE]
        if paths:RECEIPT=max(paths,key=lambda p:p.stat().st_mtime)
    if RECEIPT is None:return
    r=load(RECEIPT,{})
    if not r:return
    elapsed=r.get('actual_control_s',r.get('elapsed_s'))
    if elapsed is None and r.get('control_started_unix'):elapsed=time.time()-r['control_started_unix']
    summary={'receipt':RECEIPT.name,'latency':latency_summary(r),'elapsed_s':elapsed or 0,'remaining_s':None,'continuous':r.get('continuous',False),
             'action_frequency_hz':r.get('action_frequency_hz',15),
             'average_action_hz':r.get('action_count',len(r.get('actions',[])))/elapsed if elapsed and elapsed>1 else 0,
             'average_inference_hz':r.get('chunk_count',len(r.get('chunks',[])))/elapsed if elapsed and elapsed>1 else 0,
             'actions':r.get('action_count',len(r.get('actions',[]))),'inferences':r.get('chunk_count',len(r.get('chunks',[]))),'camera_holds':r.get('camera_holds',0),
             'joint_holds':r.get('joint_limit_holds',0),'validation_holds':r.get('validation_holds',0),'workspace_holds':r.get('cartesian_target_holds',0),
             'external_view':r.get('latest_snapshot_external_view'),'snapshot_unix':r.get('snapshot_unix'),
             'pause_events':r.get('pause_events',[]),'complete':r.get('complete',False),'failure':failure_detail(r)}
    with LOCK:
        STATE['run']=summary
        if STATE['busy'] and not STOP_REQUESTED and r.get('control_started_unix') and not r.get('finished_at_unix'):
            if PAUSE_REQUESTED:
                STATE.update(phase='paused' if r.get('paused') else 'pausing',message='Arm and gripper held. Press Play to resume.' if r.get('paused') else 'Holding the arm and gripper…')
            elif not r.get('pause_requested'):STATE.update(phase='playing',message='Executing policy actions.')

def monitor():
    last_robot=0
    while True:
        try:
            refresh_receipt()
            if time.monotonic()-last_robot>3:
                result=subprocess.run(['/usr/bin/python3','/var/lib/spring-data/franka-network-20260923/read-franka.py'],capture_output=True,text=True,timeout=5)
                with LOCK:STATE['robot']=json.loads(result.stdout) if not result.returncode else {'error':'Robot status unavailable'}
                last_robot=time.monotonic()
        except Exception:pass
        time.sleep(.5)

def camera_status():
    global CAMERA_CACHE,CAMERA_CACHE_TIME
    with CAMERA_LOCK:
        now=time.monotonic()
        if CAMERA_CACHE is not None and now-CAMERA_CACHE_TIME<.4:return CAMERA_CACHE
        c=config();rows=[];seen=set()
        for role,enabled,items in [('external',True,c['external_cameras']),
                                   ('external',False,c.get('external_cameras_disabled',[])),
                                   ('wrist',True,[c['wrist_camera']])]:
            for item in items:
                serial=str(item['serial'])
                if serial in seen:continue
                seen.add(serial)
                rows.append(dict(id=serial,serial=serial,role=role,eye=item.get('eye','left'),
                                 inference_enabled=enabled,fps=0.,healthy=False,age_seconds=None,
                                 frame_available=False,status='Waiting for capture',source=None))
        capture=load(ROOT/'evidence/camera-preview.json',{})
        age=now-capture.get('updated_monotonic',0)
        pid=capture.get('owner_pid')
        owner_alive=type(pid) is int and (Path('/proc')/str(pid)).exists()
        if owner_alive and 0<=age<3:
            for row in rows:
                data=next((x for x in capture.get('cameras',[]) if str(x.get('serial'))==row['serial']),None)
                if not data:continue
                frame_age=data.get('age_seconds')
                row.update(source='capture',fps=data.get('fps',0.),healthy=bool(data.get('healthy')),
                           age_seconds=frame_age+age if frame_age is not None else None,
                           stream_generation=str(pid),_stream_port=capture.get('stream_port'),
                           status=data.get('error') or ('Live' if data.get('healthy') else 'No fresh frames'))
                image=f'camera-preview-{pid}-{row["serial"]}.jpg'
                if data.get('image')==image and row['healthy']:
                    row.update(frame_available=True,_image=image)
        else:
            try:
                with urlopen('http://127.0.0.1:8765/api/status',timeout=1) as response:
                    viewer_status=json.load(response);viewer=viewer_status.get('cameras',[])
                for row in rows:
                    data=next((x for x in viewer if str(x.get('serial'))==row['serial']),None)
                    # Some wrist UVC nodes lack a serial; the sole wrist is unambiguous.
                    wrists=[x for x in viewer if x.get('role')=='wrist']
                    if data is None and row['role']=='wrist' and len(wrists)==1:data=wrists[0]
                    if data is None:
                        row['status']='Disconnected';continue
                    fresh=bool(data.get('healthy')) and not data.get('error')
                    row.update(source='viewer',fps=data.get('fps',0.) if fresh else 0.,healthy=fresh,
                               age_seconds=data.get('age_seconds'),frame_available=fresh,_viewer_id=data['id'],
                               stream_generation=str(viewer_status.get('pid',data['id'])),
                               status=data.get('error') or ('Live' if fresh else 'No fresh frames'))
            except (OSError,ValueError):
                for row in rows:row['status']='Waiting for camera owner'
        for row in rows:
            if not row['healthy']:row['fps']=0.
        CAMERA_CACHE={'cameras':rows,'updated_unix':time.time()};CAMERA_CACHE_TIME=time.monotonic()
        return CAMERA_CACHE

def camera_frame(camera_id):
    row=next((x for x in camera_status()['cameras'] if x['id']==camera_id),None)
    if not row or not row['frame_available']:raise ValueError('No fresh frame for this camera')
    if row['source']=='capture':return (ROOT/'evidence'/row['_image']).read_bytes()
    query=urlencode({'camera':row['_viewer_id'],'eye':row['eye']})
    with urlopen('http://127.0.0.1:8765/snapshot.jpg?'+query,timeout=2) as response:data=response.read()
    if row['role']=='wrist':
        zoom=config()['wrist_camera'].get('digital_zoom',1.0)
        if zoom!=1:
            with Image.open(BytesIO(data)) as image:
                output=BytesIO();image.crop(center_crop_bounds(image.width,image.height,zoom)).save(output,format='JPEG',quality=90);data=output.getvalue()
    return data

def frame(view):
    if view not in ('external','wrist'):raise ValueError('Unknown camera view')
    if STATE['phase'] in ('starting','playing','pausing','paused','resuming','stopping') and RECEIPT:
        for suffix in ('latest','before'):
            p=RECEIPT.with_name(RECEIPT.stem+'-'+view+'-'+suffix+'.jpg')
            if p.exists():return p.read_bytes()
        raise ValueError('Waiting for first camera observation')
    with urlopen('http://127.0.0.1:8765/api/status',timeout=2) as response:camera_status=json.load(response)
    camera=next(c for c in camera_status['cameras'] if c['role']==('wrist' if view=='wrist' else 'external'))
    eye=config()['wrist_camera'].get('eye','color') if view=='wrist' else 'left'
    with urlopen('http://127.0.0.1:8765/snapshot.jpg?camera='+camera['id']+'&eye='+eye,timeout=2) as response:data=response.read()
    if view=='wrist':
        zoom=config()['wrist_camera'].get('digital_zoom',1.0)
        if zoom!=1:
            with Image.open(BytesIO(data)) as image:
                cropped=image.crop(center_crop_bounds(image.width,image.height,zoom))
                output=BytesIO();cropped.save(output,format='JPEG',quality=90);data=output.getvalue()
    return data

class Handler(BaseHTTPRequestHandler):
    def stream_camera(self,camera_id):
        row=next((x for x in camera_status()['cameras'] if x['id']==camera_id),None)
        if not row or not row['frame_available']:return self.reply({'error':'No fresh camera stream'},code=503)
        if row['source']=='capture':
            port=row.get('_stream_port')
            if type(port) is not int or not 1<=port<=65535:return self.reply({'error':'Camera owner stream unavailable'},code=503)
            upstream=f'http://127.0.0.1:{port}/stream.mjpg?'+urlencode({'camera':camera_id})
        else:
            upstream='http://127.0.0.1:8765/stream.mjpg?'+urlencode({'camera':row['_viewer_id'],'eye':row['eye']})
        # Relay the owner's latest-frame stream with bounded socket buffers.
        # This listener never opens a camera, queues video, or executes robot work.
        try:response=urlopen(upstream,timeout=3)
        except OSError:return self.reply({'error':'Camera stream unavailable'},code=503)
        with response:
            kind=response.headers.get('Content-Type','')
            if not kind.startswith('multipart/x-mixed-replace;'):return self.reply({'error':'Invalid camera stream'},code=503)
            self.connection.settimeout(2)
            self.connection.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,131072)
            self.send_response(200);self.send_header('Content-Type',kind)
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Connection','close');self.end_headers();self.close_connection=True
            try:
                while True:
                    chunk=response.read1(65536)
                    if not chunk:break
                    self.wfile.write(chunk);self.wfile.flush()
            except (OSError,TimeoutError):pass
    def reply(self,body,kind='application/json',code=200):
        if isinstance(body,(dict,list)):body=json.dumps(body).encode()
        self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(body)
    def valid_host(self):return self.headers.get('Host','') in self.server.allowed_hosts
    def do_GET(self):
        if not self.valid_host():return self.reply({'error':'Host not allowed'},code=403)
        u=urlparse(self.path)
        try:
            if u.path=='/api/status':return self.reply(public_state())
            if u.path=='/api/cameras':
                data=camera_status()
                return self.reply({'updated_unix':data['updated_unix'],
                                   'cameras':[{k:v for k,v in row.items() if not k.startswith('_')} for row in data['cameras']]})
            if u.path=='/api/stream':return self.stream_camera(parse_qs(u.query).get('camera',[''])[0])
            if u.path=='/api/telemetry':
                with LOCK:path,phase=RECEIPT,STATE['phase']
                return self.reply(TELEMETRY.read(path,phase))
            if u.path=='/api/frame':
                query=parse_qs(u.query)
                return self.reply(camera_frame(query['camera'][0]) if 'camera' in query else frame(query.get('view',['external'])[0]),'image/jpeg')
            if u.path=='/api/log':
                name=STATE.get('runtime_log') if STATE['phase'] in ('loading','warming') else STATE.get('run_log',STATE.get('runtime_log'))
                text=(ROOT/'evidence'/name).read_text(errors='replace')[-6000:] if name else ''
                if STATE.get('runtime_log')==name:
                    p=ROOT/'evidence/policy-service.log'
                    text+='\n\nPOLICY SERVICE\n'+(p.read_text(errors='replace')[-10000:] if p.exists() else '')
                if RECEIPT and STATE['phase'] not in ('loading','warming'):
                    p=RECEIPT.with_suffix('.stderr.log')
                    if p.exists():
                        with p.open('rb') as log:
                            log.seek(0,2);size=log.tell();log.seek(max(0,size-10000))
                            text+='\n\nCONTROLLER STDERR\n'+log.read().decode(errors='replace')
                return self.reply({'text':text})
            paths={'/':'index.html','/app.js':'app.js','/style.css':'style.css','/graphs.js':'graphs.js'}
            if u.path not in paths:return self.reply({'error':'Not found'},code=404)
            kind='text/html' if u.path=='/' else 'text/javascript' if u.path.endswith('.js') else 'text/css'
            return self.reply((ROOT/'ui'/paths[u.path]).read_bytes(),kind)
        except Exception as e:return self.reply({'error':str(e)},code=503)
    def do_POST(self):
        if not self.valid_host() or self.headers.get('Origin') not in ('http://'+h for h in self.server.allowed_hosts) or not secrets.compare_digest(self.headers.get('X-Control-Token',''),TOKEN):
            return self.reply({'error':'Reload this panel before sending commands'},code=403)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n<0 or n>8192:raise ValueError('Invalid request length')
            data=json.loads(self.rfile.read(n) or b'{}')
            actions={'/api/start':start,'/api/pause':pause,'/api/play':lambda:pause(True),'/api/stop':stop}
            if self.path=='/api/settings':configure(data)
            elif self.path in actions:actions[self.path]()
            else:raise ValueError('Unknown operation')
            self.reply(public_state())
        except (ValueError,ProcessLookupError) as e:self.reply({'error':str(e)},code=409)
        except Exception as e:self.reply({'error':str(e)},code=500)
    def log_message(self,fmt,*args):
        if args and any(p in str(args[0]) for p in ('/api/status','/api/telemetry','/api/cameras','/api/frame','/api/stream')):return
        super().log_message(fmt,*args)

def restore_last_session():
    # Keep the last finished session visible across panel restarts; never resume it.
    global RECEIPT,RUN_SINCE
    if clients() or controller_running():return
    paths=list((ROOT/'evidence').glob('stream-policy-*.json'))
    if not paths:return
    path=max(paths,key=lambda p:p.stat().st_mtime);r=load(path,{})
    if not r.get('finished_at_unix'):return
    c=config()
    if r.get('prompt')!=c['task']['prompt'] or r.get('policy_metadata',{}).get('backend')!=c['policy']['backend']:return
    RECEIPT=path;RUN_SINCE=r.get('started_at_unix',path.stat().st_mtime)
    refresh_receipt()
    if r.get('complete'):STATE.update(phase='completed',message='Last session completed.')
    elif r.get('stopped_by_user'):STATE.update(phase='stopped',message='Last session stopped.')
    else:STATE.update(phase='error',message=failure_detail(r) or 'Last session stopped before motion began.')
    logs=list((ROOT/'evidence').glob('ui-arm-*.log'))
    if logs:STATE['run_log']=max(logs,key=lambda p:p.stat().st_mtime).name

def main():
    p=argparse.ArgumentParser();p.add_argument('--bind',default='100.95.186.107');p.add_argument('--port',type=int,default=8787);args=p.parse_args()
    STATE['ready']=usable_readiness()
    if not maintenance() and not clients() and not controller_running():userctl('start','zed-camera-viewer.service')
    if STATE['ready']:STATE['message']=BACKENDS[config()['policy']['backend']]+' ready.'
    restore_last_session()
    def shutdown(*_):
        if STATE['phase'] not in ('loading','warming'):stop()
        elif PROCESS and PROCESS.poll() is None:os.killpg(PROCESS.pid,signal.SIGTERM)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,shutdown);signal.signal(signal.SIGINT,shutdown)
    threading.Thread(target=monitor,daemon=True).start()
    server=ThreadingHTTPServer((args.bind,args.port),Handler)
    server.allowed_hosts={f'{h}:{args.port}' for h in (args.bind,'rome','rome.taila2d385.ts.net')}
    server.serve_forever()
if __name__=='__main__':main()
