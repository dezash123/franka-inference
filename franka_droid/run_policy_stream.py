#!/usr/bin/env python3
"""Configurable action playback, synchronous inference refill, continuous FCI with explicit Pause/Play/Stop."""
import argparse,json,os,random,secrets,signal,subprocess,threading,time
import cv2
import numpy as np
import rclpy
from moveit_msgs.srv import GetStateValidity
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from cameras import ZedSource,wrist_source,LatestCamera
from contract import ROOT,settings,make_observation
from gripper_control import GripperSession
from runtime_contract import RUNTIMES,ABSOLUTE,validate_runtime

MOTION_LIMITS=None
CONSTRAINT_PROFILE=json.loads((ROOT/'config/droid-fr3-active.json').read_text())
FEEDBACK_CONTROLLER='droid_hybrid_joint_cartesian_impedance'
OPERATING_LIMITS=None

class Feedback:
    def __init__(self,process,path):
        self.process=process;self.lock=threading.Lock();self.latest=None;self.error=None;self.complete=False;self.target_rejections=0;self.fault=None
        self.log=path.open('w');self.thread=threading.Thread(target=self.read,daemon=True);self.thread.start()
    def read(self):
        try:
            for line in self.process.stdout:
                item=json.loads(line)
                with self.lock:
                    if item.get('ok') is False and self.error is None:self.error=RuntimeError(item['error'])
                    diagnostic=item.get('control_error_record')
                    if diagnostic and diagnostic.get('current_errors'):self.fault=diagnostic
                    if 'state' in item:self.latest=(item,time.monotonic())
                    if item.get('complete'):self.complete=True
                    if item.get('target_rejected'):self.target_rejections+=1
                self.log.write(line);self.log.flush()
        except BaseException as e:
            if self.error is None:self.error=e
        finally:self.log.close()
    def get(self,timeout=.15):
        if self.error:raise self.error
        if self.process.poll() is not None and not self.complete:raise RuntimeError('FCI process exited')
        with self.lock:item=self.latest
        if item is None or time.monotonic()-item[1]>timeout:raise RuntimeError('FCI feedback stale')
        if 'stamp' in item[0] and time.monotonic()-item[0]['stamp']>timeout:raise RuntimeError('FCI state timestamp stale')
        return item[0]

def main():
    cfg=settings()
    external_cfgs=cfg['external_cameras']
    if cfg['policy'].get('external_view_selection')!='uniform_random_camera_left_eye_per_inference':
        raise ValueError('Expected random external camera selection with left eyes only')
    if not external_cfgs or len({c['serial'] for c in external_cfgs})!=len(external_cfgs):
        raise ValueError('At least one external camera with distinct serials is required')
    if any(c.get('eyes')!=['left'] for c in external_cfgs):
        raise ValueError('External cameras must use left eyes only')
    views=[{'camera_index':i,'serial':c['serial'],'eye':'left','key':c['serial']+':left'}
           for i,c in enumerate(external_cfgs)]
    selection_seed=secrets.randbits(64);view_rng=random.Random(selection_seed)
    selected_view=views[0]
    p=argparse.ArgumentParser();p.add_argument('--execute-confirmed',action='store_true')
    p.add_argument('--hold-test',action='store_true');p.add_argument('--duration',type=float,default=0,help='0 runs continuously until paused or stopped; positive values request a timed session')
    p.add_argument('--horizon',type=int,default=cfg['policy'].get('execution_horizon',cfg['policy']['action_horizon']));p.add_argument('--prompt',default=cfg.get('task',{}).get('prompt'))
    p.add_argument('--action-hz',type=float,default=cfg['policy'].get('control_hz',15),help='Positive action playback frequency in Hz; no configured upper bound; inference remains synchronous')
    a=p.parse_args()
    if not np.isfinite(a.action_hz) or a.action_hz<=0:p.error('Action playback frequency must be a finite number greater than 0 Hz')
    action_period=1./a.action_hz
    if not isinstance(a.prompt,str) or not a.prompt.strip():p.error('A nonempty task prompt is required')
    if not a.execute_confirmed:p.error('Explicit execution confirmation required')
    if not np.isfinite(a.duration) or a.duration<0 or (0<a.duration<1) or not 1<=a.horizon<=15:p.error('Invalid duration or action horizon')
    if a.hold_test and not 1<=a.duration<=5:p.error('Hold test limited to five seconds')
    path=ROOT/'evidence'/f'stream-policy-{time.time_ns()}.json'
    receipt={'prompt':a.prompt,'requested_duration_s':a.duration or None,'continuous':a.duration==0,'action_frequency_hz':a.action_hz,'inference_mode':'synchronous_chunk_refill',
             'open_loop_horizon':a.horizon,'generated_action_horizon':cfg['policy']['action_horizon'],
             'execute_full_action_chunk':a.horizon==cfg['policy']['action_horizon'],'denoise_steps':RUNTIMES[cfg['policy']['backend']]['denoise_steps'],'hold_test':a.hold_test,'chunks':[],'actions':[],'action_count':0,'chunk_count':0,
             'motion_command_mode':'droid_joint_torques','application_smoothing':False,
             'feedback_controller':FEEDBACK_CONTROLLER,'operating_profile':'droid_fr3',
             'operating_motion_limits':OPERATING_LIMITS,
             'application_motion_limits':MOTION_LIMITS,'franka_rate_limiting':True,'franka_filter_cutoff_hz':100.,
             'max_droid_joint_increment_rad':.2,'additional_action_scale':1.,
             'action_decode':RUNTIMES[cfg['policy']['backend']]['action_space'], 'started_at_unix':time.time(),
             'constraint_profile':CONSTRAINT_PROFILE,
             'external_view_selection':cfg['policy']['external_view_selection'],
             'external_view_selection_seed':selection_seed,'external_views':views,
             'external_view_counts':{v['key']:0 for v in views},'wrist_in_every_observation':True,'pause_supported':True,'pause_requested':False,'paused':False,'pause_events':[]}

    streams=[];gripper=None;process=None;feedback=None;stderr=None;started=None
    pause_requested=False;pause_started=None;pause_sequence=None;last_pause_save=0.
    journal=path.with_suffix('.events.jsonl').open('a',buffering=1) if a.duration==0 else None
    receipt['history_storage']='append_only_journal_with_recent_receipt_entries' if journal else 'full_receipt'
    if journal:receipt['event_journal']=str(path.with_suffix('.events.jsonl'))
    def log_event(kind,entry):
        if journal:journal.write(json.dumps({'kind':kind,'entry':entry},separators=(',',':'))+'\n')
    def record(kind,entry):
        counter={'actions':'action_count','chunks':'chunk_count'}.get(kind)
        if counter:
            if kind=='chunks':entry['id']=receipt[counter]
            receipt[counter]+=1
        log_event(kind,entry)
        entries=receipt.setdefault(kind,[]);entries.append(entry)
        if journal:
            limit=128 if kind=='actions' else 32
            if len(entries)>limit:del entries[:-limit]
    def save():
        temp=path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(receipt,indent=2));os.replace(temp,path)
    def request_pause(*_):
        nonlocal pause_requested
        pause_requested=True
    def request_play(*_):
        nonlocal pause_requested
        pause_requested=False
    def stop_signal(*_):
        receipt['stopped_by_user']=True
        raise KeyboardInterrupt('Stop requested')
    signal.signal(signal.SIGUSR1,request_pause);signal.signal(signal.SIGUSR2,request_play)
    signal.signal(signal.SIGINT,stop_signal);signal.signal(signal.SIGTERM,stop_signal)
    status=json.loads(subprocess.check_output(['python3','/var/lib/spring-data/franka-network-20260923/read-franka.py'],text=True))
    if status['operating_mode']!='Execution' or not status['fci_active'] or status['robot_errors'] or status['brakes']!=['Unlocked']*7 or status['arm']['robotSerial']!=cfg['robot']['serial']:
        receipt['robot_status']=status;receipt['failure']='Robot readiness mismatch';receipt['finished_at_unix']=time.time();save()
        raise RuntimeError('Robot readiness mismatch')
    rclpy.init();node=rclpy.create_node('droid_stream_validity_only')
    validity=node.create_client(GetStateValidity,'/franka/check_state_validity')
    sequence=0
    def send(kind,expected=None,target=None):
        nonlocal sequence
        sequence+=1
        text=f'{kind} {sequence}'
        if target is not None:text+=' '+' '.join(format(float(v),'.17g') for v in np.r_[expected,target])
        process.stdin.write(text+'\n');process.stdin.flush()
    def cameras(timeout=.15):
        end=time.monotonic()+timeout;holding=False
        while True:
            if gripper and gripper.error:raise gripper.error
            try:
                all_frames=[s.get(.25) for s in streams]
                wrist=all_frames[-1]
                if any(abs(frame[1]-wrist[1])>.1 for frame in all_frames[:-1]):
                    raise RuntimeError('External/wrist camera pair skew exceeds 100 ms')
                external=all_frames[selected_view['camera_index']]
                return [external,wrist]
            except RuntimeError:
                if time.monotonic()>=end or any(s.error for s in streams):raise
                if not holding and started and process and process.poll() is None:
                    send('HOLD');holding=True
                    receipt['camera_holds']=receipt.get('camera_holds',0)+1
                time.sleep(.01)
    def wait_for_action_tick(deadline):
        # Low playback rates still need fresh feedback/cameras/gripper and a live
        # command producer. Keep the same validated target; do not advance policy.
        holds_before=receipt.get('camera_holds',0)
        if a.duration>0:deadline=min(deadline,started+a.duration-.3)
        next_heartbeat=time.monotonic()+.1
        while not pause_requested and time.monotonic()<deadline:
            if time.monotonic()>=next_heartbeat:
                receipt['latest_controller']=feedback.get();cameras();gs=gripper.read()
                if gs['fault'] or gs['activation_status']!=3:raise RuntimeError('Gripper fault between actions')
                send('KEEPALIVE')
                receipt['playback_keepalives']=receipt.get('playback_keepalives',0)+1
                next_heartbeat=time.monotonic()+.1
            time.sleep(min(.02,max(0,deadline-time.monotonic())))
        return receipt.get('camera_holds',0)!=holds_before
    def infer(state,gs):
        nonlocal selected_view
        # One independent uniform draw per inference; all actions in the chunk
        # use this observation. Only the selected camera's left eye enters the policy.
        selected_view=view_rng.choice(views)
        receipt['last_attempted_external_view']=dict(selected_view)
        frames=cameras()
        observed_at=time.monotonic()
        observation_frames={name:{'sequence':f[2],'timestamp_monotonic':f[1],'age_s':observed_at-f[1]}
                            for name,f in zip(('external','wrist'),frames)}
        obs=make_observation(frames[0][0],frames[1][0],state['q'],gs['closure'],a.prompt)
        obs['_session_id']=path.stem;obs['_external_view']=selected_view['key']
        begin=time.monotonic();result=client.infer(obs);dt=time.monotonic()-begin
        chunk=np.asarray(result['actions'])
        if dt>.25 or chunk.shape!=(15,8) or not np.isfinite(chunk).all() or (RUNTIMES[cfg['policy']['backend']]['action_space']!=ABSOLUTE and np.max(np.abs(chunk[:,:7]))>1.2):
            raise RuntimeError('Invalid or late policy chunk')
        record('chunks',{'t':time.monotonic()-started if started else None,'inference_ms':dt*1000,
                                  'external_view':dict(selected_view),'observation_frames':observation_frames,
                                  'camera_pair_skew_s':abs(frames[0][1]-frames[1][1]),'actions':chunk.tolist(),
                                  'policy_timing':result.get('policy_timing'),'server_timing':result.get('server_timing')})
        receipt['external_view_counts'][selected_view['key']]+=1
        return chunk,obs,frames
    def valid_path(q,target):
        intervals=max(4,int(np.ceil(np.max(np.abs(target-q))/.01)))
        for fraction in np.linspace(0,1,intervals+1):
            req=GetStateValidity.Request();req.group_name='fr3v2_1_arm'
            req.robot_state.joint_state.name=cfg['robot']['joints']
            req.robot_state.joint_state.position=(q+fraction*(target-q)).tolist()
            f=validity.call_async(req);rclpy.spin_until_future_complete(node,f,timeout_sec=.15)
            if not f.done() or not f.result().valid:
                receipt['validity_failure']={'fraction':fraction,'q':q.tolist(),'target':target.tolist(),
                    'timeout':not f.done(),'response':str(f.result()) if f.done() else None}
                save()
                raise RuntimeError('Collision validity service timed out' if not f.done() else 'Collision validity rejected target')
    try:
        if not validity.wait_for_service(timeout_sec=10):raise RuntimeError('MoveIt validity service unavailable')
        client=WebsocketClientPolicy(host=cfg['policy']['host'],port=cfg['policy']['port'])
        metadata=client.get_server_metadata();receipt['policy_metadata']=metadata
        runtime=validate_runtime(metadata,cfg['policy']['backend']);absolute_targets=runtime['action_space']==ABSOLUTE
        for camera_cfg in external_cfgs:
            streams.append(LatestCamera(ZedSource({**camera_cfg,'eye':'left'})))
        streams.append(LatestCamera(wrist_source(cfg['wrist_camera'])))
        frames=cameras(8)
        receipt['camera_sources']=[s.source.info for s in streams]
        for view in views:
            image=streams[view['camera_index']].get(.25)[0]
            cv2.imwrite(str(path.with_name(path.stem+'-'+view['serial']+'-'+view['eye']+'-before.jpg')),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        gripper=GripperSession(cfg['robot']['gripper']);time.sleep(.25)
        gs=gripper.read()
        receipt["gripper_initial"]=gs;save()
        if gs["fault"]==9 or (gs["fault"]==0 and gs["activation_status"]!=3):
            # Startup only: recover communication timeout or finish activation
            # before the arm controller is launched. Serialize the activation
            # sequence against the heartbeat; other faults still abort.
            with gripper.lock:gs=gripper.activate()
            receipt["gripper_startup_reactivation"]=gs;save()
        if gs["fault"] or gs["activation_status"]!=3:
            raise RuntimeError("Gripper unready: "+json.dumps(gs))
        stderr=path.with_suffix('.stderr.log').open('w')
        controller_args=[str(ROOT/'stream_fci'),'--session-confirmed']
        process=subprocess.Popen(controller_args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                 stderr=stderr,text=True,bufsize=1)
        feedback=Feedback(process,path.with_suffix('.states.jsonl'))
        ready_deadline=time.monotonic()+10
        while feedback.latest is None:
            if feedback.error:raise feedback.error
            if process.poll() is not None or time.monotonic()>ready_deadline:raise RuntimeError('Controller did not become ready')
            time.sleep(.01)
        initial=feedback.get();receipt['initial']=initial;save()
        expected_controller={'continuous_sessions_supported':True,'command_keepalive_supported':True,'motion_command_mode':'droid_joint_torques','application_smoothing':False,
                             'feedback_controller':FEEDBACK_CONTROLLER,'operating_profile':'droid_fr3',
                             'operating_motion_limits':OPERATING_LIMITS,
                             'application_motion_limits':MOTION_LIMITS,'franka_rate_limiting':True,
                             'franka_filter_cutoff_hz':100.,'max_droid_joint_increment_rad':.2}
        if any(key not in initial or initial[key]!=value for key,value in expected_controller.items()):
            raise RuntimeError('Controller motion configuration does not match client')
        if initial.get('constraint_profile')!=CONSTRAINT_PROFILE:raise RuntimeError('Controller constraint profile mismatch')
        target_lower=np.asarray(initial['joint_target_lower_rad'],dtype=float)
        target_upper=np.asarray(initial['joint_target_upper_rad'],dtype=float)
        if (target_lower.shape!=(7,) or target_upper.shape!=(7,) or
            not np.isfinite(target_lower).all() or not np.isfinite(target_upper).all() or
            np.any(target_lower>=target_upper)):
            raise RuntimeError('Controller joint target bounds are invalid')
        chunk,obs,frames=infer(initial['state'],gs)
        np.savez_compressed(path.with_name(path.stem+'-initial-observation.npz'),**obs)
        for name,f in zip(('external','wrist'),frames):
            cv2.imwrite(str(path.with_name(path.stem+'-'+name+'-before.jpg')),cv2.cvtColor(f[0],cv2.COLOR_RGB2BGR))
        process.stdin.write(f'START {a.duration}\n');process.stdin.flush()
        deadline=time.monotonic()+2
        while True:
            item=feedback.get(timeout=2)
            if 'stamp' in item:break
            if time.monotonic()>deadline:raise RuntimeError('Streaming feedback did not start')
            time.sleep(.005)
        started=time.monotonic();receipt['control_started_unix']=time.time()
        receipt['control_started_monotonic']=started
        index=0;step=0;next_tick=started;last_snapshot=started-10;last_progress=started;last_rejections=0
        while a.duration==0 or time.monotonic()-started<a.duration-.3:
            if pause_requested:
                send('HOLD')
                if pause_started is None:
                    pause_started=time.monotonic();pause_sequence=sequence
                    gripper.hold_current()
                    receipt['pause_requested']=True;receipt['paused']=False
                    record('pause_events',{'t':pause_started-started,'action_count':receipt['action_count']})
                    save()
                item=feedback.get();gs=gripper.read()
                if gs['fault'] or gs['activation_status']!=3:raise RuntimeError('Gripper fault while paused')
                frames=cameras()
                receipt['paused']=item.get('applied',0)>=pause_sequence and max(abs(v) for v in item['state']['dq'])<=.02
                if time.monotonic()-last_pause_save>=.5:
                    receipt['latest_controller']=item
                    receipt['pause_elapsed_s']=time.monotonic()-pause_started
                    for name,f in zip(('external','wrist'),frames):
                        cv2.imwrite(str(path.with_name(path.stem+'-'+name+'-latest.jpg')),cv2.cvtColor(f[0],cv2.COLOR_RGB2BGR))
                    receipt['latest_snapshot_external_view']=dict(selected_view);receipt['snapshot_unix']=time.time()
                    save();last_pause_save=time.monotonic()
                time.sleep(.04)
                continue
            if pause_started is not None:
                receipt['pause_events'][-1]['resumed_t']=time.monotonic()-started
                receipt['pause_events'][-1]['duration_s']=time.monotonic()-pause_started
                log_event('pause_resumed',dict(receipt['pause_events'][-1]))
                receipt['pause_requested']=False;receipt['paused']=False;receipt['pause_elapsed_s']=0
                pause_started=None;pause_sequence=None
                index=a.horizon;next_tick=time.monotonic();save()
            if feedback.target_rejections!=last_rejections:
                last_rejections=feedback.target_rejections;receipt['cartesian_target_holds']=last_rejections;index=a.horizon
            item=feedback.get();holds_before=receipt.get('camera_holds',0);frames=cameras();gs=gripper.read()
            if receipt.get('camera_holds',0)!=holds_before:index=a.horizon
            if gs['fault'] or gs['activation_status']!=3:raise RuntimeError('Gripper fault')
            if pause_requested:continue
            if index>=a.horizon:
                send('HOLD');chunk,obs,frames=infer(item['state'],gs);index=0
                item=feedback.get();frames=cameras();next_tick=time.monotonic()
            if pause_requested:continue
            q=np.array(item['state']['q']);action=chunk[index]
            delta=.2*action[:7]/max(1.,float(np.max(np.abs(action[:7]))))
            target=action[:7].copy() if absolute_targets else q+delta
            if a.hold_test:target=q
            rejected_target=False
            for recheck in range(3):
                outside=(target<target_lower)|(target>target_upper)
                if absolute_targets:outside |= np.abs(target-q)>.2
                if np.any(outside):
                    # Keep the existing C++ margin. Hold and discard the chunk
                    # before dispatching an out-of-bounds policy target.
                    send('HOLD');index=a.horizon;rejected_target=True
                    receipt['joint_limit_holds']=receipt.get('joint_limit_holds',0)+1
                    record('joint_limit_events',{
                        't':time.monotonic()-started,'joints':(np.flatnonzero(outside)+1).tolist(),
                        'q':q.tolist(),'rejected_target':target.tolist(),
                        'reason':'joint range or existing 0.2 rad target increment'})
                    receipt['latest_controller']=feedback.get();save()
                    if time.monotonic()-last_progress>=10:
                        print(json.dumps({'elapsed_s':time.monotonic()-started,'steps':step,
                                          'chunks':receipt['chunk_count'],'holding':'joint_target_limit',
                                          'joint_limit_holds':receipt['joint_limit_holds']}),flush=True)
                        last_progress=time.monotonic()
                    wait_for_action_tick(time.monotonic()+action_period)
                    break
                valid_path(q,target)
                fresh=feedback.get()
                if np.max(np.abs(np.array(fresh['state']['q'])-q))<=.002:break
                # A slow RPC may outlive the state it checked during streaming.
                # Discard its target, hold and validate a newly anchored target.
                send('HOLD')
                receipt['state_refreshes']=receipt.get('state_refreshes',0)+1
                q=np.array(fresh['state']['q']);target=q if a.hold_test else action[:7].copy() if absolute_targets else q+delta
            else:
                # The old path stopped the whole session after three moving-state
                # retries. Keep HOLD active until measured motion settles, then
                # discard this plan and infer from a new observation.
                send('HOLD');hold_sequence=sequence;hold_started=time.monotonic();settled_since=None
                index=a.horizon;rejected_target=True
                receipt['validation_holds']=receipt.get('validation_holds',0)+1
                while not pause_requested:
                    send('HOLD');fresh=feedback.get();cameras();gs=gripper.read()
                    if gs['fault'] or gs['activation_status']!=3:raise RuntimeError('Gripper fault during validation hold')
                    now=time.monotonic()
                    settled=fresh.get('applied',0)>=hold_sequence and max(abs(v) for v in fresh['state']['dq'])<=.02
                    settled_since=(now if settled_since is None else settled_since) if settled else None
                    if settled_since is not None and now-settled_since>=.05:break
                    if now-hold_started>2:raise RuntimeError('Arm did not settle during validation hold')
                    time.sleep(.01)
                record('validation_hold_events',{'t':hold_started-started,'duration_s':time.monotonic()-hold_started,
                    'reason':'moving_start_state','settled':not pause_requested})
                receipt['latest_controller']=feedback.get();save()
            if rejected_target or pause_requested:continue
            cameras()
            if np.max(np.abs(np.array(feedback.get()['state']['q'])-q))>.002:
                send('HOLD');receipt['state_refreshes']=receipt.get('state_refreshes',0)+1
                continue
            if pause_requested:continue
            dispatched=time.monotonic()
            send('HOLD' if a.hold_test else 'TARGET',q,None if a.hold_test else target)
            if not a.hold_test:gripper.command_closure(float(action[7]))
            now=time.monotonic()
            record('actions',{'step':step,'chunk':receipt['chunk_count']-1,'index':index,'t':now-started,
                                       'scale':1.,'dispatch_t':dispatched-started,
                                       'normalized_action_clipped':bool(not absolute_targets and np.any(np.abs(action[:7])>1.)),
                                       'q':q.tolist(),'target':target.tolist(),'xyz':item['state']['xyz'],
                                       'gripper':gs['closure'],'applied':item.get('applied')})
            if now-last_snapshot>=2:
                for name,f in zip(('external','wrist'),frames):
                    cv2.imwrite(str(path.with_name(path.stem+'-'+name+'-latest.jpg')),cv2.cvtColor(f[0],cv2.COLOR_RGB2BGR))
                receipt['latest_controller']=item
                receipt['latest_snapshot_external_view']=dict(selected_view);receipt['snapshot_unix']=time.time()
                save();last_snapshot=now
            if now-last_progress>=10:
                print(json.dumps({'elapsed_s':now-started,'steps':step+1,'chunks':receipt['chunk_count'],
                                  'xyz':item['state']['xyz'],'gripper_closure':gs['closure'],'external_view_counts':receipt['external_view_counts'],'cartesian_target_holds':feedback.target_rejections,'safety_feedback_seen_mask':item.get('safety_feedback_seen_mask',0),'torque_saturation_seen_mask':item.get('torque_saturation_seen_mask',0)}),flush=True);last_progress=now
            index+=1;step+=1;next_tick+=action_period
            delay=next_tick-time.monotonic()
            if delay>0:
                if wait_for_action_tick(next_tick):index=a.horizon
            elif delay<-action_period:next_tick=time.monotonic()
        process.stdin.write('STOP\n');process.stdin.flush();process.wait(timeout=3);feedback.thread.join(timeout=1)
        if process.returncode or feedback.error or not feedback.complete:raise RuntimeError('Controller did not stop cleanly')
        receipt['complete']=True;receipt['actual_control_s']=time.monotonic()-started
        receipt['final']=feedback.latest[0]
        for name,f in zip(('external','wrist'),cameras()):
            cv2.imwrite(str(path.with_name(path.stem+'-'+name+'-after.jpg')),cv2.cvtColor(f[0],cv2.COLOR_RGB2BGR))
    except BaseException as e:
        receipt['failure']=str(e)
        if process and process.poll() is None:
            try:process.stdin.write('STOP\n');process.stdin.flush()
            except BrokenPipeError:pass
        raise
    finally:
        if process and process.poll() is None:
            try:process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.send_signal(signal.SIGINT)
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        if process:receipt['controller_exit_code']=process.returncode
        if feedback:feedback.thread.join(timeout=1)
        if feedback:
            receipt['cartesian_target_holds']=feedback.target_rejections
            if feedback.fault:receipt['controller_fault']=feedback.fault
            if feedback.error:
                receipt['controller_error']=str(feedback.error)
                secondary=('FCI feedback stale','FCI state timestamp stale','FCI process exited',
                           'Streaming FCI stopped; see controller stderr','Controller did not stop cleanly')
                if receipt.get('failure') in secondary:
                    receipt['client_failure']=receipt['failure']
                    receipt['failure']=str(feedback.error)
        if feedback and feedback.latest:
            receipt['final']=feedback.latest[0]
            receipt['controller_complete']=feedback.complete
        if started:receipt['elapsed_s']=time.monotonic()-started
        if pause_started is not None:
            receipt['pause_events'][-1]['duration_s']=time.monotonic()-pause_started
            log_event('pause_ended',dict(receipt['pause_events'][-1]))
        receipt['paused']=False;receipt['pause_requested']=False
        receipt['finished_at_unix']=time.time();save()
        if stderr:stderr.close()
        if gripper:
            try:gripper.close()
            except Exception as e:
                receipt.setdefault('cleanup_errors',[]).append(f'Gripper: {e}');save()
        for s in reversed(streams):
            try:s.close()
            except Exception as e:
                receipt.setdefault('cleanup_errors',[]).append(f'{type(s.source).__name__}: {e}')
                save()
        print(json.dumps({'receipt':str(path),'complete':receipt.get('complete',False),'failure':receipt.get('failure'),'steps':receipt['action_count']}),flush=True)
        if journal:journal.close()
        node.destroy_node();rclpy.try_shutdown()

if __name__=='__main__':main()
