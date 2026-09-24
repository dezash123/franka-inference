"""Explicit Robotiq commands, following installed ros2_robotiq_gripper driver."""
import argparse
import json
import time
import threading
from gripper import GripperReader, crc16
from contract import ROOT, settings

class GripperControl(GripperReader):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.lock=threading.RLock()
    def read(self):
        with self.lock:return super().read()
    def write_command(self, action, position=0, speed=32, force=0):
        with self.lock:return self._write_command(action,position,speed,force)
    def _write_command(self, action, position=0, speed=32, force=0):
        if action not in (0,1,9): raise ValueError('Unsupported action')
        if not all(0<=v<=255 for v in (position,speed,force)): raise ValueError('Invalid command')
        payload=bytes([self.cfg['slave_id'],16,3,232,0,3,6,action,0,0,position,speed,force])
        request=payload+crc16(payload).to_bytes(2,'little')
        self.port.reset_input_buffer();self.port.write(request);self.port.flush()
        response=self.port.read(8)
        expected=bytes([self.cfg['slave_id'],16,3,232,0,3])
        if len(response)!=8 or response[:6]!=expected or crc16(response[:6])!=int.from_bytes(response[6:],'little'):
            raise RuntimeError('Gripper write acknowledgement invalid')

    def activate(self):
        self.write_command(0)
        time.sleep(.15)
        self.write_command(1)
        end=time.monotonic()+12
        while time.monotonic()<end:
            state=self.read()
            if state['activation']==1 and state['activation_status']==3 and state['fault']==0:
                return state
            if state['fault'] not in (0,5,7,9): raise RuntimeError(f'Gripper fault {state["fault"]}')
            time.sleep(.05)
        raise RuntimeError(f'Gripper activation timed out: {state}')

    def command_closure(self, closure):
        self.write_command(9,230 if closure>.5 else 3,speed=32,force=0)

    def hold(self):
        state=self.read()
        self.write_command(1,state['actual_position'],speed=0,force=0)

class GripperSession(GripperControl):
    """Keep the last authorized command alive while inference is running."""
    def __init__(self,cfg):
        super().__init__(cfg)
        self.command=(1,0,32,0)
        self.stop_event=threading.Event();self.error=None
        self.worker=threading.Thread(target=self._heartbeat,daemon=True)
        self.worker.start()
    def _heartbeat(self):
        try:
            while not self.stop_event.is_set():
                with self.lock:self.write_command(*self.command)
                self.stop_event.wait(.2)
        except BaseException as e:self.error=e
    def command_closure(self,closure):
        if self.error:raise self.error
        with self.lock:
            self.command=(9,230 if closure>.5 else 3,32,0)
            self.write_command(*self.command)
    def hold_current(self):
        # Replace the heartbeat command too, so pause cannot reopen/reclose fingers.
        if self.error:raise self.error
        with self.lock:
            state=self.read()
            self.command=(1,state['actual_position'],0,0)
            self.write_command(*self.command)

    def close(self):
        self.stop_event.set();self.worker.join(timeout=1)
        try:self.hold()
        finally:super().close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--activate-confirmed',action='store_true');a=p.parse_args()
    if not a.activate_confirmed:p.error('Explicit --activate-confirmed required')
    reader=GripperControl(settings()['robot']['gripper'])
    receipt={'started_at_unix':time.time(),'activation_command_sent':True}
    try:
        receipt['state']=reader.activate();print(json.dumps(receipt),flush=True)
    except BaseException as e:
        receipt['error']=str(e);raise
    finally:
        reader.close();(ROOT/'evidence/gripper_activation.json').write_text(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
