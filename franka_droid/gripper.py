"""Robotiq 2F-85 status reader. Only Modbus FC03 is implemented; no activation."""
import json
import time
import serial
from contract import ROOT,settings

def crc16(data):
    value=0xffff
    for byte in data:
        value^=byte
        for _ in range(8): value=(value>>1)^0xa001 if value&1 else value>>1
    return value

def position_to_width(gpo,stroke=.085):
    # Match Polymetis Robotiq2FingerGripper.get_pos used by DROID.
    return min(stroke,max(0.,stroke/(3.-230.)*(gpo-230.)))

class GripperReader:
    def __init__(self,cfg):
        if cfg['model']!='Robotiq 2F-85':raise ValueError('Expected configured Robotiq 2F-85')
        self.cfg=cfg
        self.port=serial.Serial(cfg['port'],cfg['baud'],timeout=.25,write_timeout=.25,exclusive=True)
    def read(self):
        # Read holding registers 0x07D0..0x07D2; never write actuator registers.
        message=bytes([self.cfg['slave_id'],3,7,208,0,3])
        request=message+crc16(message).to_bytes(2,'little')
        self.port.reset_input_buffer();self.port.write(request);self.port.flush()
        reply=self.port.read(11)
        if len(reply)!=11:raise RuntimeError(f'Gripper status timeout: received {len(reply)} of 11 bytes')
        if reply[:3]!=bytes([self.cfg['slave_id'],3,6]) or crc16(reply[:-2])!=int.from_bytes(reply[-2:],'little'):
            raise RuntimeError('Invalid Modbus gripper reply')
        data=reply[3:9];width=position_to_width(data[4],self.cfg['max_width_m'])
        return {'activation':data[0]&1,'activation_status':(data[0]>>4)&3,
            'go_to':(data[0]>>3)&1,'object_status':(data[0]>>6)&3,'fault':data[2],
            'requested_position':data[3],'actual_position':data[4],'current_10mA':data[5],
            'width_m':width,'closure':1-width/self.cfg['max_width_m'],
            'received_monotonic':time.monotonic(),'timestamp_unix':time.time(),'actuation_sent':False}
    def close(self):self.port.close()

def main():
    reader=GripperReader(settings()['robot']['gripper'])
    try:
        result=reader.read()
        (ROOT/'evidence/gripper_read.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally:reader.close()

if __name__=='__main__':main()
