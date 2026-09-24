#!/usr/bin/env python3
"""Headless ZED stereo and RealSense D405 color viewer. Loopback by default."""
import argparse
import collections
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
from capture import Capture, has_yuyv

ROOT = Path(__file__).resolve().parent
STOP = threading.Event()
LOCK = threading.Lock()
CAMERAS = {}
cv2.setNumThreads(1)


def discover():
    devices = []
    for node in sorted(Path('/sys/class/video4linux').glob('video*')):
        try:
            usb = next(p for p in node.resolve().parents if (p / 'idVendor').exists())
            vendor = (usb / 'idVendor').read_text().strip()
            product = (usb / 'idProduct').read_text().strip()
            index = int((node / 'index').read_text().strip())
            d405 = (vendor, product) == ('8086', '0b5b')
            if vendor == '2b03' and index == 0:
                layout = 'stereo_pair'
            elif d405 and index % 2 == 0 and has_yuyv('/dev/' + node.name):
                layout = 'color'
            else:
                continue
            model = 'RealSense D405' if d405 else (usb / 'product').read_text().strip()
            # ZED 2i UVC exposes generic serial OV0001. Its adjacent HID
            # interface on the same internal USB2 hub exposes the real serial.
            serial = (usb / 'serial').read_text().strip() if (usb / 'serial').exists() else None
            serial_source = 'uvc' if serial and serial != 'OV0001' else 'unavailable'
            if serial_source == 'unavailable':
                for sibling in usb.parent.iterdir():
                    if sibling != usb and (sibling / 'idVendor').exists() and (sibling / 'serial').exists():
                        if (sibling / 'idVendor').read_text().strip() == '2b03' and (sibling / 'idProduct').read_text().strip() in ('f881', 'f681', 'f781'):
                            serial = (sibling / 'serial').read_text().strip()
                            serial_source = 'adjacent_sensor_same_internal_hub'
                            break
            if serial_source == 'unavailable':
                serial = None
            key = hashlib.sha256(str(usb).encode()).hexdigest()[:12]
            devices.append(dict(id=key, device='/dev/' + node.name, model=model,
                                usb_path=usb.name, usb_speed_mbps=int((usb / 'speed').read_text()),
                                serial=serial, serial_source=serial_source,
                                layout=layout,
                                role='wrist' if d405 or 'ZED-M' in model or 'ZED Mini' in model else 'external',
                                views=['color'] if d405 else ['left','right','stereo'],
                                default_view='color' if d405 else 'left'))
        except (OSError, StopIteration, ValueError):
            continue
    return devices


class Camera:
    def __init__(self, info, args):
        self.info, self.args = info, args
        self.condition = threading.Condition()
        self.stop = threading.Event()
        self.jpeg = {}
        self.frame_count = 0
        self.rejected_frames = 0
        self.driver_error_frames = 0
        self.incomplete_frames = 0
        self.sequence_gaps = 0
        self.last_rejected = 0
        self.negotiated_fps = 0
        self.last_frame = 0
        self.times = collections.deque(maxlen=90)
        self.width = self.height = 0
        self.error = 'Opening camera'
        self.thread = threading.Thread(target=self.capture, name=info['device'])
        self.thread.start()

    def status(self):
        with self.condition:
            age = time.monotonic() - self.last_frame if self.last_frame else None
            fps = (len(self.times)-1)/(self.times[-1]-self.times[0]) if len(self.times)>1 else 0
            return dict(self.info, frame_count=self.frame_count, width=self.width, height=self.height,
                        fps=round(fps, 2), age_seconds=round(age, 3) if age is not None else None,
                        healthy=age is not None and age < 2, error=self.error,
                        rejected_frames=self.rejected_frames,
                        driver_error_frames=self.driver_error_frames,
                        incomplete_frames=self.incomplete_frames,
                        sequence_gaps=self.sequence_gaps,
                        recent_corruption=time.monotonic()-self.last_rejected < 10,
                        negotiated_fps=self.negotiated_fps,
                        image_type='uvc_color' if self.info['layout']=='color' else 'raw_unrectified',
                        output_color='JPEG RGB')

    def capture(self):
        while not STOP.is_set() and not self.stop.is_set():
            cap = None
            try:
                hd = self.args.resolution == '720p' or (self.args.resolution == 'auto' and self.info['usb_speed_mbps'] >= 5000)
                if self.info['layout'] == 'color':
                    w,h = (1280,720) if hd else (640,480)
                else:
                    w,h = (2560,720) if hd else (1344,376)
                cap = Capture(self.info['device'],w,h,self.args.fps)
                self.negotiated_fps = cap.fps
                previous_sequence = None
                last_received = time.monotonic()
                logging.info('Capture %s %s negotiated %sx%s@%s with frame validation', self.info['model'],self.info['serial'],cap.width,cap.height,cap.fps)
                while not STOP.is_set() and not self.stop.is_set():
                    result,raw,meta = cap.read()
                    if result == 0:
                        if time.monotonic()-last_received > 3:
                            raise RuntimeError('Camera stopped delivering frames; retrying')
                        continue
                    last_received = time.monotonic()
                    with self.condition:
                        if previous_sequence is not None and meta.sequence > previous_sequence + 1:
                            self.sequence_gaps += meta.sequence-previous_sequence-1
                        previous_sequence = meta.sequence
                        if result == 2:
                            self.rejected_frames += 1
                            self.driver_error_frames += bool(meta.flags & 0x40)
                            self.incomplete_frames += meta.bytesused != raw.nbytes
                            self.last_rejected = time.monotonic()
                            if self.rejected_frames == 1 or self.rejected_frames % 100 == 0:
                                logging.warning('%s rejected %d bad USB frames (latest bytes=%d flags=%#x)',self.info['serial'],self.rejected_frames,meta.bytesused,meta.flags)
                            continue
                    frame = cv2.cvtColor(raw,cv2.COLOR_YUV2BGR_YUYV)
                    h,w = frame.shape[:2]
                    if self.info['layout'] == 'color':
                        pictures = {'color':frame}
                    else:
                        if w % 2:
                            raise RuntimeError('Unexpected odd stereo frame width')
                        pictures = {'stereo':frame,'left':frame[:,:w//2],'right':frame[:,w//2:]}
                    encoded = {}
                    for eye,pic in pictures.items():
                        success,jpg=cv2.imencode('.jpg',pic,[cv2.IMWRITE_JPEG_QUALITY,self.args.quality])
                        if not success:
                            raise RuntimeError('JPEG encoding failed')
                        encoded[eye]=jpg.tobytes()
                    with self.condition:
                        self.jpeg=encoded
                        self.width,self.height=w,h
                        self.frame_count+=1
                        self.last_frame=time.monotonic()
                        self.times.append(self.last_frame)
                        self.error=None
                        self.condition.notify_all()
            except Exception as exc:
                logging.warning('%s: %s',self.info['device'],exc)
                with self.condition:
                    self.error=str(exc)
            finally:
                if cap is not None:
                    cap.close()
            if not self.stop.is_set():
                STOP.wait(2)


def scan(args):
    while not STOP.is_set():
        found={i['id']:i for i in discover()}
        with LOCK:
            for key in list(CAMERAS):
                if key not in found or CAMERAS[key].info != found[key]:
                    CAMERAS.pop(key).stop.set()
            for key,info in found.items():
                if key not in CAMERAS:
                    CAMERAS[key]=Camera(info,args)
        STOP.wait(2)


class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'

    def log_message(self, fmt, *args):
        if args and str(args[1] if len(args)>1 else '') not in ('200','304'):
            logging.info(fmt,*args)

    def reply(self,data,content_type,status=200):
        self.send_response(status)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url=urlparse(self.path)
        if url.path=='/':
            return self.reply((ROOT/'web/index.html').read_bytes(),'text/html; charset=utf-8')
        if url.path=='/api/status':
            with LOCK:
                status=[c.status() for c in CAMERAS.values()]
            return self.reply(json.dumps({'cameras':status,'timestamp':time.time(),'pid':os.getpid()}).encode(),'application/json')
        if url.path not in ('/stream.mjpg','/snapshot.jpg'):
            return self.reply(b'Not found','text/plain',404)
        query=parse_qs(url.query)
        key=query.get('camera',[''])[0]
        with LOCK:
            cam=CAMERAS.get(key)
        if cam is None:
            return self.reply(b'Camera/view not found','text/plain',404)
        eye=query.get('eye',[cam.info['default_view']])[0]
        if eye not in cam.info['views']:
            return self.reply(b'Camera/view not found','text/plain',404)
        if url.path=='/snapshot.jpg':
            with cam.condition:
                data=cam.jpeg.get(eye)
                fresh=time.monotonic()-cam.last_frame<2
            return self.reply(data,'image/jpeg') if data and fresh else self.reply(b'No fresh frame','text/plain',503)
        self.send_response(200)
        self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control','no-store')
        self.send_header('Connection','close')
        self.end_headers()
        self.close_connection=True
        previous=-1
        last_sent=time.monotonic()
        try:
            while not STOP.is_set() and not cam.stop.is_set():
                with cam.condition:
                    cam.condition.wait_for(lambda:(cam.frame_count!=previous and time.monotonic()-cam.last_frame<2) or cam.stop.is_set(),timeout=.5)
                    if time.monotonic()-last_sent>3:break
                    if cam.frame_count==previous or time.monotonic()-cam.last_frame>2:
                        continue
                    data=cam.jpeg.get(eye)
                    previous=cam.frame_count
                if data:
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(data)).encode()+b'\r\n\r\n'+data+b'\r\n')
                    self.wfile.flush()
                    last_sent=time.monotonic()
        except (BrokenPipeError,ConnectionResetError,TimeoutError):
            pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bind',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--resolution',choices=['auto','vga','720p'],default='auto')
    parser.add_argument('--fps',type=int,choices=[15,30],default=15)
    parser.add_argument('--quality',type=int,default=80)
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    scanner=threading.Thread(target=scan,args=(args,))
    scanner.start()
    server=ThreadingHTTPServer((args.bind,args.port),Handler)
    server.daemon_threads=True
    def stop(*_):
        STOP.set()
        threading.Thread(target=server.shutdown,daemon=True).start()
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    logging.info('Camera viewer listening on http://%s:%d',args.bind,args.port)
    server.serve_forever()
    STOP.set()
    scanner.join()
    with LOCK:
        cameras=list(CAMERAS.values())
    for camera in cameras:
        camera.stop.set()
        with camera.condition:
            camera.condition.notify_all()
    for camera in cameras:
        camera.thread.join()
    server.server_close()

if __name__=='__main__':
    main()
