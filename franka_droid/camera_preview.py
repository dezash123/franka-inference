"""Bounded camera telemetry/JPEG publication on a housekeeping thread.

Reuses the inference camera owner. Preview-only cameras never enter observations
or the required-camera freshness checks, and their failure cannot stop motion.
"""
import json
import logging
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
from cameras import LatestCamera, ZedSource
from contract import ROOT


class CameraPreview:
    def __init__(self, streams, cfg):
        self.stop = threading.Event()
        self.pid = os.getpid()
        self.directory = ROOT / 'evidence'
        self.status_path = self.directory / 'camera-preview.json'
        self.entries = []
        self.owned = []
        self.files = set()
        self.condition = threading.Condition()
        self.frames = {}  # At most one encoded frame per camera; no video queue.
        self.last_status = 0.
        self.server = None
        active = [*cfg['external_cameras'], cfg['wrist_camera']]
        for index, (stream, camera) in enumerate(zip(streams, active)):
            self.entries.append(dict(serial=camera['serial'], role='wrist' if index == len(active)-1 else 'external',
                                     stream=stream, inference_enabled=True, config=camera))
        serials = {entry['serial'] for entry in self.entries}
        for camera in cfg.get('external_cameras_disabled', []):
            if camera['serial'] not in serials:
                self.entries.append(dict(serial=camera['serial'], role='external', stream=None,
                                         inference_enabled=False, config=camera, retry=0., error=None))
                serials.add(camera['serial'])
        try:
            self.server = ThreadingHTTPServer(('127.0.0.1', 0), PreviewHandler)
            self.server.daemon_threads = True
            self.server.preview = self
            threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .1},
                             name='preview-http', daemon=True).start()
        except OSError:
            logging.exception('Live preview listener unavailable; snapshots remain available')
        self.thread = threading.Thread(target=self._run, name='camera-preview', daemon=True)
        self.thread.start()

    def _write(self, path, data):
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_bytes(data)
        os.replace(temporary, path)
        self.files.add(path)

    def _publish(self):
        now = time.monotonic()
        publish_status = now - self.last_status >= 1.
        cameras = []
        for entry in self.entries:
            if entry['stream'] is None and now >= entry.get('retry', 0):
                try:
                    stream = LatestCamera(ZedSource({**entry['config'], 'eye': 'left'}))
                    entry['stream'] = stream
                    self.owned.append(stream)
                    entry['error'] = None
                except Exception as exc:
                    entry['error'] = str(exc)
                    entry['retry'] = now + 5
            camera = {key: entry[key] for key in ('serial', 'role', 'inference_enabled')}
            camera.update(fps=0., healthy=False, age_seconds=None, frame_count=0,
                          error=entry.get('error'), image=None)
            stream = entry['stream']
            if stream is not None:
                frame, metrics = stream.preview()
                camera.update(metrics)
                if frame is not None and metrics['healthy']:
                    cached = self.frames.get(entry['serial'])
                    # Polling is fast; encode only new frames, at most 30 FPS.
                    if (cached is None or cached[0] != metrics['frame_count']) and (cached is None or now-cached[1] >= 1/30):
                        rgb = frame[0]
                        if isinstance(rgb, dict):
                            rgb = rgb['left']
                        if rgb.shape[1] > 672:
                            rgb = cv2.resize(rgb, (672, round(rgb.shape[0] * 672 / rgb.shape[1])))
                        ok, jpeg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                                 [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok:
                            cached = (metrics['frame_count'], now, jpeg.tobytes())
                            with self.condition:
                                self.frames[entry['serial']] = cached
                                self.condition.notify_all()
                    if cached is not None:
                        # Keep the legacy snapshot route, without per-frame disk writes.
                        name = f'camera-preview-{self.pid}-{entry["serial"]}.jpg'
                        if publish_status:self._write(self.directory / name, cached[2])
                        camera['image'] = name
                else:
                    with self.condition:self.frames.pop(entry['serial'], None)
            cameras.append(camera)
        if publish_status:
            self._write(self.status_path, json.dumps(dict(owner_pid=self.pid, updated_monotonic=time.monotonic(),
                                                         stream_port=self.server.server_address[1] if self.server else None,
                                                         cameras=cameras)).encode())
            self.last_status = now

    def _run(self):
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                try:
                    self._publish()
                except Exception:
                    logging.exception('Camera preview update failed')
                self.stop.wait(max(.001, 1/60-(time.monotonic()-started)))
        finally:
            if self.server:
                self.server.shutdown()
                self.server.server_close()
            for stream in self.owned:
                try:
                    stream.close()
                except Exception:
                    logging.exception('Preview-only camera cleanup failed')
            for path in self.files:
                if path == self.status_path:
                    try:
                        if json.loads(path.read_text()).get('owner_pid') != self.pid:
                            continue
                    except (OSError, ValueError):
                        continue
                path.unlink(missing_ok=True)

    def close(self):
        self.stop.set()
        with self.condition:self.condition.notify_all()
        self.thread.join(timeout=5)


class PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        preview = self.server.preview
        serial = parse_qs(url.query).get('camera', [''])[0]
        if url.path != '/stream.mjpg' or serial not in {e['serial'] for e in preview.entries}:
            self.send_error(404)
            return
        self.connection.settimeout(2)
        self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Connection', 'close')
        self.end_headers()
        self.close_connection = True
        previous = None
        last_sent = time.monotonic()
        try:
            while not preview.stop.is_set():
                with preview.condition:
                    preview.condition.wait_for(lambda: preview.stop.is_set() or
                        (preview.frames.get(serial) is not None and preview.frames[serial][0] != previous
                         and time.monotonic()-preview.frames[serial][1] <= .5), timeout=.5)
                    current = preview.frames.get(serial)
                if preview.stop.is_set():break
                if time.monotonic()-last_sent > 3:break
                if current is None or current[0] == previous or time.monotonic()-current[1] > .5:continue
                previous, _, jpeg = current
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+
                                 b'\r\nX-Frame-Id: '+str(previous).encode()+b'\r\n\r\n'+jpeg+b'\r\n')
                self.wfile.flush()
                last_sent = time.monotonic()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def log_message(self, *args):
        pass
